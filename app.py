import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

st.set_page_config(page_title="EV Simulator with Accuracy Validation", layout="wide")

def run_simulation(m, Capacity_kWh, slope_deg, initial_SOC, cycleType, csv_data=None, enable_eff_map=False, enable_thermal=False):
    # Base Constants (Do NOT modify these)
    Cd = 0.29
    A = 2.2
    rho = 1.2
    Cr = 0.015
    g = 9.81
    V_nom = 350
    R_internal_base = 0.05
    eta_base = 0.92
    regen_eff = 0.65
    
    # Advanced Model Constants
    r_wheel = 0.3      # m
    gear_ratio = 10.0
    T_ambient = 25.0   # °C
    m_batt = 100.0     # kg (reduced to show faster heating on short cycles)
    Cp = 600.0         # J/(kg*K)
    h_lumped = 5.0     # W/K (reduced cooling to allow heat buildup)
    alpha = 0.01       # 1/K (increased temp coefficient for resistance)
    
    theta = slope_deg * np.pi / 180
    Q = (Capacity_kWh * 1000 * 3600) / V_nom
    dt = 0.1
    
    # Drive Cycle Handling
    if cycleType == 'CSV Upload' and csv_data is not None:
        # Interpolate CSV to match our dt exactly
        t_max = csv_data['Time (s)'].max()
        t = np.arange(0, t_max + dt, dt)
        v_ref = np.interp(t, csv_data['Time (s)'].values, csv_data['Speed (m/s)'].values)
        v_ref = np.maximum(v_ref, 0)
    else:
        t = np.arange(0, 600 + dt, dt)
        n = len(t)
        if cycleType == 'Constant':
            v_ref = 20 * np.ones(n)
        elif cycleType == 'City':
            v_ref = 15 + 8 * np.sin(0.05 * t)
        elif cycleType == 'Highway':
            v_ref = 25 + 5 * np.sin(0.02 * t)
        elif cycleType == 'UDDS':
            v_ref = 10 + 10 * np.sin(0.05 * t) + 5 * np.sin(0.1 * t)
            v_ref = np.maximum(v_ref, 0)
        elif cycleType == 'HWFET':
            v_ref = 22 + 6 * np.sin(0.02 * t) + 2 * np.sin(0.05 * t)
            v_ref = np.maximum(v_ref, 0)
        elif cycleType == 'WLTP':
            v_ref = 15 + 15 * np.sin(0.01 * t) + 8 * np.sin(0.03 * t)
            v_ref = np.maximum(v_ref, 0)
        else:
            v_ref = 20 * np.ones(n)
            
    n = len(t)
    
    # Initialize arrays
    v = np.zeros(n)
    SOC_vec = np.zeros(n)
    P_vec = np.zeros(n)
    T_batt = np.zeros(n)
    
    SOC = initial_SOC
    SOC_vec[0] = SOC * 100
    T_batt[0] = T_ambient
    
    energy_elec = 0.0
    energy_mech = 0.0
    distance = 0.0
    
    # Synthetic Motor Efficiency Map (simplified 2D mesh)
    if enable_eff_map:
        w_points = np.linspace(0, 1500, 10)  # rad/s
        T_points = np.linspace(0, 500, 10)   # Nm
        # Provide lower efficiency at low speed/torque, peaking around mid-range
        eff_grid = 0.85 + 0.10 * np.sin(w_points / 1500 * np.pi)[:, None] * np.sin(T_points / 500 * np.pi)
        eff_grid = np.clip(eff_grid, 0.70, 0.96)
        # scipy.interpolate.interp2d is deprecated in newer SciPy, but we use it via old API or RegularGridInterpolator if preferred
        # Since requirements specifies older setups or standard scipy, interp2d works (x, y, z)
        try:
            from scipy.interpolate import RegularGridInterpolator
            eff_map_func = RegularGridInterpolator((w_points, T_points), eff_grid, bounds_error=False, fill_value=0.85)
            use_rgi = True
        except ImportError:
            eff_map_func = interp2d(T_points, w_points, eff_grid, kind='linear')
            use_rgi = False
            
    for k in range(1, n):
        # 1. Controller Logic (Unchanged)
        error = v_ref[k-1] - v[k-1]
        Kp = 700
        F_trac = Kp * error
        F_trac = max(min(F_trac, 6000), -6000)
        
        # 2. Physics Equations (Unchanged)
        F_drag = 0.5 * rho * A * Cd * v[k-1]**2
        F_roll = m * g * Cr * np.cos(theta)
        F_slope = m * g * np.sin(theta)
        
        F_net = F_trac - F_drag - F_roll - F_slope
        a = F_net / m
        
        v[k] = v[k-1] + a * dt
        if v[k] < 0:
            v[k] = 0
            
        P_mech = F_trac * v[k]
        energy_mech += P_mech * dt
        
        # 3. Efficiency Map Logic (Modular Add)
        if enable_eff_map and P_mech > 0:
            w_rad_s = v[k] * gear_ratio / r_wheel
            T_nm = abs(F_trac * r_wheel / gear_ratio)
            # Clip values within bounds to be safe
            w_rad_s = min(w_rad_s, 1499)
            T_nm = min(T_nm, 499)
            if use_rgi:
                eta_dynamic = float(eff_map_func([[w_rad_s, T_nm]])[0])
            else:
                eta_dynamic = float(eff_map_func(T_nm, w_rad_s)[0])
        else:
            eta_dynamic = eta_base
            
        if P_mech < 0:
            P_elec = P_mech * regen_eff
        else:
            P_elec = P_mech / eta_dynamic
            
        # 4. Thermal Logic applied to R_internal (Modular Add)
        if enable_thermal:
            R_actual = R_internal_base * (1 + alpha * (T_batt[k-1] - T_ambient))
        else:
            R_actual = R_internal_base
            
        # 5. Power Balance & SOC (Unchanged equations, dynamic R_actual)
        I = P_elec / V_nom
        V_actual = V_nom - I * R_actual
        P_actual = V_actual * I
        
        P_vec[k] = P_actual / 1000  # kW
        
        SOC = SOC - (I * dt) / Q
        SOC_vec[k] = SOC * 100
        
        # 6. Thermal Integration (Modular Add)
        if enable_thermal:
            P_loss = (I**2) * R_actual
            dT_batt = (P_loss - h_lumped * (T_batt[k-1] - T_ambient)) / (m_batt * Cp)
            T_batt[k] = T_batt[k-1] + dT_batt * dt
        else:
            T_batt[k] = T_ambient
            
        energy_elec += P_actual * dt
        distance += v[k] * dt
        
    # Accuracy Metrics (Unchanged)
    RMSE = np.sqrt(np.mean((v - v_ref)**2))
    
    SOC_drop = initial_SOC - SOC
    energy_from_SOC = SOC_drop * Capacity_kWh
    
    energy_calc_kWh = energy_elec / (1000 * 3600)
    energy_balance_error = abs(energy_from_SOC - energy_calc_kWh)
    
    if energy_elec != 0:
        avg_eff = energy_mech / energy_elec
    else:
        avg_eff = 0
        
    distance_km = distance / 1000
    
    if distance_km > 0:
        Wh_per_km = (energy_calc_kWh * 1000) / distance_km
        usable_capacity_kWh = Capacity_kWh * initial_SOC
        range_est = (usable_capacity_kWh * 1000) / Wh_per_km
    else:
        Wh_per_km = 0
        range_est = 0
        
    return {
        't': t,
        'v': v,
        'v_ref': v_ref,
        'SOC_vec': SOC_vec,
        'P_vec': P_vec,
        'T_batt': T_batt,
        'RMSE': RMSE,
        'energy_balance_error': energy_balance_error,
        'avg_eff': avg_eff,
        'range_est': range_est,
        'Wh_per_km': Wh_per_km
    }

# UI Layout
st.title("EV Simulator with Accuracy Validation & Advanced Features")

st.sidebar.header("Vehicle Inputs")
massField = st.sidebar.number_input("Vehicle Mass (kg)", value=1500)
batteryField = st.sidebar.number_input("Battery Capacity (kWh)", value=30)
slopeField = st.sidebar.number_input("Road Slope (deg)", value=0.0)
socField = st.sidebar.slider("Initial SOC (%)", min_value=0.0, max_value=100.0, value=90.0)

st.sidebar.header("Drive Cycle Customization")
cycleDrop = st.sidebar.selectbox("Drive Cycle", ["Constant", "City", "Highway", "UDDS", "HWFET", "WLTP", "CSV Upload"])
csv_upload = None
if cycleDrop == "CSV Upload":
    csv_upload = st.sidebar.file_uploader("Upload Drive Cycle CSV (Time (s), Speed (m/s))", type=['csv'])

st.sidebar.header("Advanced Features")
eff_map_toggle = st.sidebar.checkbox("Enable Motor Efficiency Map", value=False)
thermal_toggle = st.sidebar.checkbox("Enable Battery Thermal Model", value=False)

runBtn = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

if runBtn:
    # Error checking for CSV
    if cycleDrop == "CSV Upload" and csv_upload is None:
        st.error("Please upload a CSV file to run the CSV Drive Cycle.")
        st.stop()
        
    csv_df = None
    if csv_upload is not None:
        try:
            csv_df = pd.read_csv(csv_upload)
            if 'Time (s)' not in csv_df.columns or 'Speed (m/s)' not in csv_df.columns:
                st.error("CSV must contain columns 'Time (s)' and 'Speed (m/s)'.")
                st.stop()
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
            st.stop()

    res = run_simulation(
        m=massField, 
        Capacity_kWh=batteryField, 
        slope_deg=slopeField, 
        initial_SOC=socField/100.0, 
        cycleType=cycleDrop,
        csv_data=csv_df,
        enable_eff_map=eff_map_toggle,
        enable_thermal=thermal_toggle
    )
    
    # Custom CSS to style metrics
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    div[data-testid="metric-container"] label {
        color: #aaa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("Accuracy Metrics & Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Speed RMSE", f"{res['RMSE']:.3f} m/s")
    col2.metric("Energy Bal. Error", f"{res['energy_balance_error']:.3f} kWh")
    col3.metric("Average Efficiency", f"{res['avg_eff']:.2f}")
    col4.metric("Estimated Range", f"{res['range_est']:.1f} km")
    col5.metric("Consumption", f"{res['Wh_per_km']:.1f} Wh/km")
    
    st.markdown("---")
    
    # Use standard matplotlib styles suitable for dark/light mode
    plt.style.use('dark_background')
    
    # Row 1 Plots
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(res['t'], res['v'], color='#00a1ff', linewidth=2, label='Vehicle')
    ax1.plot(res['t'], res['v_ref'], color='#ff4d4d', linestyle='--', linewidth=2, label='Reference')
    ax1.set_title("Speed Tracking", fontsize=14, loc='left')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.plot(res['t'], res['SOC_vec'], color='#00ff9d', linewidth=2)
    ax2.set_title("Battery SOC", fontsize=14, loc='left')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("SOC (%)")
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    st.pyplot(fig1)
    
    # Row 2 Plots
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax3.plot(res['t'], res['P_vec'], color='#ff9900', linewidth=2)
    ax3.set_title("Motor Power (kW)", fontsize=14, loc='left')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Power (kW)")
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    cumsum_P = np.cumsum(res['P_vec']) * 0.1 / 3600
    ax4.plot(res['t'], cumsum_P, color='#cc00ff', linewidth=2)
    ax4.set_title("Cumulative Energy (kWh)", fontsize=14, loc='left')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Energy (kWh)")
    ax4.grid(True, alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    st.pyplot(fig2)
    
    # Optional Row 3: Thermal Plot
    if thermal_toggle:
        fig3, ax5 = plt.subplots(1, 1, figsize=(16, 5))
        ax5.plot(res['t'], res['T_batt'], color='#ff3333', linewidth=2)
        ax5.set_title("Battery Temperature (°C)", fontsize=14, loc='left')
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Temperature (°C)")
        ax5.grid(True, alpha=0.3)
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        st.pyplot(fig3)
