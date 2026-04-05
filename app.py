import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

st.set_page_config(page_title="EV Simulator with Accuracy Validation", layout="wide")

# Initialize session state for persistence
if 'res' not in st.session_state:
    st.session_state['res'] = None

def run_simulation(m, Capacity_kWh, slope_deg, initial_SOC, cycleType, csv_data=None, enable_eff_map=False, enable_thermal=False, P_aux_W=0, drivingMode='Normal', target_speed_kmh=50):
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
        v_target_ms = target_speed_kmh / 3.6
        
        # Smooth ramp-up (0 to 1 over 10 seconds)
        ramp = np.minimum(t / 10.0, 1.0)
        
        if cycleType == 'Constant':
            if drivingMode == 'City':
                # Lower average, frequent fluctuations
                v_ref = v_target_ms * (0.6 + 0.3 * np.sin(0.1 * t) + 0.1 * np.cos(0.25 * t))
            elif drivingMode == 'Highway':
                # Higher average, stable speed
                v_ref = v_target_ms * (1.0 + 0.03 * np.sin(0.015 * t))
            else: # Normal
                # Moderate speed, mixed behavior
                v_ref = v_target_ms * (0.85 + 0.1 * np.sin(0.05 * t) + 0.05 * np.cos(0.12 * t))
        elif cycleType == 'City':
            v_ref = v_target_ms * (0.6 + 0.3 * np.sin(0.08 * t))
        elif cycleType == 'Highway':
            v_ref = v_target_ms * (1.0 + 0.05 * np.sin(0.03 * t))
        elif cycleType == 'UDDS':
            v_ref = (10 + 10 * np.sin(0.05 * t) + 5 * np.sin(0.1 * t))
        elif cycleType == 'HWFET':
            v_ref = (22 + 6 * np.sin(0.02 * t) + 2 * np.sin(0.05 * t))
        elif cycleType == 'WLTP':
            v_ref = (15 + 15 * np.sin(0.01 * t) + 8 * np.sin(0.03 * t))
        else:
            v_ref = v_target_ms * np.ones(n)
            
        # Apply Ramp and ensure non-negative
        v_ref = v_ref * ramp
        v_ref = np.maximum(v_ref, 0)
            
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
        
        # 2. Physics Equations 
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
            P_elec_base = P_mech * regen_eff
        else:
            P_elec_base = P_mech / eta_dynamic
            
        # Add Constant Auxiliary Load (W)
        P_elec = P_elec_base + P_aux_W

            
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
        
    # Accuracy Metrics 
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
        'Wh_per_km': Wh_per_km,
        'm': m,
        'slope_deg': slope_deg,
        'P_aux_W': P_aux_W,
        'v_avg_ms': np.mean(v)
    }

def calculate_steady_state_wh_km(v_ms, m, slope_deg, aux_load_w, enable_eff_map=False):
    # Constants matching run_simulation
    Cd, A, rho, Cr, g, V_nom, eta_base = 0.29, 2.2, 1.2, 0.015, 9.81, 350, 0.92
    theta = slope_deg * np.pi / 180
    
    if v_ms <= 0.1: return 0
    
    F_drag = 0.5 * rho * A * Cd * v_ms**2
    F_roll = m * g * Cr * np.cos(theta)
    F_slope = m * g * np.sin(theta)
    F_trac = F_drag + F_roll + F_slope
    
    P_mech = F_trac * v_ms
    
    # Simplified efficiency (using base or rough map approximation for steady state)
    eta = eta_base 
    if enable_eff_map:
        # Simple parabolic approximation for steady-state peak efficiency
        # In a real map we'd interpolate, but for advisory we can use a representative curve
        eta = 0.7 + 0.25 * np.sin(min(v_ms / 30, 1) * np.pi / 2)
        
    P_elec = (P_mech / eta if P_mech > 0 else 0) + aux_load_w
    
    # Wh/km = (W * (1/3600) * 1000) / (m/s) = (P_elec / 3.6) / (v_ms * 3.6) = P_elec / (v_ms * 3.6)
    # Correct: Wh/km = (P_elec [W] * (1000 [m] / v_ms [m/s])) / 3600 [s/h] = P_elec / (v_ms * 3.6)
    wh_km = P_elec / (v_ms * 3.6)
    return max(wh_km, 0)


# UI Layout
st.title("EV Simulator with Accuracy Validation & Advanced Features")

st.sidebar.header("Vehicle Inputs")
massField = st.sidebar.number_input("Vehicle Mass (kg)", value=1500)
batteryField = st.sidebar.number_input("Battery Capacity (kWh)", value=30)
slopeField = st.sidebar.number_input("Road Slope (deg)", value=0.0)
socField = st.sidebar.slider("Initial SOC (%)", min_value=0.0, max_value=100.0, value=90.0)
auxField = st.sidebar.number_input("Auxiliary Load (W)", value=300, step=50)

st.sidebar.header("Eco-Advisory Override")
manualSpeed = st.sidebar.slider("Manual Average Speed (km/h)", 10, 120, 50)


st.sidebar.header("Drive Cycle Customization")
cycleDrop = st.sidebar.selectbox("Drive Cycle", ["Constant", "City", "Highway", "UDDS", "HWFET", "WLTP", "CSV Upload"])
drivingMode = st.sidebar.selectbox("Driving Mode", ["Normal", "City", "Highway"])
csv_upload = None
if cycleDrop == "CSV Upload":
    csv_upload = st.sidebar.file_uploader("Upload Drive Cycle CSV (Time (s), Speed (m/s))", type=['csv'])

eff_map_toggle = st.sidebar.checkbox("Enable Motor Efficiency Map", value=False)
thermal_toggle = st.sidebar.checkbox("Enable Battery Thermal Model", value=False)
show_dynamic_range = st.sidebar.checkbox("Show Dynamic Range Analysis", value=True)

runBtn = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

if runBtn:
    # Error checking for CSV ... (kept the same)
    csv_df = None
    if cycleDrop == "CSV Upload" and csv_upload is not None:
        try:
            csv_df = pd.read_csv(csv_upload)
            if 'Time (s)' not in csv_df.columns or 'Speed (m/s)' not in csv_df.columns:
                st.error("CSV must contain columns 'Time (s)' and 'Speed (m/s)'.")
                st.stop()
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
            st.stop()

    st.session_state['res'] = run_simulation(
        m=massField, 
        Capacity_kWh=batteryField, 
        slope_deg=slopeField, 
        initial_SOC=socField/100.0, 
        cycleType=cycleDrop,
        csv_data=csv_df,
        enable_eff_map=eff_map_toggle,
        enable_thermal=thermal_toggle,
        P_aux_W=auxField,
        drivingMode=drivingMode,
        target_speed_kmh=manualSpeed
    )

if st.session_state['res'] is not None:
    res = st.session_state['res']

    
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

    st.markdown("---")
    st.subheader("Eco-Speed Advisory Section")
    
    # Calculate Eco-Speed Curve
    v_speeds_kmh = np.linspace(10, 120, 100)
    v_speeds_ms = v_speeds_kmh / 3.6
    wh_km_curve = np.array([calculate_steady_state_wh_km(v, res['m'], res['slope_deg'], res['P_aux_W'], enable_eff_map=eff_map_toggle) for v in v_speeds_ms])
    
    # Range Curve: Range = Capacity / (Wh/km / 1000) = (Capacity * 1000) / Wh/km
    usable_kwh = batteryField * (socField / 100.0)
    range_curve = (usable_kwh * 1000) / wh_km_curve
    
    # Global Maximum Range
    global_max_idx = np.argmax(range_curve)
    v_max_range_kmh = v_speeds_kmh[global_max_idx]
    max_range_km = range_curve[global_max_idx]
    wh_km_min_global = wh_km_curve[global_max_idx]

    # Search Window Logic
    curr_v_avg_kmh = manualSpeed
    window = 0.15 * curr_v_avg_kmh # Default 15%
    
    if cycleDrop == 'UDDS': window = 10
    elif cycleDrop == 'HWFET': window = 15
    elif cycleDrop == 'WLTP': window = 20
    elif drivingMode == 'City': window = 10
    elif drivingMode == 'Highway': window = 15
    elif drivingMode == 'Normal': window = 20
    
    v_min_win = max(10, curr_v_avg_kmh - window)
    v_max_win = min(120, curr_v_avg_kmh + window)
    
    # Local Optimization within Window
    window_mask = (v_speeds_kmh >= v_min_win) & (v_speeds_kmh <= v_max_win)
    v_window = v_speeds_kmh[window_mask]
    wh_km_window = wh_km_curve[window_mask]
    range_window = range_curve[window_mask]
    
    opt_local_idx = np.argmin(wh_km_window)
    v_opt_kmh = v_window[opt_local_idx]
    wh_km_min = wh_km_window[opt_local_idx]
    range_nearby_opt = range_window[opt_local_idx]

    # Baseline Consumption
    actual_wh_km = calculate_steady_state_wh_km(curr_v_avg_kmh / 3.6, res['m'], res['slope_deg'], res['P_aux_W'], enable_eff_map=eff_map_toggle)
    range_current = (usable_kwh * 1000) / actual_wh_km
        
    # Visuals
    col_a1, col_a2 = st.columns([2, 1])
    
    with col_a1:
        fig_eco, ax_eco = plt.subplots(figsize=(10, 5))
        ax_eco.plot(v_speeds_kmh, wh_km_curve, color='#00ffcc', linewidth=3, label='Steady-State Consumption')
        ax_eco.scatter(v_opt_kmh, wh_km_min, color='gold', s=100, zorder=5, label=f'Nearby Optimal: {v_opt_kmh:.1f} km/h')
        ax_eco.axvline(curr_v_avg_kmh, color='#ff4d4d', linestyle='--', alpha=0.7, label=f'Current: {curr_v_avg_kmh} km/h')
        
        # Highlight Window
        ax_eco.axvspan(v_min_win, v_max_win, color='white', alpha=0.1, label='Optimization Window')

        ax_eco.set_title("Consumption vs Speed (Steady State)", fontsize=14, loc='left')
        ax_eco.set_xlabel("Speed (km/h)")
        ax_eco.set_ylabel("Consumption (Wh/km)")
        ax_eco.legend()
        ax_eco.grid(True, alpha=0.3)
        st.pyplot(fig_eco)
        
        if show_dynamic_range:
            fig_range, ax_range = plt.subplots(figsize=(10, 5))
            ax_range.plot(v_speeds_kmh, range_curve, color='#ff00ff', linewidth=3, label='Estimated Range')
            
            # Highlight 3 points
            # 🔴 Current speed
            ax_range.scatter(curr_v_avg_kmh, range_current, color='red', s=100, zorder=6, label=f'Current: {range_current:.1f} km')
            # 🟡 Nearby optimal speed
            ax_range.scatter(v_opt_kmh, range_nearby_opt, color='gold', s=100, zorder=6, label=f'Nearby Optimal: {range_nearby_opt:.1f} km')
            # 🟢 Maximum possible range (global)
            ax_range.scatter(v_max_range_kmh, max_range_km, color='lime', s=100, zorder=6, label=f'Max Range: {max_range_km:.1f} km')
            
            ax_range.set_title("Range vs Speed", fontsize=14, loc='left')
            ax_range.set_xlabel("Speed (km/h)")
            ax_range.set_ylabel("Range (km)")
            ax_range.legend()
            ax_range.grid(True, alpha=0.3)
            st.pyplot(fig_range)
            
    with col_a2:
        st.metric("Nearby Optimal Speed", f"{v_opt_kmh:.1f} km/h")
        st.metric("Min Consumption (Local)", f"{wh_km_min:.1f} Wh/km")
        
        # Mass-sensitive refined range calculation
        usable_kwh = batteryField * (socField / 100.0)
        refined_range = (usable_kwh * 1000) / wh_km_min
        st.metric("Refined Range", f"{refined_range:.1f} km")
        
        # Formula: ((Actual - Optimal) / Actual) * 100
        diff = actual_wh_km - wh_km_min
        if diff > 0:
            improvement = (diff / actual_wh_km) * 100
            st.success(f"Efficiency Improvement: **{improvement:.1f}%** possible by adjusting from {curr_v_avg_kmh} km/h to a nearby efficient range of {v_min_win:.0f}–{v_max_win:.0f} km/h.")
        else:
            st.info(f"Speed of {curr_v_avg_kmh} km/h is near or below peak efficiency speed.")
            
        st.write(f"**Baseline Consumption ({curr_v_avg_kmh} km/h):** {actual_wh_km:.1f} Wh/km")

        if show_dynamic_range:
            st.markdown("---")
            st.markdown("#### 🔵 Range Comparison Block")
            st.write(f"Current Speed → Range: **{range_current:.1f} km**")
            st.write(f"Nearby Optimal Speed → Range: **{range_nearby_opt:.1f} km**")
            st.write(f"Maximum Possible → Range: **{max_range_km:.1f} km**")
            
            st.markdown("---")
            efficiency_pc = (range_current / max_range_km) * 100
            st.markdown(f"#### 🟢 Efficiency Insight Line")
            st.write(f"Your current driving speed achieves ~**{efficiency_pc:.1f}%** of the maximum possible range.")

            # Energy Contribution Breakdown
            v_ms_curr = curr_v_avg_kmh / 3.6
            if v_ms_curr > 0:
                Cd, A, rho, Cr, g, eta_base = 0.29, 2.2, 1.2, 0.015, 9.81, 0.92
                eta_curr = 0.7 + 0.25 * np.sin(min(v_ms_curr / 30, 1) * np.pi / 2) if eff_map_toggle else eta_base
                
                f_drag_curr = 0.5 * rho * A * Cd * v_ms_curr**2
                f_roll_curr = res['m'] * g * Cr * np.cos(res['slope_deg'] * np.pi / 180)
                
                p_aero_curr = (f_drag_curr * v_ms_curr) / eta_curr
                p_roll_curr = (f_roll_curr * v_ms_curr) / eta_curr
                p_aux_curr = res['P_aux_W']
                
                p_total_curr = p_aero_curr + p_roll_curr + p_aux_curr
                if p_total_curr > 0:
                    aero_pct = (p_aero_curr / p_total_curr) * 100
                    roll_pct = (p_roll_curr / p_total_curr) * 100
                    aux_pct = (p_aux_curr / p_total_curr) * 100
                    
                    st.markdown("---")
                    st.subheader("🔍 Energy Contribution Breakdown")
                    st.write(f"🌬️ **Aerodynamic Loss:** {aero_pct:.1f}%")
                    st.write(f"🛞 **Rolling Resistance:** {roll_pct:.1f}%")
                    st.write(f"🔌 **Auxiliary Load:** {aux_pct:.1f}%")
                    st.info("“Higher speeds increase aerodynamic losses, reducing efficiency.”")


            
        st.write(f"**Note:** auxiliary load of {res['P_aux_W']}W and slope of {res['slope_deg']}° are factored into this advisory.")

