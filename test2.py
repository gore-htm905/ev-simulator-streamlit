from app import run_simulation

base = run_simulation(1500, 30, 0, 0.90, 'City')
eff = run_simulation(1500, 30, 0, 0.90, 'City', enable_eff_map=True)
thm = run_simulation(1500, 30, 0, 0.90, 'City', enable_thermal=True)
both = run_simulation(1500, 30, 0, 0.90, 'City', enable_eff_map=True, enable_thermal=True)

print("Baseline:")
print(f"Eff: {base['avg_eff']:.3f}, Range: {base['range_est']:.1f}, Cons: {base['Wh_per_km']:.1f}")

print("Eff Map:")
print(f"Eff: {eff['avg_eff']:.3f}, Range: {eff['range_est']:.1f}, Cons: {eff['Wh_per_km']:.1f}")

print("Thermal:")
print(f"Eff: {thm['avg_eff']:.3f}, Range: {thm['range_est']:.1f}, Cons: {thm['Wh_per_km']:.1f}")

print("Both:")
print(f"Eff: {both['avg_eff']:.3f}, Range: {both['range_est']:.1f}, Cons: {both['Wh_per_km']:.1f}")
