from app import run_simulation
print("Baseline check (Features OFF)")
res_base = run_simulation(1500, 30, 0, 0.90, 'Constant')
print(f"RMSE: {res_base['RMSE']:.3f}, Range: {res_base['range_est']:.1f}")

print("\nFeatures ON check (Eff Map)")
res_eff = run_simulation(1500, 30, 0, 0.90, 'Constant', enable_eff_map=True)
print(f"RMSE: {res_eff['RMSE']:.3f}, Range: {res_eff['range_est']:.1f}")

print("\nFeatures ON check (Thermal)")
res_thm = run_simulation(1500, 30, 0, 0.90, 'Constant', enable_thermal=True)
print(f"RMSE: {res_thm['RMSE']:.3f}, Range: {res_thm['range_est']:.1f}, Final T: {res_thm['T_batt'][-1]:.2f}C")
