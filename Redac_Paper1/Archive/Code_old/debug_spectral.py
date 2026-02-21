import numpy as np
import matplotlib.pyplot as plt
from quantum_agrivoltaics_simulations import SpectralOptimizer

# Create a spectral optimizer instance to debug
so = SpectralOptimizer()

print("Debugging spectral optimization components:")

# Check the solar spectrum
print(f"Energy range: {so.E_range[:5]}... to ...{so.E_range[-5:]}")
print(f"Solar spectrum values at start: {so.solar_spec[:5]}")
print(f"Solar spectrum values at end: {so.solar_spec[-5:]}")
print(f"Total solar integral: {np.trapezoid(so.solar_spec, dx=so.E_range[1]-so.E_range[0])}")

# Check quantum response functions
R_opv = so.opv_quantum_response(so.E_range, bandgap=1.4, max_efficiency=0.8)
R_psu = so.psu_quantum_response(so.E_range)
print(f"OPV response integral: {np.trapezoid(R_opv)}")
print(f"PSU response integral: {np.trapezoid(R_psu)}")

# Test with some sample parameters
test_params = [(1.8, 0.3, 0.5), (2.2, 0.4, 0.6), (2.8, 0.2, 0.4)]  # 3 layers
T_test = so.multi_layer_transmission(so.E_range, test_params)
print(f"Sample transmission - min: {np.min(T_test)}, max: {np.max(T_test)}, mean: {np.mean(T_test)}")

# Calculate PCE and ETR with these parameters
pce_test = so.calculate_pce(T_test)
etr_test = so.calculate_etr(T_test)
print(f"Test PCE: {pce_test}")
print(f"Test ETR: {etr_test}")

# Check what happens with the objective function
test_flat_params = [1.8, 0.3, 0.5, 2.2, 0.4, 0.6, 2.8, 0.2, 0.4]
objective_val = so.spce_objective(test_flat_params)
print(f"Test objective function value: {objective_val}")

# Also check what happens with a completely transparent layer (T=1 everywhere)
T_all_transparent = np.ones_like(so.E_range)
pce_all_transparent = so.calculate_pce(T_all_transparent)
etr_all_transparent = so.calculate_etr(T_all_transparent)
print(f"All transparent - PCE: {pce_all_transparent}, ETR: {etr_all_transparent}")

# And with a completely blocking layer (T=0 everywhere)
T_all_blocking = np.zeros_like(so.E_range)
pce_all_blocking = so.calculate_pce(T_all_blocking)
etr_all_blocking = so.calculate_etr(T_all_blocking)
print(f"All blocking - PCE: {pce_all_blocking}, ETR: {etr_all_blocking}")