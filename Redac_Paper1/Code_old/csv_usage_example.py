"""
Example usage of the CSV storage functions in the quantum coherence agrivoltaics notebook.

This script shows how to use the CSV storage functionality that has been added to the notebook.
"""

# Create the figures directory if it doesn't exist
import os
os.makedirs("figures", exist_ok=True)

# Example usage of CSV functions after they've been added to the notebook:
print("The following functions are now available in the notebook:")

print("\n1. save_simulation_data_to_csv(time_points, populations, coherences, qfi_values, etr_values, filename_prefix)")
print("   - Saves quantum dynamics simulation data to CSV files in the figures/ folder")

print("\n2. save_optimization_results_to_csv(optimization_params, optimization_results, filename)")
print("   - Saves optimization results to CSV files in the figures/ folder")

print("\n3. save_robustness_analysis_to_csv(robustness_data, temperatures, disorder_strengths, filename)")
print("   - Saves robustness analysis results to CSV files in the figures/ folder")

print("\n4. save_spectral_data_to_csv(wavelengths, solar_irradiance, transmission_funcs, filename)")
print("   - Saves spectral data to CSV files in the figures/ folder")

print("\n5. save_all_figures_to_folder(figures_dict, folder)")
print("   - Saves all matplotlib figures to the specified folder (defaults to figures/)")

print("\nExample usage in the notebook:")
print("   # After running quantum dynamics simulation")
print("   save_simulation_data_to_csv(time_points, populations, coherences, qfi_values, etr_values, 'my_simulation')")
print("   # This will create figures/my_simulation_dynamics.csv and figures/my_simulation_summary.csv")

print("\nAll figures are automatically saved to the figures/ folder when using save_all_figures_to_folder().")
