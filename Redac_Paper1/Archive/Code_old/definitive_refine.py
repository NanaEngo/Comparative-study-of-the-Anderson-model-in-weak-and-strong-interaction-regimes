
import json
import os
import re

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

def get_indent(line):
    match = re.match(r'^(\s*)', line)
    return match.group(1) if match else ''

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        
        # 1. Imports cleanup (ensure pandas and DATA_DIR are there and clean)
        if 'FIGURES_DIR = "Graphics/"' in "".join(source):
            clean_source = []
            for line in source:
                if 'import pandas as pd' in line or 'DATA_DIR =' in line or 'os.makedirs(DATA_DIR' in line:
                    continue
                clean_source.append(line)
            
            # Re-insert
            for i, line in enumerate(clean_source):
                if 'import numpy as np' in line:
                    clean_source.insert(i, "import pandas as pd\n")
                    clean_source.insert(i+1, "DATA_DIR = \"simulation_data/\"\n")
                    clean_source.insert(i+2, "os.makedirs(DATA_DIR, exist_ok=True)\n")
                    break
            cell['source'] = clean_source
            continue

        # Helper to clean a line for these top-level blocks
        def clean_plot_line(line):
            return line.lstrip()

        # 2. Hamiltonians
        if any('FMO_Site_Energies.pdf' in line for line in source):
            # Strip previous attempts and re-write
            temp_source = []
            skip = False
            for line in source:
                if "plt.figure(figsize=" in line or "plt.subplot(" in line or "pd.DataFrame(fmo_" in line:
                    continue
                temp_source.append(line)
            
            # Re-insert side-by-side
            for i, line in enumerate(temp_source):
                if "# Visualize the Hamiltonian structure" in line:
                    insert_idx = i + 1
                    temp_source.insert(insert_idx, "plt.figure(figsize=(12, 5))\n")
                    temp_source.insert(insert_idx+1, "plt.subplot(1, 2, 1)\n")
                    # Note: We need to find the imshow line. This is getting complex.
                    break
            # Let's just rewrite the whole cell content for this one
            cell['source'] = [
                "# Test the FMO Hamiltonian creation\n",
                "fmo_hamiltonian, fmo_energies = create_fmo_hamiltonian()\n",
                "print(f\"FMO Hamiltonian created with {fmo_hamiltonian.shape[0]} sites\")\n",
                "print(f\"Site energies (cm^-1): {fmo_energies}\")\n",
                "print(f\"Hamiltonian shape: {fmo_hamiltonian.shape}\")\n",
                "print(f\"Hamiltonian (first 4x4):\\n{fmo_hamiltonian[:4, :4]}\")\n",
                "\n",
                "# Visualize the Hamiltonian structure\n",
                "plt.figure(figsize=(12, 5))\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.imshow(fmo_hamiltonian, cmap='RdBu_r', aspect='equal', vmin=-150, vmax=150)\n",
                "plt.title('FMO Hamiltonian Matrix (cm⁻¹)')\n",
                "plt.colorbar(label='Energy (cm⁻¹)')\n",
                "plt.xlabel('Site Index')\n",
                "plt.ylabel('Site Index')\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.plot(fmo_energies, 'ro-', label='Site Energies')\n",
                "plt.title('FMO Site Energies')\n",
                "plt.xlabel('Site Index')\n",
                "plt.ylabel('Energy (cm⁻¹)')\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(FIGURES_DIR, \"FMO_Site_Energies.pdf\"), bbox_inches=\"tight\", dpi=300)\n",
                "# Data export\n",
                "pd.DataFrame(fmo_hamiltonian).to_csv(os.path.join(DATA_DIR, \"fmo_hamiltonian_data.csv\"))\n",
                "pd.DataFrame(fmo_energies).to_csv(os.path.join(DATA_DIR, \"fmo_site_energies.csv\"))\n",
                "plt.show()\n"
            ]
            continue

        # 3. Transmission comparison (Dusty)
        if any('PAR_Transmission__Clean_vs_Dusty_Conditions.pdf' in line for line in source):
            clean_source = []
            for line in source:
                line = line.replace("linestyle='--', linestyle='--'", "linestyle='--'")
                if 'plt.plot(wavelengths, T_dusty, linewidth=2,' in line and 'linestyle' not in line:
                    line = line.replace('linewidth=2,', "linewidth=2, linestyle='--',")
                if 'plt.legend()' in line:
                    line = "plt.legend(prop={'size': 8})\n"
                if 'pd.DataFrame' in line and 'solar_irradiance_data.csv' in line:
                    continue
                clean_source.append(line)
            
            # Add data export before show
            for i, line in enumerate(clean_source):
                if 'plt.savefig' in line:
                    clean_source.insert(i+1, "pd.DataFrame({'wavelength': wavelengths, 'irradiance': solar_irradiance}).to_csv(os.path.join(DATA_DIR, \"solar_irradiance_data.csv\"))\n")
                    break
            cell['source'] = clean_source
            continue

        # 4. Reactivity
        if any('Global_Reactivity_Indices.pdf' in line for line in source):
            # Full rewrite for safety
            cell['source'] = [
                "biodegradability_analyzer = BiodegradabilityAnalyzer(fmo_hamiltonian, n_electrons=14)\n",
                "f_plus, f_minus, f_zero = biodegradability_analyzer.calculate_fukui_functions()\n",
                "dual_descriptor = biodegradability_analyzer.calculate_dual_descriptor()\n",
                "indices = biodegradability_analyzer.calculate_global_reactivity_indices()\n",
                "\n",
                "plt.figure(figsize=(12, 10))\n",
                "sites = np.arange(len(f_plus))\n",
                "plt.subplot(2, 2, 1)\n",
                "plt.bar(sites, f_plus, alpha=0.7, label='f$^+$', color='red')\n",
                "plt.title('Fukui Function f$^+$ (Electrophilic)')\n",
                "plt.xlabel('Site Index')\n",
                "plt.ylabel('Reactivity')\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.subplot(2, 2, 2)\n",
                "plt.bar(sites, f_zero, alpha=0.7, label='f$^0$', color='green')\n",
                "plt.title('Fukui Function f$^0$ (Radical)')\n",
                "plt.xlabel('Site Index')\n",
                "plt.ylabel('Reactivity')\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.subplot(2, 2, 3)\n",
                "plt.bar(sites, dual_descriptor, alpha=0.7, label='\\\\Delta f', color='purple')\n",
                "plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n",
                "plt.title('Dual Descriptor')\n",
                "plt.xlabel('Site Index')\n",
                "plt.ylabel('Selectivity')\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.subplot(2, 2, 4)\n",
                "if np.isfinite(indices['e_homo']):\n",
                "    plt.plot(indices['e_homo'], 0, 'ro', label='HOMO', markersize=12)\n",
                "    plt.text(indices['e_homo'], 0.1, f'HOMO\\\\n{indices[\"e_homo\"]:.1f}', ha='center', va='bottom')\n",
                "if np.isfinite(indices['e_lumo']):\n",
                "    plt.plot(indices['e_lumo'], 0, 'bo', label='LUMO', markersize=12)\n",
                "    plt.text(indices['e_lumo'], 0.1, f'LUMO\\\\n{indices[\"e_lumo\"]:.1f}', ha='center', va='bottom')\n",
                "plt.title('FMOs')\n",
                "plt.xlabel('Energy (cm$^{-1}$)')\n",
                "plt.legend()\n",
                "plt.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(os.path.join(FIGURES_DIR, \"Global_Reactivity_Indices.pdf\"), bbox_inches='tight', dpi=300)\n",
                "# Data export\n",
                "pd.DataFrame({'site': sites, 'f_plus': f_plus, 'f_zero': f_zero, 'dual': dual_descriptor}).to_csv(os.path.join(DATA_DIR, \"reactivity_indices_data.csv\"))\n",
                "pd.Series(indices).to_csv(os.path.join(DATA_DIR, \"global_reactivity_summary.csv\"))\n",
                "plt.show()\n"
            ]
            continue

        # 5. Clean up other cells from duplicate exports and bad indentation
        clean_source = []
        skip_next = False
        for line in source:
            if '# Data export' in line or ('pd.DataFrame' in line and 'DATA_DIR' in line):
                continue
            # Remove leading 4 spaces from top-level savefig if present
            if 'plt.savefig' in line and line.startswith('    '):
                line = line.lstrip()
            clean_source.append(line)
        
        # Re-add clean exports
        final_source = []
        for line in clean_source:
            final_source.append(line)
            if 'plt.savefig' in line:
                indent = get_indent(line)
                if 'Quantum_Advantage_in_Energy_Transfer.pdf' in line:
                    final_source.append(f"{indent}pd.DataFrame({{'site': sites, 'pop_filtered': populations_filtered_avg[-1], 'pop_broad': populations_broad_avg[-1]}}).to_csv(os.path.join(DATA_DIR, \"energy_transfer_advantage_data.csv\"))\n")
                elif 'Pareto_Front__PCE_vs_ETR_Trade_off.pdf' in line:
                    final_source.append(f"{indent}pd.DataFrame({{'pce': results_df['pce'], 'etr': results_df['etr_mean']}}).to_csv(os.path.join(DATA_DIR, \"pareto_frontier_raw_data.csv\"))\n")
                elif 'ETR_Uncertainty_Distribution.pdf' in line:
                    final_source.append(f"{indent}pd.DataFrame({{'etr_samples': mc_results['etr']['samples']}}).to_csv(os.path.join(DATA_DIR, \"etr_uncertainty_samples.csv\"))\n")
        
        cell['source'] = final_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Definitive figure refinements and data export applied.")
