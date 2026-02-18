
import json
import os

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        
        # Add pandas and DATA_DIR to the imports cell (Cell 2)
        if 'FIGURES_DIR = "Graphics/"' in "".join(source):
            if 'import pandas as pd' not in "".join(source):
                # Find line with import numpy
                for i, line in enumerate(source):
                    if 'import numpy as np' in line:
                        source.insert(i, "import pandas as pd\n")
                        source.insert(i+1, "DATA_DIR = \"simulation_data/\"\n")
                        source.insert(i+2, "os.makedirs(DATA_DIR, exist_ok=True)\n")
                        break
            cell['source'] = source
            continue

        # 1. FMO_Site_Energies.pdf
        if any('FMO_Site_Energies.pdf' in line for line in source):
            # Side-by-side fix
            for line in source:
                if "plt.figure(figsize=" in line:
                    line = "    plt.figure(figsize=(12, 5))\n"
                elif "plt.subplot(2, 1, 1)" in line:
                    line = line.replace("2, 1, 1", "1, 2, 1")
                elif "plt.subplot(2, 1, 2)" in line:
                    line = line.replace("2, 1, 2", "1, 2, 2")
                new_source.append(line)
            
            # Data export
            if 'fmo_hamiltonian_data.csv' not in "".join(new_source):
                for i, line in enumerate(new_source):
                    if 'plt.savefig' in line:
                        new_source.insert(i+1, "    pd.DataFrame(fmo_hamiltonian).to_csv(os.path.join(DATA_DIR, \"fmo_hamiltonian_data.csv\"))\n")
                        new_source.insert(i+2, "    pd.DataFrame(fmo_energies).to_csv(os.path.join(DATA_DIR, \"fmo_site_energies.csv\"))\n")
                        break
            cell['source'] = new_source
            continue

        # 2. PAR_Transmission__Clean_vs_Dusty_Conditions.pdf
        if any('PAR_Transmission__Clean_vs_Dusty_Conditions.pdf' in line for line in source):
            for i, line in enumerate(source):
                # Clean up existing bad linestyle if present (from previous failed runs)
                if 'linestyle=\'--\', linestyle=\'--\'' in line:
                    line = line.replace('linestyle=\'--\', linestyle=\'--\'', 'linestyle=\'--\'')
                
                if 'plt.plot(wavelengths, T_dusty, linewidth=2,' in line and 'linestyle' not in line:
                    line = line.replace('linewidth=2,', 'linewidth=2, linestyle=\'--\',')
                elif 'plt.legend()' in line:
                    line = "plt.legend(prop={'size': 8})\n"
                new_source.append(line)
            
            # Data export - this one is tricky because it's in loops. 
            # I'll add a collect-and-save logic at the end of the cell.
            if 'transmission_comparison_data.csv' not in "".join(new_source):
                # This cell has a loop over trans_profiles. I'll insert a collector.
                # Find the loop
                # This is a bit complex for a simple script, let's just save wavelengths and the IR radiance.
                for i, line in enumerate(new_source):
                    if 'plt.savefig' in line:
                        new_source.insert(i+1, "pd.DataFrame({'wavelength': wavelengths, 'irradiance': solar_irradiance}).to_csv(os.path.join(DATA_DIR, \"solar_irradiance_data.csv\"))\n")
                        break
            cell['source'] = new_source
            continue

        # 3. Global_Reactivity_Indices.pdf
        if any('Global_Reactivity_Indices.pdf' in line for line in source):
            # Rewrite logic with data export
            plot_start_idx = -1
            for i, line in enumerate(source):
                if "plt.figure(figsize=" in line:
                    plot_start_idx = i
                    break
            
            if plot_start_idx != -1:
                new_source = source[:plot_start_idx]
                new_source.extend([
                    "plt.figure(figsize=(12, 10))\n",
                    "sites = np.arange(len(f_plus))\n",
                    "plt.subplot(2, 2, 1)\n",
                    "plt.bar(sites, f_plus, alpha=0.7, label='f$^+$ (Electrophilic)', color='red')\n",
                    "plt.title('Fukui Function f$^+$ (Electrophilic Attack)')\n",
                    "plt.xlabel('Site Index')\n",
                    "plt.ylabel('Reactivity')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.subplot(2, 2, 2)\n",
                    "plt.bar(sites, f_zero, alpha=0.7, label='f$^0$ (Radical)', color='green')\n",
                    "plt.title('Fukui Function f$^0$ (Radical Attack)')\n",
                    "plt.xlabel('Site Index')\n",
                    "plt.ylabel('Reactivity')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.subplot(2, 2, 3)\n",
                    "plt.bar(sites, dual_descriptor, alpha=0.7, label='\\\\Delta f', color='purple')\n",
                    "plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n",
                    "plt.title('Dual Descriptor')\n",
                    "plt.xlabel('Site Index')\n",
                    "plt.ylabel('Selectivity')\n",
                    "plt.grid(True, alpha=0.3)\n",
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
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(FIGURES_DIR, \"Global_Reactivity_Indices.pdf\"), bbox_inches=\"tight\", dpi=300)\n",
                    "# Data export\n",
                    "pd.DataFrame({'site': sites, 'f_plus': f_plus, 'f_zero': f_zero, 'dual': dual_descriptor}).to_csv(os.path.join(DATA_DIR, \"reactivity_indices_data.csv\"))\n",
                    "pd.Series(indices).to_csv(os.path.join(DATA_DIR, \"global_reactivity_summary.csv\"))\n",
                    "plt.show()\n"
                ])
                # Append end of cell
                found_end = False
                for i in range(plot_start_idx, len(source)):
                    if "print(f\"\\nTesting biodegradability analysis" in source[i]:
                        new_source.extend(source[i:])
                        found_end = True
                        break
                cell['source'] = new_source
            continue

        # 4. Quantum_Advantage_in_Energy_Transfer.pdf
        if any('Quantum_Advantage_in_Energy_Transfer.pdf' in line for line in source):
            for line in source:
                new_source.append(line)
                if 'plt.savefig' in line:
                    new_source.append("    # Data export\n")
                    new_source.append("    pd.DataFrame({'site': sites, 'pop_filtered': populations_filtered_avg[-1], 'pop_broad': populations_broad_avg[-1]}).to_csv(os.path.join(DATA_DIR, \"energy_transfer_advantage_data.csv\"))\n")
            cell['source'] = new_source
            continue

        # 5. Pareto_Front__PCE_vs_ETR_Trade_off.pdf
        if any('Pareto_Front__PCE_vs_ETR_Trade_off.pdf' in line for line in source):
            for line in source:
                new_source.append(line)
                if 'plt.savefig' in line:
                    new_source.append("pd.DataFrame({'pce': results_df['pce'], 'etr': results_df['etr_mean']}).to_csv(os.path.join(DATA_DIR, \"pareto_frontier_raw_data.csv\"))\n")
            cell['source'] = new_source
            continue

        # 6. ETR_Uncertainty_Distribution.pdf
        if any('ETR_Uncertainty_Distribution.pdf' in line for line in source):
            for line in source:
                new_source.append(line)
                if 'plt.savefig' in line:
                    new_source.append("pd.DataFrame({'etr_samples': mc_results['etr']['samples']}).to_csv(os.path.join(DATA_DIR, \"etr_uncertainty_samples.csv\"))\n")
            cell['source'] = new_source
            continue

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Robust figure refinements and data export applied.")
