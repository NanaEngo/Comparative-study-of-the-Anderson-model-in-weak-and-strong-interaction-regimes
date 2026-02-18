
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
        
        # Add pandas and DATA_DIR to the imports cell
        if 'FIGURES_DIR = "Graphics/"' in "".join(source):
            if 'import pandas as pd' not in "".join(source):
                for i, line in enumerate(source):
                    if 'import numpy as np' in line:
                        indent = get_indent(line)
                        source.insert(i, f"{indent}import pandas as pd\n")
                        source.insert(i+1, f"{indent}DATA_DIR = \"simulation_data/\"\n")
                        source.insert(i+2, f"{indent}os.makedirs(DATA_DIR, exist_ok=True)\n")
                        break
            cell['source'] = source
            continue

        # 1. FMO_Site_Energies.pdf
        if any('FMO_Site_Energies.pdf' in line for line in source):
            for line in source:
                indent = get_indent(line)
                if "plt.figure(figsize=" in line:
                    line = f"{indent}plt.figure(figsize=(12, 5))\n"
                elif "plt.subplot(2, 1, 1)" in line:
                    line = line.replace("2, 1, 1", "1, 2, 1")
                elif "plt.subplot(2, 1, 2)" in line:
                    line = line.replace("2, 1, 2", "1, 2, 2")
                new_source.append(line)
                
                if 'plt.savefig' in line and 'fmo_hamiltonian_data.csv' not in "".join(source):
                    new_source.append(f"{indent}pd.DataFrame(fmo_hamiltonian).to_csv(os.path.join(DATA_DIR, \"fmo_hamiltonian_data.csv\"))\n")
                    new_source.append(f"{indent}pd.DataFrame(fmo_energies).to_csv(os.path.join(DATA_DIR, \"fmo_site_energies.csv\"))\n")
            cell['source'] = new_source
            continue

        # 2. PAR_Transmission__Clean_vs_Dusty_Conditions.pdf
        if any('PAR_Transmission__Clean_vs_Dusty_Conditions.pdf' in line for line in source):
            for line in source:
                indent = get_indent(line)
                if 'linestyle=\'--\', linestyle=\'--\'' in line:
                    line = line.replace('linestyle=\'--\', linestyle=\'--\'', 'linestyle=\'--\'')
                
                if 'plt.plot(wavelengths, T_dusty, linewidth=2,' in line and 'linestyle' not in line:
                    line = line.replace('linewidth=2,', 'linewidth=2, linestyle=\'--\',')
                elif 'plt.legend()' in line:
                    line = f"{indent}plt.legend(prop={{'size': 8}})\n"
                new_source.append(line)
                
                if 'plt.savefig' in line and 'solar_irradiance_data.csv' not in "".join(source):
                    new_source.append(f"{indent}pd.DataFrame({{'wavelength': wavelengths, 'irradiance': solar_irradiance}}).to_csv(os.path.join(DATA_DIR, \"solar_irradiance_data.csv\"))\n")
            cell['source'] = new_source
            continue

        # 3. Global_Reactivity_Indices.pdf
        if any('Global_Reactivity_Indices.pdf' in line for line in source):
            plot_start_idx = -1
            for i, line in enumerate(source):
                if "plt.figure(figsize=" in line:
                    plot_start_idx = i
                    indent = get_indent(line)
                    break
            
            if plot_start_idx != -1:
                new_source = source[:plot_start_idx]
                new_source.extend([
                    f"{indent}plt.figure(figsize=(12, 10))\n",
                    f"{indent}sites = np.arange(len(f_plus))\n",
                    f"{indent}plt.subplot(2, 2, 1)\n",
                    f"{indent}plt.bar(sites, f_plus, alpha=0.7, label='f$^+$ (Electrophilic)', color='red')\n",
                    f"{indent}plt.title('Fukui Function f$^+$ (Electrophilic Attack)')\n",
                    f"{indent}plt.xlabel('Site Index')\n",
                    f"{indent}plt.ylabel('Reactivity')\n",
                    f"{indent}plt.grid(True, alpha=0.3)\n",
                    f"{indent}plt.subplot(2, 2, 2)\n",
                    f"{indent}plt.bar(sites, f_zero, alpha=0.7, label='f$^0$ (Radical)', color='green')\n",
                    f"{indent}plt.title('Fukui Function f$^0$ (Radical Attack)')\n",
                    f"{indent}plt.xlabel('Site Index')\n",
                    f"{indent}plt.ylabel('Reactivity')\n",
                    f"{indent}plt.grid(True, alpha=0.3)\n",
                    f"{indent}plt.subplot(2, 2, 3)\n",
                    f"{indent}plt.bar(sites, dual_descriptor, alpha=0.7, label='\\\\Delta f', color='purple')\n",
                    f"{indent}plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n",
                    f"{indent}plt.title('Dual Descriptor')\n",
                    f"{indent}plt.xlabel('Site Index')\n",
                    f"{indent}plt.ylabel('Selectivity')\n",
                    f"{indent}plt.grid(True, alpha=0.3)\n",
                    f"{indent}plt.subplot(2, 2, 4)\n",
                    f"{indent}if np.isfinite(indices['e_homo']):\n",
                    f"{indent}    plt.plot(indices['e_homo'], 0, 'ro', label='HOMO', markersize=12)\n",
                    f"{indent}    plt.text(indices['e_homo'], 0.1, f'HOMO\\\\n{{indices[\"e_homo\"]:.1f}}', ha='center', va='bottom')\n",
                    f"{indent}if np.isfinite(indices['e_lumo']):\n",
                    f"{indent}    plt.plot(indices['e_lumo'], 0, 'bo', label='LUMO', markersize=12)\n",
                    f"{indent}    plt.text(indices['e_lumo'], 0.1, f'LUMO\\\\n{{indices[\"e_lumo\"]:.1f}}', ha='center', va='bottom')\n",
                    f"{indent}plt.title('FMOs')\n",
                    f"{indent}plt.xlabel('Energy (cm$^{{-1}}$)')\n",
                    f"{indent}plt.legend()\n",
                    f"{indent}plt.grid(True, alpha=0.3)\n",
                    f"{indent}plt.tight_layout()\n",
                    f"{indent}plt.savefig(os.path.join(FIGURES_DIR, \"Global_Reactivity_Indices.pdf\"), bbox_inches=\"tight\", dpi=300)\n",
                    f"{indent}# Data export\n",
                    f"{indent}pd.DataFrame({{'site': sites, 'f_plus': f_plus, 'f_zero': f_zero, 'dual': dual_descriptor}}).to_csv(os.path.join(DATA_DIR, \"reactivity_indices_data.csv\"))\n",
                    f"{indent}pd.Series(indices).to_csv(os.path.join(DATA_DIR, \"global_reactivity_summary.csv\"))\n",
                    f"{indent}plt.show()\n"
                ])
                for i in range(plot_start_idx, len(source)):
                    if "print(f\"\\nTesting biodegradability analysis" in source[i]:
                        new_source.extend(source[i:])
                        break
                cell['source'] = new_source
            continue

        # Data export for all other savefig calls
        temp_source = []
        modified = False
        for line in source:
            temp_source.append(line)
            if 'plt.savefig' in line:
                indent = get_indent(line)
                # Extract filename
                match = re.search(r'\"([^\"]+\.pdf)\"', line)
                if match:
                    filename = match.group(1).replace('.pdf', '_data.csv')
                    if filename not in "".join(source):
                        # For now, we'll just add placeholders or generic exports where possible
                        if 'Quantum_Advantage_in_Energy_Transfer.pdf' in line:
                            temp_source.append(f"{indent}pd.DataFrame({{'site': sites, 'pop_filtered': populations_filtered_avg[-1], 'pop_broad': populations_broad_avg[-1]}}).to_csv(os.path.join(DATA_DIR, \"{filename}\"))\n")
                            modified = True
                        elif 'Pareto_Front__PCE_vs_ETR_Trade_off.pdf' in line:
                            temp_source.append(f"{indent}pd.DataFrame({{'pce': results_df['pce'], 'etr': results_df['etr_mean']}}).to_csv(os.path.join(DATA_DIR, \"{filename}\"))\n")
                            modified = True
                        elif 'ETR_Uncertainty_Distribution.pdf' in line:
                            temp_source.append(f"{indent}pd.DataFrame({{'etr_samples': mc_results['etr']['samples']}}).to_csv(os.path.join(DATA_DIR, \"{filename}\"))\n")
                            modified = True
        if modified:
            cell['source'] = temp_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Smart figure refinements and data export applied.")
