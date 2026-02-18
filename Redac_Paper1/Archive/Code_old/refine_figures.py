
import json
import os

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_to_next_if = False
        
        # 1. FMO_Site_Energies.pdf
        if any('FMO_Site_Energies.pdf' in line for line in source):
            for line in source:
                if "plt.figure(figsize=(10, 8))" in line:
                    line = line.replace("(10, 8)", "(12, 5)")
                elif "plt.subplot(2, 1, 1)" in line:
                    line = line.replace("2, 1, 1", "1, 2, 1")
                elif "plt.subplot(2, 1, 2)" in line:
                    line = line.replace("2, 1, 2", "1, 2, 2")
                new_source.append(line)
            cell['source'] = new_source
            continue

        # 2. PAR_Transmission__Clean_vs_Dusty_Conditions.pdf
        if any('PAR_Transmission__Clean_vs_Dusty_Conditions.pdf' in line for line in source):
            for line in source:
                if 'plt.plot(wavelengths, T_dusty, linewidth=2,' in line:
                    line = line.replace('linewidth=2,', 'linewidth=2, linestyle=\'--\',')
                elif 'plt.legend()' in line:
                    line = line.replace('plt.legend()', 'plt.legend(prop={\'size\': 8})')
                new_source.append(line)
            cell['source'] = new_source
            continue

        # 3. Global_Reactivity_Indices.pdf
        if any('Global_Reactivity_Indices.pdf' in line for line in source):
            # We want to remove subplots 2, 3, 5(if no LUMO), 6
            # Actually, let's rewrite the plotting section for this cell entirely for simplicity
            
            # Find the start of the plotting section
            plot_start_idx = -1
            for i, line in enumerate(source):
                if "plt.figure(figsize=(15, 10))" in line:
                    plot_start_idx = i
                    break
            
            if plot_start_idx != -1:
                # Keep everything before the plot
                new_source = source[:plot_start_idx]
                
                # New plotting logic: 2x2 grid
                new_source.extend([
                    "plt.figure(figsize=(12, 10))\n",
                    "\n",
                    "sites = np.arange(len(f_plus))\n",
                    "# Plot 1: f+\n",
                    "plt.subplot(2, 2, 1)\n",
                    "plt.bar(sites, f_plus, alpha=0.7, label='f$^+$ (Electrophilic)', color='red')\n",
                    "plt.title('Fukui Function f$^+$ (Electrophilic Attack)')\n",
                    "plt.xlabel('Site Index')\n",
                    "plt.ylabel('Reactivity')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 2: f0\n",
                    "plt.subplot(2, 2, 2)\n",
                    "plt.bar(sites, f_zero, alpha=0.7, label='f$^0$ (Radical)', color='green')\n",
                    "plt.title('Fukui Function f$^0$ (Radical Attack)')\n",
                    "plt.xlabel('Site Index')\n",
                    "plt.ylabel('Reactivity')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 3: Dual Descriptor\n",
                    "plt.subplot(2, 2, 3)\n",
                    "plt.bar(sites, dual_descriptor, alpha=0.7, label='\\\\Delta f (Dual Descriptor)', color='purple')\n",
                    "plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n",
                    "plt.title('Dual Descriptor \\\\Delta f = f$^+$ - f$^-$')\n",
                    "plt.xlabel('Site Index')\n",
                    "plt.ylabel('Selectivity')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 4: FMOs\n",
                    "plt.subplot(2, 2, 4)\n",
                    "if np.isfinite(indices['e_homo']):\n",
                    "    plt.plot(indices['e_homo'], 0, 'ro', label='HOMO', markersize=12)\n",
                    "    plt.text(indices['e_homo'], 0.1, f'HOMO\\\\n{indices[\"e_homo\"]:.1f}', ha='center', va='bottom')\n",
                    "if np.isfinite(indices['e_lumo']):\n",
                    "    plt.plot(indices['e_lumo'], 0, 'bo', label='LUMO', markersize=12)\n",
                    "    plt.text(indices['e_lumo'], 0.1, f'LUMO\\\\n{indices[\"e_lumo\"]:.1f}', ha='center', va='bottom')\n",
                    "plt.title('Frontier Molecular Orbitals')\n",
                    "plt.xlabel('Energy (cm$^{-1}$)')\n",
                    "plt.ylabel('')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(os.path.join(FIGURES_DIR, \"Global_Reactivity_Indices.pdf\"), bbox_inches=\"tight\", dpi=300)\n",
                    "plt.show()\n"
                ])
                
                # Check for any remaining lines after show() to capture the end of the cell
                for i in range(plot_start_idx, len(source)):
                    if "print(f\"\\nTesting biodegradability analysis" in source[i]:
                        new_source.extend(source[i:])
                        break
                
                cell['source'] = new_source
            continue

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Figure refinements applied.")
