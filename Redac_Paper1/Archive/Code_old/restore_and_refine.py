#!/usr/bin/env python3
"""
Restore the missing BiodegradabilityAnalyzer class from the archived notebook
and apply figure refinements + data exports to the current notebook.
"""

import json
import os
import re
import copy

BASE_DIR = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1'

CURRENT_NB = os.path.join(BASE_DIR, 'quantum_coherence_agrivoltaics_analysis_refined.ipynb')
ARCHIVE_NB = os.path.join(BASE_DIR, 'Archive/Code_old/quantum_coherence_agrivoltaics_analysis_refined.ipynb')

# --- Load notebooks ---
with open(CURRENT_NB, 'r') as f:
    current_nb = json.load(f)

with open(ARCHIVE_NB, 'r') as f:
    archive_nb = json.load(f)


def get_indent(line):
    """Get leading whitespace from a line."""
    match = re.match(r'^(\s*)', line)
    return match.group(1) if match else ''


def cell_contains(cell, text):
    """Check if a cell's source contains a given text."""
    if cell['cell_type'] != 'code':
        return False
    return text in "".join(cell['source'])


def find_cell_index(nb, text):
    """Find the index of the first cell containing the given text."""
    for i, cell in enumerate(nb['cells']):
        if cell_contains(cell, text):
            return i
    return -1


# ============================================================
# STEP 1: Restore the BiodegradabilityAnalyzer class
# ============================================================
print("=" * 60)
print("STEP 1: Restoring BiodegradabilityAnalyzer class")
print("=" * 60)

# Check if the class already exists in the current notebook
already_has_class = find_cell_index(current_nb, 'class BiodegradabilityAnalyzer:') >= 0

if already_has_class:
    print("  [SKIP] BiodegradabilityAnalyzer class already exists in current notebook.")
else:
    # Find the cell in the archive
    archive_cell_idx = find_cell_index(archive_nb, 'class BiodegradabilityAnalyzer:')
    if archive_cell_idx < 0:
        print("  [ERROR] Could not find BiodegradabilityAnalyzer in archive!")
    else:
        # Extract the cell from archive
        archive_cell = copy.deepcopy(archive_nb['cells'][archive_cell_idx])
        # Clear outputs (will be regenerated when notebook runs)
        archive_cell['outputs'] = []
        archive_cell['execution_count'] = None

        # Also extract the markdown header cell if it exists (the cell before)
        archive_md_cell = None
        if archive_cell_idx > 0:
            prev_cell = archive_nb['cells'][archive_cell_idx - 1]
            if prev_cell['cell_type'] == 'markdown' and 'Biodegradability' in "".join(prev_cell['source']):
                archive_md_cell = copy.deepcopy(prev_cell)

        # Find where to insert in the current notebook:
        # It should go BEFORE the cell that uses BiodegradabilityAnalyzer
        usage_idx = find_cell_index(current_nb, 'biodegradability_analyzer = BiodegradabilityAnalyzer(')
        if usage_idx < 0:
            print("  [ERROR] Could not find BiodegradabilityAnalyzer usage in current notebook!")
        else:
            # Insert the class cell (and optionally the markdown cell) before the usage
            insert_cells = []
            if archive_md_cell:
                insert_cells.append(archive_md_cell)
            insert_cells.append(archive_cell)

            for i, c in enumerate(insert_cells):
                current_nb['cells'].insert(usage_idx + i, c)

            print(f"  [OK] Inserted BiodegradabilityAnalyzer class at cell index {usage_idx}")
            if archive_md_cell:
                print(f"  [OK] Also inserted markdown header cell")


# ============================================================
# STEP 2: Apply figure refinements and data exports
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Applying figure refinements and data exports")
print("=" * 60)

for cell in current_nb['cells']:
    if cell['cell_type'] != 'code':
        continue

    source = cell['source']
    joined = "".join(source)

    # --- 2a. FMO_Site_Energies.pdf: side-by-side layout ---
    if 'FMO_Site_Energies.pdf' in joined:
        new_source = []
        for line in source:
            indent = get_indent(line)
            if "plt.figure(figsize=" in line and "12, 5" not in line:
                line = f"{indent}plt.figure(figsize=(12, 5))\n"
            elif "plt.subplot(2, 1, 1)" in line:
                line = line.replace("2, 1, 1", "1, 2, 1")
            elif "plt.subplot(2, 1, 2)" in line:
                line = line.replace("2, 1, 2", "1, 2, 2")
            new_source.append(line)

            # Add data export after savefig (idempotent check)
            if 'plt.savefig' in line and 'fmo_hamiltonian_data.csv' not in joined:
                new_source.append(f"{indent}pd.DataFrame(fmo_hamiltonian).to_csv(os.path.join(DATA_DIR, \"fmo_hamiltonian_data.csv\"))\n")
                new_source.append(f"{indent}pd.DataFrame(fmo_energies).to_csv(os.path.join(DATA_DIR, \"fmo_site_energies.csv\"))\n")

        cell['source'] = new_source
        print("  [OK] FMO_Site_Energies.pdf: Applied side-by-side layout + data export")

    # --- 2b. PAR_Transmission: dashed lines + smaller legend ---
    elif 'PAR_Transmission__Clean_vs_Dusty_Conditions.pdf' in joined:
        new_source = []
        for line in source:
            indent = get_indent(line)
            # Fix duplicate linestyle
            if "linestyle='--', linestyle='--'" in line:
                line = line.replace("linestyle='--', linestyle='--'", "linestyle='--'")
            # Add dashed linestyle for dusty transmission lines
            if 'plt.plot(wavelengths, T_dusty' in line and 'linestyle' not in line and 'linewidth' in line:
                line = line.replace('linewidth=2,', "linewidth=2, linestyle='--',")
            # Fix legend font size
            if 'plt.legend()' in line:
                line = f"{indent}plt.legend(prop={{'size': 8}})\n"
            new_source.append(line)

            # Add data export after savefig (idempotent check)
            if 'plt.savefig' in line and 'solar_irradiance_data.csv' not in joined:
                new_source.append(f"{indent}pd.DataFrame({{'wavelength': wavelengths, 'irradiance': solar_irradiance}}).to_csv(os.path.join(DATA_DIR, \"solar_irradiance_data.csv\"))\n")

        cell['source'] = new_source
        print("  [OK] PAR_Transmission: Applied dashed lines + smaller legend + data export")

    # --- 2c. Global_Reactivity_Indices.pdf: 2x2 grid, remove empty panels ---
    elif 'Global_Reactivity_Indices.pdf' in joined:
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
                f"{indent}plt.bar(sites, dual_descriptor, alpha=0.7, label='$\\\\Delta f$', color='purple')\n",
                f"{indent}plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)\n",
                f"{indent}plt.title('Dual Descriptor')\n",
                f"{indent}plt.xlabel('Site Index')\n",
                f"{indent}plt.ylabel('Selectivity')\n",
                f"{indent}plt.grid(True, alpha=0.3)\n",
                f"{indent}plt.subplot(2, 2, 4)\n",
                f"{indent}if np.isfinite(indices['e_homo']):\n",
                f"{indent}    plt.plot(indices['e_homo'], 0, 'ro', label='HOMO', markersize=12)\n",
                f"{indent}    plt.text(indices['e_homo'], 0.1, f'HOMO\\n{{indices[\"e_homo\"]:.1f}}', ha='center', va='bottom')\n",
                f"{indent}if np.isfinite(indices['e_lumo']):\n",
                f"{indent}    plt.plot(indices['e_lumo'], 0, 'bo', label='LUMO', markersize=12)\n",
                f"{indent}    plt.text(indices['e_lumo'], 0.1, f'LUMO\\n{{indices[\"e_lumo\"]:.1f}}', ha='center', va='bottom')\n",
                f"{indent}plt.title('Frontier Molecular Orbitals')\n",
                f"{indent}plt.xlabel('Energy (cm$^{{-1}}$)')\n",
                f"{indent}plt.legend()\n",
                f"{indent}plt.grid(True, alpha=0.3)\n",
                f"{indent}plt.tight_layout()\n",
                f"{indent}plt.savefig(os.path.join(FIGURES_DIR, \"Global_Reactivity_Indices.pdf\"), bbox_inches=\"tight\", dpi=300)\n",
                f"{indent}# Data export\n",
                f"{indent}pd.DataFrame({{'site': sites, 'f_plus': f_plus, 'f_zero': f_zero, 'dual': dual_descriptor}}).to_csv(os.path.join(DATA_DIR, \"reactivity_indices_data.csv\"))\n",
                f"{indent}pd.Series(indices).to_csv(os.path.join(DATA_DIR, \"global_reactivity_summary.csv\"))\n",
                f"{indent}plt.show()\n",
            ])
            # Append remaining code after the original plot block
            for i in range(plot_start_idx, len(source)):
                if "print(f\"\\nTesting biodegradability analysis" in source[i]:
                    new_source.extend(source[i:])
                    break
            cell['source'] = new_source
            print("  [OK] Global_Reactivity_Indices.pdf: Applied 2x2 grid layout + data export")

    # --- 2d. Generic data export for other savefig calls ---
    else:
        temp_source = []
        modified = False
        for line in source:
            temp_source.append(line)
            if 'plt.savefig' in line:
                indent = get_indent(line)
                match = re.search(r'\"([^\"]+\.pdf)\"', line)
                if match:
                    filename = match.group(1).replace('.pdf', '_data.csv')
                    if filename not in joined:
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


# ============================================================
# STEP 3: Save the modified notebook
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Saving modified notebook")
print("=" * 60)

with open(CURRENT_NB, 'w') as f:
    json.dump(current_nb, f, indent=1)

print(f"  [OK] Saved to: {CURRENT_NB}")
print("\nDone! All refinements applied successfully.")
