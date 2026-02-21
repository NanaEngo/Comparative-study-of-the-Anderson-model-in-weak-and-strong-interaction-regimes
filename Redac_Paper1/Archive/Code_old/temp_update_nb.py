import json
import os

notebook_path = '/home/taamangtchu/Documents/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modified = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'FIGURES_DIR = "Graphics/"' in line:
                new_source.append(line.replace('FIGURES_DIR = "Graphics/"', 'FIGURES_DIR = "../Graphics/"'))
                modified = True
            elif 'DATA_DIR = "simulation_data/"' in line:
                new_source.append(line.replace('DATA_DIR = "simulation_data/"', 'DATA_DIR = "../simulation_data/"'))
                modified = True
            else:
                new_source.append(line)
        cell['source'] = new_source

if modified:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Paths not found or already updated.")
