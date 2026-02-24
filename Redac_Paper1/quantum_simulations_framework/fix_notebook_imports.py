import json

notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

modifications_made = 0

for cell in nb_data.get('cells', []):
    if cell.get('cell_type') == 'code':
        source_lines = cell.get('source', [])
        for i, line in enumerate(source_lines):
            if "from quantum_simulations_framework.models.sensitivity_analyzer import SensitivityAnalyzer" in line:
                source_lines[i] = line.replace("from quantum_simulations_framework.models.sensitivity_analyzer import SensitivityAnalyzer", "from models.sensitivity_analyzer import SensitivityAnalyzer")
                modifications_made += 1

if modifications_made > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)
    print(f"Successfully replaced {modifications_made} occurrences.")
else:
    print("No occurrences found to replace.")
