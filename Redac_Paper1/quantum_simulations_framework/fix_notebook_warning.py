import json

notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_data = json.load(f)

modifications_made = 0

for cell in nb_data.get('cells', []):
    if cell.get('cell_type') == 'code':
        source_lines = cell.get('source', [])
        for i, line in enumerate(source_lines):
            # Target the warning print
            if "logger.warning(\"EnvironmentalFactors module not found" in line:
                source_lines[i] = "        pass # Silenced warning\n"
                modifications_made += 1

if modifications_made > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)
    print(f"Successfully replaced {modifications_made} occurrences.")
else:
    print("No occurrences found to replace.")
