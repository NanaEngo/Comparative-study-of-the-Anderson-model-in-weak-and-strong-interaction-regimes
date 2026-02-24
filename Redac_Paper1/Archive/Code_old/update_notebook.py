import json
import sys

notebook_path = '/home/taamangtchu/Documents/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops_complete.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the start index of the redundant block in cell 0
source = nb['cells'][0]['source']
start_idx = -1
for i, line in enumerate(source):
    if "### Real Material Data: PM6 and Y6-BO" in line:
        start_idx = i
        break

if start_idx != -1:
    # also remove empty newlines before it
    while start_idx > 0 and source[start_idx - 1].strip() == "":
        start_idx -= 1

    print(f"Removing {len(source) - start_idx} lines from the first cell.")
    nb['cells'][0]['source'] = source[:start_idx]
    
    if len(nb['cells'][0]['source']) > 0:
        nb['cells'][0]['source'][-1] = nb['cells'][0]['source'][-1].rstrip('\n')

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Could not find the text to remove.")

# Now verify the JSON
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        json.load(f)
    print("JSON validation passed.")
except Exception as e:
    print(f"JSON validation failed: {e}")
    sys.exit(1)
