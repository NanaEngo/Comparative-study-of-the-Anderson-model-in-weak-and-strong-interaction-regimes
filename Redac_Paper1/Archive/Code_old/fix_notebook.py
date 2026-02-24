import json

file_path = '/home/taamangtchu/Documents/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops_complete.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        idx_to_remove = -1
        for i, line in enumerate(source):
            if "### Real Material Data: PM6 and Y6-BO" in line:
                idx_to_remove = i
                break
        
        is_intro = any("# Quantum Agrivoltaics with MesoHOPS Framework" in line for line in source)
        is_eco_design = any("Eco-Design Analysis with Quantum Reactivity Descriptors" in line for line in source)
        
        if idx_to_remove != -1 and not is_intro and not is_eco_design:
            start_removing = idx_to_remove
            while start_removing > 0 and source[start_removing - 1] == "\n":
                start_removing -= 1
            
            new_source = source[:start_removing]
            cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook redundant blocks removed.")
