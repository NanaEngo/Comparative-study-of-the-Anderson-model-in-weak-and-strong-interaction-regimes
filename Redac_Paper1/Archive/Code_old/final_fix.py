
import json
import os

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # Fix wavelengths usage (redundant but safe)
            if 'trans = opv_transmission_parametric(wavelengths, trans_params)' in line:
                line = line.replace('wavelengths', 'self.agrivoltaic_model.wavelengths')
            elif 'trans = opv_transmission_with_dust(wavelengths, trans,' in line:
                line = line.replace('wavelengths', 'self.agrivoltaic_model.wavelengths')
            
            # Fix Indentation Error specifically
            if 'trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)' in line and 'elif' not in line:
                 # Check if it lacks correct indentation (should be 16 spaces inside the elif block if it was 12 before)
                 if line.startswith('            '): # only 12 spaces
                     line = '                ' + line.lstrip()
            
            new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook errors fixed. Restarting execution...")
