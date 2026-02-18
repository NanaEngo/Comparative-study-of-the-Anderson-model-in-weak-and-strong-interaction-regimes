
import json
import os

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Surgical fix 1: local_sensitivity_analysis indentation
            if '            trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)' in line:
                 source[i] = '                trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n'
            
            # Surgical fix 2: monte_carlo_uncertainty indentation
            if '                trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)' in line:
                 source[i] = '            trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n'
                 
        cell['source'] = source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Surgical fix applied.")
