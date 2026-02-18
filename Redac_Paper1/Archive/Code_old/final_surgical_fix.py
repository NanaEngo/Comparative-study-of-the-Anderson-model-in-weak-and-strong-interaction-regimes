
import json
import os

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            # Surgical Fix for local_sensitivity_analysis (elif block)
            if line == '                trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n':
                # This line might be used in different places.
                # In local_sensitivity_analysis, it should be 16 spaces (4 levels).
                # In monte_carlo_uncertainty, it should be 12 spaces (3 levels).
                pass # Handled below by checking context
            
            # Use specific line matching with exact spaces
            if line == '            trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n':
                # If it's in the monte_carlo_uncertainty method (after 'Calculate outputs with sampled parameters')
                if i > 0 and 'Calculate outputs with sampled parameters' in source[i-1]:
                    # This is correct for monte_carlo_uncertainty (12 spaces)
                    continue
                # If it's in the local_sensitivity_analysis (after 'base_transmission': 0.1)
                else:
                    source[i] = '                trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n'
            
            if line == '                trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n':
                # If it's in monte_carlo_uncertainty (line 4596), it should be 12 spaces
                if i > 0 and '}' in source[i-1]:
                    source[i] = '            trans = opv_transmission_parametric(self.agrivoltaic_model.wavelengths, trans_params)\n'

        cell['source'] = source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Final surgical fix applied.")
