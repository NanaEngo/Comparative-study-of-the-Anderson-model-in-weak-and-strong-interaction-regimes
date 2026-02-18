
import json
import os

nb_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_refined.ipynb'

if not os.path.exists(nb_path):
    print(f"Error: {nb_path} not found")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# 1. Update Hamiltonian
ham_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('def create_fmo_hamiltonian' in line for line in cell['source']):
        new_source = []
        for line in cell['source']:
            # Site energies RC
            if '12200, 12070, 11980, 12050, 12140, 12130, 12260, 11700' in line:
                line = '        site_energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440, 11700])\n'
            # Site energies Standard
            elif '12200, 12070, 11980, 12050, 12140, 12130, 12260' in line:
                line = '        site_energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])\n'
            # Couplings
            elif '(0, 1): 63' in line:
                line = '        (0, 1): -87.7, (0, 2): 5.5, (0, 3): -5.9, (0, 4): 6.7, (0, 5): -13.7, (0, 6): -9.9,\n'
            elif '(1, 2): 104' in line:
                line = '        (1, 2): 30.8, (1, 3): 8.2, (1, 4): 0.7, (1, 5): 11.4, (1, 6): 4.7,\n'
            elif '(2, 3): 180' in line:
                line = '        (2, 3): -53.5, (2, 4): -2.2, (2, 5): -9.6, (2, 6): 6.0,\n'
            elif '(3, 4): 60, (3, 5): 120, (3, 6): -10' in line:
                line = '        (3, 4): -70.7, (3, 5): -17.0, (3, 6): -63.3,\n'
            elif '(4, 5): 120, (4, 6): 100' in line:
                line = '        (4, 5): 81.1, (4, 6): -1.3,\n'
            elif '(5, 6): 60' in line:
                line = '        (5, 6): 39.7\n'
            # Update comment
            elif '# Standard FMO site energies (cm^-1) - from Adolphs & Renger 2006' in line:
                line = '    # Standard FMO site energies (cm^-1) - from Adolphs & Renger 2006 (SI Table 1)\n'
            
            new_source.append(line)
        cell['source'] = new_source
        ham_found = True
        break

# 2. Update QuantumDynamicsSimulator docstring
sim_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('class QuantumDynamicsSimulator' in line for line in cell['source']):
        new_source = []
        in_docstring = False
        docstring_updated = False
        for line in cell['source']:
            if '"""' in line:
                if not in_docstring:
                    in_docstring = True
                else:
                    # Closing docstring
                    if not docstring_updated:
                        new_source.append('    This implementation uses a computationally optimized Markovian Lindblad\n')
                        new_source.append('    approach to approximate the results of the full non-Markovian\n')
                        new_source.append('    PT-HOPS+LTC method described in the EES manuscript.\n')
                    in_docstring = False
            
            if in_docstring and 'Quantum dynamics simulator' in line:
                docstring_updated = True
                new_source.append(line)
                new_source.append('    \n')
                new_source.append('    Scientific Note: This implementation uses a computationally optimized Markovian\n')
                new_source.append('    Lindblad approach to approximate the results of the full non-Markovian\n')
                new_source.append('    PT-HOPS+LTC method described in the main manuscript.\n')
            else:
                new_source.append(line)
        
        cell['source'] = new_source
        sim_found = True
        break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Update complete. Hamiltonian found: {ham_found}, Simulator found: {sim_found}")
