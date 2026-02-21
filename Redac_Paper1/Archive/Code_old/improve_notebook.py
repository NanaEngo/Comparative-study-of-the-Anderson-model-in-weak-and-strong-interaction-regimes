#!/usr/bin/env python3
import json
import re

with open('quantum_coherence_agrivoltaics_mesohops.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # Fix spacing
        src = re.sub(r'\s*=\s*', ' = ', src)
        src = re.sub(r'\s*,\s*', ', ', src)
        # Fix comparison operators
        src = re.sub(r'\s*==\s*', ' == ', src)
        src = re.sub(r'\s*!=\s*', ' != ', src)
        # Remove multiple blank lines
        src = re.sub(r'\n{3,}', '\n\n', src)
        cell['source'] = [src]

with open('quantum_coherence_agrivoltaics_mesohops.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook improved")
