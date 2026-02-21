#!/usr/bin/env python3
import json
import re

with open('quantum_coherence_agrivoltaics_mesohops.ipynb') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # Fix escape sequences in strings
        src = re.sub(r'\\([SoliDeP])', r'\\\\\\1', src)
        # Fix indentation issues
        lines = src.split('\n')
        fixed = []
        for line in lines:
            if line.strip():
                fixed.append(line)
            else:
                fixed.append('')
        cell['source'] = ['\n'.join(fixed)]

with open('quantum_coherence_agrivoltaics_mesohops.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Fixed remaining issues")
