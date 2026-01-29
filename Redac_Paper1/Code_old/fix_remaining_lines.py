#!/usr/bin/env python3
"""
Script to fix all remaining malformed lines in the notebook
"""

def fix_remaining_lines(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all remaining instances of the malformed pattern
    import re
    content = re.sub(r'\\n"\\n",\\\\', '\\\\n",\n  ', content)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed all remaining malformed lines")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_remaining_lines(notebook_path)
