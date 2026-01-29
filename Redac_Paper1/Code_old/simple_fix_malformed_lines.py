#!/usr/bin/env python3
"""
Script to find and fix all malformed lines in the notebook using simple string replacement
"""

def fix_all_malformed_lines(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all instances of the malformed pattern
    # The pattern in the file is: \\n"\\n",\ 
    # Should become: \\n",
    
    fixed_content = content.replace('\\\\n"\\\\n",\\', '\\\\n",')
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed all malformed lines in the notebook")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_all_malformed_lines(notebook_path)