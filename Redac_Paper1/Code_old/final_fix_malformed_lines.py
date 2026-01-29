#!/usr/bin/env python3
"""
Script to find and fix all malformed lines in the notebook
"""

import re

def fix_all_malformed_lines(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace all instances of the malformed pattern
    # Pattern: text ending with \n"\n",
    # Should become: text ending with \n",
    
    # The malformed pattern is: some content + \n"\n", where the \n", part is what needs to be changed
    
    # Let's use a more specific pattern: find lines ending with the specific malformed part
    # We need to match the pattern: "any content here\n"\n",
    fixed_content = re.sub(r'\\"\\n\\"\\n",\\', '\\"\\n",', content)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed all malformed lines in the notebook")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_all_malformed_lines(notebook_path)
