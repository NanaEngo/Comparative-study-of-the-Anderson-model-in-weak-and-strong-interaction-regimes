#!/usr/bin/env python3
"""
Script to systematically fix all remaining malformed lines in the notebook
"""

def fix_all_malformed_lines(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all instances of the malformed pattern
    # The pattern is: some text ending with \n"\n",
    # Should become: some text ending with \n",
    
    # First, let's try a more comprehensive regex approach
    import re
    
    # Pattern: any text ending with \n"\n",
    # This should match: "    populations: 2D array of site populations over time\n"\n",
    content = re.sub(r'\\n"\\n",\\', '\\n",', content)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed all malformed lines in the notebook")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_all_malformed_lines(notebook_path)