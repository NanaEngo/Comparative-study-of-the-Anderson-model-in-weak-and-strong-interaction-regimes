#!/usr/bin/env python3
"""
Script to fix the specific malformed lines in the notebook
"""

def fix_specific_lines(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # Fix the specific malformed pattern in the CSV section
        if '"# CSV Data Storage and Figure Saving Functions\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "# CSV Data Storage and Figure Saving Functions\\\\n",\n')
        elif '"# These functions save simulation results to CSV files and figures to the figures/ folder\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "# These functions save simulation results to CSV files and figures to the figures/ folder\\\\n",\n')
        elif '"\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "\\\\n",\n')
        elif '"import numpy as np\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "import numpy as np\\\\n",\n')
        elif '"import pandas as pd\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "import pandas as pd\\\\n",\n')
        elif '"import os\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "import os\\\\n",\n')
        elif '"from datetime import datetime\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "from datetime import datetime\\\\n",\n')
        elif '"import matplotlib.pyplot as plt\\\\n"\\\\n",\\' in line:
            fixed_lines.append('  "import matplotlib.pyplot as plt\\\\n",\n')
        elif '"def save_simulation_data_to_csv(' in line and '\\\\n"\\\\n",\\' in line:
            fixed_lines.append(line.replace('\\\\n"\\\\n",\\', '\\\\n",\n'))
        elif '"    Save quantum dynamics simulation data to CSV files.' in line and '\\\\n"\\\\n",\\' in line:
            fixed_lines.append(line.replace('\\\\n"\\\\n",\\', '\\\\n",\n'))
        else:
            fixed_lines.append(line)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("Fixed the specific malformed lines in the CSV section")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_specific_lines(notebook_path)