#!/usr/bin/env python3
"""
Script to fix the JSON formatting in the notebook and update all necessary fixes
"""

import json
import re

def fix_notebook_completely(notebook_path):
    # Read the file as text first to fix the JSON formatting issues
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The issue is with malformed lines containing `\\` at the end which creates invalid JSON
    # Replace the malformed CSV section with properly formatted JSON
    problematic_section = ''' "source": [
  "# CSV Data Storage and Figure Saving Functions\\n"\\n",
  "# These functions save simulation results to CSV files and figures to the figures/ folder\\n"\\n",
  "\\n"\\n",
  "import numpy as np\\n"\\n",
  "import pandas as pd\\n"\\n",
  "import os\\n"\\n",
  "from datetime import datetime\\n"\\n",
  "import matplotlib.pyplot as plt\\n"\\n",
'''
    
    fixed_section = ''' "source": [
  "# CSV Data Storage and Figure Saving Functions\\n",
  "\\n",
  "# These functions save simulation results to CSV files and figures to the figures/ folder\\n",
  "\\n",
  "import numpy as np\\n",
  "import pandas as pd\\n",
  "import os\\n",
  "from datetime import datetime\\n",
  "import matplotlib.pyplot as plt\\n",
  "\\n",
'''
    
    content = content.replace(problematic_section, fixed_section)
    
    # Write the fixed content back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Now load the JSON properly
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Process each cell to fix the issues
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Join the source lines to process as a single string
            source = ''.join(cell['source'])
            
            # Fix 1: workers parameter issue
            if 'workers=mp.cpu_count()' in source:
                source = source.replace('workers=mp.cpu_count()', 'workers=1')
                source = source.replace(
                    '# Run differential evolution with parallel workers',
                    '# Run differential evolution with single worker to avoid issues in Jupyter notebooks'
                )
                print("Fixed workers parameter in optimization function")
            
            # Fix 2: Update QFI calculation threshold
            if 'def calculate_qfi' in source and '1e-12' in source:
                # Replace the threshold with a more conservative value
                source = re.sub(r'if eigenvals\[i\] \+ eigenvals\[j\] > 1e-12:', 
                               'if eigenvals[i] + eigenvals[j] > 1e-8:', source)
                
                # Add upper bound protection
                if 'return 2 * qfi' in source:
                    # Add the upper bound check before the return statement
                    lines = source.split('\n')
                    new_lines = []
                    for line in lines:
                        if 'return 2 * qfi' in line and line.strip() == 'return 2 * qfi':
                            new_lines.append('        # Apply upper bound to prevent extremely large values')
                            new_lines.append('        qfi = 2 * qfi')
                            new_lines.append('        if qfi > 1e10:  # Cap extremely large values')
                            new_lines.append('            qfi = 1e10')
                            new_lines.append('        return qfi')
                        else:
                            new_lines.append(line)
                    source = '\n'.join(new_lines)
                print("Updated QFI calculation with more conservative threshold and upper bound")
            
            # Fix 3: Complete the transmission_profiles list
            if 'transmission_profiles = [' in source and '# Run optimization to get optimized parameters' in source:
                # Replace the incomplete transmission_profiles assignment
                source = source.replace(
                    "transmission_profiles = [\n# Run optimization to get optimized parameters",
                    "transmission_profiles = [\n    {'name': 'Optimized', 'params': optimized_params},\n    {'name': 'Broad Bandpass', 'params': {'center_wls': [600], 'widths': [120], 'peak_transmissions': [0.8], 'base_transmission': 0.1}},\n    {'name': 'Chlorophyll Matching', 'params': {'center_wls': [430, 680], 'widths': [30, 40], 'peak_transmissions': [0.6, 0.8], 'base_transmission': 0.05}}\n]\n\n# Run optimization to get optimized parameters"
                )
                print("Completed transmission_profiles list assignment")
            
            # Fix 4: Fix robustness analysis overlapping y-axis issue
            if 'plt.twinx().plot(' in source and 'Dephasing Sensitivity' in source:
                # Replace multiple plt.twinx() calls with proper twin axis usage
                source = source.replace(
                    "plt.twinx().plot(dephasing_rates, etr_at_dephasing, 'b.--', label='Dephasing Sensitivity', \n                 linewidth=2, markersize=8)\n\nplt.title('Robustness Analysis: Sensitivity to Parameters')\nplt.xlabel('Temperature (K) / Dephasing Rate (cm\\u207b\\u00b9)')\nplt.ylabel('ETR per Photon (Temp)', color='red')\nplt.twinx().set_ylabel('ETR per Photon (Dephasing)', color='blue')\nplt.grid(True, alpha=0.3)\n\n# Add legends\nplt.legend(loc='upper left')\nplt.twinx().legend(loc='upper right')",
                    "# Create proper twin axis\nax1 = plt.gca()\nax2 = ax1.twinx()\nax2.plot(dephasing_rates, etr_at_dephasing, 'b.--', label='Dephasing Sensitivity', \n         linewidth=2, markersize=8)\n\nplt.title('Robustness Analysis: Sensitivity to Parameters')\nplt.xlabel('Temperature (K) / Dephasing Rate (cm\\u207b\\u00b9)')\nplt.ylabel('ETR per Photon (Temp)', color='red')\nax2.set_ylabel('ETR per Photon (Dephasing)', color='blue')\nplt.grid(True, alpha=0.3)\n\n# Add legends\nplt.legend(loc='upper left')\nax2.legend(loc='upper right')"
                )
                print("Fixed robustness analysis overlapping y-axis issue")
            
            # Split back into lines for the notebook format
            cell['source'] = source.splitlines(keepends=True)
    
    # Write the updated notebook with proper formatting
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)  # Use indent=1 for notebook format
    
    print("Notebook updated successfully with all fixes")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_notebook_completely(notebook_path)
