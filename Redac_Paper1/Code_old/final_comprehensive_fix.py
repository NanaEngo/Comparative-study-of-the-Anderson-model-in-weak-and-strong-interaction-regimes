#!/usr/bin/env python3
"""
Comprehensive fix for the quantum coherence agrivoltaics analysis notebook
"""

import json
import re
import sys

def create_fixed_notebook(notebook_path):
    """Create a properly formatted version of the notebook with all fixes applied"""
    
    # Read the original notebook content
    with open(notebook_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Parse the JSON to make targeted changes
    try:
        notebook = json.loads(original_content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return False
    
    # Fix 1: Update optimization function to use workers=1  
    cells = notebook['cells']
    for cell in cells:
        if 'source' in cell:
            source = ''.join(cell['source'])
            
            # Fix multi-objective optimization function workers parameter
            if 'multi_objective_optimization' in source and 'workers=' in source:
                # Update the source array to change workers=mp.cpu_count() to workers=1
                new_source = []
                for line in cell['source']:
                    if 'workers=mp.cpu_count()' in line:
                        line = line.replace('workers=mp.cpu_count()', 'workers=1  # Use single process to avoid issues in Jupyter notebooks')
                    new_source.append(line)
                cell['source'] = new_source
            elif 'def optimize_transmission_function' in source:
                # Update the other optimization function too
                new_source = []
                for line in cell['source']:
                    if 'workers=mp.cpu_count()' in line:
                        line = line.replace('workers=mp.cpu_count()', 'workers=1  # Use single process to avoid issues in Jupyter notebooks')
                    new_source.append(line)
                cell['source'] = new_source
    
    # Fix 2: Complete the transmission_profiles list assignment
    for cell in cells:
        if 'source' in cell:
            source_text = ''.join(cell['source'])
            if 'transmission_profiles = [' in source_text and 'Optimized' in source_text:
                # Find the cell that starts the transmission_profiles list
                source = cell['source']
                new_source = []
                
                i = 0
                while i < len(source):
                    line = source[i]
                    new_source.append(line)
                    
                    if 'transmission_profiles = [' in line:
                        # Add the missing entries right after
                        new_source.extend([
                            "    {'name': 'Optimized', 'params': optimized_params},\n",
                            "    {'name': 'Broad Bandpass', 'params': {'center_wls': [600], 'widths': [120], 'peak_transmissions': [0.8], 'base_transmission': 0.1}},\n",
                            "    {'name': 'Chlorophyll Matching', 'params': {'center_wls': [430, 680], 'widths': [30, 40], 'peak_transmissions': [0.6, 0.8], 'base_transmission': 0.05}}\n",
                            "]\n"
                        ])
                        
                        # Skip the lines that were incorrectly placed after the opening bracket
                        # until we find where the actual entries should be
                        i += 1
                        while i < len(source) and not any(x in source[i] for x in ["{'name': 'Optimized'", "{'name': 'Broad Bandpass'", "{'name': 'Chlorophyll Matching'"]):
                            i += 1
                        continue
                    i += 1
                
                cell['source'] = new_source
                break
    
    # Fix 3 & 5: Update QFI calculation with conservative threshold and upper bound
    for cell in cells:
        if 'source' in cell:
            source = cell['source']
            new_source = []
            
            in_qfi_function = False
            for line in source:
                if 'def calculate_qfi' in line:
                    in_qfi_function = True
                    new_source.append(line)
                elif in_qfi_function and 'eigenvals[i] + eigenvals[j] > 1e-12' in line:
                    # Fix the threshold and add upper bound
                    fixed_line = line.replace('> 1e-12', '> 1e-8  # More conservative threshold to prevent numerical instabilities')
                    new_source.append(fixed_line)
                    
                    # Add upper bound protection after the qfi calculation
                    # We need to find where to add the cap
                elif 'return 2 * qfi' in line and in_qfi_function:
                    # Add upper bound before return
                    new_source.append("        # Apply upper bound to prevent extremely large QFI values\n")
                    new_source.append("        qfi = min(qfi, 1e10)  # Cap extremely large values\n")
                    new_source.append(line)
                    in_qfi_function = False  # Reset flag after return
                else:
                    new_source.append(line)
                    # Reset flag when we get to the end of the function
                    if in_qfi_function and 'def ' in line and 'calculate_qfi' not in line:
                        in_qfi_function = False
            
            cell['source'] = new_source
    
    # Fix 4: Fix robustness analysis overlapping y-axis
    for cell in cells:
        if 'source' in cell:
            source = cell['source']
            new_source = []
            
            for line in source:
                if 'plt.twinx().plot(' in line:
                    line = '    ax1 = plt.gca()\n    ax2 = ax1.twinx()\n    ax2.plot(dephasing_rates, etr_at_dephasing, \'b.--\', label=\'Dephasing Sensitivity\', \n                 linewidth=2, markersize=8)\n'
                elif 'plt.twinx().set_ylabel(' in line:
                    line = '    ax2.set_ylabel(\'ETR per Photon (Dephasing)\', color=\'blue\')\n'
                elif 'plt.twinx().legend(' in line:
                    line = '    ax2.legend(loc=\'upper right\')\n'
                new_source.append(line)
            
            cell['source'] = new_source
    
    # Write the fixed notebook back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print("All fixes applied successfully!")
    return True

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    success = create_fixed_notebook(notebook_path)
    if success:
        print("Notebook has been successfully updated with all fixes!")
    else:
        print("Failed to update the notebook.")