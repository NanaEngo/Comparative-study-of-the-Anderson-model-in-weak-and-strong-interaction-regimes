#!/usr/bin/env python3
"""
Script to fix the workers parameter in the Jupyter notebook
"""

import json

def fix_workers_parameter(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Join the source lines to process as a single string
            source = ''.join(cell['source'])
            
            # Check if this is the cell containing the optimization function
            if 'workers=mp.cpu_count()' in source:
                # Replace the workers parameter
                updated_source = source.replace('workers=mp.cpu_count()', 'workers=1')
                
                # Add comment explaining the change
                updated_source = updated_source.replace(
                    '# Run differential evolution with parallel workers',
                    '# Run differential evolution with single worker to avoid issues in Jupyter notebooks'
                )
                
                # Split back into lines for the notebook format
                cell['source'] = updated_source.splitlines(keepends=True)
                print("Fixed workers parameter in optimization function")
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)  # Use indent=1 for notebook format
    
    print("Notebook updated successfully")

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    fix_workers_parameter(notebook_path)