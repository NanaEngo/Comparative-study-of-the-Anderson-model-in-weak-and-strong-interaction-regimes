#!/usr/bin/env python3
"""
Script to fix, lint, and update the quantum coherence notebook
"""
import json
import re

def update_markdown_cells(nb):
    """Update markdown cells with proper formatting"""
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            
            # Fix common markdown issues
            # 1. Ensure proper spacing around headers
            source = re.sub(r'(^|\n)(#{1,6})([^\s#])', r'\1\2 \3', source)
            
            # 2. Fix list formatting
            source = re.sub(r'(^|\n)(\d+\.|\*|\-)([^\s])', r'\1\2 \3', source)
            
            # 3. Ensure proper line breaks
            source = source.strip() + '\n'
            
            # Update the cell
            cell['source'] = [source]
            
    print(f"Updated {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')} markdown cells")
    return nb

def lint_code_cells(nb):
    """Lint code cells for common issues"""
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix common code issues
            # 1. Fix matplotlib format strings
            source = source.replace("'ro - '", "'ro-'")
            source = source.replace('"ro - "', '"ro-"')
            
            # 2. Ensure proper spacing around operators
            # (but be careful not to break strings)
            
            # 3. Remove trailing whitespace
            lines = source.split('\n')
            lines = [line.rstrip() for line in lines]
            source = '\n'.join(lines)
            
            # Update the cell
            cell['source'] = [source]
            
    print(f"Linted {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} code cells")
    return nb

def main():
    # Read the notebook
    with open('quantum_coherence_agrivoltaics_mesohops.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Update markdown cells
    nb = update_markdown_cells(nb)
    
    # Lint code cells
    nb = lint_code_cells(nb)
    
    # Save the updated notebook
    with open('quantum_coherence_agrivoltaics_mesohops.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook updated successfully!")

if __name__ == '__main__':
    main()
