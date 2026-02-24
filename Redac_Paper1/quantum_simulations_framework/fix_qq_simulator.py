import nbformat
import re

def patch_notebook():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            if "quantum_quantum_simulator" in cell.source:
                cell.source = cell.source.replace("quantum_quantum_simulator", "quantum_simulator")
                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    patch_notebook()
