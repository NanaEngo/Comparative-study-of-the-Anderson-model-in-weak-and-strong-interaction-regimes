import nbformat
import json

def patch_notebook():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            if "plt.rcParams['text.usetex'] = False" in source:
                new_source = []
                for line in source.split('\n'):
                    if "plt.rcParams['text.usetex'] = False" not in line:
                        new_source.append(line)
                cell.source = '\n'.join(new_source)
                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    patch_notebook()
