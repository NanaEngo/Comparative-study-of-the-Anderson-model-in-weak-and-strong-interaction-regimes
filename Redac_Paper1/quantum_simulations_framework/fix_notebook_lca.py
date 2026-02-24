import nbformat

def main():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            if "lca_analyzer.calculate_lca" in cell.source:
                cell.source = cell.source.replace("lca_results = lca_analyzer.calculate_lca", "lca_results = lca_analyzer.calculate_lca_impact")
            if "lca_analyzer.save_results" in cell.source:
                cell.source = cell.source.replace("lca_analyzer.save_results", "lca_analyzer.save_lca_results_to_csv")
            if "lca_analyzer.plot_results" in cell.source:
                cell.source = cell.source.replace("lca_analyzer.plot_results", "lca_analyzer.plot_lca_results")
                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    main()
