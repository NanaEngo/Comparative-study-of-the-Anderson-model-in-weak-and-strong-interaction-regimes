import nbformat

def main():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            if "csv_storage.save_agrivoltaic_results(agrivoltaic_data)" in cell.source:
                 cell.source = cell.source.replace("""agrivoltaic_data = {
    'pce': opt_results.get('optimal_pce', material_result['pce']),
    'etr': opt_results.get('optimal_etr', 0.85),
    'timestamp': datetime.now().isoformat(),
    'temperature': DEFAULT_TEMPERATURE,
    'max_hierarchy': DEFAULT_MAX_HIERARCHY,
    'n_sites': H_fmo.shape[0]
}
csv_path = csv_storage.save_agrivoltaic_results(agrivoltaic_data)""",
"""pce = opt_results.get('optimal_pce', material_result['pce']) if 'opt_results' in locals() else material_result['pce']
etr = opt_results.get('optimal_etr', 0.85) if 'opt_results' in locals() else 0.85
metadata = {
    'timestamp': datetime.now().isoformat(),
    'temperature': DEFAULT_TEMPERATURE,
    'max_hierarchy': DEFAULT_MAX_HIERARCHY,
    'n_sites': H_fmo.shape[0]
}
csv_path = csv_storage.save_agrivoltaic_results(pce, etr, {}, **metadata)""")

            if "fig_path = lca_analyzer.plot_lca_results(lca_results)" in cell.source:
                 cell.source = cell.source.replace("""    try:
        fig_path = lca_analyzer.plot_lca_results(lca_results)
        print(f"  - LCA plots saved to: {fig_path}")
    except Exception as e:
        print(f"  ⚠ Could not plot LCA results: {e}")""",
"""    try:
        import matplotlib.pyplot as plt
        plt.rcParams['text.usetex'] = False
        fig_path = lca_analyzer.plot_lca_results(lca_results)
        print(f"  - LCA plots saved to: {fig_path}")
    except Exception as e:
        print(f"  ⚠ Could not plot LCA results: {e}")""")

                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    main()
