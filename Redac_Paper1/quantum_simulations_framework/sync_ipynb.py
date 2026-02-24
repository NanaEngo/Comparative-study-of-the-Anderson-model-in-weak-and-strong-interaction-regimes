import nbformat
import json

def patch_notebook():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # --- 1. LCA Analyzer Patch ---
            if "lca_analyzer.calculate_lca(" in source:
                source = source.replace("calculate_lca(", "calculate_lca_impact(")
                source = source.replace("save_results(lca_results, filename_prefix=\"lca_analysis\")", "save_lca_results_to_csv(lca_results)")
                plot_old = "fig_path = lca_analyzer.plot_results(lca_results, filename_prefix=\"lca_analysis\")"
                plot_new = "import matplotlib.pyplot as plt\n        plt.rcParams['text.usetex'] = False\n        fig_path = lca_analyzer.plot_lca_results(lca_results)"
                source = source.replace(plot_old, plot_new)
            
            # --- 2. CSV Data Storage Patch ---
            if "csv_storage.save_quantum_dynamics_results(results)" in source:
                qdyn_old = "csv_path = csv_storage.save_quantum_dynamics_results(results)"
                qdyn_new = "time_fs = results['t_axis']\n    populations = results['populations']\n    coherences = results.get('coherences', [])\n    quantum_metrics = {k: v for k, v in results.items() if k not in ['t_axis', 'populations', 'coherences']}\n    csv_path = csv_storage.save_quantum_dynamics_results(time_fs, populations, coherences, quantum_metrics)"
                source = source.replace(qdyn_old, qdyn_new)
                
                agri_old = "csv_path = csv_storage.save_agrivoltaic_results({\n    'pce': pce,\n    'etr': etr,\n    'spectral_data': {},\n    **metadata\n})"
                agri_new = "csv_path = csv_storage.save_agrivoltaic_results(pce, etr, {}, **metadata)"
                source = source.replace(agri_old, agri_new)
                
                eco_old = "csv_path = csv_storage.save_eco_design_results(eco_data, filename_prefix=\"eco_design_results\")"
                
                eco_new = """if 'global_indices' in material_result:
    eco_data['chemical_potential'] = material_result['global_indices'].get('chemical_potential', 0)
    eco_data['chemical_hardness'] = material_result['global_indices'].get('chemical_hardness', 0)
    eco_data['electrophilicity'] = material_result['global_indices'].get('electrophilicity', 0)
    
csv_path = csv_storage.save_biodegradability_analysis(eco_data, filename_prefix="eco_design_results")"""
                source = source.replace(eco_old, eco_new)

            # --- 3. Sensitivity Analyzer Patch ---
            if "sensitivity_analyzer = SensitivityAnalyzer(" in source and "simulator=simulator" in source:
                source = source.replace("simulator=simulator,", "quantum_simulator=simulator,")
                
                sens_old = """    sensitivity_results = sensitivity_analyzer.analyze_sensitivity(
        parameters=['temperature', 'dephasing_rate', 'coupling_strength'],
        parameter_ranges={
            'temperature': [273, 320],
            'dephasing_rate': [10, 50],
            'coupling_strength': [0.8, 1.2]
        },
        n_samples=10  # Reduced for notebook
    )

    print(f"✓ Sensitivity analysis completed:")
    print(f"  - Parameters analyzed: {list(sensitivity_results.get('sensitivity_indices', {}).keys())}")
    print(f"  - Total samples: {sensitivity_results.get('n_samples', 0)}")"""
                sens_new = """    # Update parameter ranges
    sensitivity_analyzer.param_ranges.update({
        'temperature': (273, 320),
        'dephasing_rate': (10, 50),
    })
    
    # Run comprehensive report
    report = sensitivity_analyzer.comprehensive_sensitivity_report(n_points=10)
    
    print(f"✓ Sensitivity analysis completed:")
    print(f"  - Parameters analyzed: {list(report.keys())}")
    print(f"  - Total samples: 10")"""
                source = source.replace(sens_old, sens_new)
                
            cell.source = source
            
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    patch_notebook()
