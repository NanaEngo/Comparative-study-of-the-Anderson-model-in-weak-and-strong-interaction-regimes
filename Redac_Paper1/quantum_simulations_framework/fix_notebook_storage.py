import nbformat

def main():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            if "csv_storage.save_quantum_dynamics_results(dynamics_data)" in cell.source:
                cell.source = cell.source.replace("""    dynamics_data = {
        'time_fs': results['t_axis'],
        'populations': results['populations'],
        'coherences': results.get('coherences', []),
        'qfi': results.get('qfi', []),
        'entropy': results.get('entropy', []),
        'purity': results.get('purity', []),
        'linear_entropy': results.get('linear_entropy', []),
        'bipartite_ent': results.get('bipartite_ent', []),
        'multipartite_ent': results.get('multipartite_ent', []),
        'pairwise_concurrence': results.get('pairwise_concurrence', []),
        'discord': results.get('discord', []),
        'fidelity': results.get('fidelity', []),
        'mandel_q': results.get('mandel_q', [])
    }
    csv_path = csv_storage.save_quantum_dynamics_results(dynamics_data)""",
"""    time_fs = results['t_axis']
    populations = results['populations']
    coherences = results.get('coherences', [])
    quantum_metrics = {k: v for k, v in results.items() if k not in ['t_axis', 'populations', 'coherences']}
    csv_path = csv_storage.save_quantum_dynamics_results(time_fs, populations, coherences, quantum_metrics)"""
                )
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
            if "csv_storage.save_eco_design_results(eco_data)" in cell.source:
                 cell.source = cell.source.replace("""eco_data = {
    'material_name': material_result['material_name'],
    'pce': material_result['pce'],
    'b_index': material_result['b_index'],
    'sustainability_score': material_result['sustainability_score'],
    'chemical_potential': material_result['global_indices']['chemical_potential'],
    'chemical_hardness': material_result['global_indices']['chemical_hardness'],
    'electrophilicity': material_result['global_indices']['electrophilicity'],
    'timestamp': datetime.now().isoformat()
}
csv_path = csv_storage.save_eco_design_results(eco_data)""",
"""eco_data = {
    'material_name': material_result['material_name'],
    'pce': material_result['pce'],
    'b_index': material_result['b_index'],
    'sustainability_score': material_result['sustainability_score'],
    'timestamp': datetime.now().isoformat()
}
if 'global_indices' in material_result:
    eco_data['chemical_potential'] = material_result['global_indices'].get('chemical_potential', 0)
    eco_data['chemical_hardness'] = material_result['global_indices'].get('chemical_hardness', 0)
    eco_data['electrophilicity'] = material_result['global_indices'].get('electrophilicity', 0)
    
csv_path = csv_storage.save_biodegradability_analysis(eco_data, filename_prefix="eco_design_results")""")

                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    main()
