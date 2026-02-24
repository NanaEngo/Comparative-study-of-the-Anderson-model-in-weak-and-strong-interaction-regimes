import nbformat

def main():
    notebook_path = "quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            if "csv_storage.save_eco_design_results(eco_data)" in cell.source:
                 cell.source = cell.source.replace("""csv_path = csv_storage.save_eco_design_results(eco_data)""",
"""if 'global_indices' in material_result:
    eco_data['chemical_potential'] = material_result['global_indices'].get('chemical_potential', 0)
    eco_data['chemical_hardness'] = material_result['global_indices'].get('chemical_hardness', 0)
    eco_data['electrophilicity'] = material_result['global_indices'].get('electrophilicity', 0)
csv_path = csv_storage.save_biodegradability_analysis(eco_data, filename_prefix="eco_design_results")""")

            if "sensitivity_analyzer.analyze_sensitivity(" in cell.source:
                 cell.source = cell.source.replace("""sensitivity_analyzer = SensitivityAnalyzer(
    simulator=simulator,
    agrivoltaic_model=agrivoltaic_model
)

# Run sensitivity analysis for key parameters
try:
    sensitivity_results = sensitivity_analyzer.analyze_sensitivity(
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
    print(f"  - Total samples: {sensitivity_results.get('n_samples', 0)}")""",
"""sensitivity_analyzer = SensitivityAnalyzer(
    quantum_simulator=simulator,
    agrivoltaic_model=agrivoltaic_model
)

# Run sensitivity analysis for key parameters
try:
    # Update parameter ranges
    sensitivity_analyzer.param_ranges.update({
        'temperature': (273, 320),
        'dephasing_rate': (10, 50),
    })
    
    # Run comprehensive report
    report = sensitivity_analyzer.comprehensive_sensitivity_report(n_points=10)
    
    print(f"✓ Sensitivity analysis completed:")
    print(f"  - Parameters analyzed: {list(report.keys())}")
    print(f"  - Total samples: 10")""")

                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    main()
