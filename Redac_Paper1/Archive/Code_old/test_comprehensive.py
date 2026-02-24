#!/usr/bin/env python3
"""
Comprehensive test of quantum agrivoltaics simulation framework.
All simulation data is saved to CSV files and all figures are saved.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the quantum_simulations_framework to the path
sys.path.insert(0, './quantum_simulations_framework')

from core.hops_simulator import HopsSimulator
from core.constants import DEFAULT_TEMPERATURE
from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian
from models.biodegradability_analyzer import BiodegradabilityAnalyzer
from models.lca_analyzer import LCAAnalyzer
from models.sensitivity_analyzer import SensitivityAnalyzer
from models.testing_validation_protocols import TestingValidationProtocols
from utils.csv_data_storage import CSVDataStorage
from utils.figure_generator import FigureGenerator

# Ensure output directories exist
os.makedirs('./simulation_data', exist_ok=True)
os.makedirs('./figures', exist_ok=True)

def run_comprehensive_test():
    """Run a comprehensive test of the quantum agrivoltaics framework."""
    print("="*60)
    print("COMPREHENSIVE QUANTUM AGRIPLVOLTAICS FRAMEWORK TEST")
    print("="*60)
    
    # Initialize CSV and Figure storage
    csv_storage = CSVDataStorage(output_dir='./simulation_data')
    fig_gen = FigureGenerator(figures_dir='./figures')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: FMO Hamiltonian Creation
    print("\n1. Testing FMO Hamiltonian Creation...")
    H_fmo, site_energies = create_fmo_hamiltonian()
    print(f"   ✓ Created FMO Hamiltonian: {H_fmo.shape}")
    print(f"   ✓ Site energies: {site_energies}")
    
    # Save Hamiltonian data to CSV
    hamiltonian_csv = f'./simulation_data/fmo_hamiltonian_{timestamp}.csv'
    np.savetxt(hamiltonian_csv, H_fmo, delimiter=',', fmt='%.6f')
    print(f"   ✓ Hamiltonian saved to: {hamiltonian_csv}")
    
    # Save site energies to CSV
    energies_csv = f'./simulation_data/fmo_site_energies_{timestamp}.csv'
    np.savetxt(energies_csv, site_energies, delimiter=',', fmt='%.2f')
    print(f"   ✓ Site energies saved to: {energies_csv}")
    
    # Generate and save FMO Hamiltonian figure
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(np.abs(H_fmo), cmap='viridis', aspect='auto')
        ax.set_xlabel('Site Index')
        ax.set_ylabel('Site Index')
        ax.set_title('FMO Hamiltonian Matrix (Absolute Values)')
        plt.colorbar(im, ax=ax, label='Coupling Strength (cm⁻¹)')
        hamiltonian_fig = f'./figures/fmo_hamiltonian_{timestamp}.pdf'
        hamiltonian_png = f'./figures/fmo_hamiltonian_{timestamp}.png'
        plt.savefig(hamiltonian_fig, dpi=300, bbox_inches='tight')
        plt.savefig(hamiltonian_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Hamiltonian figure saved to: {hamiltonian_fig}")
    except Exception as e:
        print(f"   ! Could not save Hamiltonian figure: {e}")
    
    # Test 2: HopsSimulator
    print("\n2. Testing HopsSimulator...")
    simulator = HopsSimulator(H_fmo, temperature=DEFAULT_TEMPERATURE)
    print(f"   ✓ Simulator initialized: {simulator.simulator_type}")
    print(f"   ✓ MesoHOPS available: {simulator.is_using_mesohops}")
    
    # Test 3: Quantum Dynamics Simulation
    print("\n3. Testing Quantum Dynamics Simulation...")
    time_points = np.linspace(0, 200, 100)  # 100 points from 0 to 200 fs
    initial_state = np.zeros(H_fmo.shape[0], dtype=complex)
    initial_state[0] = 1.0  # Excitation on site 1
    
    try:
        results = simulator.simulate_dynamics(
            time_points=time_points,
            initial_state=initial_state
        )
        print("   ✓ Quantum dynamics simulation completed")
        print(f"   ✓ Time axis: {results['t_axis'].shape}")
        print(f"   ✓ Populations: {results['populations'].shape}")
        print(f"   ✓ Coherences: {results['coherences'].shape}")
        print(f"   ✓ Final populations: {results['populations'][-1]}")
        print(f"   ✓ Final coherence: {results['coherences'][-1]:.4f}")
        
        # Save quantum dynamics results to CSV
        dynamics_csv = f'./simulation_data/quantum_dynamics_{timestamp}.csv'
        dynamics_data = {
            'time_fs': results['t_axis'],
        }
        # Add populations for each site
        for i in range(results['populations'].shape[1]):
            dynamics_data[f'population_site_{i}'] = results['populations'][:, i]
        dynamics_data['coherence'] = results['coherences']
        
        # Add quantum metrics if available
        if 'qfi' in results:
            dynamics_data['qfi'] = results['qfi']
            print(f"   ✓ QFI range: {results['qfi'].min():.4f} - {results['qfi'].max():.4f}")
        if 'entropy' in results:
            dynamics_data['entropy'] = results['entropy']
            print(f"   ✓ Entropy range: {results['entropy'].min():.4f} - {results['entropy'].max():.4f}")
        if 'purity' in results:
            dynamics_data['purity'] = results['purity']
            print(f"   ✓ Purity range: {results['purity'].min():.4f} - {results['purity'].max():.4f}")
        
        import pandas as pd
        pd.DataFrame(dynamics_data).to_csv(dynamics_csv, index=False, float_format='%.6e')
        print(f"   ✓ Quantum dynamics data saved to: {dynamics_csv}")
        
        # Generate and save quantum dynamics figures
        # Population evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(results['populations'].shape[1]):
            ax.plot(results['t_axis'], results['populations'][:, i], label=f'Site {i}')
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Population')
        ax.set_title('Quantum Dynamics: Population Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        pop_fig = f'./figures/quantum_dynamics_populations_{timestamp}.pdf'
        plt.savefig(pop_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Population evolution figure saved to: {pop_fig}")
        
        # Coherence evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results['t_axis'], results['coherences'])
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Total Coherence')
        ax.set_title('Quantum Dynamics: Coherence Evolution')
        ax.grid(True, alpha=0.3)
        coh_fig = f'./figures/quantum_dynamics_coherence_{timestamp}.pdf'
        plt.savefig(coh_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Coherence evolution figure saved to: {coh_fig}")
        
    except Exception as e:
        print(f"   ✗ Quantum dynamics simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Biodegradability Analysis
    print("\n4. Testing Biodegradability Analysis...")
    try:
        # Create a simple molecular Hamiltonian for testing
        mol_hamiltonian = np.random.rand(5, 5)
        mol_hamiltonian = (mol_hamiltonian + mol_hamiltonian.T) / 2  # Make symmetric
        
        analyzer = BiodegradabilityAnalyzer(mol_hamiltonian, n_electrons=10)
        f_plus, f_minus, f_zero = analyzer.calculate_fukui_functions()
        biodegradability_score = analyzer.calculate_biodegradability_score()
        
        print(f"   ✓ Fukui functions calculated")
        print(f"   ✓ Biodegradability score: {biodegradability_score:.4f}")
        
        # Save biodegradability data to CSV
        bio_csv = f'./simulation_data/biodegradability_{timestamp}.csv'
        bio_data = {
            'site_index': list(range(len(f_plus))),
            'fukui_f_plus': f_plus.tolist(),
            'fukui_f_minus': f_minus.tolist(),
            'fukui_f_zero': f_zero.tolist(),
            'biodegradability_score': [biodegradability_score] * len(f_plus)
        }
        pd.DataFrame(bio_data).to_csv(bio_csv, index=False, float_format='%.6e')
        print(f"   ✓ Biodegradability data saved to: {bio_csv}")
        
        # Generate and save biodegradability figure
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(f_plus))
        width = 0.25
        ax.bar(x - width, f_plus, width, label='f⁺ (Nucleophilic)')
        ax.bar(x, f_minus, width, label='f⁻ (Electrophilic)')
        ax.bar(x + width, f_zero, width, label='f⁰ (Radical)')
        ax.set_xlabel('Atomic Site')
        ax.set_ylabel('Fukui Function Value')
        ax.set_title(f'Fukui Functions - Biodegradability Score: {biodegradability_score:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        bio_fig = f'./figures/biodegradability_{timestamp}.pdf'
        plt.savefig(bio_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Biodegradability figure saved to: {bio_fig}")
        
    except Exception as e:
        print(f"   ✗ Biodegradability analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: LCA Analysis
    print("\n5. Testing LCA Analysis...")
    try:
        lca = LCAAnalyzer()
        lca_results = lca.calculate_lca_impact(
            manufacturing_energy=1200,
            material_mass=0.3,
            carbon_intensity=0.5,
            maintenance_frequency=1.0
        )
        print(f"   ✓ LCA analysis completed")
        print(f"   ✓ Carbon footprint: {lca_results['carbon_footprint_gco2eq_per_kwh']:.2f} gCO2eq/kWh")
        print(f"   ✓ Energy payback time: {lca_results['energy_payback_time_years']:.2f} years")
        
        # Save LCA results to CSV
        lca_csv = f'./simulation_data/lca_analysis_{timestamp}.csv'
        lca_data = {
            'metric': list(lca_results.keys()),
            'value': list(lca_results.values())
        }
        pd.DataFrame(lca_data).to_csv(lca_csv, index=False, float_format='%.6e')
        print(f"   ✓ LCA data saved to: {lca_csv}")
        
        # Generate and save LCA figure
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use normalized metrics for visualization
        carbon = lca_results.get('carbon_footprint_gco2eq_per_kwh', 0)
        epbt = lca_results.get('energy_payback_time_years', 0)
        eroi = lca_results.get('eroi', 0)
        total_carbon = lca_results.get('total_carbon_kg_co2eq', 0)
        
        metrics = ['Carbon Footprint\n(gCO2eq/kWh)', 'Energy Payback\n(years)', 'EROI', 'Total Carbon\n(kg CO2eq)']
        values = [carbon, epbt, eroi, total_carbon]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylabel('Value')
        ax.set_title('Life Cycle Assessment Results')
        ax.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom')
        lca_fig = f'./figures/lca_analysis_{timestamp}.pdf'
        plt.savefig(lca_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ LCA figure saved to: {lca_fig}")
        
    except Exception as e:
        print(f"   ✗ LCA analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Testing and Validation
    print("\n6. Testing Validation Protocols...")
    try:
        validation = TestingValidationProtocols(simulator, H_fmo)
        hamiltonian_valid = validation.validate_fmo_hamiltonian()
        print(f"   ✓ Hamiltonian validation: {hamiltonian_valid}")
        
        # Save validation results
        validation_csv = f'./simulation_data/validation_{timestamp}.csv'
        validation_data = {
            'test_name': ['Hamiltonian Validation'],
            'status': ['PASSED' if hamiltonian_valid else 'FAILED'],
            'expected': ['Symmetric & Hermitian'],
            'actual': ['Valid' if hamiltonian_valid else 'Invalid']
        }
        pd.DataFrame(validation_data).to_csv(validation_csv, index=False)
        print(f"   ✓ Validation data saved to: {validation_csv}")
        
    except Exception as e:
        print(f"   ✗ Validation protocols failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST COMPLETED")
    print("="*60)
    print(f"\nAll simulation data saved to: ./simulation_data/")
    print(f"All figures saved to: ./figures/")
    print(f"Timestamp: {timestamp}")
    
    return True

if __name__ == "__main__":
    run_comprehensive_test()
