#!/usr/bin/env python3
"""
Test script for quantum dynamics simulation using MesoHOPS in agrivoltaic systems.
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
from core.constants import DEFAULT_TEMPERATURE, DEFAULT_REORGANIZATION_ENERGY, DEFAULT_DRUDE_CUTOFF
from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian

# Ensure output directories exist
os.makedirs('./simulation_data', exist_ok=True)
os.makedirs('./figures', exist_ok=True)

def test_quantum_simulation():
    """Test the quantum dynamics simulation and save all data/figures."""
    print("Testing Quantum Dynamics Simulation with MesoHOPS...")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create FMO Hamiltonian
    print("1. Creating FMO Hamiltonian...")
    H_fmo, site_energies = create_fmo_hamiltonian()
    print(f"   Hamiltonian shape: {H_fmo.shape}")
    print(f"   Site energies: {site_energies}")
    
    # Save Hamiltonian data to CSV
    hamiltonian_csv = f'./simulation_data/fmo_hamiltonian_{timestamp}.csv'
    np.savetxt(hamiltonian_csv, H_fmo, delimiter=',', fmt='%.6f')
    print(f"   ✓ Hamiltonian saved to: {hamiltonian_csv}")
    
    # Save site energies to CSV
    energies_csv = f'./simulation_data/fmo_site_energies_{timestamp}.csv'
    np.savetxt(energies_csv, site_energies, delimiter=',', fmt='%.2f', header='site_energy_cm-1')
    print(f"   ✓ Site energies saved to: {energies_csv}")
    
    # Generate and save FMO Hamiltonian figure
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
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
    
    # Initialize HopsSimulator with specific parameters
    print("\n2. Initializing HopsSimulator...")
    try:
        simulator = HopsSimulator(
            H_fmo,
            temperature=DEFAULT_TEMPERATURE,
            reorganization_energy=DEFAULT_REORGANIZATION_ENERGY,
            drude_cutoff=DEFAULT_DRUDE_CUTOFF,
            max_hierarchy=6  # Use a smaller hierarchy for testing
        )
        print(f"   Simulator type: {simulator.simulator_type}")
        print(f"   Using MesoHOPS: {simulator.is_using_mesohops}")
    except Exception as e:
        print(f"   Error initializing simulator: {e}")
        return
    
    # Define time points for simulation
    print("\n3. Setting up simulation parameters...")
    time_points = np.linspace(0, 100, 50)  # 50 time points from 0 to 100 fs
    print(f"   Time points: {len(time_points)} points from {time_points[0]} to {time_points[-1]} fs")
    
    # Set up initial state (excitation on first site)
    initial_state = np.zeros(H_fmo.shape[0], dtype=complex)
    initial_state[0] = 1.0  # Start with excitation on site 1
    print(f"   Initial state: excitation on site 0")
    
    # Run simulation
    print("\n4. Running quantum dynamics simulation...")
    try:
        results = simulator.simulate_dynamics(
            time_points=time_points,
            initial_state=initial_state
        )
        
        print("   Simulation completed successfully!")
        print(f"   Time axis shape: {results['t_axis'].shape}")
        print(f"   Populations shape: {results['populations'].shape}")
        print(f"   Coherences shape: {results['coherences'].shape}")
        
        # Print some results
        print(f"   Final populations: {results['populations'][-1]}")
        print(f"   Final coherence: {results['coherences'][-1]}")
        
        # Save simulation results to CSV
        print("\n5. Saving simulation data to CSV...")
        dynamics_csv = f'./simulation_data/quantum_dynamics_{timestamp}.csv'
        
        # Prepare data for CSV
        import pandas as pd
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
            print(f"   ✓ QFI included in CSV")
        if 'entropy' in results:
            dynamics_data['entropy'] = results['entropy']
            print(f"   ✓ Entropy included in CSV")
        if 'purity' in results:
            dynamics_data['purity'] = results['purity']
            print(f"   ✓ Purity included in CSV")
        
        pd.DataFrame(dynamics_data).to_csv(dynamics_csv, index=False, float_format='%.6e')
        print(f"   ✓ Simulation data saved to: {dynamics_csv}")
        
        # Generate and save figures
        print("\n6. Generating and saving figures...")
        
        # Population evolution figure
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(results['populations'].shape[1]):
            ax.plot(results['t_axis'], results['populations'][:, i], label=f'Site {i}', linewidth=2)
        ax.set_xlabel('Time (fs)', fontsize=12)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title('Quantum Dynamics: Population Evolution', fontsize=14)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        pop_fig_pdf = f'./figures/quantum_dynamics_populations_{timestamp}.pdf'
        pop_fig_png = f'./figures/quantum_dynamics_populations_{timestamp}.png'
        plt.savefig(pop_fig_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(pop_fig_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Population evolution saved to: {pop_fig_pdf}")
        
        # Coherence evolution figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results['t_axis'], results['coherences'], linewidth=2, color='#e74c3c')
        ax.set_xlabel('Time (fs)', fontsize=12)
        ax.set_ylabel('Total Coherence', fontsize=12)
        ax.set_title('Quantum Dynamics: Coherence Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        coh_fig_pdf = f'./figures/quantum_dynamics_coherence_{timestamp}.pdf'
        coh_fig_png = f'./figures/quantum_dynamics_coherence_{timestamp}.png'
        plt.savefig(coh_fig_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(coh_fig_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Coherence evolution saved to: {coh_fig_pdf}")
        
        # Quantum metrics figures if available
        if 'qfi' in results or 'entropy' in results or 'purity' in results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            if 'qfi' in results:
                axes[0].plot(results['t_axis'], results['qfi'], linewidth=2, color='#3498db')
                axes[0].set_xlabel('Time (fs)')
                axes[0].set_ylabel('QFI')
                axes[0].set_title('Quantum Fisher Information')
                axes[0].grid(True, alpha=0.3)
            
            if 'entropy' in results:
                axes[1].plot(results['t_axis'], results['entropy'], linewidth=2, color='#2ecc71')
                axes[1].set_xlabel('Time (fs)')
                axes[1].set_ylabel('Entropy')
                axes[1].set_title('Von Neumann Entropy')
                axes[1].grid(True, alpha=0.3)
            
            if 'purity' in results:
                axes[2].plot(results['t_axis'], results['purity'], linewidth=2, color='#9b59b6')
                axes[2].set_xlabel('Time (fs)')
                axes[2].set_ylabel('Purity')
                axes[2].set_title('State Purity')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            metrics_fig_pdf = f'./figures/quantum_metrics_{timestamp}.pdf'
            metrics_fig_png = f'./figures/quantum_metrics_{timestamp}.png'
            plt.savefig(metrics_fig_pdf, dpi=300, bbox_inches='tight')
            plt.savefig(metrics_fig_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Quantum metrics saved to: {metrics_fig_pdf}")
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nAll simulation data saved to: ./simulation_data/")
        print(f"All figures saved to: ./figures/")
        print(f"Timestamp: {timestamp}")
        
        return results
        
    except Exception as e:
        print(f"   Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_quantum_simulation()