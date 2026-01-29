import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set up matplotlib parameters for better plots
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['lines.linewidth'] = 2
rcParams['figure.figsize'] = [10, 6]

def plot_quantum_metrics():
    # Run the simulation to get the quantum metrics
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Import the required functions and classes
    from quantum_agrivoltaics_simulations import QuantumDynamicsSimulator, create_fmo_hamiltonian
    
    # Initialize the simulator with the FMO Hamiltonian
    fmo_hamiltonian, fmo_energies = create_fmo_hamiltonian()
    qd_sim = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=295, dephasing_rate=10)
    
    # Run the enhanced simulation to get quantum metrics
    print("Running simulation to generate quantum metrics data...")
    time_points, density_matrices, populations, coherences, qfi_values, entropy_values, purity_values, linear_entropy_values, bipartite_ent_values, multipartite_ent_values, pairwise_concurrence_values = qd_sim.simulate_dynamics(
        time_points=np.linspace(0, 500, 500)  # fs
    )
    
    print(f"Simulation completed.")
    print(f"  Final QFI: {qfi_values[-1]:.4f}")
    print(f"  Final von Neumann entropy: {entropy_values[-1]:.4f}")
    print(f"  Final purity: {purity_values[-1]:.4f}")
    print(f"  Final linear entropy: {linear_entropy_values[-1]:.4f}")
    print(f"  Final bipartite entanglement: {bipartite_ent_values[-1]:.4f}")
    print(f"  Final multipartite entanglement: {multipartite_ent_values[-1]:.4f}")
    print(f"  Final pairwise concurrence: {pairwise_concurrence_values[-1]:.4f}")
    
    # Create subplots for essential quantum metrics (reduced from 6 to 4 for better clarity)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantum Metrics Evolution in Agrivoltaic System', fontsize=16, fontweight='bold')
    
    # Plot 1: Quantum Fisher Information
    axes[0, 0].plot(time_points, qfi_values, 'r-', label='QFI', linewidth=2)
    axes[0, 0].set_xlabel('Time (fs)')
    axes[0, 0].set_ylabel('Quantum Fisher Information')
    axes[0, 0].set_title('Quantum Fisher Information (QFI) Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: von Neumann Entropy and Purity (combined for information content)
    axes[0, 1].plot(time_points, entropy_values, 'b-', label='von Neumann Entropy', linewidth=2)
    ax2_twin = axes[0, 1].twinx()
    ax2_twin.plot(time_points, purity_values, 'g--', label='Purity', linewidth=2)
    axes[0, 1].set_xlabel('Time (fs)')
    axes[0, 1].set_ylabel('Entropy (kB)', color='b')
    ax2_twin.set_ylabel('Purity', color='g')
    axes[0, 1].set_title('Information Content (Entropy & Purity)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='upper left')
    ax2_twin.legend(loc='lower right')
    
    # Plot 3: Bipartite Entanglement
    axes[1, 0].plot(time_points, bipartite_ent_values, 'c-', label='Bipartite Entanglement', linewidth=2)
    axes[1, 0].set_xlabel('Time (fs)')
    axes[1, 0].set_ylabel('Entanglement Entropy')
    axes[1, 0].set_title('Bipartite Entanglement Evolution (Dimer Sites)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Multipartite Entanglement
    axes[1, 1].plot(time_points, multipartite_ent_values, 'orange', label='Multipartite Entanglement', linewidth=2)
    axes[1, 1].set_xlabel('Time (fs)')
    axes[1, 1].set_ylabel('Entanglement Measure')
    axes[1, 1].set_title('Multipartite Entanglement Evolution (FMO System)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fig_path = os.path.join(output_path, 'quantum_metrics_evolution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Quantum metrics figure saved to: {fig_path}")
    
    # Create another plot focusing on entanglement measures
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Entanglement Measures in FMO System', fontsize=16, fontweight='bold')
    
    # Bipartite entanglement
    axes2[0].plot(time_points, bipartite_ent_values, 'c-', label='Bipartite Entanglement', linewidth=2)
    axes2[0].set_xlabel('Time (fs)')
    axes2[0].set_ylabel('Entanglement Entropy')
    axes2[0].set_title('Bipartite Entanglement Evolution (Dimer Sites)')
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()
    
    # Multipartite entanglement
    axes2[1].plot(time_points, multipartite_ent_values, 'orange', label='Multipartite Entanglement', linewidth=2)
    axes2[1].set_xlabel('Time (fs)')
    axes2[1].set_ylabel('Entanglement Measure')
    axes2[1].set_title('Multipartite Entanglement Evolution (FMO System)')
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()
    
    fig2.tight_layout()
    
    fig2_path = os.path.join(output_path, 'entanglement_evolution.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Entanglement evolution figure saved to: {fig2_path}")
    
    # Create another plot focusing on QFI behavior
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_points, qfi_values, 'r-', label='QFI', linewidth=2)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Quantum Fisher Information')
    ax.set_title('Quantum Fisher Information Evolution (FMO System)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations for high QFI values
    max_qfi_idx = np.argmax(qfi_values)
    ax.annotate(f'Max QFI: {qfi_values[max_qfi_idx]:.2f}', 
                xy=(time_points[max_qfi_idx], qfi_values[max_qfi_idx]),
                xytext=(time_points[max_qfi_idx]+50, qfi_values[max_qfi_idx]+10),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
                fontsize=10, color='red')
    
    # Add horizontal line indicating expected range for FMO systems
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Expected range O(10-100)')
    ax.legend()
    
    fig3.tight_layout()
    
    fig3_path = os.path.join(output_path, 'qfi_evolution.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"QFI evolution figure saved to: {fig3_path}")
    
    plt.show()
    
    return time_points, qfi_values, entropy_values, purity_values, linear_entropy_values, bipartite_ent_values, multipartite_ent_values, pairwise_concurrence_values

if __name__ == "__main__":
    plot_quantum_metrics()