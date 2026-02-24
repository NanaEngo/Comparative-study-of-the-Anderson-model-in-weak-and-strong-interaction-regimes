import numpy as np
from core.hops_simulator import HopsSimulator
from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian

def main():
    H_fmo, energies = create_fmo_hamiltonian()
    
    # We must shift the Hamiltonian to center the energies around 0 to avoid highly oscillatory terms exp(-i E t / hbar)
    # The FMO site energies are ~12000 cm^-1.
    E_mean = np.mean(np.diag(H_fmo))
    print(f"Mean energy: {E_mean}")
    H_shifted = H_fmo - E_mean * np.eye(H_fmo.shape[0])
    
    # Create simulator
    simulator = HopsSimulator(
        H_shifted,
        temperature=295.0,
        max_hierarchy=2
    )
    
    time_points = np.linspace(0, 100, 10)
    initial_state = np.zeros(H_shifted.shape[0], dtype=complex)
    initial_state[0] = 1.0
    
    # Add seed to parameters to test
    print("Testing dynamics computation...")
    res = simulator.simulate_dynamics(time_points, initial_state=initial_state, seed=42)
    print("Populations at final step:")
    print(res['populations'][-1])

if __name__ == "__main__":
    main()
