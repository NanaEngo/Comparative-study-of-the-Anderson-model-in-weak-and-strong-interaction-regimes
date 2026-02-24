"""
Simple quantum dynamics simulator for educational purposes.
This provides a minimal fallback that doesn't require MesoHOPS.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimpleQuantumDynamicsSimulator:
    """
    A simple quantum dynamics simulator that doesn't require MesoHOPS.
    
    This uses a basic Schrodinger equation solver to simulate quantum dynamics.
    It's not as accurate as the MesoHOPS implementation but serves as a fallback.
    """
    
    def __init__(self, hamiltonian, temperature=295):
        self.hamiltonian = np.array(hamiltonian, dtype=complex)
        self.temperature = temperature
        self.n_sites = self.hamiltonian.shape[0]
        
        # Shift Hamiltonian to improve numerical stability
        self.energy_shift = np.mean(np.diag(self.hamiltonian).real)
        self.h_shifted = self.hamiltonian - self.energy_shift * np.eye(self.n_sites)
        
        logger.info(f"Simple quantum simulator initialized with {self.n_sites} sites")
    
    def simulate_dynamics(self, time_points=None, initial_state=None, dt=0.1):
        """
        Simulate quantum dynamics using the time-dependent Schrödinger equation.
        
        iħ d/dt |ψ⟩ = H |ψ⟩
        
        The solution is |ψ(t)⟩ = exp(-iHt/ħ) |ψ(0)⟩
        """
        # Set default time points if not provided
        if time_points is None:
            time_points = np.linspace(0, 100, 101)  # 101 points from 0 to 100 fs
        
        # Set initial state if not provided (excitation on first site)
        if initial_state is None:
            initial_state = np.zeros(self.n_sites, dtype=complex)
            initial_state[0] = 1.0
        
        initial_state = np.array(initial_state, dtype=complex)
        
        # Number of time points
        n_times = len(time_points)
        
        # Initialize result arrays
        populations = np.zeros((n_times, self.n_sites))
        coherences = np.zeros(n_times)
        qfi_values = np.zeros(n_times)
        entropy_values = np.zeros(n_times)
        purity_values = np.zeros(n_times)
        linear_entropy_values = np.zeros(n_times)
        
        # Calculate time evolution
        logger.info(f"Simulating quantum dynamics for {n_times} time points...")
        
        for i, t in enumerate(time_points):
            # Time evolution operator: U(t) = exp(-iHt)
            # Using units where ħ=1 and energy is in cm^-1, time in fs
            # We need to account for units: E*t in units of (cm^-1) * fs
            # 1 cm^-1 = 3.3356e-12 fs^-1 (in natural units)
            # So we multiply by appropriate conversion factor
            time_factor = t * 0.0333564  # Conversion factor to match units
            
            # Calculate U(t) = exp(-i * H_shifted * time_factor)
            eigenvals, eigenvecs = np.linalg.eigh(self.h_shifted)
            exp_evals = np.exp(-1j * eigenvals * time_factor)
            
            # U = V * exp(-i*E*t) * V^dagger
            U = eigenvecs @ np.diag(exp_evals) @ eigenvecs.T.conj()
            
            # Evolved state
            psi_t = U @ initial_state
            
            # Density matrix
            rho_t = np.outer(psi_t, psi_t.conj())
            
            # Calculate observables
            populations[i, :] = np.real(np.diag(rho_t))
            
            # Calculate coherence measure (L1 norm of off-diagonal elements)
            off_diag_sum = 0.0
            for m in range(self.n_sites):
                for n in range(self.n_sites):
                    if m != n:
                        off_diag_sum += np.abs(rho_t[m, n])
            coherences[i] = off_diag_sum
            
            # Calculate purity: Tr[rho^2]
            purity = np.real(np.trace(rho_t @ rho_t))
            purity_values[i] = purity
            
            # Calculate von Neumann entropy: -Tr[rho * log(rho)]
            eigenvals_rho = np.linalg.eigvals(rho_t)
            eigenvals_rho = np.real(eigenvals_rho)  # Take real part to handle numerical errors
            # Ensure non-negative and normalize
            eigenvals_rho = np.clip(eigenvals_rho, 0, None)
            total = np.sum(eigenvals_rho)
            if total > 1e-12:
                eigenvals_rho = eigenvals_rho / total
            else:
                eigenvals_rho = np.ones_like(eigenvals_rho) / len(eigenvals_rho)
            
            # Calculate entropy, avoiding log(0)
            entropy = 0.0
            for ev in eigenvals_rho:
                if ev > 1e-12:
                    entropy += -ev * np.log(ev)
            entropy_values[i] = entropy
            
            # Calculate linear entropy: 1 - Tr[rho^2]
            linear_entropy_values[i] = 1.0 - purity_values[i]
            
            # Simplified QFI calculation
            qfi_values[i] = 4 * np.var(np.real(np.diag(rho_t)))
            
            if (i + 1) % max(1, n_times // 10) == 0:  # Print progress every 10%
                logger.info(f"  {i+1}/{n_times} time steps completed")
        
        logger.info("Quantum dynamics simulation completed")
        
        return {
            't_axis': time_points,
            'populations': populations,
            'coherences': coherences,
            'qfi': qfi_values,
            'entropy': entropy_values,
            'purity': purity_values,
            'linear_entropy': linear_entropy_values,
            'bipartite_ent': np.zeros(n_times),
            'multipartite_ent': np.zeros(n_times),
            'pairwise_concurrence': np.zeros(n_times),
            'discord': np.zeros(n_times),
            'fidelity': np.ones(n_times),  # Dummy fidelity
            'mandel_q': np.zeros(n_times),
            'simulator': 'Simple Quantum Simulator',
        }