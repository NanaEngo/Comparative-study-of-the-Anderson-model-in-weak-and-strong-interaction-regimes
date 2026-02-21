#!/usr/bin/env python3
"""Complete MesoHOPS integration test with all required parameters."""

import numpy as np
import scipy.sparse as sp

print("=== MesoHOPS Complete Integration Test ===\n")

from mesohops import HopsSystem, HopsBasis, HopsEOM, HopsTrajectory

H = np.array([[12410, 50, 0], 
              [50, 12530, 50],
              [0, 50, 12210]], dtype=float)
n_sites = 3

lambda_reorg = 35.0
gamma_cutoff = 50.0
temperature = 295.0

gw_sysbath = [(lambda_reorg * gamma_cutoff / np.pi, gamma_cutoff)]
L_hier = [sp.csr_matrix(np.eye(n_sites))]
L_noise1 = [sp.csr_matrix(np.eye(n_sites))]  # Noise coupling operator

def drude_correlation(t, lambda_reorg, gamma_cutoff, temperature):
    return (lambda_reorg / np.pi) * gamma_cutoff * np.exp(-gamma_cutoff * np.abs(t))

system_param = {
    'HAMILTONIAN': H,
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': L_hier,
    'L_NOISE1': L_noise1,
    'ALPHA_NOISE1': drude_correlation,
    'PARAM_NOISE1': [lambda_reorg, gamma_cutoff, temperature],
}

print("Initializing HopsSystem...")
try:
    system = HopsSystem(system_param)
    print("✓ HopsSystem initialized\n")
    
    hierarchy_param = {'MAXHIER': 2}
    basis = HopsBasis(system, hierarchy_param)
    print("✓ HopsBasis created\n")
    
    eom_param = {'TIME_DEPENDENCE': False, 'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    eom = HopsEOM(basis, eom_param)
    print("✓ HopsEOM created\n")
    
    trajectory_param = {'TMAX': 100.0, 'TAU': 10.0}
    trajectory = HopsTrajectory(eom, trajectory_param)
    
    initial_state = np.zeros(n_sites, dtype=complex)
    initial_state[0] = 1.0
    
    trajectory.initialize(initial_state)
    trajectory.propagate(100.0, 10.0)
    
    psi_traj = np.array(trajectory.psi_traj)
    populations = np.abs(psi_traj)**2
    
    print(f"✓ Simulation completed")
    print(f"  Steps: {len(populations)}")
    print(f"  Site 1: {populations[0, 0]:.4f} → {populations[-1, 0]:.4f}")
    print(f"  Transfer: {(1 - populations[-1, 0]) * 100:.2f}%\n")
    print("=== SUCCESS ===")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
