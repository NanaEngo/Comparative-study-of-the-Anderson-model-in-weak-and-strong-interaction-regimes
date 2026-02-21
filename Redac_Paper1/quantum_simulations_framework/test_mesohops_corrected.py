#!/usr/bin/env python3
"""Corrected MesoHOPS integration test."""

import numpy as np
import scipy.sparse as sp

print("=== MesoHOPS Integration Test (Corrected) ===\n")

# Import
print("1. Importing MesoHOPS...")
from mesohops import HopsSystem, HopsBasis, HopsEOM, HopsTrajectory
print("   ✓ Imported")

# Hamiltonian
print("\n2. Creating Hamiltonian...")
H = np.array([[12410, 50, 0], 
              [50, 12530, 50],
              [0, 50, 12210]], dtype=float)
n_sites = 3
print(f"   ✓ {H.shape}")

# Parameters
print("\n3. Configuring parameters...")
lambda_reorg = 35.0
gamma_cutoff = 50.0
temperature = 295.0

# CRITICAL: L_HIER length must match GW_SYSBATH length
# For single bath mode, use one coupling operator
gw_sysbath = [(lambda_reorg * gamma_cutoff / np.pi, gamma_cutoff)]
L_hier = [sp.csr_matrix(np.eye(n_sites))]  # Single operator for single bath mode

def drude_correlation(t, lambda_reorg, gamma_cutoff, temperature):
    return (lambda_reorg / np.pi) * gamma_cutoff * np.exp(-gamma_cutoff * np.abs(t))

system_param = {
    'HAMILTONIAN': H,
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': L_hier,
    'ALPHA_NOISE1': drude_correlation,
    'PARAM_NOISE1': [lambda_reorg, gamma_cutoff, temperature],
}
print(f"   ✓ Bath modes: {len(gw_sysbath)}, Coupling ops: {len(L_hier)}")

# Initialize
print("\n4. Initializing HopsSystem...")
try:
    system = HopsSystem(system_param)
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Basis
print("\n5. Creating HopsBasis...")
try:
    hierarchy_param = {'MAXHIER': 2}
    basis = HopsBasis(system, hierarchy_param)
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# EOM
print("\n6. Creating HopsEOM...")
try:
    eom_param = {
        'TIME_DEPENDENCE': False,
        'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR',
    }
    eom = HopsEOM(basis, eom_param)
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Trajectory
print("\n7. Running simulation...")
try:
    trajectory_param = {
        'TMAX': 100.0,
        'TAU': 10.0,
    }
    trajectory = HopsTrajectory(eom, trajectory_param)
    
    initial_state = np.zeros(n_sites, dtype=complex)
    initial_state[0] = 1.0
    
    trajectory.initialize(initial_state)
    trajectory.propagate(100.0, 10.0)
    
    psi_traj = np.array(trajectory.psi_traj)
    populations = np.abs(psi_traj)**2
    
    print(f"   ✓ Completed")
    print(f"   Steps: {len(populations)}")
    print(f"   Site 1: {populations[0, 0]:.4f} → {populations[-1, 0]:.4f}")
    print(f"   Transfer: {(1 - populations[-1, 0]) * 100:.2f}%")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== SUCCESS ===")
