#!/usr/bin/env python3
"""Direct test of MesoHOPS HopsSimulator class."""

import numpy as np
import scipy.sparse as sp

print("=== Direct MesoHOPS Integration Test ===\n")

# Test 1: Import MesoHOPS
print("1. Testing MesoHOPS import...")
try:
    from mesohops import HopsSystem, HopsBasis, HopsEOM, HopsTrajectory
    print("   ✓ MesoHOPS classes imported")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Create FMO Hamiltonian
print("\n2. Creating FMO Hamiltonian...")
H = np.array([[12410, 50, 0], 
              [50, 12530, 50],
              [0, 50, 12210]], dtype=float)
n_sites = 3
print(f"   ✓ Hamiltonian: {H.shape}")

# Test 3: Setup MesoHOPS parameters
print("\n3. Setting up MesoHOPS parameters...")
lambda_reorg = 35.0  # cm^-1
gamma_cutoff = 50.0  # cm^-1
temperature = 295.0  # K

L_hier = [sp.csr_matrix(np.eye(n_sites)) for _ in range(n_sites)]
gw_sysbath = [(lambda_reorg * gamma_cutoff / np.pi, gamma_cutoff)]

def drude_correlation(t, lambda_reorg, gamma_cutoff, temperature):
    return (lambda_reorg / np.pi) * gamma_cutoff * np.exp(-gamma_cutoff * np.abs(t))

system_param = {
    'HAMILTONIAN': H,
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': L_hier,
    'ALPHA_NOISE1': drude_correlation,
    'PARAM_NOISE1': [lambda_reorg, gamma_cutoff, temperature],
}
print("   ✓ Parameters configured")

# Test 4: Initialize HopsSystem
print("\n4. Initializing HopsSystem...")
try:
    system = HopsSystem(system_param)
    print("   ✓ HopsSystem initialized")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Create Basis
print("\n5. Creating HopsBasis...")
try:
    hierarchy_param = {'MAXHIER': 2}
    basis = HopsBasis(system, hierarchy_param)
    print(f"   ✓ HopsBasis created")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Create EOM
print("\n6. Creating HopsEOM...")
try:
    eom_param = {
        'TIME_DEPENDENCE': False,
        'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR',
    }
    eom = HopsEOM(basis, eom_param)
    print("   ✓ HopsEOM created")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Create and run Trajectory
print("\n7. Running HopsTrajectory...")
try:
    trajectory_param = {
        'TMAX': 100.0,  # fs
        'TAU': 10.0,    # fs
    }
    trajectory = HopsTrajectory(eom, trajectory_param)
    
    # Initial state (site 1 excited)
    initial_state = np.zeros(n_sites, dtype=complex)
    initial_state[0] = 1.0
    
    trajectory.initialize(initial_state)
    trajectory.propagate(100.0, 10.0)
    
    psi_traj = np.array(trajectory.psi_traj)
    populations = np.abs(psi_traj)**2
    
    print(f"   ✓ Trajectory completed")
    print(f"   Time steps: {len(populations)}")
    print(f"   Initial pop (site 1): {populations[0, 0]:.4f}")
    print(f"   Final pop (site 1): {populations[-1, 0]:.4f}")
    print(f"   Transfer: {(1 - populations[-1, 0]) * 100:.2f}%")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== ALL TESTS PASSED ===")
print("MesoHOPS is working correctly!")
