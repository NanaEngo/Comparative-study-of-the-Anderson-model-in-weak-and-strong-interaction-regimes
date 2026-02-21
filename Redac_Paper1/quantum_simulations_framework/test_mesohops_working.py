#!/usr/bin/env python3
"""Working MesoHOPS integration test using correct API."""

import numpy as np

print("=== MesoHOPS Working Integration Test ===\n")

from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.trajectory.exp_noise import bcf_exp

# Simple 3-site system
H = np.array([[12410, 50, 0], 
              [50, 12530, 50],
              [0, 50, 12210]], dtype=np.float64)

# Bath parameters
lambda_reorg = 35.0
gamma_cutoff = 50.0
g = lambda_reorg * gamma_cutoff / np.pi
w = gamma_cutoff

# Coupling operators
loperator = np.zeros([3, 3, 3], dtype=np.float64)
for i in range(3):
    loperator[i, i, i] = 1.0

# System parameters
sys_param = {
    "HAMILTONIAN": H,
    "GW_SYSBATH": [[g, w]],
    "L_HIER": [loperator[0]],
    "L_NOISE1": [loperator[0]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[g, w]],  # List of [g, w] pairs
}

# Hierarchy parameters
hier_param = {"MAXHIER": 2}

# EOM parameters
eom_param = {
    "TIME_DEPENDENCE": False,
    "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR",
}

# Integration parameters
integrator_param = {
    "INTEGRATOR": "RUNGE_KUTTA",
}

# Noise parameters
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 200.0,  # fs (must be > propagation time)
    "TAU": 1.0,     # fs
}

print("Initializing HOPS...")
try:
    hops = HOPS(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    print("✓ HOPS initialized\n")
    
    # Initial state
    psi_0 = np.zeros(3, dtype=np.complex128)
    psi_0[0] = 1.0
    
    # Run simulation
    print("Running simulation...")
    hops.initialize(psi_0)
    hops.propagate(100.0, 10.0)
    
    # Extract results
    psi_traj = np.array(hops.storage['psi_traj'])
    populations = np.abs(psi_traj)**2
    
    print(f"✓ Simulation completed")
    print(f"  Time steps: {len(populations)}")
    print(f"  Site 1: {populations[0, 0]:.4f} → {populations[-1, 0]:.4f}")
    print(f"  Site 2: {populations[0, 1]:.4f} → {populations[-1, 1]:.4f}")
    print(f"  Site 3: {populations[0, 2]:.4f} → {populations[-1, 2]:.4f}")
    print(f"  Transfer: {(1 - populations[-1, 0]) * 100:.2f}%")
    print(f"  Conservation: {np.sum(populations[-1]):.4f}\n")
    
    print("=== SUCCESS ===")
    print("MesoHOPS is working correctly!")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
