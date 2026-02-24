#!/usr/bin/env python3
"""
Test script for MesoHOPS using the proper API from test examples.
"""

import numpy as np
import scipy as sp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.trajectory.exp_noise import bcf_exp

# System parameters (based on test examples)
nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 295.0
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

# Define system-bath coupling operators (Lindblad operators)
loperator = np.zeros([nsite, nsite, nsite], dtype=np.float64)
for i in range(nsite):
    loperator[i, i, i] = 1.0

# Build operator lists
lop_list = []
gw_sysbath = []
for i in range(nsite):
    gw_sysbath.append([g_0, w_0])
    lop_list.append(sp.sparse.coo_matrix(loperator[i]))
    gw_sysbath.append([-1j * np.imag(g_0), 500.0])
    lop_list.append(loperator[i])

# Hamiltonian (2-site system for simplicity)
hs = np.zeros([nsite, nsite])
hs[0, 1] = 40
hs[1, 0] = 40
hs[1, 2] = 10
hs[2, 1] = 10
hs[2, 3] = 40
hs[3, 2] = 40

# System parameters
sys_param = {
    "HAMILTONIAN": np.array(hs, dtype=np.complex128),
    "GW_SYSBATH": gw_sysbath,
    "L_HIER": lop_list,
    "L_NOISE1": lop_list,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": gw_sysbath,
}

# Noise parameters
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 100.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

# Hierarchy parameters
hier_param = {"MAXHIER": 4}

# EOM parameters
eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

# Integrator parameters
integrator_param = {
    "INTEGRATOR": "RUNGE_KUTTA",
}

# Initial state
psi_0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

print("Testing MesoHOPS simulation with proper API...")

# Create HopsTrajectory
hops = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
)

print("✓ HopsTrajectory created successfully")

# Initialize with initial state
hops.initialize(psi_0)
print("✓ Initial state set")

# Propagate the system
t_max = 50.0  # fs
dt_save = 2.0  # fs
hops.propagate(t_max, dt_save)
print(f"✓ Propagation completed to t_max = {t_max} fs")

# Extract results
t_axis = np.array(hops.storage.data['t_axis'])
psi_traj = np.array(hops.storage.data['psi_traj'])

print(f"✓ Results extracted:")
print(f"  - Time points: {len(t_axis)}")
print(f"  - Trajectory shape: {psi_traj.shape}")

# Calculate populations
n_times = len(t_axis)
populations = np.zeros((n_times, nsite))
for i in range(n_times):
    psi = psi_traj[i, :nsite]
    rho = np.outer(psi, np.conj(psi))
    populations[i, :] = np.real(np.diag(rho))

print(f"  - Populations shape: {populations.shape}")
print(f"  - Initial populations: {populations[0]}")
print(f"  - Final populations: {populations[-1]}")

print("\nMesoHOPS test completed successfully!")