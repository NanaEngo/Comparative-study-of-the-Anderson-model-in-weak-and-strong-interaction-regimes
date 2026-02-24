#!/usr/bin/env python3
"""
Test script for HopsSimulator with MesoHOPS after installation.
"""

import sys
import os
sys.path.insert(0, '/media/taamangtchu/MYDATA/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework')

import numpy as np

print("Testing HopsSimulator with MesoHOPS...")

# Check MesoHOPS availability
try:
    from mesohops.basis.hops_system import HopsSystem
    from mesohops.basis.hops_basis import HopsBasis
    from mesohops.trajectory.hops_trajectory import HopsTrajectory
    print("✓ MesoHOPS modules imported successfully")
except ImportError as e:
    print(f"✗ MesoHOPS import failed: {e}")
    sys.exit(1)

# Create a simple Hamiltonian
H = np.array([[1.0, 0.5], [0.5, 1.0]])

# Initialize HopsSimulator with MesoHOPS
from core.hops_simulator import HopsSimulator
simulator = HopsSimulator(H, temperature=295, use_mesohops=True)

print(f"✓ HopsSimulator initialized successfully")
print(f"  - Simulator type: {simulator.simulator_type}")
print(f"  - Using MesoHOPS: {simulator.is_using_mesohops}")

if simulator.is_using_mesohops:
    print("✓ MesoHOPS is being used for simulation")
else:
    print("✗ MesoHOPS is not being used (fallback)")

# Run a quick simulation
time_points = np.linspace(0, 50, 11)
initial_state = np.array([1.0, 0.0])

try:
    results = simulator.simulate_dynamics(time_points, initial_state)
    print(f"✓ Dynamics simulation completed")
    print(f"  - Simulator used: {results.get('simulator', 'unknown')}")
    print(f"  - Time points: {len(results.get('t_axis', []))}")
    print(f"  - Populations shape: {results.get('populations', np.array([])).shape}")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")