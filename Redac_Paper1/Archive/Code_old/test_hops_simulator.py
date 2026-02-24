#!/usr/bin/env python3
"""
Test script to verify that HopsSimulator can initialize with fallback simulators.
"""

import sys
import os
sys.path.insert(0, '/media/taamangtchu/MYDATA/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework')

# Import the HopsSimulator
from core.hops_simulator import HopsSimulator

# Create a simple test Hamiltonian (2x2 for testing)
import numpy as np
H_test = np.array([[1.0, 0.5], [0.5, 1.0]])

print("Testing HopsSimulator initialization...")

try:
    # Initialize the simulator
    simulator = HopsSimulator(
        hamiltonian=H_test,
        temperature=295,
        use_mesohops=False  # Force use of fallback
    )
    
    print(f"✓ HopsSimulator initialized successfully")
    print(f"  - Simulator type: {simulator.simulator_type}")
    print(f"  - Using MesoHOPS: {simulator.is_using_mesohops}")
    
    # Test a simple dynamics simulation
    time_points = np.linspace(0, 10, 11)  # 11 points from 0 to 10 fs
    initial_state = np.array([1.0, 0.0])  # Excite first site
    
    print("Running test dynamics simulation...")
    results = simulator.simulate_dynamics(
        time_points=time_points,
        initial_state=initial_state
    )
    
    print(f"✓ Dynamics simulation completed")
    print(f"  - Simulator used: {results.get('simulator', 'Unknown')}")
    print(f"  - Time points: {len(results.get('t_axis', []))}")
    print(f"  - Populations shape: {results.get('populations', np.array([])).shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()