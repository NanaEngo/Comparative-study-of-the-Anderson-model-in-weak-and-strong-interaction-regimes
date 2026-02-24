#!/usr/bin/env python3
"""
Test script to check simulator availability.
"""

import sys
import os
sys.path.insert(0, '/media/taamangtchu/MYDATA/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework')

try:
    from models.quantum_dynamics_simulator import QuantumDynamicsSimulator
    print("QuantumDynamicsSimulator imported successfully")
    
    import numpy as np
    dummy_ham = np.array([[0.0]])
    try:
        test_sim = QuantumDynamicsSimulator(dummy_ham)
        print("QuantumDynamicsSimulator instantiated successfully")
    except ImportError as e:
        print(f"QuantumDynamicsSimulator instantiation failed with ImportError: {e}")
    except Exception as e:
        print(f"QuantumDynamicsSimulator instantiation failed with other error: {e}")
        
except ImportError as e:
    print(f"QuantumDynamicsSimulator import failed: {e}")

try:
    from models.simple_quantum_dynamics_simulator import SimpleQuantumDynamicsSimulator
    print("SimpleQuantumDynamicsSimulator imported successfully")
    
    import numpy as np
    dummy_ham = np.array([[0.0]])
    try:
        test_sim = SimpleQuantumDynamicsSimulator(dummy_ham)
        print("SimpleQuantumDynamicsSimulator instantiated successfully")
    except Exception as e:
        print(f"SimpleQuantumDynamicsSimulator instantiation failed: {e}")
        
except ImportError as e:
    print(f"SimpleQuantumDynamicsSimulator import failed: {e}")