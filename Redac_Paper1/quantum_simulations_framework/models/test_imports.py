import sys
import os
import traceback

def test_imports():
    print("Testing SensitivityAnalyzer import debugging...")
    print(f"Current SYS PATH: {sys.path}")
    print(f"Current CWD: {os.getcwd()}")
    
    # Try the first import route
    try:
        from core.hops_simulator import HopsSimulator
        print("1. Success: from core.hops_simulator import HopsSimulator")
    except Exception as e:
        print(f"1. Failed: from core.hops_simulator import HopsSimulator")
        print(f"Reason: {type(e).__name__}: {e}")
        
    # Try the second import route
    try:
        from quantum_simulations_framework.core.hops_simulator import HopsSimulator
        print("2. Success: from quantum_simulations_framework.core.hops_simulator import HopsSimulator")
    except Exception as e:
        print(f"2. Failed: from quantum_simulations_framework.core.hops_simulator import HopsSimulator")
        print(f"Reason: {type(e).__name__}: {e}")
        
    # Try the third import route (the sys.path fallback route)
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.hops_simulator import HopsSimulator
        print("3. Success: sys.path.append(...) -> from core.hops_simulator import HopsSimulator")
    except Exception as e:
        print(f"3. Failed: sys.path.append(...) -> from core.hops_simulator import HopsSimulator")
        print(f"Reason: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_imports()
