#!/usr/bin/env python3
"""Validate updated notebook MesoHOPS implementation."""

import numpy as np
import sys

print("=== Notebook MesoHOPS Validation ===\n")

# Test 1: Import and initialize
print("1. Testing HopsSimulator from notebook...")
exec(open('quantum_coherence_agrivoltaics_mesohops.py').read(), globals())

# Create FMO Hamiltonian
FMO_ENERGIES = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])
n_sites = len(FMO_ENERGIES)
H = np.diag(FMO_ENERGIES.astype(float))

# Add couplings
couplings = 50.0
for i in range(n_sites-1):
    H[i, i+1] = couplings
    H[i+1, i] = couplings

print(f"   ✓ FMO Hamiltonian: {H.shape}")

# Test 2: Initialize simulator
print("\n2. Initializing HopsSimulator...")
try:
    sim = HopsSimulator(H, temperature=295, reorganization_energy=35.0, drude_cutoff=50.0)
    print(f"   ✓ Initialized (using MesoHOPS: {sim.use_mesohops})")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Run simulation
print("\n3. Running simulation (0-500 fs)...")
try:
    time_points = np.linspace(0, 500, 51)  # 0-500 fs, 51 points
    initial_state = np.zeros(n_sites, dtype=complex)
    initial_state[0] = 1.0
    
    results = sim.simulate_dynamics(time_points, initial_state, max_hierarchy=2)
    
    t, pops, _, coh, qfi, ent, pur, _, _, _, _ = results
    
    print(f"   ✓ Simulation completed")
    print(f"   Time points: {len(t)}")
    print(f"   Initial pop (site 1): {pops[0, 0]:.4f}")
    print(f"   Final pop (site 1): {pops[-1, 0]:.4f}")
    print(f"   Transfer: {(1 - pops[-1, 0]) * 100:.2f}%")
    print(f"   Coherence: {coh[0]:.4f} → {coh[-1]:.4f}")
    print(f"   Purity: {pur[0]:.4f} → {pur[-1]:.4f}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Validate results
print("\n4. Validating results...")
try:
    assert len(t) >= 50, f"Too few time points: {len(t)}"
    assert pops.shape == (len(t), n_sites), f"Wrong shape: {pops.shape}"
    assert np.allclose(np.sum(pops, axis=1), 1.0, atol=0.1), "Population not conserved"
    assert pops[-1, 0] < pops[0, 0], "No energy transfer"
    assert 0 <= pur[-1] <= 1, f"Invalid purity: {pur[-1]}"
    
    print("   ✓ All validations passed")
    
except AssertionError as e:
    print(f"   ✗ Validation failed: {e}")
    sys.exit(1)

print("\n=== VALIDATION PASSED ===")
print("Notebook MesoHOPS implementation is working correctly!")
print(f"\nKey metrics:")
print(f"  - Time resolution: {len(t)} points ✓")
print(f"  - Energy transfer: {(1 - pops[-1, 0]) * 100:.1f}% ✓")
print(f"  - Population conservation: {np.sum(pops[-1]):.4f} ✓")
print(f"  - Coherence decay: {coh[0]:.3f} → {coh[-1]:.3f} ✓")
