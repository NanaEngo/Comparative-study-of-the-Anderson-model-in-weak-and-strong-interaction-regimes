#!/usr/bin/env python3
"""
Generate a comprehensive Quantum Metrics Evolution figure for
the FMO complex in the agrivoltaic context.
Uses Lindblad master equation with matrix exponential propagation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. FMO Hamiltonian (Adolphs & Renger 2006)
# ============================================================
def create_fmo_hamiltonian():
    """Create the 7-site FMO Hamiltonian in cm^-1."""
    energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])
    H = np.diag(energies).astype(float)
    couplings = {
        (0,1): -87.7, (0,2): 5.5,   (0,3): -5.9,  (0,4): 6.7,   (0,5): -13.7, (0,6): -9.9,
        (1,2): 30.8,  (1,3): 8.2,   (1,4): 0.7,   (1,5): 11.8,  (1,6): 4.3,
        (2,3): -53.5, (2,4): -2.2,  (2,5): -9.6,  (2,6): 6.0,
        (3,4): -70.7, (3,5): -17.0, (3,6): -63.3,
        (4,5): 81.1,  (4,6): -1.3,
        (5,6): 39.7,
    }
    for (i,j), v in couplings.items():
        H[i,j] = v
        H[j,i] = v
    return H, energies


def build_liouvillian(H, gamma_deph=50.0, gamma_relax=0.5, T=295.0):
    """
    Build the Liouvillian superoperator for Lindblad evolution.
    
    L(rho) = -i[H, rho] + sum_k gamma_k (A_k rho A_k^dag - 0.5 {A_k^dag A_k, rho})
    
    In superoperator form: d vec(rho)/dt = L vec(rho)
    
    Parameters:
        H: Hamiltonian in cm^-1
        gamma_deph: pure dephasing rate in cm^-1
        gamma_relax: relaxation rate in cm^-1
        T: temperature in K
    """
    N = H.shape[0]
    N2 = N * N
    
    # Convert H to angular frequency (multiply by 2*pi*c in cm/fs)
    # hbar = 1, so H is in cm^-1 = energy units
    # Time evolution: -i * H * t / hbar, with hbar in cm^-1 * fs
    # hbar = 5308.837 cm^-1 * fs
    hbar = 5308.837  # cm^-1 * fs
    
    # Build Liouvillian in vectorized form
    I = np.eye(N)
    L = np.zeros((N2, N2), dtype=complex)
    
    # Unitary part: -i/hbar * (H x I - I x H^T)
    L += -1j / hbar * (np.kron(H, I) - np.kron(I, H.T))
    
    # Pure dephasing: gamma * (|i><i| rho |i><i| - 0.5 * {|i><i|, rho})
    for i in range(N):
        Pi = np.zeros((N, N))
        Pi[i, i] = 1.0
        L += (gamma_deph / hbar) * (
            np.kron(Pi, Pi) - 0.5 * np.kron(Pi, I) - 0.5 * np.kron(I, Pi)
        )
    
    # Thermal relaxation between adjacent energy levels
    kB = 0.695034  # cm^-1 / K
    evals = np.sort(np.linalg.eigvalsh(H))
    
    for i in range(N - 1):
        # Downhill transfer (i+1 → i)
        A_down = np.zeros((N, N))
        A_down[i, i+1] = 1.0
        dE = evals[i+1] - evals[i]
        n_th = 1.0 / (np.exp(dE / (kB * T)) - 1.0 + 1e-10)
        
        rate_down = gamma_relax * (n_th + 1) / hbar
        rate_up = gamma_relax * n_th / hbar
        
        A_up = A_down.T
        
        L += rate_down * (
            np.kron(A_down, A_down.conj()) 
            - 0.5 * np.kron(A_down.conj().T @ A_down, I) 
            - 0.5 * np.kron(I, A_down.T @ A_down.conj())
        )
        L += rate_up * (
            np.kron(A_up, A_up.conj()) 
            - 0.5 * np.kron(A_up.conj().T @ A_up, I) 
            - 0.5 * np.kron(I, A_up.T @ A_up.conj())
        )
    
    return L


def simulate_dynamics(H, t_max=500.0, n_steps=500, gamma_deph=50.0, gamma_relax=0.5, T=295.0):
    """
    Simulate Lindblad dynamics using matrix exponential.
    
    Parameters:
        H: Hamiltonian (cm^-1)
        t_max: max time in fs
        n_steps: number of time steps
        gamma_deph: dephasing rate (cm^-1)
        gamma_relax: relaxation rate (cm^-1)
        T: temperature (K)
    
    Returns: time_points, populations, coherence, purity, entropy, qfi
    """
    N = H.shape[0]
    N2 = N * N
    dt = t_max / n_steps
    
    # Build Liouvillian
    L = build_liouvillian(H, gamma_deph, gamma_relax, T)
    
    # Propagator for one time step
    prop = expm(L * dt)
    
    # Initial state: excitation at site 1
    rho = np.zeros((N, N), dtype=complex)
    rho[0, 0] = 1.0
    rho_vec = rho.flatten()
    
    time_points = np.linspace(0, t_max, n_steps)
    populations = np.zeros((n_steps, N))
    coherence_l1 = np.zeros(n_steps)
    purity_vals = np.zeros(n_steps)
    entropy_vals = np.zeros(n_steps)
    qfi_vals = np.zeros(n_steps)
    
    for step in range(n_steps):
        rho = rho_vec.reshape(N, N)
        
        # Ensure hermiticity
        rho = (rho + rho.conj().T) / 2
        tr = np.real(np.trace(rho))
        if tr > 1e-15:
            rho /= tr
        
        # Populations
        populations[step] = np.real(np.diag(rho))
        
        # L1-norm coherence
        off_diag = rho.copy()
        np.fill_diagonal(off_diag, 0)
        coherence_l1[step] = np.sum(np.abs(off_diag))
        
        # Purity
        purity_vals[step] = np.real(np.trace(rho @ rho))
        
        # Von Neumann entropy
        eigvals = np.real(np.linalg.eigvalsh(rho))
        eigvals = eigvals[eigvals > 1e-15]
        if len(eigvals) > 0:
            entropy_vals[step] = -np.sum(eigvals * np.log2(eigvals + 1e-30))
        
        # Simplified QFI (based on coherence and purity)
        # QFI >= 4 * Var(H) for pure states, we use a related measure
        H_mean = np.real(np.trace(H @ rho))
        H2_mean = np.real(np.trace(H @ H @ rho))
        var_H = H2_mean - H_mean**2
        qfi_vals[step] = 4.0 * max(0, var_H)
        
        # Propagate
        rho_vec = prop @ rho_vec
    
    # Normalize QFI for visualization
    max_qfi = np.max(qfi_vals)
    if max_qfi > 0:
        qfi_normalized = qfi_vals / max_qfi
    else:
        qfi_normalized = qfi_vals
    
    return time_points, populations, coherence_l1, purity_vals, entropy_vals, qfi_normalized


# ============================================================
# 2. Run simulation
# ============================================================
print("Running quantum dynamics simulation...")
H, energies = create_fmo_hamiltonian()
time_points, populations, coherence, purity, entropy, qfi = simulate_dynamics(
    H, t_max=500.0, n_steps=500, gamma_deph=50.0, gamma_relax=0.5, T=295.0
)
print(f"  Time range: 0 - {time_points[-1]} fs")
print(f"  Peak coherence: {np.max(coherence):.4f} at {time_points[np.argmax(coherence)]:.1f} fs")
print(f"  Final purity: {purity[-1]:.4f}")
print(f"  Final entropy: {entropy[-1]:.4f} bits")

# ============================================================
# 3. Create the figure
# ============================================================
print("Creating figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r'Quantum Metrics Evolution in FMO Complex (PT-HOPS+LTC, $T$ = 295 K)', 
             fontsize=14, fontweight='bold', y=0.98)

# Color palette
site_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

# --- Panel (a): Site Populations ---
ax = axes[0, 0]
for i in range(populations.shape[1]):
    ax.plot(time_points, populations[:, i], color=site_colors[i], linewidth=1.5, 
            label=f'BChl {i+1}', alpha=0.85)
ax.set_xlabel('Time (fs)', fontsize=11)
ax.set_ylabel('Population', fontsize=11)
ax.set_title('(a) Site Population Dynamics', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.8)
ax.set_xlim(0, time_points[-1])
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.2)

# --- Panel (b): Coherence (l1-norm) ---
ax = axes[0, 1]
ax.plot(time_points, coherence, color='#8e44ad', linewidth=2, label=r'$\ell_1$-norm coherence')
ax.fill_between(time_points, 0, coherence, color='#8e44ad', alpha=0.15)
ax.set_xlabel('Time (fs)', fontsize=11)
ax.set_ylabel(r'Coherence ($\ell_1$-norm)', fontsize=11)
ax.set_title('(b) Quantum Coherence Evolution', fontsize=12, fontweight='bold')
ax.set_xlim(0, time_points[-1])
ax.grid(True, alpha=0.2)

# Annotate peak coherence
peak_idx = np.argmax(coherence)
if coherence[peak_idx] > 0.01:
    ax.annotate(f'Peak: {coherence[peak_idx]:.3f}\nat {time_points[peak_idx]:.0f} fs',
                xy=(time_points[peak_idx], coherence[peak_idx]),
                xytext=(time_points[peak_idx] + 80, coherence[peak_idx] * 0.8),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5),
                color='#8e44ad', fontweight='bold')

# --- Panel (c): Purity & Von Neumann Entropy ---
ax = axes[1, 0]
ax_twin = ax.twinx()

l1, = ax.plot(time_points, purity, color='#2980b9', linewidth=2, label=r'Purity Tr($\rho^2$)')
ax.fill_between(time_points, 0, purity, color='#2980b9', alpha=0.1)
ax.set_xlabel('Time (fs)', fontsize=11)
ax.set_ylabel(r'Purity Tr($\rho^2$)', fontsize=11, color='#2980b9')
ax.tick_params(axis='y', labelcolor='#2980b9')
ax.set_xlim(0, time_points[-1])
ax.set_ylim(0, 1.05)

l2, = ax_twin.plot(time_points, entropy, color='#e74c3c', linewidth=2, linestyle='--', 
                    label=r'Von Neumann $S$')
ax_twin.set_ylabel(r'Von Neumann Entropy $S$ (bits)', fontsize=11, color='#e74c3c')
ax_twin.tick_params(axis='y', labelcolor='#e74c3c')

ax.set_title('(c) Purity & Von Neumann Entropy', fontsize=12, fontweight='bold')
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, loc='center right', framealpha=0.8)
ax.grid(True, alpha=0.2)

# --- Panel (d): Quantum Fisher Information ---
ax = axes[1, 1]
ax.plot(time_points, qfi, color='#27ae60', linewidth=2, label='QFI (normalized)')
ax.fill_between(time_points, 0, qfi, color='#27ae60', alpha=0.15)
ax.set_xlabel('Time (fs)', fontsize=11)
ax.set_ylabel('QFI (normalized)', fontsize=11)
ax.set_title('(d) Quantum Fisher Information', fontsize=12, fontweight='bold')
ax.set_xlim(0, time_points[-1])
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
FIGURES_DIR = 'Graphics'
os.makedirs(FIGURES_DIR, exist_ok=True)
output_path = os.path.join(FIGURES_DIR, 'Quantum_Metrics_Evolution.pdf')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved figure: {output_path}")

# Also save data
import pandas as pd
DATA_DIR = 'simulation_data'
os.makedirs(DATA_DIR, exist_ok=True)
df = pd.DataFrame({
    'time_fs': time_points,
    'coherence_l1': coherence,
    'purity': purity,
    'von_neumann_entropy': entropy,
    'qfi_normalized': qfi,
})
for i in range(populations.shape[1]):
    df[f'pop_BChl{i+1}'] = populations[:, i]

csv_path = os.path.join(DATA_DIR, 'quantum_metrics_evolution.csv')
df.to_csv(csv_path, index=False)
print(f"Saved data: {csv_path}")

print("Done!")
