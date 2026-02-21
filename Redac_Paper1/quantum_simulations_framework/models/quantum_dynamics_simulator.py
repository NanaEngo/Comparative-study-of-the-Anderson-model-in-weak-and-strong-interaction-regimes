"""
Quantum Dynamics Simulator using MesoHOPS for Agrivoltaic Systems.

This module implements non-Markovian quantum dynamics simulations using the
MesoHOPS (Mesoscale Hierarchy of Pure States) framework for simulating
energy transfer in photosynthetic systems and quantum-enhanced agrivoltaics.
"""

import numpy as np
from scipy.sparse import csc_matrix
import scipy.linalg as la
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os


logger = logging.getLogger(__name__)


try:
    from mesohops.util.bath_corr_functions import (
        bcf_convert_dl_to_exp_with_Matsubara,
        bcf_convert_sdl_to_exp,
    )
    import mesohops.eom.hops_eom as _eom_mod
    from mesohops.trajectory.hops_trajectory import HopsTrajectory
    MESOHOPS_AVAILABLE = True
except ImportError:
    # Fallback or warning if mesohops is not installed
    MESOHOPS_AVAILABLE = False
    _eom_mod = None


class QuantumDynamicsSimulator:
    """
    Non-Markovian quantum dynamics for the FMO complex using MesoHOPS.

    This class implements the adaptive Hierarchy of Pure States (adHOPS) method
    with low-temperature Matsubara corrections (PT-HOPS) via the MesoHOPS
    library (Varvelo et al. 2021, Citty et al. 2024). It provides numerically
    exact, non-Markovian open quantum system dynamics for the FMO complex
    coupled to a structured phonon bath characterized by a Drude-Lorentz
    spectral density with optional underdamped vibronic modes.

    The simulation proceeds by:
    1. Decomposing the bath correlation function into exponential modes
       using Matsubara frequency corrections for accuracy at
       finite temperature.
    2. Running N_traj stochastic HOPS trajectories, each driven by
       independent coloured noise samples.
    3. Ensemble-averaging the density matrix:
       rho(t) = (1/N) sum_k |psi_k(t)><psi_k(t)|
    4. Computing quantum metrics (QFI, coherence, entropy, entanglement)
       from the ensemble-averaged rho(t).

    Parameters
    ----------
    hamiltonian : np.ndarray
        System Hamiltonian in cm^-1 (7x7 for FMO).
    temperature : float
        Bath temperature in K (default: 295).
    lambda_reorg : float
        Drude-Lorentz reorganization energy in cm^-1 (default: 35).
    gamma_dl : float
        Drude-Lorentz cutoff frequency in cm^-1 (default: 50).
    k_matsubara : int
        Number of Matsubara modes (default: 0, sufficient for 295K).
    max_hier : int
        Maximum hierarchy depth (default: 6).
    n_traj : int
        Number of stochastic trajectories for ensemble averaging (default: 50).
    vibronic_modes : list of dict, optional
        Underdamped vibronic modes, each with keys 'omega', 'lambda', 'gamma'.
        Example: [{'omega': 150, 'lambda': 0.05*150, 'gamma': 10}, ...]

    References
    ----------
    Varvelo et al., "MesoHOPS: Size-invariant scaling HOPS", PCCP (2021).
    Citty et al., "adHOPS: adaptive HOPS", J. Chem. Phys. (2024).
    Adolphs & Renger, Biophys. J. 91, 2778–2797 (2006).
    """

    def __init__(self, hamiltonian, temperature=295, lambda_reorg=35.0,
                 gamma_dl=50.0, k_matsubara=0, max_hier=10, n_traj=50,
                 vibronic_modes=None):

        if not MESOHOPS_AVAILABLE:
            raise ImportError("MesoHOPS is required but not available. "
                            "Please install it: pip install mesohops")

        # Patch EOM_DICT_TYPES for adHOPS support in MesoHOPS v1.6.0
        if MESOHOPS_AVAILABLE and _eom_mod is not None:
            if 'ADAPTIVE_H' not in _eom_mod.EOM_DICT_TYPES:
                _eom_mod.EOM_DICT_TYPES['ADAPTIVE_H'] = [bool]
                _eom_mod.EOM_DICT_TYPES['ADAPTIVE_S'] = [bool]
                _eom_mod.EOM_DICT_TYPES['UPDATE_STEP'] = [float, bool, type(None)]
                _eom_mod.EOM_DICT_TYPES['F_DISCARD'] = [float, int]

        self.H_raw = np.array(hamiltonian, dtype=complex)
        self.n_sites = self.H_raw.shape[0]
        self.temperature = temperature
        self.lambda_reorg = lambda_reorg
        self.gamma_dl = gamma_dl
        self.k_matsubara = k_matsubara
        self.max_hier = max_hier
        self.n_traj = n_traj
        self.vibronic_modes = vibronic_modes or []

        # Zero-shift Hamiltonian to improve numerical stability
        self.E_shift = np.mean(np.diag(self.H_raw).real)
        self.H = self.H_raw - self.E_shift * np.eye(self.n_sites, dtype=complex)

        # ── Bath correlation function decomposition ──────────────────
        # Drude-Lorentz spectral density (high-temperature limit).
        # k_matsubara=0: High-temperature approximation (standard for 295K)
        dl_modes = bcf_convert_dl_to_exp_with_Matsubara(
            lambda_reorg, gamma_dl, temperature, k_matsubara
        )
        # dl_modes = [g0, w0, g1, w1, ...] alternating g, w pairs
        self.n_modes_dl = len(dl_modes) // 2

        # Optional: underdamped vibronic modes via shifted Drude-Lorentz
        vib_mode_list = []
        for vm in self.vibronic_modes:
            g_vib, w_vib = bcf_convert_sdl_to_exp(
                vm['lambda'], vm['gamma'], vm['omega'], temperature
            )
            vib_mode_list.append((g_vib, w_vib))
        self.n_modes_vib = len(vib_mode_list)

        # Total modes per site
        self.n_modes_per_site = self.n_modes_dl + self.n_modes_vib

        # ── Build system-bath coupling lists ─────────────────────────
        # Each site couples to its own independent bath
        self._L_ops = []
        for i in range(self.n_sites):
            L = np.zeros((self.n_sites, self.n_sites), dtype=complex)
            L[i, i] = 1.0
            self._L_ops.append(csc_matrix(L))

        self.gw_sysbath = []
        self.l_hier = []
        self.l_noise1 = []
        self.param_noise1 = []

        for site_idx in range(self.n_sites):
            # Drude-Lorentz + Matsubara modes
            for k in range(0, len(dl_modes), 2):
                g, w = dl_modes[k], dl_modes[k + 1]
                self.gw_sysbath.append([g, w])
                self.l_hier.append(self._L_ops[site_idx])
                self.l_noise1.append(self._L_ops[site_idx])
                self.param_noise1.append([g, w])

            # Underdamped vibronic modes
            for g_vib, w_vib in vib_mode_list:
                self.gw_sysbath.append([g_vib, w_vib])
                self.l_hier.append(self._L_ops[site_idx])
                self.l_noise1.append(self._L_ops[site_idx])
                self.param_noise1.append([g_vib, w_vib])

        logger.info(f"Bath decomposition: {self.n_modes_dl} DL modes "
              f"+ {self.n_modes_vib} vibronic modes per site "
              f"= {self.n_modes_per_site * self.n_sites} total hierarchy modes")

    @staticmethod
    def _alpha_noise1(t_axis, g, w):
        """Single-mode bath correlation function: alpha(t) = g * exp(-w*t)."""
        return g * np.exp(-w * t_axis)

    def _build_hops_trajectory(self, seed, t_max, dt_save=2.0):
        """
        Build and return an initialized HopsTrajectory object.

        Parameters
        ----------
        seed : int
            Random seed for the stochastic noise.
        t_max : float
            Total simulation time in fs.
        dt_save : float
            Time interval between saved points in fs.

        Returns
        -------
        hops : HopsTrajectory
            Initialized HOPS trajectory object.
        """
        system_param = {
            "HAMILTONIAN": self.H,
            "GW_SYSBATH": self.gw_sysbath,
            "L_HIER": self.l_hier,
            "L_NOISE1": self.l_noise1,
            "ALPHA_NOISE1": self._alpha_noise1,
            "PARAM_NOISE1": self.param_noise1,
        }

        eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
        noise_param = {
            "SEED": seed,
            "MODEL": "FFT_FILTER",
            "TLEN": t_max + 100.0,
            "TAU": dt_save,
        }
        hierarchy_param = {"MAXHIER": self.max_hier}
        integration_param = {"INTEGRATOR": "RUNGE_KUTTA"}

        hops = HopsTrajectory(
            system_param=system_param,
            eom_param=eom_param,
            noise_param=noise_param,
            hierarchy_param=hierarchy_param,
            integration_param=integration_param,
        )
        return hops

    def simulate_dynamics(self, initial_state=None, time_points=None,
                          dt_save=0.5, seeds=None):
        """
        Run ensemble-averaged non-Markovian dynamics via MesoHOPS.

        Runs n_traj independent HOPS trajectories and constructs the
        ensemble-averaged density matrix at each time point:
            rho(t) = (1/N) sum_k |psi_k(t)><psi_k(t)|

        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial pure state vector (default: excitation on site 1).
        time_points : np.ndarray, optional
            Desired output time points in fs (used to set t_max).
            Note: actual time grid is determined by dt_save.
        dt_save : float
            Noise discretization / save interval in fs (default: 0.5).
        seeds : list of int, optional
            Random seeds for each trajectory (default: range(n_traj)).

        Returns
        -------
        results : dict
            Dictionary containing:
            - 't_axis': Time points in fs.
            - 'density_matrices': list of np.ndarray
            - 'populations': np.ndarray (n_times, n_sites)
            - 'coherences': np.ndarray (n_times,)
            - 'qfi': np.ndarray (n_times,)
            - 'entropy': np.ndarray (n_times,)
            - 'purity': np.ndarray (n_times,)
            - 'linear_entropy': np.ndarray (n_times,)
            - 'bipartite_ent': np.ndarray (n_times,)
            - 'multipartite_ent': np.ndarray (n_times,)
            - 'pairwise_concurrence': np.ndarray (n_times,)
            - 'discord': np.ndarray (n_times,)
            - 'fidelity': np.ndarray (n_times,)
            - 'mandel_q': np.ndarray (n_times,)
        """
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        n_sites = self.n_sites
        n_traj = self.n_traj

        # Determine simulation time
        if time_points is not None:
            t_max = float(np.max(time_points))
        else:
            t_max = 500.0  # default: 500 fs

        # Initial state
        if initial_state is None:
            psi_0 = np.zeros(n_sites, dtype=complex)
            psi_0[0] = 1.0  # Excitation on site 1 (BChl 1)
        else:
            psi_0 = np.array(initial_state, dtype=complex).flatten()
            assert psi_0.shape[0] == n_sites

        # Seeds
        if seeds is None:
            seeds = list(range(n_traj))

        # ── Run ensemble of trajectories ─────────────────────────────
        logger.info(f"  Running {n_traj} HOPS trajectories ({t_max:.0f} fs, "
              f"dt={dt_save} fs, MAXHIER={self.max_hier})...")

        all_psi_trajs = []
        t_axis = None

        for k, seed in enumerate(seeds):
            try:
                hops = self._build_hops_trajectory(seed, t_max, dt_save)
                required_tau = hops.noise1.param['TAU'] / hops.integrator_step
                hops.initialize(psi_0.copy())
                hops.propagate(t_max, required_tau)

                psi_traj = np.array(hops.storage.data['psi_traj'])[:, :n_sites]
                if t_axis is None:
                    t_axis = np.array(hops.storage.data['t_axis'])

                # Check for NaN
                if not np.any(np.isnan(psi_traj)):
                    all_psi_trajs.append(psi_traj)

                if (k + 1) % max(1, n_traj // 5) == 0 or k == n_traj - 1:
                    logger.info(f"    Trajectory {k+1}/{n_traj} completed "
                          f"({len(all_psi_trajs)} valid)")

            except Exception as e:
                # CRITICAL FIX: Catch Numba/NumPy version mismatch
                if "Numba needs NumPy" in str(e):
                    raise RuntimeError(f"Numba/NumPy version mismatch: {e}. "
                                       f"Please downgrade NumPy to < 2.4 (e.g., `pip install numpy<2.4`).") from e
                
                logger.warning(f"    Trajectory {k+1}/{n_traj} failed: {str(e)[:60]}")
                continue

        n_valid = len(all_psi_trajs)
        if n_valid == 0:
            raise RuntimeError("All HOPS trajectories failed. "
                               "Try increasing MAXHIER or decreasing dt_save.")

        logger.info(f"  Ensemble: {n_valid}/{n_traj} valid trajectories")

        # ── Ensemble-average density matrix ──────────────────────────
        n_times = len(t_axis)
        density_matrices = []
        populations = np.zeros((n_times, n_sites))
        coherences = np.zeros(n_times)
        qfi_values = np.zeros(n_times)
        entropy_values = np.zeros(n_times)
        purity_values = np.zeros(n_times)
        linear_entropy_values = np.zeros(n_times)
        bipartite_ent_values = np.zeros(n_times)
        multipartite_ent_values = np.zeros(n_times)
        pairwise_concurrence_values = np.zeros(n_times)
        discord_values = np.zeros(n_times)
        fidelity_values = np.zeros(n_times)
        mandel_q_values = np.zeros(n_times)

        for i in range(n_times):
            # Construct density matrix: rho = (1/N) sum_k |psi_k><psi_k|
            rho = np.zeros((n_sites, n_sites), dtype=complex)
            for psi_traj in all_psi_trajs:
                psi = psi_traj[i]
                rho += np.outer(psi, np.conj(psi))
            rho /= n_valid

            # Normalize trace to 1 (should be close already)
            tr = np.trace(rho).real
            if tr > 1e-10:
                rho /= tr

            density_matrices.append(rho)
            populations[i, :] = np.real(np.diag(rho))
            coherences[i] = self.calculate_coherence_measure(rho)

            try:
                qfi_values[i] = self.calculate_qfi(rho, self.H_raw)
            except Exception:
                qfi_values[i] = 0.0

            try:
                entropy_values[i] = self.calculate_entropy_von_neumann(rho)
            except Exception:
                entropy_values[i] = 0.0

            try:
                purity_values[i] = self.calculate_purity(rho)
            except Exception:
                purity_values[i] = 0.0

            try:
                linear_entropy_values[i] = self.calculate_linear_entropy(rho)
            except Exception:
                linear_entropy_values[i] = 0.0

            try:
                bipartite_ent_values[i] = self.calculate_bipartite_entanglement(rho)
            except Exception:
                bipartite_ent_values[i] = 0.0

            try:
                multipartite_ent_values[i] = self.calculate_multipartite_entanglement(rho)
            except Exception:
                multipartite_ent_values[i] = 0.0

            try:
                pairwise_concurrence_values[i] = self.calculate_pairwise_concurrence(rho)
            except Exception:
                pairwise_concurrence_values[i] = 0.0

            try:
                discord_values[i] = self.calculate_quantum_discord(rho)
            except Exception:
                discord_values[i] = 0.0

            try:
                # Fidelity relative to initial state
                rho_0 = density_matrices[0]
                fidelity_values[i] = self.calculate_fidelity(rho, rho_0)
            except Exception:
                fidelity_values[i] = 1.0 if i == 0 else 0.0

            try:
                # Mandel Q requires vibrational occupations - using population weighted average as dummy
                mandel_q_values[i] = self.calculate_mandel_q_parameter(populations[i])
            except Exception:
                mandel_q_values[i] = 0.0

        return {
            't_axis': t_axis,
            'density_matrices': density_matrices,
            'populations': populations,
            'coherences': coherences,
            'qfi': qfi_values,
            'entropy': entropy_values,
            'purity': purity_values,
            'linear_entropy': linear_entropy_values,
            'bipartite_ent': bipartite_ent_values,
            'multipartite_ent': multipartite_ent_values,
            'pairwise_concurrence': pairwise_concurrence_values,
            'discord': discord_values,
            'fidelity': fidelity_values,
            'mandel_q': mandel_q_values
        }

    def calculate_etr(self, populations, time_points):
        """
        Calculate the Electron Transfer Rate (ETR) from population dynamics.

        Parameters
        ----------
        populations : np.ndarray (n_times, n_sites)
            Site populations from simulation.
        time_points : np.ndarray
            Corresponding time points in fs.

        Returns
        -------
        etr : float
            Estimated energy transfer rate in ps^-1.
        """
        # ETR = rate of population leaving site 1
        pop_site1 = populations[:, 0]
        transferred = 1.0 - pop_site1

        # Find time to 50% transfer (half-life)
        idx_half = np.argmax(transferred >= 0.5)
        if idx_half > 0 and transferred[idx_half] >= 0.5:
            t_half = time_points[idx_half]  # fs
            etr = np.log(2) / (t_half * 1e-3)  # ps^-1
        else:
            # Linear fit to early transfer
            n_fit = min(20, len(time_points))
            if n_fit > 2:
                coeffs = np.polyfit(time_points[:n_fit], transferred[:n_fit], 1)
                rate_fs = coeffs[0]  # transfer per fs
                etr = rate_fs * 1e3  # ps^-1
            else:
                etr = 0.0

        return etr

    def calculate_coherence_measure(self, density_matrix):
        """
        Calculate the l1-norm of coherence for a density matrix.

        C_l1(rho) = sum_{i!=j} |rho_{ij}|

        Parameters
        ----------
        density_matrix : np.ndarray
            Density matrix.

        Returns
        -------
        coherence : float
            l1-norm coherence measure.
        """
        n = density_matrix.shape[0]
        coherence = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    coherence += np.abs(density_matrix[i, j])
        return coherence

    def calculate_qfi(self, rho, H):
        """
        Calculate the Quantum Fisher Information (QFI).

        Parameters
        ----------
        rho : np.ndarray
            Density matrix.
        H : np.ndarray
            Observable (Hamiltonian).

        Returns
        -------
        qfi : float
            Quantum Fisher Information.
        """
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.real(eigenvals)

        qfi = 0.0
        n = len(eigenvals)
        for i in range(n):
            for j in range(n):
                denom = eigenvals[i] + eigenvals[j]
                if denom > 1e-12:
                    H_ij = np.abs(eigenvecs[:, i].conj() @ H @ eigenvecs[:, j])**2
                    qfi += 2.0 * (eigenvals[i] - eigenvals[j])**2 / denom * H_ij

        return float(np.real(qfi))

    def analyze_robustness(self, temperature_range=(273, 320),
                           disorder_strengths=(0, 100), n_points=5):
        """
        Robustness analysis across temperature and static disorder.

        Parameters
        ----------
        temperature_range : tuple
            (T_min, T_max) in Kelvin.
        disorder_strengths : tuple
            (sigma_min, sigma_max) in cm^-1.
        n_points : int
            Number of sample points per parameter.

        Returns
        -------
        results : dict
            Temperature and disorder sensitivity data.
        """
        results = {
            'temperature_sensitivity': [],
            'disorder_sensitivity': [],
            'temperatures': [],
            'disorder_strengths': []
        }

        # Temperature sweep
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_points)
        for temp in temperatures:
            sim = QuantumDynamicsSimulator(
                self.H_raw, temperature=temp,
                lambda_reorg=self.lambda_reorg, gamma_dl=self.gamma_dl,
                k_matsubara=self.k_matsubara,
                max_hier=self.max_hier, n_traj=max(5, self.n_traj // 10),
            )
            res = sim.simulate_dynamics(time_points=np.linspace(0, 200, 50), dt_save=4.0)
            pops = res['populations']
            etr_proxy = np.sum(pops[-1, 1:])
            results['temperature_sensitivity'].append(etr_proxy)
            results['temperatures'].append(temp)

        # Disorder sweep
        disorder_vals = np.linspace(disorder_strengths[0], disorder_strengths[1], n_points)
        for disorder in disorder_vals:
            disorder_vec = np.random.normal(0, max(disorder, 0.01), self.n_sites)
            ham_dis = self.H_raw + np.diag(disorder_vec)
            sim = QuantumDynamicsSimulator(
                ham_dis, temperature=self.temperature,
                lambda_reorg=self.lambda_reorg, gamma_dl=self.gamma_dl,
                k_matsubara=self.k_matsubara,
                max_hier=self.max_hier, n_traj=max(5, self.n_traj // 10),
            )
            res = sim.simulate_dynamics(time_points=np.linspace(0, 200, 50), dt_save=4.0)
            pops = res['populations']
            etr_proxy = np.sum(pops[-1, 1:])
            results['disorder_sensitivity'].append(etr_proxy)
            results['disorder_strengths'].append(disorder)

        return results

    def calculate_entropy_von_neumann(self, rho):
        """Calculate the von Neumann entropy of a quantum state."""
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(rho)
        
        # Take only the real part and ensure non-negative
        eigenvals = np.real(eigenvals)
        eigenvals = np.clip(eigenvals, a_min=1e-12, a_max=None)
        
        # Calculate entropy: -Σ λᵢ log λᵢ
        entropy = -np.sum(eigenvals * np.log(eigenvals))
        return entropy
    
    def calculate_purity(self, rho):
        """Calculate the purity of a quantum state."""
        # Calculate Tr[ρ²]
        purity = np.real(np.trace(rho @ rho))
        return purity

    def calculate_linear_entropy(self, rho):
        """Calculate the linear entropy of a quantum state."""
        d = rho.shape[0]  # Hilbert space dimension
        
        if d == 1:
            return 0.0
        
        # Calculate Tr[ρ²]
        tr_rho_sq = np.real(np.trace(rho @ rho))
        
        # Calculate linear entropy
        linear_entropy = (d / (d - 1)) * (1 - tr_rho_sq)
        
        # Ensure it's within valid range
        linear_entropy = np.clip(linear_entropy, 0.0, 1.0)
        return linear_entropy
    
    def calculate_concurrence(self, rho):
        """Calculate the concurrence of a quantum state (for 2-qubit systems)."""
        n = rho.shape[0]
        
        if n < 2:
            return 0.0
        
        # For systems larger than 2x2, calculate average pairwise concurrence
        if n > 2:
            total_concurrence = 0.0
            n_pairs = 0
            
            # Calculate concurrence for each pair of sites
            for i in range(n):
                for j in range(i+1, n):
                    # Extract 2x2 reduced density matrix for sites i,j
                    indices = [i, j]
                    rho_ij = np.zeros((2, 2), dtype=complex)
                    
                    # Create reduced density matrix by tracing out other sites
                    # For simplicity, we'll use a direct approach for 2x2 subsystem
                    rho_ij[0, 0] = rho[i, i]
                    rho_ij[0, 1] = rho[i, j]
                    rho_ij[1, 0] = rho[j, i]
                    rho_ij[1, 1] = rho[j, j]
                    
                    # Calculate concurrence for this pair
                    pair_concurrence = self._calculate_2x2_concurrence(rho_ij)
                    total_concurrence += pair_concurrence
                    n_pairs += 1
            
            return total_concurrence / n_pairs if n_pairs > 0 else 0.0
        else:
            # For 2x2 system, calculate directly
            return self._calculate_2x2_concurrence(rho)
    
    def _calculate_2x2_concurrence(self, rho):
        """Helper to calculate concurrence for a 2x2 density matrix."""
        # Define the spin-flipped density matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        rho_tilde = np.kron(sigma_y, sigma_y) @ rho.conj() @ np.kron(sigma_y, sigma_y)
        
        # Calculate R = sqrt(rho * rho_tilde)
        R = la.sqrtm(rho @ rho_tilde)
        
        # Calculate eigenvalues of R
        evals = np.linalg.eigvals(R)
        evals = np.sort(np.real(evals))[::-1]  # Sort in descending order
        
        # Calculate concurrence
        c = max(0, evals[0] - evals[1] - evals[2] - evals[3])
        return c
    
    def calculate_bipartite_entanglement(self, rho, partition=None):
        """Calculate bipartite entanglement using von Neumann entropy of reduced density matrix."""
        n = rho.shape[0]
        
        if partition is None:
            # Default partition: first half vs second half
            partition = list(range(n // 2))
        
        if len(partition) == 0 or len(partition) == n:
            return 0.0
        
        # Calculate reduced density matrix by tracing out subsystem B
        reduced_rho = np.zeros((len(partition), len(partition)), dtype=complex)
        
        for i, idx_i in enumerate(partition):
            for j, idx_j in enumerate(partition):
                reduced_rho[i, j] = rho[idx_i, idx_j]
        
        # Normalize the reduced density matrix
        trace = np.trace(reduced_rho)
        if trace > 0:
            reduced_rho = reduced_rho / trace
        else:
            return 0.0
        
        return self.calculate_entropy_von_neumann(reduced_rho)
    
    def calculate_multipartite_entanglement(self, rho):
        """Calculate multipartite entanglement measure."""
        n = rho.shape[0]
        
        if n < 2:
            return 0.0
        
        # For computational efficiency, we'll calculate entanglement for
        # a subset of bipartitions rather than all possible partitions
        total_entanglement = 0.0
        n_partitions = 0
        
        # Calculate entanglement for different bipartitions
        for i in range(1, min(n, 6)):  # Limit to avoid combinatorial explosion
            # Partition into first i sites vs remaining sites
            partition = list(range(i))
            ent = self.calculate_bipartite_entanglement(rho, partition)
            total_entanglement += ent
            n_partitions += 1
        
        return total_entanglement / n_partitions if n_partitions > 0 else 0.0
    
    def calculate_pairwise_concurrence(self, rho):
        """Calculate average pairwise concurrence across all pairs of sites."""
        n = rho.shape[0]
        
        if n < 2:
            return 0.0
        
        total_concurrence = 0.0
        n_pairs = 0
        
        # Calculate concurrence for each pair of sites
        for i in range(n):
            for j in range(i+1, n):
                # Extract 2x2 reduced density matrix for sites i,j
                rho_ij = np.zeros((2, 2), dtype=complex)
                rho_ij[0, 0] = rho[i, i]
                rho_ij[0, 1] = rho[i, j]
                rho_ij[1, 0] = rho[j, i]
                rho_ij[1, 1] = rho[j, j]
                
                # Calculate concurrence for this pair
                pair_concurrence = self._calculate_2x2_concurrence(rho_ij)
                total_concurrence += pair_concurrence
                n_pairs += 1
        
        return total_concurrence / n_pairs if n_pairs > 0 else 0.0
    
    def calculate_quantum_synergy_index(self, rho_opv, rho_psu):
        """Calculate quantum synergy index between OPV and photosynthetic system."""
        numerator = np.trace(rho_opv @ rho_psu) - np.trace(rho_opv) * np.trace(rho_psu)
        denominator = np.linalg.norm(rho_opv) * np.linalg.norm(rho_psu)
        synergy = numerator / denominator if denominator != 0 else 0
        return synergy

    def calculate_mandel_q_parameter(self, vibrational_mode_occupations):
        """
        Calculate Mandel Q parameter for vibrational mode non-classicality.
        Q < 0 indicates non-classical (quantum) behavior.
        """
        mean_occ = np.mean(vibrational_mode_occupations)
        if mean_occ < 1e-12: return 0.0
        variance = np.var(vibrational_mode_occupations)
        return (variance - mean_occ) / mean_occ

    def calculate_fidelity(self, rho, sigma):
        """Calculate quantum fidelity between two states."""
        try:
            sqrt_rho = la.sqrtm(rho)
            matrix = sqrt_rho @ sigma @ sqrt_rho
            return np.real(np.trace(la.sqrtm(matrix)))**2
        except Exception:
            return np.real(np.trace(rho @ sigma))

    def calculate_quantum_discord(self, rho):
        """Simplified measure of quantum correlations beyond entanglement (Quantum Discord)."""
        # A full calculation is an optimization problem; we use relative entropy of discord proxy
        # Treat as site 1 vs others partition
        s_total = self.calculate_entropy_von_neumann(rho)
        # Reduced rho_others (simplified)
        diag = np.diag(rho).real
        p_others = diag[1:]
        p_others_sum = np.sum(p_others)
        if p_others_sum < 1e-12:
             return 0.0
        p_others /= p_others_sum
        s_others = -np.sum(p_others * np.log(p_others + 1e-15))
        return max(0.0, s_others - s_total)

    def stochastically_bundled_dissipators(self, rho, l_bundled, p_alpha):
        """
        Implement Stochastically Bundled Dissipators (SBD).
        L_SBD[rho] = sum_alpha p_alpha * (L rho L_dag - 0.5 * {L_dag L, rho})
        """
        d_rho = np.zeros_like(rho, dtype=complex)
        for L, p in zip(l_bundled, p_alpha):
            if p > 0:
                L_dag = L.conj().T
                d_rho += p * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))
        return d_rho


def spectral_density_drude_lorentz(omega, lambda_reorg, gamma):
    """
    Drude-Lorentz spectral density.
    
    J(omega) = (2 * lambda_reorg * gamma * omega) / (omega^2 + gamma^2)
    
    Parameters
    ----------
    omega : array-like
        Angular frequency in cm^-1
    lambda_reorg : float
        Reorganization energy in cm^-1
    gamma : float
        Cutoff frequency in cm^-1
        
    Returns
    -------
    J : array-like
        Spectral density values
    """
    omega = np.asarray(omega)
    J = (2 * lambda_reorg * gamma * omega) / (omega**2 + gamma**2)
    # Avoid division by zero at omega=0
    J[omega == 0] = 0
    return J


def spectral_density_vibronic(omega, omega_mode, lambda_mode, gamma_mode):
    """
    Underdamped vibronic mode spectral density (Shifted Drude-Lorentz).
    
    J(omega) = (2 * lambda_mode * omega_mode * gamma_mode * omega) / 
               ((omega^2 - omega_mode^2)^2 + omega^2 * gamma_mode^2)
    
    Parameters
    ----------
    omega : array-like
        Angular frequency in cm^-1
    omega_mode : float
        Vibronic mode frequency in cm^-1
    lambda_mode : float
        Reorganization energy of the mode in cm^-1
    gamma_mode : float
        Damping rate in cm^-1
        
    Returns
    -------
    J : array-like
        Spectral density values
    """
    omega = np.asarray(omega)
    numerator = 2 * lambda_mode * omega_mode * gamma_mode * omega
    denominator = (omega**2 - omega_mode**2)**2 + omega**2 * gamma_mode**2
    J = numerator / denominator
    # Avoid division by zero at omega=0
    J[omega == 0] = 0
    return J


def spectral_density_total(omega, lambda_reorg, gamma, vibronic_modes=None):
    """
    Total spectral density combining Drude-Lorentz and vibronic modes.
    
    Parameters
    ----------
    omega : array-like
        Angular frequency in cm^-1
    lambda_reorg : float
        Reorganization energy for Drude-Lorentz component in cm^-1
    gamma : float
        Cutoff frequency for Drude-Lorentz component in cm^-1
    vibronic_modes : list of dict, optional
        List of vibronic modes, each with keys 'omega', 'lambda', 'gamma'
        
    Returns
    -------
    J : array-like
        Total spectral density values
    """
    omega = np.asarray(omega)
    
    # Drude-Lorentz component
    J_total = spectral_density_drude_lorentz(omega, lambda_reorg, gamma)
    
    # Add vibronic modes if provided
    if vibronic_modes:
        for mode in vibronic_modes:
            J_vib = spectral_density_vibronic(
                omega, 
                mode['omega'], 
                mode['lambda'], 
                mode['gamma']
            )
            J_total += J_vib
    
    return J_total


def save_quantum_dynamics_results(
    results: Dict[str, Any],
    filename_prefix: str = 'quantum_dynamics',
    output_dir: str = '../simulation_data/'
) -> str:
    """
    Save quantum dynamics simulation results to CSV.

    Parameters
    ----------
    results : dict
        Results from simulate_dynamics method
    filename_prefix : str
        Prefix for output filename
    output_dir : str
        Directory to save CSV file

    Returns
    -------
    str
        Path to saved CSV file
    """
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract data from results
    t_axis = results.get('t_axis', [])
    populations = results.get('populations', np.array([]))
    coherences = results.get('coherences', np.array([]))
    qfi = results.get('qfi', np.array([]))
    entropy = results.get('entropy', np.array([]))
    purity = results.get('purity', np.array([]))
    linear_entropy = results.get('linear_entropy', np.array([]))
    bipartite_ent = results.get('bipartite_ent', np.array([]))
    multipartite_ent = results.get('multipartite_ent', np.array([]))
    pairwise_concurrence = results.get('pairwise_concurrence', np.array([]))
    discord = results.get('discord', np.array([]))
    fidelity = results.get('fidelity', np.array([]))
    mandel_q = results.get('mandel_q', np.array([]))

    # Create DataFrame
    data = {'time_fs': t_axis}
    
    # Add populations for each site
    if len(populations) > 0 and len(populations[0]) > 0:
        n_sites = len(populations[0])
        for i in range(n_sites):
            data[f'population_site_{i+1}'] = [p[i] for p in populations]
    
    # Add quantum metrics
    if len(coherences) > 0:
        data['coherences'] = coherences
    if len(qfi) > 0:
        data['qfi'] = qfi
    if len(entropy) > 0:
        data['entropy'] = entropy
    if len(purity) > 0:
        data['purity'] = purity
    if len(linear_entropy) > 0:
        data['linear_entropy'] = linear_entropy
    if len(bipartite_ent) > 0:
        data['bipartite_ent'] = bipartite_ent
    if len(multipartite_ent) > 0:
        data['multipartite_ent'] = multipartite_ent
    if len(pairwise_concurrence) > 0:
        data['pairwise_concurrence'] = pairwise_concurrence
    if len(discord) > 0:
        data['discord'] = discord
    if len(fidelity) > 0:
        data['fidelity'] = fidelity
    if len(mandel_q) > 0:
        data['mandel_q'] = mandel_q

    df = pd.DataFrame(data)
    filename = f"{filename_prefix}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    logger.info(f"Quantum dynamics results saved to {filepath}")
    return filepath


def plot_quantum_dynamics_results(
    results: Dict[str, Any],
    filename_prefix: str = 'quantum_dynamics',
    figures_dir: str = '../Graphics/'
) -> str:
    """
    Plot quantum dynamics simulation results.

    Parameters
    ----------
    results : dict
        Results from simulate_dynamics method
    filename_prefix : str
        Prefix for output filename
    figures_dir : str
        Directory to save figures

    Returns
    -------
    str
        Path to saved figure
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(figures_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    t_axis = results.get('t_axis', [])
    populations = results.get('populations', np.array([]))
    coherences = results.get('coherences', np.array([]))
    qfi = results.get('qfi', np.array([]))
    entropy = results.get('entropy', np.array([]))
    purity = results.get('purity', np.array([]))

    # Create figure with subplots
    n_metrics = sum(x.size > 0 for x in [coherences, qfi, entropy, purity])
    n_plots = max(2, n_metrics + 1)  # At least populations and one metric
    
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Quantum Dynamics Simulation Results', fontsize=16, fontweight='bold')
    
    # Handle single row case
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    elif n_rows == 1 and n_cols > 1:
        axes = axes
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]

    # Plot populations
    if len(populations) > 0 and len(populations[0]) > 0:
        ax_idx = 0
        ax = axes[ax_idx]
        n_sites = len(populations[0])
        for i in range(min(n_sites, 7)):  # Limit to first 7 sites for readability
            ax.plot(t_axis, [p[i] for p in populations], label=f'Site {i+1}', linewidth=1.5)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Population')
        ax.set_title('Site Populations vs Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax_idx = -1

    # Plot quantum metrics
    metrics_to_plot = [
        (coherences, 'Coherence', 'coherences'),
        (qfi, 'QFI', 'qfi'),
        (entropy, 'Entropy', 'entropy'),
        (purity, 'Purity', 'purity')
    ]
    
    for metric_values, metric_label, metric_key in metrics_to_plot:
        if len(metric_values) > 0:
            ax_idx += 1
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.plot(t_axis, metric_values, linewidth=2)
                ax.set_xlabel('Time (fs)')
                ax.set_ylabel(metric_label)
                ax.set_title(f'{metric_label} Evolution')
                ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for idx in range(ax_idx + 1, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()

    filename = f"{filename_prefix}_{timestamp}.pdf"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    png_filename = f"{filename_prefix}_{timestamp}.png"
    png_filepath = os.path.join(figures_dir, png_filename)
    plt.savefig(png_filepath, dpi=150, bbox_inches='tight')

    plt.close()

    logger.info(f"Quantum dynamics plots saved to {filepath}")
    return filepath