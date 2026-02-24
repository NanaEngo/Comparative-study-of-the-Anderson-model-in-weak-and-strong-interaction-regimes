"""
HopsSimulator: Unified quantum dynamics simulator with MesoHOPS integration.

This module provides the HopsSimulator class which uses MesoHOPS as the default
simulation engine with automatic fallback to QuantumDynamicsSimulator.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
import sys

# Import MesoHOPS modules
try:
    from mesohops.basis.hops_system import HopsSystem
    from mesohops.basis.hops_basis import HopsBasis
    from mesohops.trajectory.hops_trajectory import HopsTrajectory
    MESOHOPS_AVAILABLE = True
except ImportError:
    MESOHOPS_AVAILABLE = False
    HopsSystem = None
    HopsBasis = None
    HopsTrajectory = None

# Import fallback simulators - we'll check if they can be instantiated in the init method
try:
    from .quantum_dynamics_simulator import QuantumDynamicsSimulator as QDS
    # Check if it can be used by testing if MesoHOPS is available
    try:
        # Create a minimal test instance to see if it fails due to missing MesoHOPS
        dummy_ham = np.array([[0.0]])
        test_sim = QDS(dummy_ham)
        QuantumDynamicsSimulator = QDS
    except ImportError:
        # MesoHOPS not available, so this simulator won't work
        QuantumDynamicsSimulator = None
    except Exception:
        # Some other error during instantiation
        QuantumDynamicsSimulator = None
except ImportError:
    try:
        from models.quantum_dynamics_simulator import QuantumDynamicsSimulator as QDS
        # Check if it can be used by testing if MesoHOPS is available
        try:
            # Create a minimal test instance to see if it fails due to missing MesoHOPS
            dummy_ham = np.array([[0.0]])
            test_sim = QDS(dummy_ham)
            QuantumDynamicsSimulator = QDS
        except ImportError:
            # MesoHOPS not available, so this simulator won't work
            QuantumDynamicsSimulator = None
        except Exception:
            # Some other error during instantiation
            QuantumDynamicsSimulator = None
    except ImportError:
        QuantumDynamicsSimulator = None

# Import simple fallback simulator
try:
    from .simple_quantum_dynamics_simulator import SimpleQuantumDynamicsSimulator
except ImportError:
    try:
        from models.simple_quantum_dynamics_simulator import SimpleQuantumDynamicsSimulator
    except ImportError:
        SimpleQuantumDynamicsSimulator = None

from .constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_HIERARCHY, DEFAULT_REORGANIZATION_ENERGY, DEFAULT_DRUDE_CUTOFF

logger = logging.getLogger(__name__)


class HopsSimulator:
    """
    Unified simulator that uses MesoHOPS by default with fallback.

    This class provides a consistent interface for quantum dynamics simulations,
    automatically using MesoHOPS when available and falling back to the custom
    QuantumDynamicsSimulator when needed.

    Parameters
    ----------
    hamiltonian : NDArray[np.float64]
        System Hamiltonian matrix
    temperature : float, optional
        Temperature in Kelvin (default: 295.0)
    use_mesohops : bool, optional
        Whether to use MesoHOPS if available (default: True)
    max_hierarchy : int, optional
        Maximum hierarchy level for MesoHOPS (default: 10)
    **kwargs : dict
        Additional arguments passed to underlying simulator

    Attributes
    ----------
    hamiltonian : NDArray[np.float64]
        The system Hamiltonian
    temperature : float
        Simulation temperature
    use_mesohops : bool
        Whether MesoHOPS is being used
    system : Optional[HopsSystem]
        MesoHOPS system object (if initialized)
    fallback_sim : Optional[QuantumDynamicsSimulator]
        Fallback simulator (if initialized)

    Examples
    --------
    >>> import numpy as np
    >>> from core.hops_simulator import HopsSimulator
    >>> hamiltonian = np.array([[1, 0.5], [0.5, 1]])
    >>> simulator = HopsSimulator(hamiltonian, temperature=295)
    >>> time_points = np.linspace(0, 100, 100)
    >>> results = simulator.simulate_dynamics(time_points)
    """

    def __init__(
        self,
        hamiltonian: NDArray[np.float64],
        temperature: float = DEFAULT_TEMPERATURE,
        use_mesohops: bool = True,
        max_hierarchy: int = DEFAULT_MAX_HIERARCHY,
        **kwargs: Any
    ):
        """Initialize the simulator."""
        self.hamiltonian = hamiltonian
        self.temperature = temperature
        self.max_hierarchy = max_hierarchy
        self.use_mesohops = MESOHOPS_AVAILABLE and use_mesohops
        self.system: Optional[HopsSystem] = None
        self.fallback_sim: Optional[Any] = None

        logger.debug(
            f"Initializing HopsSimulator (MesoHOPS available: {MESOHOPS_AVAILABLE}, "
            f"use_mesohops: {use_mesohops})"
        )

        if self.use_mesohops:
            self._init_mesohops(**kwargs)
        else:
            self._init_fallback(**kwargs)

    def _drude_correlation_function(self, t_axis, lambda_reorg, gamma_cutoff, temperature):
        """
        Drude-Lorentz bath correlation function for MesoHOPS.
        
        Mathematical Framework:
        C(t) = (λ/π) * [coth(βωc/2) * cos(ωc*t) - i*sin(ωc*t)] * exp(-ωc*|t|)
        
        For high temperature approximation:
        C(t) ≈ (λ*ωc) * exp(-ωc*|t|) / (β*π*ωc/2)
        """
        import numpy as np
        # Boltzmann constant in appropriate units
        kb = 0.695  # cm^-1/K
        beta = 1.0 / (kb * temperature)
        
        # Standard Drude-Lorentz correlation function
        # For high-temperature approximation
        if temperature > 77:  # Room temperature
            corr = (lambda_reorg / np.pi) * gamma_cutoff * np.exp(-gamma_cutoff * np.abs(t_axis))
        else:  # Include quantum correction at low temperature
            coth_term = 1.0 / np.tanh(beta * gamma_cutoff / 2.0)
            corr = (lambda_reorg / np.pi) * gamma_cutoff * coth_term * np.exp(-gamma_cutoff * np.abs(t_axis))
            
        return corr

    def _init_mesohops(self, **kwargs: Any) -> None:
        """Initialize MesoHOPS simulator with proper system parameters."""
        try:
            # Prepare system parameters for MesoHOPS - correct API format
            n_sites = self.hamiltonian.shape[0]

            # Get bath parameters from kwargs or use defaults
            lambda_reorg = kwargs.get('reorganization_energy', DEFAULT_REORGANIZATION_ENERGY)
            gamma_cutoff = kwargs.get('drude_cutoff', DEFAULT_DRUDE_CUTOFF)

            # Define system-bath coupling operators (Lindblad operators)
            # MesoHOPS uses dense arrays for L operators based on test examples
            L_hier = []
            L_noise = []
            for i in range(n_sites):
                L_op = np.zeros((n_sites, n_sites), dtype=np.float64)
                L_op[i, i] = 1.0  # Diagonal operator for site i
                L_hier.append(L_op)
                L_noise.append(L_op)

            # Shift Hamiltonian to center energies around 0.
            # Large absolute energies (like ~12000 cm^-1 for FMO) cause rapid oscillations
            # exp(-i H t / hbar) which numerical integrators struggle with, causing overflows.
            # Physical observables (populations) are invariant to uniform energy shifts.
            E_mean = np.mean(np.diag(self.hamiltonian))
            H_shifted = self.hamiltonian - E_mean * np.eye(n_sites)

            # Bath correlation function parameters - Drude-Lorentz model
            # Using the bcf_convert_dl_to_exp function from MesoHOPS
            try:
                from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
                dl_modes = bcf_convert_dl_to_exp(
                    lambda_reorg, gamma_cutoff, self.temperature
                )
                # dl_modes is a list like [g0, w0] where g0 is complex and w0 is real
                # Build gw_sysbath - one entry per site
                gw_sysbath = []
                for i in range(n_sites):
                    gw_sysbath.append([dl_modes[0], dl_modes[1]])
            except Exception as e:
                # Fallback: simplified single mode
                logger.warning(f"MesoHOPS bath correlation functions not available, using simplified model: {e}")
                gw_sysbath = [[lambda_reorg, gamma_cutoff]]  # Simplified single mode

            # System parameters dictionary - MesoHOPS format
            # Use the bcf_exp correlation function from MesoHOPS
            from mesohops.trajectory.exp_noise import bcf_exp

            self.system_param = {
                'HAMILTONIAN': H_shifted,
                'GW_SYSBATH': gw_sysbath,
                'L_HIER': L_hier,
                'L_NOISE1': L_noise,
                'ALPHA_NOISE1': bcf_exp,
                'PARAM_NOISE1': gw_sysbath,
            }

            logger.info("MesoHOPS system parameters prepared successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize MesoHOPS: {e}")
            logger.info("Falling back to QuantumDynamicsSimulator")
            self.use_mesohops = False
            self._init_fallback(**kwargs)

    def _init_fallback(self, **kwargs: Any) -> None:
        """Initialize fallback simulator."""
        # First try the simple simulator as it doesn't require MesoHOPS
        if SimpleQuantumDynamicsSimulator is not None:
            try:
                self.fallback_sim = SimpleQuantumDynamicsSimulator(
                    self.hamiltonian,
                    temperature=self.temperature
                )
                logger.info("SimpleQuantumDynamicsSimulator initialized successfully")
                return
            except Exception as e:
                logger.error(f"Failed to initialize SimpleQuantumDynamicsSimulator: {e}")
        
        # Then try the full QuantumDynamicsSimulator (this may fail if MesoHOPS is not available)
        if QuantumDynamicsSimulator is not None:
            try:
                # Filter kwargs to only pass parameters that QuantumDynamicsSimulator accepts
                qds_kwargs = {}
                valid_params = {'temperature', 'lambda_reorg', 'gamma_dl', 'k_matsubara', 
                               'max_hier', 'n_traj', 'vibronic_modes'}
                
                for key, value in kwargs.items():
                    if key in valid_params:
                        qds_kwargs[key] = value
                
                # Use default values if not provided
                qds_kwargs.setdefault('temperature', self.temperature)
                qds_kwargs.setdefault('lambda_reorg', kwargs.get('reorganization_energy', DEFAULT_REORGANIZATION_ENERGY))
                qds_kwargs.setdefault('gamma_dl', kwargs.get('drude_cutoff', DEFAULT_DRUDE_CUTOFF))
                qds_kwargs.setdefault('max_hier', self.max_hierarchy)
                qds_kwargs.setdefault('n_traj', 10)  # Use fewer trajectories for testing
                
                self.fallback_sim = QuantumDynamicsSimulator(
                    self.hamiltonian,
                    **qds_kwargs
                )
                logger.info("QuantumDynamicsSimulator initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize QuantumDynamicsSimulator: {e}")
        
        # If both fail, log error
        logger.error("Failed to initialize any fallback simulator")
        self.fallback_sim = None

    def simulate_dynamics(
        self,
        time_points: NDArray[np.float64],
        initial_state: Optional[NDArray[np.float64]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Simulate quantum dynamics.

        Uses MesoHOPS if available, otherwise falls back to custom simulator.

        Parameters
        ----------
        time_points : NDArray[np.float64]
            Array of time points for simulation
        initial_state : Optional[NDArray[np.float64]]
            Initial quantum state (default: None)
        **kwargs : dict
            Additional simulation parameters

        Returns
        -------
        Dict[str, Any]
            Simulation results containing populations, coherences, and other metrics

        Raises
        ------
        RuntimeError
            If no simulator is available
        """
        logger.info(f"Starting dynamics simulation for {len(time_points)} time points")

        if self.use_mesohops and hasattr(self, 'system_param') and self.system_param is not None:
            try:
                logger.debug("Attempting MesoHOPS simulation")
                return self._simulate_with_mesohops(
                    time_points, initial_state, **kwargs
                )
            except Exception as e:
                logger.warning(f"MesoHOPS simulation failed: {e}")
                logger.info("Falling back to custom simulator...")

        # Fallback to custom simulator
        if self.fallback_sim is not None:
            logger.debug("Using fallback simulator")
            return self.fallback_sim.simulate_dynamics(
                time_points, initial_state, **kwargs
            )
        else:
            raise RuntimeError("No simulator available")

    def _simulate_with_mesohops(
        self,
        time_points: NDArray[np.float64],
        initial_state: Optional[NDArray[np.float64]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Internal method to run MesoHOPS simulation with proper quantum dynamics.
        
        Mathematical Framework:
        Uses the Hierarchy of Pure States (HOPS) method to solve the
        stochastic Schrödinger equation for non-Markovian open quantum systems:
        
        |ψ(t)⟩ = |ψ(0)⟩ + ∫₀ᵗ dt' [Ĥ_eff(t') + noise terms] |ψ(t')⟩
        
        where Ĥ_eff includes the system Hamiltonian and system-bath interactions.
        """
        # Import required MesoHOPS classes
        if HopsTrajectory is None:
            raise RuntimeError("MesoHOPS not available for simulation")
        
        try:
            # Get simulation parameters
            max_hierarchy = kwargs.get('max_hierarchy', self.max_hierarchy)
            
            # Set up hierarchy parameters
            hierarchy_param = {'MAXHIER': max_hierarchy}
            
            # Set up EOM parameters - use LINEAR for simplicity
            eom_param = {
                'EQUATION_OF_MOTION': 'LINEAR',
                'TIME_DEPENDENCE': False,
            }
            
            # Set up noise parameters based on test examples
            t_max = float(np.max(time_points)) if len(time_points) > 0 else 500.0
            dt_save = 1.0  # Time step for saving
            
            noise_param = {
                'SEED': kwargs.get('seed', 42),
                'MODEL': 'FFT_FILTER',
                'TLEN': t_max + 100.0,  # Add buffer
                'TAU': dt_save,
            }
            
            # Set up integrator parameters based on test examples
            integrator_param = {
                'INTEGRATOR': 'RUNGE_KUTTA',
                'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
                'EARLY_INTEGRATOR_STEPS': 5,
                'INCHWORM_CAP': 5,
                'STATIC_BASIS': None,
            }
            
            # Create HOPS trajectory with all parameters
            trajectory = HopsTrajectory(
                self.system_param,
                noise_param=noise_param,
                hierarchy_param=hierarchy_param,
                eom_param=eom_param,
                integration_param=integrator_param,
            )
            
            # Set initial state
            if initial_state is None:
                # Default: excitation on site 0 (first site)
                initial_state = np.zeros(self.hamiltonian.shape[0], dtype=complex)
                initial_state[0] = 1.0
            
            trajectory.initialize(initial_state.copy())
            
            # Propagate the system
            # Note: The actual timestep is tau * integrator_step
            # We want actual timestep = TAU, so we pass tau = TAU / integrator_step
            # Since default integrator_step is 0.5, we pass tau = 2.0 * TAU
            integrator_step = 0.5  # Default RK integrator step
            tau = dt_save / integrator_step
            
            import sys
            import io
            # Suppress normal stdout completely during trajectory propagation to avoid 
            # the spamming warnings, e.g., 'WARNING: circulant embedding is NOT positive semidefinite.'
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                trajectory.propagate(t_max, tau)
            finally:
                sys.stdout = old_stdout
            
            # Extract results from trajectory
            t_axis = np.array(trajectory.storage.data['t_axis'])
            psi_traj = np.array(trajectory.storage.data['psi_traj'])
            
            # Calculate ensemble averages if multiple trajectories are used
            # For simplicity here, using single trajectory results
            n_times = len(t_axis)
            n_sites = self.hamiltonian.shape[0]
            
            # Calculate density matrices and observables
            populations = np.zeros((n_times, n_sites))
            coherences = np.zeros(n_times)
            
            for i in range(n_times):
                # Calculate density matrix: |psi><psi|
                psi = psi_traj[i, :n_sites]  # Extract physical state
                rho = np.outer(psi, np.conj(psi))
                
                # Calculate populations
                populations[i, :] = np.real(np.diag(rho))
                
                # Calculate coherence measure (L1 norm)
                coherence = 0.0
                for m in range(n_sites):
                    for n in range(n_sites):
                        if m != n:
                            coherence += np.abs(rho[m, n])
                coherences[i] = coherence
            
            # Calculate other quantum metrics
            qfi_values = np.zeros(n_times)
            entropy_values = np.zeros(n_times)
            purity_values = np.zeros(n_times)
            linear_entropy_values = np.zeros(n_times)
            
            for i in range(n_times):
                # Calculate density matrix from ensemble average if multiple trajectories
                psi = psi_traj[i, :n_sites]
                rho = np.outer(psi, np.conj(psi))  # Pure state
                
                # Normalize density matrix
                trace_rho = np.trace(rho).real
                if trace_rho > 1e-10:
                    rho = rho / trace_rho
                
                purity_values[i] = self._calculate_purity(rho)
                entropy_values[i] = self._calculate_von_neumann_entropy(rho)
                linear_entropy_values[i] = self._calculate_linear_entropy(rho)
                
                # QFI calculation (simplified)
                try:
                    qfi_values[i] = self._calculate_qfi(rho, self.hamiltonian)
                except Exception:
                    qfi_values[i] = 0.0
            
            logger.debug("MesoHOPS simulation completed successfully")
            
            return {
                't_axis': t_axis,
                'populations': populations,
                'coherences': coherences,
                'qfi': qfi_values,
                'entropy': entropy_values,
                'purity': purity_values,
                'linear_entropy': linear_entropy_values,
                'simulator': 'MesoHOPS',
                'n_traj_used': 1,  # Single trajectory
                'max_hierarchy_used': max_hierarchy
            }
            
        except Exception as e:
            logger.error(f"MesoHOPS simulation failed: {e}")
            raise RuntimeError(f"MesoHOPS simulation failed: {e}")

    def _calculate_purity(self, rho: NDArray[np.float64]) -> float:
        """Calculate the purity of a density matrix: Tr[rho^2]."""
        return float(np.real(np.trace(np.dot(rho, rho))))

    def _calculate_von_neumann_entropy(self, rho: NDArray[np.float64]) -> float:
        """Calculate the von Neumann entropy: -Tr[rho * log(rho)]."""
        # Get eigenvalues of the density matrix
        eigenvals = np.linalg.eigvals(rho)
        # Take real part and ensure non-negative
        eigenvals = np.real(eigenvals)
        eigenvals = np.clip(eigenvals, 1e-12, None)
        # Calculate entropy: -Σ λᵢ log(λᵢ)
        entropy = -np.sum(eigenvals * np.log(eigenvals))
        return float(entropy)

    def _calculate_linear_entropy(self, rho: NDArray[np.float64]) -> float:
        """Calculate the linear entropy: 1 - Tr[rho^2]."""
        purity = self._calculate_purity(rho)
        return float(1.0 - purity)

    def _calculate_qfi(self, rho: NDArray[np.float64], H: NDArray[np.float64]) -> float:
        """
        Calculate the Quantum Fisher Information for parameter estimation.
        
        QFI = 2 * Σᵢ Σⱼ (λᵢ - λⱼ)² / (λᵢ + λⱼ) * |⟨i|H|j⟩|²
        where λᵢ are eigenvalues and |i⟩ are eigenvectors of rho.
        """
        try:
            eigenvals, eigenvecs = np.linalg.eigh(rho)
            eigenvals = np.real(eigenvals)
            
            # Calculate QFI
            qfi = 0.0
            n = len(eigenvals)
            for i in range(n):
                for j in range(n):
                    denom = eigenvals[i] + eigenvals[j]
                    if denom > 1e-12:
                        Hij = np.abs(np.vdot(eigenvecs[:, i], H @ eigenvecs[:, j]))**2
                        qfi += 2.0 * (eigenvals[i] - eigenvals[j])**2 / denom * Hij
            
            return float(qfi)
        except Exception:
            return 0.0

    def create_basis(self) -> Optional[HopsBasis]:
        """
        Create MesoHOPS basis for the Hamiltonian.

        Returns
        -------
        Optional[HopsBasis]
            MesoHOPS basis object or None if not available
        """
        if not MESOHOPS_AVAILABLE or HopsBasis is None:
            logger.warning("MesoHOPS not available for basis creation")
            return None

        try:
            system_dict = {
                'hamiltonian': self.hamiltonian,
                'temperature': self.temperature,
                'n_site': self.hamiltonian.shape[0],
                'n_exciton': 1,
            }
            basis = HopsBasis(system_dict, self.max_hierarchy)
            logger.debug("MesoHOPS basis created successfully")
            return basis
        except Exception as e:
            logger.error(f"Failed to create MesoHOPS basis: {e}")
            return None

    def create_trajectory(self, basis: HopsBasis) -> Optional[HopsTrajectory]:
        """
        Create MesoHOPS trajectory for quantum dynamics.

        Parameters
        ----------
        basis : HopsBasis
            MesoHOPS basis object

        Returns
        -------
        Optional[HopsTrajectory]
            MesoHOPS trajectory object or None if not available
        """
        if not MESOHOPS_AVAILABLE or HopsTrajectory is None:
            logger.warning("MesoHOPS not available for trajectory creation")
            return None

        try:
            trajectory = HopsTrajectory(
                basis,
                temperature=self.temperature,
                max_hier=self.max_hierarchy
            )
            logger.debug("MesoHOPS trajectory created successfully")
            return trajectory
        except Exception as e:
            logger.error(f"Failed to create MesoHOPS trajectory: {e}")
            return None

    @property
    def is_using_mesohops(self) -> bool:
        """Check if MesoHOPS is being used."""
        return self.use_mesohops and hasattr(self, 'system_param') and self.system_param is not None

    @property
    def simulator_type(self) -> str:
        """Get the type of simulator being used."""
        if self.is_using_mesohops:
            return "MesoHOPS"
        elif self.fallback_sim is not None:
            return "QuantumDynamicsSimulator (fallback)"
        else:
            return "None (not initialized)"
