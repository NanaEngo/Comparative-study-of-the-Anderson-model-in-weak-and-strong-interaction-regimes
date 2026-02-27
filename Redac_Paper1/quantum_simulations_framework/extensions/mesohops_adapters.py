import logging

import numpy as np

try:
    from mesohops.noise.hops_noise import HopsNoise
except ImportError:
    logging.warning(
        "MesoHOPS not found in environment. Using mock classes for adapter structural mapping."
    )

    class HopsNoise:
        def __init__(self, *args, **kwargs):
            pass


try:
    from mesohops.trajectory.hops_trajectory import HopsTrajectory
except ImportError:

    class HopsTrajectory:
        def __init__(self, *args, **kwargs):
            pass


from .spectral_bundling import SpectrallyBundledDissipator

try:
    import jax
    import jax.numpy as jnp
    import quimb.tensor as qtn

    QUIMB_JAX_AVAILABLE = True
except ImportError:
    QUIMB_JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


def _construct_mpo_from_bcf(time_grid, correlation_func, bond_dim=10):
    """
    Constructs an MPO representation of the non-Markovian influence functional
    using Singular Value Decomposition (SVD) of the correlation matrix.
    """
    if not QUIMB_JAX_AVAILABLE:
        return None

    n_steps = len(time_grid)
    dt = time_grid[1] - time_grid[0]

    # 1. Build the correlation matrix C(t_i - t_j)
    t_row = time_grid[:, None]
    t_col = time_grid[None, :]
    delta_t = np.abs(t_row - t_col)

    # Evaluate correlation function over the time grid differences
    # Vectorize the correlation function to handle array inputs properly
    C = np.array(correlation_func(delta_t))
    
    # Ensure C is at least 2D (handle scalar or 0-dimensional results)
    if C.ndim < 2:
        # If correlation_func returned a scalar, build diagonal matrix
        if C.ndim == 0:
            C = np.full((n_steps, n_steps), float(C))
        else:
            C = np.atleast_2d(C)
    
    # Ensure C is properly shaped as (n_steps, n_steps)
    if C.shape != (n_steps, n_steps):
        # Reshape or tile to get correct shape
        C = np.resize(C, (n_steps, n_steps))
    
    # Fill any NaNs that might arise from numerical issues at t=0
    C = np.nan_to_num(C, nan=0.0)

    # 2. Perform Singular Value Decomposition
    # C = U * S * V^H
    U, S, Vh = np.linalg.svd(C, full_matrices=False)

    # 3. Truncate based on desired bond dimension
    # (or n_steps if bond_dim is larger than the matrix)
    eff_bond_dim = min(bond_dim, n_steps)
    U_trunc = U[:, :eff_bond_dim]
    S_trunc = S[:eff_bond_dim]
    Vh_trunc = Vh[:eff_bond_dim, :]

    # 4. Map the decomposed modes into an MPO structure
    # Start with an identity MPO as the structural base
    mpo = qtn.MPO_identity(n_steps, phys_dim=2)

    # Map the truncated singular modes to the tensors
    for i, t in enumerate(mpo.tensors):
        # Effective amplitude for this time step across all retained modes
        mode_amplitude = np.sum(U_trunc[i, :] * np.sqrt(S_trunc))
        mode_coupling = np.sum(Vh_trunc[:, i] * np.sqrt(S_trunc))

        # Construct the local influence propagator scaling factor
        scale_factor = np.abs(mode_amplitude * np.exp(-dt * np.abs(mode_coupling)))

        # Prevent numerical zeroing out
        if scale_factor < 1e-12:
            scale_factor = 1e-12

        t.modify(data=t.data * scale_factor)

    return mpo


class PT_HopsNoise(HopsNoise):
    """
    Process Tensor HOPS adapter for tensor-network based environment representation.

    STATUS
    -------
    This class provides the structural interface for PT-HOPS but currently delegates
    to standard HOPS dynamics. Full tensor network implementation requires integration
    with libraries such as Quimb, ITensor, or custom MPO contraction routines.

    MATHEMATICAL FRAMEWORK
    ----------------------
    The Process Tensor (PT) approach represents the environmental influence functional
    as a Matrix Product Operator (MPO):

        I[ψ*, ψ] = ∏_{k=1}^{N} M_k

    where M_k are MPO tensors encoding the non-Markovian memory kernel. This allows
    for efficient simulation of long-time dynamics with reduced memory scaling:

    - Standard HOPS: O(N_hierarchy × 2^N_sites) memory
    - PT-HOPS: O(N_time × D_bond^2) memory, where D_bond << N_hierarchy

    WHEN PT-HOPS IS ADVANTAGEOUS
    ----------------------------
    - Systems with >1000 chromophores (full chloroplast modeling)
    - Long-time dynamics where hierarchy depth becomes prohibitive
    - Systems requiring repeated simulations with different initial conditions

    CURRENT IMPLEMENTATION SCOPE
    ----------------------------
    For the current FMO-based simulations (7-8 sites, ~500 fs dynamics), the
    combination of standard HOPS with Spectrally Bundled Dissipators (SBD) provides
    equivalent accuracy with better numerical stability. PT-HOPS is planned for
    future extensions to full chloroplast modeling.

    Parameters
    ----------
    noise_param : dict
        Noise parameters for HOPS trajectory
    noise_corr : callable or list
        Bath correlation function(s)

    Attributes
    ----------
    is_pt_hops : bool
        Flag indicating PT-HOPS mode is active
    process_tensor_influence : Any
        Storage for MPO representation (future use)

    See Also
    --------
    SBD_HopsTrajectory : Spectrally Bundled Dissipators for mode compression
    SpectrallyBundledDissipator : Core SBD implementation
    core.hops_simulator.HopsSimulator : Unified simulator interface

    References
    ----------
    .. [1] Pollock et al., "Non-Markovian quantum dynamics: The quantum process
           tensor", Phys. Rev. Lett. 122, 040401 (2019)
    .. [2] Fux et al., "Efficient exploration of Hamiltonian parameter space
           using the process tensor", Phys. Rev. E 104, 045310 (2021)
    """

    _default_param = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,
        "TAU": 1.0,
        "INTERPOLATE": False,
        "RAND_MODEL": "SUM_GAUSSIAN",
        "STORE_RAW_NOISE": False,
        "NOISE_WINDOW": None,
        "ADAPTIVE": False,
        "FLAG_REAL": False,
    }
    _required_param = set()
    _param_types = {
        "SEED": (int, type(None), str, type(np.array([]))),
        "MODEL": (str,),
        "TLEN": (float,),
        "TAU": (float,),
        "INTERPOLATE": (bool,),
        "RAND_MODEL": (str,),
        "STORE_RAW_NOISE": (bool,),
        "NOISE_WINDOW": (type(None), float, int),
        "ADAPTIVE": (bool,),
        "FLAG_REAL": (bool,),
    }

    def __init__(self, noise_param, noise_corr, bond_dim=12):
        self.__locked__ = False
        super().__init__(noise_param, {})
        self.noise_corr = noise_corr
        self.is_pt_hops = True
        self.bond_dim = bond_dim
        self.process_tensor_mpo = None
        self.t_axis = None
        logger.info(f"Initialized PT_HOPS (Quimb+Jax) with bond_dim={bond_dim}")

    def _prepare_noise(self, new_lop, time_points=None):
        """
        Constructs the MPO representation of the influence functional.
        """
        if time_points is None:
            time_points = np.linspace(0, 500, 501)

        self.t_axis = time_points
        logger.info(f"Building PT-MPO for {len(time_points)} steps...")

        # Integration of actual correlation functions from self.noise_corr
        def combined_correlation(t):
            if callable(self.noise_corr):
                return self.noise_corr(t)
            elif isinstance(self.noise_corr, (list, tuple)):
                # Sum of multiple memory kernels (e.g. Drude + Vibronic)
                return sum(c(t) for c in self.noise_corr if callable(c))
            return 0.0

        if QUIMB_JAX_AVAILABLE:
            self.process_tensor_mpo = _construct_mpo_from_bcf(
                time_points, combined_correlation, bond_dim=self.bond_dim
            )

        self._noise = 0

        # Array-safe deduplication of L-operators
        if self._lop_active is None:
            self._lop_active = []

        for op in new_lop:
            # Check if this operator is already in the active list
            is_duplicate = False
            for active_op in self._lop_active:
                if np.array_equal(op, active_op):
                    is_duplicate = True
                    break
            if not is_duplicate:
                self._lop_active.append(op)

    def get_influence_propagator(self, t_step: int) -> float:
        """
        Contracts the PT-MPO using Jax-accelerated routines.
        """
        if not QUIMB_JAX_AVAILABLE or self.process_tensor_mpo is None:
            return 1.0

        # PT Contraction logic: I(t) = Tr[ M_0 * M_1 * ... * M_t ]
        # Using Jax backends via Quimb
        try:
            # Sub-mesh contraction for the specified time step
            tn_slice = self.process_tensor_mpo.slice(0, t_step + 1)
            # Contract using Jax
            res = tn_slice.contract(backend="jax")
            return float(jnp.abs(res))
        except KeyboardInterrupt:
            # Allow user to interrupt
            raise
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"PT Contraction failed at step {t_step}: {e}")
            logger.debug("PT contraction traceback:", exc_info=True)
            return 1.0


class SBD_HopsTrajectory(HopsTrajectory):
    """
    A Trajectory wrapper that triggers the Spectral Bundling Routine on initialization
    before delegating to standard or custom EOM integrators.
    """

    def __init__(
        self,
        system_param=None,
        eom_param=None,
        noise_param=None,
        noise2_param=None,
        hierarchy_param=None,
        storage_param=None,
        integration_param=None,
        n_bundles: int = 5,
    ):

        # 1. Intercept system parameters to apply SBD compression
        if system_param and "PARAM_NOISE1" in system_param:
            logger.info("SBD INTERCEPT: Preparing to compress spectral modes...")
            raw_modes = system_param["PARAM_NOISE1"]

            # The structure of raw_modes is defined in specific BCFs.
            # Often [g_1, w_1, g_2, w_2...]
            if len(raw_modes) > n_bundles * 2:
                self.sbd = SpectrallyBundledDissipator(n_bundles=n_bundles)

                # Zip into (w, g) or similar structure.
                # MesoHOPS usually passes correlations as [g_exp, w_exp, g_mats1, w_mats1...]
                modes_tuple = []
                for i in range(0, len(raw_modes), 2):
                    g = raw_modes[i]
                    w = raw_modes[i + 1]
                    modes_tuple.append((w, g))

                self.sbd.discretize_spectral_density(modes_tuple)
                bundled_ws, bundled_gs = self.sbd.get_bundle_parameters()

                # Re-flatten back into PARAM_NOISE1 format
                flattened = []
                for g, w in zip(bundled_gs, bundled_ws, strict=False):
                    flattened.append(g)
                    flattened.append(w)

                system_param["PARAM_NOISE1"] = flattened
                logger.info(
                    f"SBD Compression complete: reduced to {len(bundled_ws)} structural dissipator groups."
                )

        # Fallback to standard trajectory initialization with the bundled parameters
        super().__init__(
            system_param=system_param,
            eom_param=eom_param,
            noise_param=noise_param,
            noise2_param=noise2_param,
            hierarchy_param=hierarchy_param,
            storage_param=storage_param,
            integration_param=integration_param,
        )
