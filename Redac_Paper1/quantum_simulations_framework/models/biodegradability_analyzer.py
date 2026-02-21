"""
BiodegradabilityAnalyzer: Quantum reactivity descriptors for eco-design.

This module provides tools for analyzing molecular biodegradability using
quantum chemical calculations and Fukui functions.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig

logger = logging.getLogger(__name__)


class BiodegradabilityAnalyzer:
    """
    Class for analyzing biodegradability using quantum reactivity descriptors.

    Mathematical Framework:
    The biodegradability analysis is based on quantum reactivity descriptors
    that quantify the susceptibility of molecular structures to degradation
    processes such as hydrolysis and oxidation. The key descriptors are:

    1. Fukui functions: f^+(r), f^-(r), f^0(r) for electrophilic, nucleophilic,
       and radical attack reactivity, respectively
    2. Dual descriptor: Δf(r) = f^+(r) - f^-(r) for nucleophile vs electrophile
       selectivity
    3. Local spin density: for radical reactivity assessment

    These descriptors are calculated from the electronic structure of the
    molecular system and provide quantitative measures of reactivity at each
    site, which correlates with biodegradability.

    Parameters
    ----------
    molecular_hamiltonian : NDArray[np.float64]
        Molecular Hamiltonian matrix (n_orbitals x n_orbitals)
    n_electrons : int
        Number of electrons in the neutral system

    Attributes
    ----------
    molecular_hamiltonian : NDArray[np.float64]
        The molecular Hamiltonian matrix
    n_electrons : int
        Number of electrons
    n_orbitals : int
        Number of molecular orbitals
    evals : NDArray[np.float64]
        Orbital energies (eigenvalues)
    evecs : NDArray[np.complex128]
        Orbital coefficients (eigenvectors)
    density_n : NDArray[np.complex128]
        Electron density matrix for N electrons

    Examples
    --------
    >>> import numpy as np
    >>> from models.biodegradability_analyzer import BiodegradabilityAnalyzer
    >>> hamiltonian = np.random.rand(7, 7)
    >>> hamiltonian = (hamiltonian + hamiltonian.T) / 2  # Make symmetric
    >>> analyzer = BiodegradabilityAnalyzer(hamiltonian, n_electrons=14)
    >>> f_plus, f_minus, f_zero = analyzer.calculate_fukui_functions()
    >>> score = analyzer.calculate_biodegradability_score()
    """

    def __init__(
        self,
        molecular_hamiltonian: NDArray[np.float64],
        n_electrons: int
    ):
        """Initialize the biodegradability analyzer."""
        self.molecular_hamiltonian = molecular_hamiltonian
        self.n_electrons = n_electrons
        self.n_orbitals = molecular_hamiltonian.shape[0]

        logger.debug(
            f"Initializing BiodegradabilityAnalyzer with {n_electrons} electrons "
            f"and {self.n_orbitals} orbitals"
        )

        # Calculate reference electronic structure
        self.evals, self.evecs = eig(molecular_hamiltonian)
        self.evals = np.real(self.evals)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(self.evals)
        self.evals = self.evals[idx]
        self.evecs = self.evecs[:, idx]

        # Calculate reference electron density
        self.density_n = self._calculate_density_matrix(n_electrons)

        logger.info("BiodegradabilityAnalyzer initialized successfully")

    def _calculate_density_matrix(
        self,
        n_electrons: int
    ) -> NDArray[np.complex128]:
        """
        Calculate electron density matrix for a given number of electrons.

        Parameters
        ----------
        n_electrons : int
            Number of electrons in the system

        Returns
        -------
        NDArray[np.complex128]
            Electron density matrix (n_orbitals x n_orbitals)
        """
        density_matrix = np.zeros(
            (self.n_orbitals, self.n_orbitals),
            dtype=complex
        )
        n_filled_orbitals = min(n_electrons // 2, self.n_orbitals)

        # Fill lowest energy orbitals with 2 electrons each (closed shell)
        for i in range(n_filled_orbitals):
            orbital = self.evecs[:, i]
            density_matrix += 2 * np.outer(orbital, orbital.conj())

        # For odd number of electrons, add 1 electron to HOMO
        if n_electrons % 2 == 1 and n_filled_orbitals < self.n_orbitals:
            orbital = self.evecs[:, n_filled_orbitals]
            density_matrix += np.outer(orbital, orbital.conj())

        return density_matrix

    def calculate_fukui_functions(
        self
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculate Fukui functions for the molecular system.

        Mathematical Framework:
        The Fukui functions describe the change in electron density upon
        addition or removal of an electron:

        f^+(r) = ρ(N-1)(r) - ρ(N)(r)  # For electrophilic attack
        f^-(r) = ρ(N)(r) - ρ(N+1)(r)  # For nucleophilic attack
        f^0(r) = (f^+(r) + f^-(r)) / 2        # For radical attack

        where ρ(N)(r) is the electron density for a system with N electrons.
        In the discrete molecular orbital representation, these become:

        f^+_i = ρ_{N-1, ii} - ρ_{N, ii}
        f^-_i = ρ_{N, ii} - ρ_{N+1, ii}

        where ρ_{N, ii} is the diagonal element of the density matrix for site i.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            - f_plus: Fukui function for electrophilic attack
            - f_minus: Fukui function for nucleophilic attack
            - f_zero: Fukui function for radical attack
        """
        logger.debug("Calculating Fukui functions")

        # Calculate density matrices for N-1 and N+1 electron systems
        density_n_minus_1 = self._calculate_density_matrix(self.n_electrons - 1)
        density_n_plus_1 = self._calculate_density_matrix(self.n_electrons + 1)

        # Extract diagonal elements (atomic / molecular site densities)
        rho_n = np.real(np.diag(self.density_n))
        rho_n_minus_1 = np.real(np.diag(density_n_minus_1))
        rho_n_plus_1 = np.real(np.diag(density_n_plus_1))

        # Calculate Fukui functions
        f_plus = rho_n_minus_1 - rho_n    # Electrophilic attack
        f_minus = rho_n - rho_n_plus_1    # Nucleophilic attack
        f_zero = (f_plus + f_minus) / 2   # Radical attack

        logger.debug(
            f"Fukui functions calculated: max f+={np.max(f_plus):.3f}, "
            f"max f-={np.max(f_minus):.3f}"
        )

        return f_plus, f_minus, f_zero

    def calculate_dual_descriptor(self) -> NDArray[np.float64]:
        """
        Calculate the dual descriptor for nucleophile vs electrophile selectivity.

        Mathematical Framework:
        The dual descriptor Δf(r) measures the difference between
        electrophilic and nucleophilic reactivity:

        Δf(r) = f^+(r) - f^-(r)

        Positive values indicate sites more prone to nucleophilic attack,
        negative values indicate sites more prone to electrophilic attack.

        Returns
        -------
        NDArray[np.float64]
            Δf values for each site
        """
        f_plus, f_minus, _ = self.calculate_fukui_functions()
        dual_descriptor = f_plus - f_minus

        logger.debug(
            f"Dual descriptor range: {np.min(dual_descriptor):.3f} to "
            f"{np.max(dual_descriptor):.3f}"
        )

        return dual_descriptor

    def calculate_global_reactivity_indices(self) -> Dict[str, float]:
        """
        Calculate global reactivity indices.

        Mathematical Framework:
        Global reactivity indices provide system-wide measures of
        reactivity based on frontier molecular orbital theory:

        Chemical potential (μ): μ = (IP + EA) / 2 = -(ε_HOMO + ε_LUMO) / 2
        Chemical hardness (η): η = (IP - EA) / 2 = (ε_LUMO - ε_HOMO) / 2
        Chemical softness (S): S = 1 / (2η)
        Electronegativity (χ): χ = -μ

        where IP is ionization potential, EA is electron affinity,
        ε_HOMO is HOMO energy, and ε_LUMO is LUMO energy.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - chemical_potential: Chemical potential (μ)
            - chemical_hardness: Chemical hardness (η)
            - chemical_softness: Chemical softness (S)
            - electronegativity: Electronegativity (χ)
            - e_homo: HOMO energy
            - e_lumo: LUMO energy
        """
        logger.debug("Calculating global reactivity indices")

        # Find HOMO and LUMO indices
        n_filled_orbitals = self.n_electrons // 2

        if n_filled_orbitals > 0:
            e_homo = self.evals[n_filled_orbitals - 1]
        else:
            e_homo = -np.inf  # No occupied orbitals

        if n_filled_orbitals < self.n_orbitals:
            e_lumo = self.evals[n_filled_orbitals]
        else:
            e_lumo = np.inf   # No unoccupied orbitals

        # Calculate global reactivity indices
        if np.isfinite(e_homo) and np.isfinite(e_lumo):
            chemical_potential = -(e_homo + e_lumo) / 2
            chemical_hardness = (e_lumo - e_homo) / 2
            chemical_softness = (
                1.0 / (2 * chemical_hardness) if chemical_hardness != 0 else 0
            )
            electronegativity = -chemical_potential
        else:
            chemical_potential = 0.0
            chemical_hardness = 0.0
            chemical_softness = 0.0
            electronegativity = 0.0

        indices = {
            'chemical_potential': chemical_potential,
            'chemical_hardness': chemical_hardness,
            'chemical_softness': chemical_softness,
            'electronegativity': electronegativity,
            'e_homo': e_homo,
            'e_lumo': e_lumo
        }

        logger.debug(f"Global reactivity indices: μ={chemical_potential:.3f}, "
                    f"η={chemical_hardness:.3f}")

        return indices

    def calculate_biodegradability_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate a composite biodegradability score based on quantum descriptors.

        Mathematical Framework:
        The biodegradability score combines multiple quantum descriptors
        with different weightings to predict the relative susceptibility
        of a molecule to biodegradation:

        B_score = Σ_i w_i * descriptor_i

        where w_i are weighting factors and descriptor_i are the quantum
        reactivity descriptors (Fukui functions, global indices, etc.).

        The score is normalized to the range [0, 1] where higher values
        indicate greater biodegradability potential.

        Parameters
        ----------
        weights : Optional[Dict[str, float]]
            Optional weights for different descriptors.
            If None, uses default weights.

        Returns
        -------
        float
            Biodegradability score between 0 and 1
        """
        if weights is None:
            # Default weights based on literature for biodegradability prediction
            weights = {
                'fukui_nucleophilic': 0.3,
                'fukui_electrophilic': 0.2,
                'dual_descriptor': 0.2,
                'global_softness': 0.15,
                'max_fukui': 0.15
            }

        logger.debug("Calculating biodegradability score")

        # Calculate all descriptors
        f_plus, f_minus, f_zero = self.calculate_fukui_functions()
        dual_desc = self.calculate_dual_descriptor()
        global_indices = self.calculate_global_reactivity_indices()

        # Calculate individual descriptor contributions
        fukui_nucleophilic = np.mean(np.abs(f_minus)) if len(f_minus) > 0 else 0
        fukui_electrophilic = np.mean(np.abs(f_plus)) if len(f_plus) > 0 else 0
        dual_avg = np.mean(np.abs(dual_desc)) if len(dual_desc) > 0 else 0
        global_softness = (
            global_indices['chemical_softness']
            if global_indices['chemical_hardness'] != 0 else 0
        )
        max_fukui = (
            max(np.max(np.abs(f_plus)), np.max(np.abs(f_minus)))
            if len(f_plus) > 0 else 0
        )

        # Calculate weighted score
        score = (
            weights['fukui_nucleophilic'] * fukui_nucleophilic +
            weights['fukui_electrophilic'] * fukui_electrophilic +
            weights['dual_descriptor'] * dual_avg +
            weights['global_softness'] * min(1.0, global_softness * 10) +
            weights['max_fukui'] * min(1.0, max_fukui * 5)
        )

        # Ensure score is in [0, 1] range
        final_score = min(1.0, max(0.0, score))

        logger.info(f"Biodegradability score calculated: {final_score:.3f}")

        return final_score
