"""
Quantum Agrivoltaics Simulations: Refined Framework with Process Tensor-HOPS

This module implements a complete simulation framework for quantum-enhanced
agrivoltaic systems based on the paper "Process Tensor-HOPS: A non-recursive
framework for quantum-enhanced agrivoltaic design". The implementation incorporates the
Fenna-Matthews-Olsen (FMO) complex model, Process Tensor-HOPS,
and advanced spectral optimization for enhanced photosynthetic efficiency.

Key Features:
- FMO complex Hamiltonian with 7-site model
- Process Tensor-HOPS quantum dynamics simulation using MesoHOPS
- Stochastically Bundled Dissipators (SBD) for mesoscale systems
- E(n)-Equivariant Graph Neural Networks for physical symmetry preservation
- Quantum Reactivity Descriptors (Fukui functions) for eco-design
- Spectral optimization with multi-objective approach
- Data storage to CSV files with comprehensive metadata
- Publication-ready figure generation
- Parallel processing capabilities

Authors: Based on research by Nana Engo et al.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging first
from utils.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging(level=20)  # INFO level
logger = get_logger(__name__)

# Import constants
from core.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_HIERARCHY,
    DEFAULT_REORGANIZATION_ENERGY,
    DEFAULT_DRUDE_CUTOFF,
    FMO_SITE_ENERGIES_7,
    FMO_SITE_ENERGIES_8,
    FMO_COUPLINGS,
    MINIMAL_OMEGA,
    KB_CM_K,
)

# Import core classes
from core.hops_simulator import HopsSimulator

# Import models
from models.biodegradability_analyzer import BiodegradabilityAnalyzer
from models.sensitivity_analyzer import SensitivityAnalyzer
from models.testing_validation_protocols import TestingValidationProtocols
from models.lca_analyzer import LCAAnalyzer

# Import MesoHOPS modules with fallback
try:
    from mesohops.trajectory.hops_trajectory import HopsTrajectory
    from mesohops.basis.hops_basis import HopsBasis
    MESOHOPS_AVAILABLE = True
    logger.info("MesoHOPS modules imported successfully")
except ImportError as e:
    logger.warning(f"MesoHOPS not available: {e}")
    MESOHOPS_AVAILABLE = False

# Import other framework modules
try:
    from models.quantum_dynamics_simulator import QuantumDynamicsSimulator
    from models.agrivoltaic_coupling_model import AgrivoltaicCouplingModel
    from models.spectral_optimizer import SpectralOptimizer
    from models.eco_design_analyzer import EcoDesignAnalyzer
    from utils.csv_data_storage import CSVDataStorage
    from utils.figure_generator import FigureGenerator
    logger.info("Framework modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import framework modules: {e}")
    raise

# Import with fallback for missing modules
try:
    from environmental_factors import EnvironmentalFactors
except ImportError:
    logger.warning("EnvironmentalFactors module not found, using full implementation")

    class EnvironmentalFactors:
        r"""
        Class to model environmental factors that affect agrivoltaic system performance.

        Mathematical Framework:
        Environmental factors in agrivoltaic systems include:

        1. Dust accumulation: modeled as time-dependent attenuation following a power law
           A(t) = A_0 * (1 + \alpha * t^\beta)

        2. Temperature effects: affect OPV efficiency and photosynthetic performance
           \eta(T) = \eta_ref * [1 - \gamma * (T - T_ref)]

        3. Humidity effects: impact charge transport and photosynthetic activity

        4. Wind effects: influence heat dissipation and dust removal

        5. Precipitation: affects dust levels and temperature

        These factors are combined into a comprehensive environmental impact model
        that modifies the base performance metrics of the agrivoltaic system.
        """

        def __init__(self):
            """Initialize environmental factors with default parameters."""
            # Environmental parameters
            self.dust_accumulation_rate = 0.02  # Units per day
            self.dust_saturation_thickness = 5.0  # Max dust thickness
            self.temperature_coefficient_opv = 0.004  # % / K for OPV efficiency
            self.temperature_coefficient_psu = 0.002  # % / K for PSU efficiency
            self.humidity_coefficient = 0.001  # Effect of humidity on performance
            self.wind_speed_factor = 0.01  # Factor for dust removal
            logger.debug("EnvironmentalFactors initialized with default parameters")

        def dust_accumulation_model(self, time_days, initial_dust=0.1, weather_conditions='normal'):
            """
            Model dust accumulation over time with weather effects.

            Mathematical Framework:
            Dust accumulation is modeled as a saturating process with different
            accumulation rates based on weather conditions:

            d(t) = d_sat * (1 - exp(-k * t))

            where d_sat is the saturation dust thickness, k is the accumulation rate
            constant, and t is time. Precipitation events reset dust accumulation.

            Parameters
            ----------
            time_days : np.ndarray
                Time points in days
            initial_dust : float
                Initial dust thickness
            weather_conditions : str
                Weather type ('arid', 'normal', 'humid', 'dusty')

            Returns
            -------
            np.ndarray
                Dust thickness over time
            """
            # Adjust accumulation rate based on weather
            if weather_conditions == 'arid':
                k = self.dust_accumulation_rate * 1.5
            elif weather_conditions == 'dusty':
                k = self.dust_accumulation_rate * 2.0
            elif weather_conditions == 'humid':
                k = self.dust_accumulation_rate * 0.5
            else:  # normal
                k = self.dust_accumulation_rate

            # Simulate some random precipitation events that reset dust
            dust_thickness = np.zeros_like(time_days, dtype=float)
            current_dust = initial_dust

            for i, t in enumerate(time_days):
                # Apply accumulation
                current_dust = min(
                    self.dust_saturation_thickness,
                    current_dust + k * (1 - current_dust / self.dust_saturation_thickness)
                )

                # Random precipitation event (5% chance per day)
                if np.random.random() < 0.05:
                    current_dust *= 0.3  # Reduce dust by 70%

                dust_thickness[i] = current_dust

            return dust_thickness

        def temperature_effects_model(self, temperatures, base_efficiency, efficiency_type='opv'):
            r"""
            Model temperature effects on system efficiency.

            Mathematical Framework:
            Temperature effects on efficiency follow a linear model:

            \eta(T) = \eta_ref * [1 - \alpha * (T - T_ref)]

            where \eta_ref is the reference efficiency at T_ref, \alpha is the
            temperature coefficient, and T is the operating temperature.

            Parameters
            ----------
            temperatures : np.ndarray
                Temperature values in Kelvin
            base_efficiency : float
                Reference efficiency at reference temperature
            efficiency_type : str
                Type of efficiency ('opv', 'psu')

            Returns
            -------
            np.ndarray
                Temperature-adjusted efficiency
            """
            # Reference temperature (25°C = 298K)
            T_ref = 298  # K

            # Select temperature coefficient based on system type
            if efficiency_type == 'opv':
                alpha = self.temperature_coefficient_opv
            else:  # psu
                alpha = self.temperature_coefficient_psu

            # Calculate efficiency with temperature correction
            efficiency = base_efficiency * (1 - alpha * (temperatures - T_ref))

            # Ensure efficiency remains positive
            efficiency = np.clip(efficiency, 0, base_efficiency)

            return efficiency

        def humidity_effects_model(self, humidity_values, base_efficiency):
            r"""
            Model humidity effects on system efficiency.

            Mathematical Framework:
            Humidity effects can be complex, but are often approximated as:

            \eta_h = \eta_0 * (1 - \beta * |h - h_opt|)

            where h is relative humidity, h_opt is optimal humidity, and \beta is
            the humidity sensitivity coefficient.

            Parameters
            ----------
            humidity_values : np.ndarray
                Relative humidity values (0-1)
            base_efficiency : float
                Base efficiency without humidity effects

            Returns
            -------
            np.ndarray
                Humidity-adjusted efficiency
            """
            # Optimal humidity is around 45%
            optimal_humidity = 0.45

            # Calculate deviation from optimal humidity
            humidity_deviation = np.abs(humidity_values - optimal_humidity)

            # Apply humidity effects
            efficiency = base_efficiency * (1 - self.humidity_coefficient * humidity_deviation)

            # Ensure efficiency remains positive
            efficiency = np.clip(efficiency, 0, base_efficiency)

            return efficiency

        def wind_effects_model(self, wind_speeds, dust_thickness):
            """
            Model wind effects on dust removal and heat dissipation.

            Mathematical Framework:
            Wind affects dust accumulation through removal rate and influences
            heat dissipation. The dust removal rate is modeled as:

            dr/dt = -k_wind * v * d

            where v is wind speed and k_wind is the removal rate constant.

            Parameters
            ----------
            wind_speeds : np.ndarray
                Wind speeds in m/s
            dust_thickness : np.ndarray
                Current dust thickness

            Returns
            -------
            np.ndarray
                Dust thickness after wind effects
            """
            # Apply wind dust removal
            removal_factor = np.exp(-self.wind_speed_factor * wind_speeds)
            adjusted_dust = dust_thickness * removal_factor

            return adjusted_dust

        def combined_environmental_effects(
            self, time_days, temperatures, humidity_values, wind_speeds,
            base_pce, base_etr, weather_conditions='normal'
        ):
            r"""
            Combine all environmental effects into a comprehensive model.

            Mathematical Framework:
            The combined environmental effect is approximated as a product of
            individual effects:

            \eta_combined = \eta_base * \eta_dust * \eta_temp * \eta_humidity * \eta_wind

            This assumes that the environmental effects are approximately
            independent and multiplicative.

            Parameters
            ----------
            time_days : np.ndarray
                Time points in days
            temperatures : np.ndarray
                Temperature values in Kelvin
            humidity_values : np.ndarray
                Relative humidity values (0-1)
            wind_speeds : np.ndarray
                Wind speeds in m/s
            base_pce : float
                Base PCE without environmental effects
            base_etr : float
                Base ETR without environmental effects
            weather_conditions : str
                Weather type ('arid', 'normal', 'humid', 'dusty')

            Returns
            -------
            tuple
                (pce_env, etr_env, dust_profile) - Adjusted PCE, ETR, and dust profile
            """
            # Calculate dust accumulation
            dust_profile = self.dust_accumulation_model(time_days, weather_conditions=weather_conditions)

            # Apply wind effects to dust
            dust_profile = self.wind_effects_model(wind_speeds, dust_profile)

            # Calculate temperature effects
            pce_temp = self.temperature_effects_model(temperatures, base_pce, 'opv')
            etr_temp = self.temperature_effects_model(temperatures, base_etr, 'psu')

            # Calculate humidity effects
            pce_humidity = self.humidity_effects_model(humidity_values, base_pce)
            etr_humidity = self.humidity_effects_model(humidity_values, base_etr)

            # Combine all effects
            # For simplicity, using average dust effect across wavelengths
            dust_factor = 1 - (dust_profile / self.dust_saturation_thickness) * 0.3

            pce_env = base_pce * dust_factor * (pce_temp / base_pce) * (pce_humidity / base_pce)
            etr_env = base_etr * dust_factor * (etr_temp / base_etr) * (etr_humidity / base_etr)

            return pce_env, etr_env, dust_profile

        def save_environmental_data_to_csv(
            self,
            time_days: np.ndarray,
            temperatures: np.ndarray,
            humidity_values: np.ndarray,
            wind_speeds: np.ndarray,
            pce_env: np.ndarray,
            etr_env: np.ndarray,
            dust_profile: np.ndarray,
            filename_prefix: str = 'environmental_effects',
            output_dir: str = '../simulation_data/'
        ) -> str:
            """
            Save environmental effects data to CSV.

            Parameters
            ----------
            time_days : np.ndarray
                Time points in days
            temperatures : np.ndarray
                Temperature values in Kelvin
            humidity_values : np.ndarray
                Relative humidity values (0-1)
            wind_speeds : np.ndarray
                Wind speeds in m/s
            pce_env : np.ndarray
                PCE with environmental effects
            etr_env : np.ndarray
                ETR with environmental effects
            dust_profile : np.ndarray
                Dust thickness over time
            filename_prefix : str
                Prefix for output filename
            output_dir : str
                Directory to save CSV file

            Returns
            -------
            str
                Path to saved CSV file
            """
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            df = pd.DataFrame({
                'time_days': time_days,
                'temperature_k': temperatures,
                'humidity': humidity_values,
                'wind_speed_m_s': wind_speeds,
                'dust_thickness': dust_profile,
                'pce_with_environment': pce_env,
                'etr_with_environment': etr_env
            })

            filename = f"{filename_prefix}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False, float_format='%.6e')

            logger.info(f"Environmental data saved to {filepath}")
            return filepath

        def plot_environmental_effects(
            self,
            time_days: np.ndarray,
            temperatures: np.ndarray,
            humidity_values: np.ndarray,
            wind_speeds: np.ndarray,
            pce_env: np.ndarray,
            etr_env: np.ndarray,
            dust_profile: np.ndarray,
            filename_prefix: str = 'environmental_effects',
            figures_dir: str = '../Graphics/'
        ) -> str:
            """
            Plot environmental effects and save to file.

            Parameters
            ----------
            time_days : np.ndarray
                Time points in days
            temperatures : np.ndarray
                Temperature values in Kelvin
            humidity_values : np.ndarray
                Relative humidity values (0-1)
            wind_speeds : np.ndarray
                Wind speeds in m/s
            pce_env : np.ndarray
                PCE with environmental effects
            etr_env : np.ndarray
                ETR with environmental effects
            dust_profile : np.ndarray
                Dust thickness over time
            filename_prefix : str
                Prefix for output filename
            figures_dir : str
                Directory to save figures

            Returns
            -------
            str
                Path to saved figure
            """
            os.makedirs(figures_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle('Environmental Effects on Agrivoltaic System', fontsize=16, fontweight='bold')

            # Row 1: Environmental conditions
            ax1 = axes[0, 0]
            ax1.plot(time_days, temperatures, 'r-', linewidth=2)
            ax1.set_xlabel('Time (days)', fontsize=10)
            ax1.set_ylabel('Temperature (K)', fontsize=10)
            ax1.set_title('Temperature Variation', fontsize=11)
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            ax2.plot(time_days, humidity_values, 'b-', linewidth=2)
            ax2.set_xlabel('Time (days)', fontsize=10)
            ax2.set_ylabel('Relative Humidity', fontsize=10)
            ax2.set_title('Humidity Variation', fontsize=11)
            ax2.grid(True, alpha=0.3)

            ax3 = axes[0, 2]
            ax3.plot(time_days, wind_speeds, 'g-', linewidth=2)
            ax3.set_xlabel('Time (days)', fontsize=10)
            ax3.set_ylabel('Wind Speed (m/s)', fontsize=10)
            ax3.set_title('Wind Speed Variation', fontsize=11)
            ax3.grid(True, alpha=0.3)

            # Row 2: Effects on system performance
            ax4 = axes[1, 0]
            ax4.plot(time_days, dust_profile, 'orange', linewidth=2)
            ax4.set_xlabel('Time (days)', fontsize=10)
            ax4.set_ylabel('Dust Thickness', fontsize=10)
            ax4.set_title('Dust Accumulation', fontsize=11)
            ax4.grid(True, alpha=0.3)

            ax5 = axes[1, 1]
            ax5.plot(time_days, pce_env, 'purple', linewidth=2, label='PCE with Env. Effects')
            ax5.axhline(y=np.mean(pce_env), color='gray', linestyle='--', alpha=0.7, label='Average')
            ax5.set_xlabel('Time (days)', fontsize=10)
            ax5.set_ylabel('PCE', fontsize=10)
            ax5.set_title('PCE Under Environmental Effects', fontsize=11)
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)

            ax6 = axes[1, 2]
            ax6.plot(time_days, etr_env, 'brown', linewidth=2, label='ETR with Env. Effects')
            ax6.axhline(y=np.mean(etr_env), color='gray', linestyle='--', alpha=0.7, label='Average')
            ax6.set_xlabel('Time (days)', fontsize=10)
            ax6.set_ylabel('ETR', fontsize=10)
            ax6.set_title('ETR Under Environmental Effects', fontsize=11)
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)

            plt.tight_layout()

            filename = f"{filename_prefix}_{timestamp}.pdf"
            filepath = os.path.join(figures_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')

            png_filename = f"{filename_prefix}_{timestamp}.png"
            png_filepath = os.path.join(figures_dir, png_filename)
            plt.savefig(png_filepath, dpi=150, bbox_inches='tight')

            plt.close()

            logger.info(f"Environmental effects plots saved to {filepath}")
            return filepath


# Set publication style plots
plt.style.use(['science', 'notebook'])


def create_fmo_hamiltonian(include_reaction_center: bool = False) -> tuple:
    """
    Create the FMO Hamiltonian matrix based on standard parameters from the literature.

    Mathematical Framework:
    The Fenna-Matthews-Olsen (FMO) complex is modeled as an excitonic system
    with the Hamiltonian:

    H_FMO = Σᵢ εᵢ |i⟩⟨i| + Σᵢⱼ Jᵢⱼ |i⟩⟨j|

    where:
    - |i⟩ represents the electronic excited state of bacteriochlorophyll (BChl) a
    - εᵢ is the site energy of site i (relative to a reference energy)
    - Jᵢⱼ is the electronic coupling between sites i and j

    Parameters
    ----------
    include_reaction_center : bool
        Whether to include the reaction center state

    Returns
    -------
    tuple
        (H, site_energies) where H is the Hamiltonian matrix and site_energies
        is the array of site energies, both in cm^-1
    """
    logger.debug(f"Creating FMO Hamiltonian (include_reaction_center={include_reaction_center})")

    # Select appropriate site energies
    if include_reaction_center:
        site_energies = FMO_SITE_ENERGIES_8.copy()
        logger.debug("Using 8-site FMO with reaction center")
    else:
        site_energies = FMO_SITE_ENERGIES_7.copy()
        logger.debug("Using 7-site FMO")

    n_sites = len(site_energies)
    H = np.zeros((n_sites, n_sites))

    # Set diagonal elements (site energies)
    np.fill_diagonal(H, site_energies)

    # Fill in the coupling values
    for (i, j), value in FMO_COUPLINGS.items():
        if i < n_sites and j < n_sites:
            H[i, j] = value
            H[j, i] = value  # Ensure Hermitian

    logger.info(f"FMO Hamiltonian created with {n_sites} sites")
    return H, site_energies


def create_meso_hops_basis(H_fmo: np.ndarray, temperature: float = DEFAULT_TEMPERATURE,
                           max_hier: int = DEFAULT_MAX_HIERARCHY) -> 'HopsBasis':
    """
    Create MesoHOPS basis for the FMO Hamiltonian.

    Parameters
    ----------
    H_fmo : np.ndarray
        FMO Hamiltonian matrix
    temperature : float
        Temperature in Kelvin
    max_hier : int
        Maximum hierarchy level

    Returns
    -------
    HopsBasis
        MesoHOPS basis object

    Raises
    ------
    ImportError
        If MesoHOPS is not available
    """
    if not MESOHOPS_AVAILABLE:
        raise ImportError("MesoHOPS is not available. Please install it first.")

    logger.debug(f"Creating MesoHOPS basis (T={temperature}K, max_hier={max_hier})")

    system = {
        'hamiltonian': H_fmo,
        'temperature': temperature,
        'n_site': H_fmo.shape[0],
        'n_exciton': 1,
    }

    basis = HopsBasis(system, max_hier)
    logger.info("MesoHOPS basis created successfully")

    return basis


def create_meso_hops_trajectory(basis: 'HopsBasis', temperature: float = DEFAULT_TEMPERATURE,
                                max_hier: int = DEFAULT_MAX_HIERARCHY) -> 'HopsTrajectory':
    """
    Create MesoHOPS trajectory for quantum dynamics simulation.

    Parameters
    ----------
    basis : HopsBasis
        MesoHOPS basis object
    temperature : float
        Temperature in Kelvin
    max_hier : int
        Maximum hierarchy level

    Returns
    -------
    HopsTrajectory
        MesoHOPS trajectory object

    Raises
    ------
    ImportError
        If MesoHOPS is not available
    """
    if not MESOHOPS_AVAILABLE:
        raise ImportError("MesoHOPS is not available. Please install it first.")

    logger.debug("Creating MesoHOPS trajectory")

    trajectory = HopsTrajectory(
        basis,
        temperature=temperature,
        max_hier=max_hier
    )

    logger.info("MesoHOPS trajectory created successfully")

    return trajectory


def spectral_density_drude_lorentz(omega: np.ndarray, lambda_reorg: float,
                                   gamma: float, temperature: float) -> np.ndarray:
    """
    Calculate Drude-Lorentz spectral density.

    Mathematical Framework:
    J(ω) = 2λγω / (ω² + γ²)

    where λ is the reorganization energy, γ is the cutoff frequency,
    and ω is the frequency.

    Parameters
    ----------
    omega : np.ndarray
        Frequency array in cm^-1
    lambda_reorg : float
        Reorganization energy in cm^-1
    gamma : float
        Drude cutoff in cm^-1
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    np.ndarray
        Spectral density values
    """
    # Convert temperature to appropriate units (kT in cm^-1)
    kT = KB_CM_K * temperature

    # Drude-Lorentz spectral density
    J = 2 * lambda_reorg * gamma * omega / (omega**2 + gamma**2)

    # Apply detailed balance at finite temperature
    n_th = 1.0 / (np.exp(np.maximum(omega, MINIMAL_OMEGA) / kT) - 1)
    J *= (1 + n_th) if np.any(omega >= 0) else n_th - 1

    return J


def spectral_density_vibronic(omega: np.ndarray, omega_k: np.ndarray,
                              S_k: np.ndarray, Gamma_k: np.ndarray) -> np.ndarray:
    """
    Calculate spectral density for discrete vibronic modes.

    Mathematical Framework:
    J_vib(ω) = Σ_k S_k * ω_k² * Γ_k / [(ω - ω_k)² + Γ_k²]

    Parameters
    ----------
    omega : np.ndarray
        Frequency array in cm^-1
    omega_k : np.ndarray
        Vibronic mode frequencies in cm^-1
    S_k : np.ndarray
        Huang-Rhys factors
    Gamma_k : np.ndarray
        Damping parameters in cm^-1

    Returns
    -------
    np.ndarray
        Vibronic spectral density
    """
    J_vib = np.zeros_like(omega, dtype=float)

    for wk, Sk, Gk in zip(omega_k, S_k, Gamma_k):
        J_vib += Sk * wk**2 * Gk / ((omega - wk)**2 + Gk**2)

    return J_vib


def total_spectral_density(omega: np.ndarray, lambda_reorg: float = DEFAULT_REORGANIZATION_ENERGY,
                           gamma: float = DEFAULT_DRUDE_CUTOFF, temperature: float = DEFAULT_TEMPERATURE,
                           omega_vib: np.ndarray = None, S_vib: np.ndarray = None,
                           Gamma_vib: np.ndarray = None) -> np.ndarray:
    """
    Calculate total spectral density combining Drude-Lorentz and vibronic contributions.

    Parameters
    ----------
    omega : np.ndarray
        Frequency array in cm^-1
    lambda_reorg : float
        Reorganization energy in cm^-1
    gamma : float
        Drude cutoff in cm^-1
    temperature : float
        Temperature in Kelvin
    omega_vib : np.ndarray, optional
        Vibronic mode frequencies
    S_vib : np.ndarray, optional
        Huang-Rhys factors
    Gamma_vib : np.ndarray, optional
        Damping parameters

    Returns
    -------
    np.ndarray
        Total spectral density
    """
    from core.constants import (
        DEFAULT_VIBRONIC_FREQUENCIES,
        DEFAULT_HUANG_RHYS_FACTORS,
        DEFAULT_VIBRONIC_DAMPING,
    )

    J_drude = spectral_density_drude_lorentz(omega, lambda_reorg, gamma, temperature)

    if omega_vib is None:
        omega_vib = DEFAULT_VIBRONIC_FREQUENCIES
        S_vib = DEFAULT_HUANG_RHYS_FACTORS
        Gamma_vib = DEFAULT_VIBRONIC_DAMPING

    J_vib = spectral_density_vibronic(omega, omega_vib, S_vib, Gamma_vib)

    return J_drude + J_vib


# Import other classes from their modules
__all__ = [
    'HopsSimulator',
    'BiodegradabilityAnalyzer',
    'SensitivityAnalyzer',
    'TestingValidationProtocols',
    'EnvironmentalFactors',
    'LCAAnalyzer',
    'create_fmo_hamiltonian',
    'create_meso_hops_basis',
    'create_meso_hops_trajectory',
    'spectral_density_drude_lorentz',
    'spectral_density_vibronic',
    'total_spectral_density',
    'MESOHOPS_AVAILABLE',
]


if __name__ == "__main__":
    logger.info("Quantum Agrivoltaics Simulations - MesoHOPS Framework")
    logger.info("=" * 60)

    # Example usage
    logger.info("Creating FMO Hamiltonian...")
    H, energies = create_fmo_hamiltonian()
    logger.info(f"Hamiltonian shape: {H.shape}")
    logger.info(f"Site energies: {energies}")

    logger.info("\nInitializing HopsSimulator...")
    simulator = HopsSimulator(H, temperature=DEFAULT_TEMPERATURE)
    logger.info(f"Simulator type: {simulator.simulator_type}")

    logger.info("\nFramework ready for simulations!")