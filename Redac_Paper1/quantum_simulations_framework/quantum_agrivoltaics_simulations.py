"""
Quantum Agrivoltaics Simulations: Refined Framework with Process Tensor-HOPS

This module implements a complete simulation framework for quantum-enhanced agrivoltaic systems
based on the paper "Process Tensor-HOPS: A non-recursive 
framework for quantum-enhanced agrivoltaic design". The implementation incorporates the 
Fenna-Matthews-Olsen (FMO) complex model, Process Tensor-HOPS,
and advanced spectral optimization for enhanced photosynthetic efficiency.

Key Features:
- FMO complex Hamiltonian with 7-site model
- Process Tensor-HOPS quantum dynamics simulation
- Stochastically Bundled Dissipators (SBD) for mesoscale systems
- E(n)-Equivariant Graph Neural Networks for physical symmetry preservation
- Quantum Reactivity Descriptors (Fukui functions) for eco-design
- Enhanced optimization algorithms with differential evolution
- Comprehensive quantum metrics (QFI, von Neumann entropy, purity, linear entropy, concurrence)
- Eco-design analysis for sustainable materials
- Improved numerical stability and computational efficiency
- Spectral optimization with multi-objective approach
- Data storage to CSV files with comprehensive metadata
- Publication-ready figure generation
- Parallel processing capabilities

Authors: Nana Engo et al.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # For publication-quality plots
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eig, expm
from scipy.integrate import quad, trapezoid
import multiprocessing as mp
from functools import partial
import os
import warnings
import json
warnings.filterwarnings('ignore')

# Import all the refactored classes
from quantum_dynamics_simulator import (
    QuantumDynamicsSimulator, 
    spectral_density_drude_lorentz, 
    spectral_density_vibronic, 
    spectral_density_total
)
from agrivoltaic_coupling_model import AgrivoltaicCouplingModel
from spectral_optimizer import SpectralOptimizer
from eco_design_analyzer import EcoDesignAnalyzer
from unified_figures import UnifiedFigures
from csv_data_storage import CSVDataStorage
from csv_data_storage import CSVDataStorage
from figure_generator import FigureGenerator
from quantum_advantage_calculator import calculate_quantum_advantage, analyze_robustness, multi_objective_optimization


def create_fmo_hamiltonian(include_reaction_center=False):
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
    
    The site energies εᵢ account for the local electrostatic environment of
    each BChl a molecule, while the coupling elements Jᵢⱼ describe the
    Förster (dipole-dipole) and Dexter (exchange) interactions that enable
    electronic energy transfer between the pigments.
    
    The coupling strength is calculated as:
    
    Jᵢⱼ = (μᵢ·μⱼ)/rᵢⱼ³ - (3(μᵢ·rᵢⱼ)(μⱼ·rᵢⱼ))/rᵢⱼ⁵
    
    where μᵢ is the transition dipole moment of site i and rᵢⱼ is the
    distance vector between sites i and j.
    
    Parameters:
    include_reaction_center (bool): Whether to include the reaction center state
    
    Returns:
    H (2D array): Hamiltonian matrix in units of cm^-1
    site_energies (1D array): Site energies in cm^-1
    """
    # FMO site energies (cm^-1) - Adolphs & Renger 2006, Biophys. J. 91:2778-2797
    # DOI: 10.1529/biophysj.105.079483
    if include_reaction_center:
        # Include 8 sites with reaction center
        site_energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440, 11700])  # Last is RC
    else:
        # Standard 7-site FMO complex
        site_energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])
    
    # FMO coupling parameters (cm^-1) - Adolphs & Renger 2006
    n_sites = len(site_energies)
    H = np.zeros((n_sites, n_sites))
    
    # Set diagonal elements (site energies)
    np.fill_diagonal(H, site_energies)
    
    # Off-diagonal elements (couplings) - symmetric matrix
    # Excitonic couplings from Adolphs & Renger 2006 (cm^-1)
    couplings = {
        (0, 1): -87.7, (0, 2): 5.5, (0, 3): -5.9, (0, 4): 6.7, (0, 5): -13.7, (0, 6): -9.9,
        (1, 2): 30.8, (1, 3): 8.2, (1, 4): 0.7, (1, 5): 11.8, (1, 6): 4.3,
        (2, 3): -53.5, (2, 4): -2.2, (2, 5): -9.6, (2, 6): 6.0,
        (3, 4): -70.7, (3, 5): -17.0, (3, 6): -63.3,
        (4, 5): 81.1, (4, 6): -1.3,
        (5, 6): 39.7
    }
    
    # Fill in the coupling values
    for (i, j), value in couplings.items():
        if i < n_sites and j < n_sites:
            H[i, j] = value
            H[j, i] = value  # Ensure Hermitian
    
    return H, site_energies


def solar_spectrum_am15g(wavelengths):
    """
    Standard AM1.5G solar spectrum (mW/cm²/nm) as implemented in the notebook.
    
    Mathematical Framework:
    The AM1.5G (Air Mass 1.5 Global) solar spectrum represents the standard
    reference solar irradiance at the Earth's surface under specific conditions:
    - 1.5 air masses (sun at 48.2° from zenith)
    - Global irradiance including direct and diffuse components
    - Integrated over the entire sky hemisphere
    
    The spectral irradiance S(λ) is typically given in units of W/m²/nm or
    W/m²/µm. The total power density is approximately 1000 W/m²:
    
    ∫₀^∞ S_AM1.5G(λ) dλ ≈ 1000 W/m²
    
    For photosynthetic applications, the PAR (Photosynthetically Active Radiation)
    range is critical: 400-700 nm, which contains the wavelengths most effective
    for photosynthesis. This range typically contains about 43% of the total
    solar power.
    
    The AM1.5G spectrum is used as the reference for testing and rating
    photovoltaic devices and is essential for modeling the incident light
    conditions in agrivoltaic systems.
    
    Parameters:
    wavelengths (array): Wavelengths in nm
    
    Returns:
    irradiance (array): Solar irradiance values
    """
    # Simple model for AM1.5G, normalized to match typical values
    # This is a simplified representation; in practice, use tabulated values
    
    # Create a spectrum with appropriate shape
    irradiance = np.zeros_like(wavelengths, dtype=float)
    
    # Add main features of solar spectrum
    for i, wl in enumerate(wavelengths):
        if wl < 300:
            # UV cutoff
            irradiance[i] = 0
        elif wl < 400:
            # UV region
            irradiance[i] = 0.5 * np.exp(-(wl-300)/50)
        elif wl < 700:
            # Visible region - approximate solar maximum
            irradiance[i] = 1.5 * np.exp(-((wl-600)/150)**2) + 1.2
        elif wl < 1100:
            # Near-IR region
            irradiance[i] = 1.0 * np.exp(-((wl-850)/150)**2) + 0.8
        elif wl < 1500:
            # IR region
            irradiance[i] = 0.6 * np.exp(-((wl-1200)/200)**2) + 0.4
        else:
            # Far-IR - decreasing
            irradiance[i] = 0.2 * np.exp(-(wl-1500)/300)
    
    # Normalize to approximate AM1.5G total (about 1000 W/m²)
    irradiance = irradiance * 1000 / trapezoid(irradiance, wavelengths) * (wavelengths[1]-wavelengths[0])
    
    return irradiance

def load_simulation_params(param_file_path='./data_input/quantum_agrivoltaics_params.json'):
    """
    Load simulation parameters from JSON file.
    
    Parameters:
    param_file_path (str): Path to the parameters JSON file
    
    Returns:
    dict: Dictionary containing all simulation parameters
    """
    try:
        with open(param_file_path, 'r') as f:
            params = json.load(f)
        print(f"Loaded simulation parameters from {param_file_path}")
        return params
    except FileNotFoundError:
        print(f"Parameter file not found at {param_file_path}, using default parameters")
        # Return default parameters
        return {
            "simulation_params": {
                "temperature": 295,
                "dephasing_rate": 20,
                "time_points": 500,
                "max_time": 500,
                "n_filters": 3
            },
            "fmo_hamiltonian_params": {
                "include_reaction_center": False
            },
            "opv_params": {
                "opv_bandgap": 1.4,
                "opv_absorption_coeff": 1.0
            },
            "quantum_metrics": {
                "calculate_qfi": True,
                "calculate_entropy": True,
                "calculate_purity": True,
                "calculate_concurrence": True,
                "calculate_entanglement": True
            },
            "optimization_params": {
                "maxiter": 100,
                "popsize": 15,
                "strategy": "best1bin",
                "mutation": 0.5,
                "recombination": 0.7,
                "workers": -1
            },
            "solar_spectrum_params": {
                "wavelength_min": 300,
                "wavelength_max": 800,
                "n_points": 801
            },
            "bath_params": {
                "lambda_reorg": 35,
                "gamma": 50,
                "temperature": 295,
                "omega_vib": [150, 200, 575, 1185],
                "S_vib": [0.05, 0.02, 0.01, 0.005],
                "Gamma_vib": [10, 10, 20, 20]
            },
            "process_tensor_params": {
                "N_Mat": 0
            },
            "sbd_params": {
                "bundle_count": 50,
                "adaptive_error_control": True,
                "max_chromophores": 1000
            }
        }

def create_solar_spectrum(wavelength_min=None, wavelength_max=None, n_points=None):
    """
    Create a standard solar spectrum (AM1.5G) for use in calculations.
    Based on the notebook implementation but with realistic values.
    
    Parameters:
    wavelength_min, wavelength_max, n_points: Can override values from params file
    """
    # Load parameters
    params = load_simulation_params()
    
    # Use values from params file or override with provided values
    wavelength_min = wavelength_min or params['solar_spectrum_params']['wavelength_min']
    wavelength_max = wavelength_max or params['solar_spectrum_params']['wavelength_max']
    n_points = n_points or params['solar_spectrum_params']['n_points']
    
    wavelengths = np.linspace(wavelength_min, wavelength_max, n_points)  # nm
    
    # Create a realistic solar spectrum approximation
    solar_irradiance = np.zeros_like(wavelengths, dtype=float)
    
    # Add main features of solar spectrum
    for i, wl in enumerate(wavelengths):
        if wl < 300:
            # UV cutoff
            solar_irradiance[i] = 0
        elif wl < 400:
            # UV region
            solar_irradiance[i] = 0.5 * np.exp(-(wl-300)/50)
        elif wl < 700:
            # Visible region - approximate solar maximum
            solar_irradiance[i] = 1.5 * np.exp(-((wl-600)/150)**2) + 1.2
        elif wl < 1100:
            # Near-IR region
            solar_irradiance[i] = 1.0 * np.exp(-((wl-850)/150)**2) + 0.8
        elif wl < 1500:
            # IR region
            solar_irradiance[i] = 0.6 * np.exp(-((wl-1200)/200)**2) + 0.4
        else:
            # Far-IR - decreasing
            solar_irradiance[i] = 0.2 * np.exp(-(wl-1500)/300)
    
    # Normalize to approximate AM1.5G total (about 1000 W/m²)
    solar_irradiance = solar_irradiance * 1000 / np.trapezoid(solar_irradiance, wavelengths) * (wavelengths[1]-wavelengths[0])
    
    return wavelengths, solar_irradiance


def calculate_etrs_for_transmission(transmission_func, wavelengths, solar_irradiance, 
                                  fmo_hamiltonian, max_time=500, num_time_points=500):
    """
    Calculate ETR for a given transmission function based on the notebook implementation.
    
    Mathematical Framework:
    The photosynthetic light harvesting efficiency in the presence of an OPV filter
    is calculated by considering the modified incident light spectrum:
    
    S_transmitted(λ) = S₀(λ) * T(λ)
    
    where S₀(λ) is the original solar spectrum and T(λ) is the OPV transmission
    function. The total number of absorbed photons in the photosynthetically
    active radiation (PAR) range (400-700 nm) is:
    
    N_absorbed = ∫⁷⁰⁰₄₀₀ S_transmitted(λ) dλ
    
    The light harvesting efficiency is then defined as the ETR per absorbed photon:
    
    η_LH = ETR / N_absorbed
    
    In this model, we account for the fact that different FMO sites have different
    absorption cross-sections across the spectrum, so the initial excitation
    distribution depends on the transmitted spectrum.
    
    Parameters:
    transmission_func (array): Transmission values for each wavelength
    wavelengths (array): Wavelengths in nm
    solar_irradiance (array): Solar irradiance values
    fmo_hamiltonian: FMO Hamiltonian matrix
    max_time, num_time_points: Simulation parameters
    
    Returns:
    etr_per_photon (float): ETR per absorbed photon
    absorbed_photons (float): Total number of absorbed photons
    """
    # Calculate transmitted spectrum
    if callable(transmission_func):
        transmission_values = transmission_func(wavelengths)
    else:
        transmission_values = transmission_func
        
    transmitted_spectrum = solar_irradiance * transmission_values
    
    # Calculate total number of absorbed photons in PAR range (400-700 nm)
    par_mask = (wavelengths >= 400) & (wavelengths <= 700)
    absorbed_photons = trapezoid(transmitted_spectrum[par_mask], wavelengths[par_mask])
    
    # For simplicity, we'll create a frequency-dependent excitation rate
    # based on the transmitted spectrum and pigment absorption
    
    # We'll use a simplified approach: scale the initial excitation 
    # based on the transmitted spectrum
    avg_transmitted_par = np.mean(transmitted_spectrum[par_mask]) if np.any(par_mask) else 0
    
    # Create a modified initial state based on transmission
    # For the FMO complex, different sites absorb at different wavelengths
    fmo_simulator = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=295, dephasing_rate=10)
    
    # Modify initial state based on transmission
    initial_state = np.zeros((fmo_simulator.n_sites, fmo_simulator.n_sites), dtype=complex)
    
    # Distribute initial excitation based on site absorption characteristics
    # In FMO, sites 1 and 6 are primary absorbers
    site_weights = np.array([0.3, 0.2, 0.05, 0.1, 0.1, 0.2, 0.05])  # Example weights
    site_weights = site_weights / np.sum(site_weights)
    
    # Apply transmission effect to initial excitation
    for i in range(fmo_simulator.n_sites):
        initial_state[i, i] = site_weights[i] * (0.5 + 0.5 * 
            np.mean(transmission_values[par_mask]))  # Scale based on average transmission
    
    # Normalize
    initial_state = initial_state / np.trace(initial_state)
    
    # Run simulation
    time_points = np.linspace(0, max_time, num_time_points)  # fs
    _, _, populations, _, _, _, _, _, _, _, _ = fmo_simulator.simulate_dynamics(
        initial_state=initial_state,
        time_points=time_points
    )
    
    # Calculate ETR
    _, etr_avg, etr_per_photon = fmo_simulator.calculate_etr(populations, time_points)
    
    return etr_per_photon, absorbed_photons


def run_complete_analysis():
    """
    Run a complete analysis of the quantum agrivoltaics system.
    
    This function demonstrates the complete workflow of the quantum agrivoltaics
    simulation framework, from quantum dynamics simulation to eco-design analysis.
    """
    import os
    print("QUANTUM AGRIPLVOLTAICS SIMULATION FRAMEWORK")
    print("="*60)
    
    # Set default n_processes based on the system's CPU count
    total_cores = os.cpu_count()
    if total_cores >= 8:
        n_processes = total_cores - 4
    else:
        n_processes = total_cores - 2
    print(f"Using {n_processes} processes out of {total_cores} available cores (default logic: nproc-4 if nproc>=8; else nproc-2)")
    
    # Part 1: Quantum Dynamics Simulation
    print("\n" + "="*50)
    print("PART 1: QUANTUM DYNAMICS SIMULATION")
    print("="*50)
    
    # Create FMO Hamiltonian
    fmo_hamiltonian, site_energies = create_fmo_hamiltonian()
    print(f"  FMO Hamiltonian created: {fmo_hamiltonian.shape[0]} sites")
    print(f"  Site energies range: {np.min(site_energies):.1f} to {np.max(site_energies):.1f} cm⁻¹")
    
    # Initialize quantum simulator
    quantum_sim = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=295, dephasing_rate=20)
    print(f"  Quantum dynamics simulator initialized")
    print(f"  Temperature: {quantum_sim.temperature} K")
    print(f"  Dephasing rate: {quantum_sim.dephasing_rate} cm⁻¹")
    
    # Run quantum dynamics simulation
    time_points, density_matrices, populations, coherences, qfi_values, \
    entropy_values, purity_values, linear_entropy_values, bipartite_ent_values, \
    multipartite_ent_values, pairwise_concurrence_values = quantum_sim.simulate_dynamics(
        time_points=np.linspace(0, 500, 100)  # fs
    )
    
    print(f"  Quantum dynamics simulation completed")
    print(f"    Time evolution: {len(time_points)} points from {time_points[0]:.1f} to {time_points[-1]:.1f} fs")
    print(f"    Final populations: {populations[-1, :]}")
    print(f"    Max QFI: {np.max(qfi_values):.3f}")
    print(f"    Max entropy: {np.max(entropy_values):.3f}")
    print(f"    Min purity: {np.min(purity_values):.3f}")
    print(f"    Max bipartite entanglement: {np.max(bipartite_ent_values):.3f}")
    print(f"    Max multipartite entanglement: {np.max(multipartite_ent_values):.3f}")
    print(f"    Max pairwise concurrence: {np.max(pairwise_concurrence_values):.3f}")
    
    # Calculate ETR over time (simplified)
    _, _, etr_per_photon_vals = quantum_sim.calculate_etr(populations, time_points)  # Use the new method
    etr_time = np.full(len(time_points), etr_per_photon_vals)  # Use calculated ETR
    
    # Part 2: Agrivoltaic Coupling Model
    print("\n" + "="*50)
    print("PART 2: AGRIPLVOLTAIC COUPLING MODEL")
    print("="*50)
    
    # Create solar spectrum
    wavelengths, solar_irradiance = create_solar_spectrum()
    print(f"  Solar spectrum created: {len(wavelengths)} points from {wavelengths[0]:.0f} to {wavelengths[-1]:.0f} nm")
    
    # Initialize agrivoltaic coupling model
    agrivoltaic_model = AgrivoltaicCouplingModel(
        fmo_hamiltonian, 
        solar_spectrum=(wavelengths, solar_irradiance),
        opv_bandgap=1.4,
        opv_absorption_coeff=1.0
    )
    print(f"  Agrivoltaic coupling model initialized")
    print(f"    OPV bandgap: {agrivoltaic_model.opv_bandgap} eV")
    print(f"    OPV absorption coefficient: {agrivoltaic_model.opv_absorption_coeff}")
    print(f"    PSU response range: {np.min(agrivoltaic_model.R_psu):.3f} to {np.max(agrivoltaic_model.R_psu):.3f}")
    
    # Part 3: Spectral Optimization
    print("\n" + "="*50)
    print("PART 3: SPECTRAL OPTIMIZATION")
    print("="*50)
    
    # Initialize spectral optimizer
    spec_opt = SpectralOptimizer(agrivoltaic_model, quantum_sim, n_processes=n_processes)
    print(f"  Spectral optimizer initialized with {spec_opt.n_filters} filters")
    print(f"  Optimization bounds: {len(spec_opt.bounds)} parameters")
    
    # Define initial transmission function (neutral - all light to OPV)
    initial_params = [0.01, 600, 50] * spec_opt.n_filters  # Minimal filtering
    T_initial = agrivoltaic_model.calculate_spectral_transmission(initial_params)
    
    # Calculate initial performance
    etr_initial, _ = calculate_etrs_for_transmission(
        T_initial, wavelengths, solar_irradiance, fmo_hamiltonian, max_time=500, num_time_points=500
    )
    pce_initial = spec_opt.calculate_pce(T_initial)
    spce_initial = 0.5 * pce_initial + 0.5 * etr_initial
    
    print(f"  Initial parameters set")
    print(f"    PCE (initial): {pce_initial:.4f}")
    print(f"    ETR (initial): {etr_initial:.4f}")
    print(f"    SPCE (initial): {spce_initial:.4f}")
    
    # Perform optimization
    try:
        opt_results = spec_opt.optimize_spectral_splitting(
            method='differential_evolution',
            maxiter=15,  # Reduced for demonstration
            popsize=15   # Reduced for demonstration
        )
        
        if opt_results['success']:
            print(f"  Spectral optimization completed successfully")
            print(f"    Success: {opt_results['success']}")
            print(f"    PCE (optimized): {opt_results['pce']:.4f}")
            print(f"    ETR (optimized): {opt_results['etr']:.4f}")
            print(f"    SPCE (optimized): {opt_results['spce']:.4f}")
            print(f"    Optimal parameters: {opt_results['optimal_params'][:6]}...")  # Show first few
        else:
            print(f"  Spectral optimization did not converge, using initial parameters")
            print(f"    Message: {opt_results.get('message', 'No message')}")
            
            # Calculate performance with initial parameters as fallback
            pce_initial = spec_opt.calculate_pce(T_initial)
            etr_initial = spec_opt.calculate_etr(T_initial)
            spce_initial = 0.5 * pce_initial + 0.5 * etr_initial
            
            opt_results = {
                'success': False,
                'pce': pce_initial,
                'etr': etr_initial,
                'spce': spce_initial,
                'final_transmission': T_initial,
                'final_params': initial_params
            }
            print(f"    PCE (initial): {opt_results['pce']:.4f}")
            print(f"    ETR (initial): {opt_results['etr']:.4f}")
            print(f"    SPCE (initial): {opt_results['spce']:.4f}")
    except Exception as e:
        print(f"  Spectral optimization encountered an error: {str(e)}, using initial parameters")
        # Calculate performance with initial parameters as fallback
        pce_initial = spec_opt.calculate_pce(T_initial)
        etr_initial = spec_opt.calculate_etr(T_initial)
        spce_initial = 0.5 * pce_initial + 0.5 * etr_initial
        
        opt_results = {
            'success': False,
            'pce': pce_initial,
            'etr': etr_initial,
            'spce': spce_initial,
            'final_transmission': T_initial,
            'final_params': initial_params
        }
        print(f"    PCE (initial): {opt_results['pce']:.4f}")
        print(f"    ETR (initial): {opt_results['etr']:.4f}")
        print(f"    SPCE (initial): {opt_results['spce']:.4f}")
    
    # Part 4: Eco-Design Analysis
    print("\n" + "="*50)
    print("PART 4: ECO-DESIGN ANALYSIS")
    print("="*50)
    
    # Initialize eco-design analyzer
    eco_analyzer = EcoDesignAnalyzer(agrivoltaic_model, quantum_sim)
    print(f"  Eco-design analyzer initialized")
    
    # Find eco-friendly candidates
    eco_candidates = eco_analyzer.find_eco_friendly_candidates(
        min_biodegradability=0.7, 
        min_pce_potential=0.12
    )
    
    print(f"  Eco-design analysis completed")
    print(f"    Number of eco-friendly candidates: {len(eco_candidates)}")
    if eco_candidates:
        print(f"    Top candidate: {eco_candidates[0]['material_name']}")
        print(f"      Biodegradability: {eco_candidates[0]['biodegradability']:.3f}")
        print(f"      PCE potential: {eco_candidates[0]['pce_potential']:.3f}")
        print(f"      Multi-objective score: {eco_candidates[0]['multi_objective_score']:.3f}")
    
    # Part 5: Data Storage
    print("\n" + "="*50)
    print("PART 5: DATA STORAGE")
    print("="*50)
    
    # Initialize data storage
    data_storage = CSVDataStorage(output_dir='data_output')
    print(f"  Data storage initialized, output directory: {data_storage.output_dir}")
    
    # Save quantum dynamics
    data_storage.save_simulation_data_to_csv(
        time_points, 
        populations, 
        coherences, 
        qfi_values, 
        etr_time,
        filename_prefix='fmo_dynamics'
    )
    
    # Save extended quantum metrics
    data_storage.save_quantum_metrics_to_csv(
        time_points, 
        entropy_values, 
        purity_values, 
        linear_entropy_values, 
        bipartite_ent_values, 
        multipartite_ent_values, 
        pairwise_concurrence_values,
        filename='extended_quantum_metrics'
    )
    
    # Save spectral optimization
    data_storage.save_optimization_results_to_csv(
        {'initial_params': initial_params}, 
        opt_results, 
        filename='spectral_optimization'
    )
    
    # Save eco-design analysis
    if eco_candidates:
        eco_df = pd.DataFrame(eco_candidates)
        data_storage.save_eco_analysis_to_csv(
            eco_candidates,
            filename='eco_design_analysis'
        )
    
    # Part 6: Figure Generation
    print("\n" + "="*50)
    print("PART 6: FIGURE GENERATION")
    print("="*50)
    
    # Initialize figure generator
    fig_gen = FigureGenerator()
    print(f"  Figure generator initialized")
    
    # Generate quantum dynamics figure
    quantum_fig = fig_gen.plot_quantum_dynamics(
        time_points, 
        populations, 
        coherences, 
        qfi_values, 
        etr_time
    )
    
    # Generate extended quantum metrics figure
    ext_metrics_fig = fig_gen.plot_quantum_metrics_extended(
        time_points, 
        entropy_values, 
        purity_values, 
        linear_entropy_values, 
        bipartite_ent_values, 
        multipartite_ent_values, 
        pairwise_concurrence_values
    )
    
    # Generate spectral optimization figure
    spec_fig = fig_gen.plot_spectral_optimization(
        wavelengths, 
        T_initial, 
        opt_results.get('transmission_function', T_initial), 
        agrivoltaic_model.R_opv, 
        agrivoltaic_model.R_psu,
        title="Spectral Optimization Results"
    )
    
    # Generate eco-design analysis figure if there are candidates
    if eco_candidates:
        eco_df = pd.DataFrame(eco_candidates)
        eco_fig = fig_gen.plot_eco_design_analysis(eco_df, title="Eco-Design Analysis")
    
    # Generate robustness analysis
    robustness_data, temperatures, disorder_strengths = analyze_robustness(
        fmo_hamiltonian, wavelengths, solar_irradiance
    )
    robustness_fig = fig_gen.plot_robustness_analysis(robustness_data, title="Robustness Analysis")
    
    # Calculate quantum advantage
    T_comparison = opt_results.get('final_transmission', T_initial)
    if hasattr(T_comparison, '__call__'):
        T_comparison = T_comparison(wavelengths)
    eta_quantum, etr_quantum, etr_markovian = calculate_quantum_advantage(
        fmo_hamiltonian, wavelengths, solar_irradiance, T_comparison
    )
    
    print(f"\nQUANTUM ADVANTAGE ANALYSIS")
    print(f"{'=' * 40}")
    print(f"ETR (Quantum/Non-Markovian): {etr_quantum:.4f}")
    print(f"ETR (Markovian Approximation): {etr_markovian:.4f}")
    print(f"Quantum Advantage: {eta_quantum:.4f} ({eta_quantum*100:.2f}%)")
    
    # Perform multi-objective optimization
    print(f"\nMULTI-OBJECTIVE OPTIMIZATION")
    print(f"{'=' * 40}")
    try:
        optimal_params, best_score = multi_objective_optimization(
            wavelengths, solar_irradiance, fmo_hamiltonian
        )
        print(f"Optimal parameters found with score: {best_score:.4f}")
    except Exception as e:
        print(f"Multi-objective optimization failed: {str(e)}")
        optimal_params = None
    
    print(f"\n" + "="*60)
    print("QUANTUM AGRIPLVOLTAICS SIMULATION COMPLETED")
    print("="*60)
    print("Analysis included:")
    print("  - Quantum dynamics with multiple metrics")
    print("  - Agrivoltaic coupling model")
    print("  - Spectral optimization")
    print("  - Eco-design analysis with sustainability metrics")
    print("  - Quantum advantage quantification")
    print("  - Robustness analysis across parameters")
    print("  - Multi-objective optimization")
    print("  - Data storage in CSV format")
    print("  - Publication-quality figures")
    print("\nAll results saved to the data_output directory.")


def run_complete_analysis_with_params():
    """
    Run the complete quantum agrivoltaics analysis using parameters from JSON file
    and the unified figures class.
    """
    import os
    print("="*60)
    print("QUANTUM AGRIVOLTAICS SIMULATION ANALYSIS WITH PARAMETERS")
    print("="*60)
    
    # Load simulation parameters
    params = load_simulation_params()
    
    # Handle parallelization parameters
    total_cores = os.cpu_count()
    if 'parallelization_params' in params and params['parallelization_params']['nproc'] is not None:
        n_processes = params['parallelization_params']['nproc']
    else:
        # Apply default logic: (nproc-4) if nproc>=8; else (nproc-2)
        if total_cores >= 8:
            n_processes = total_cores - 4
        else:
            n_processes = total_cores - 2
        print(f"Using {n_processes} processes out of {total_cores} available cores (default logic: nproc-4 if nproc>=8; else nproc-2)")
    
    # Set the number of processes for any parallel operations
    print(f"Parallel processing configured with {n_processes} processes")
    
    # Part 1: Quantum Dynamics Simulation
    print("\n" + "="*50)
    print("PART 1: QUANTUM DYNAMICS SIMULATION")
    print("="*50)
    
    # Create FMO Hamiltonian
    fmo_hamiltonian, fmo_energies = create_fmo_hamiltonian()
    print(f"  FMO Hamiltonian created with {fmo_hamiltonian.shape[0]} sites")
    print(f"  Site energies range: {min(fmo_energies):.1f} to {max(fmo_energies):.1f} cm⁻¹")
    
    # Initialize quantum simulator with parameters from JSON
    temp = params['simulation_params']['temperature']
    dephasing = params['simulation_params']['dephasing_rate']
    quantum_sim = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=temp, dephasing_rate=dephasing)
    print(f"  Quantum simulator initialized")
    print(f"    Temperature: {quantum_sim.temperature} K")
    print(f"    Dephasing rate: {quantum_sim.dephasing_rate} cm⁻¹")
    
    # Run quantum dynamics simulation
    max_time = params['simulation_params']['max_time']
    n_time_points = params['simulation_params']['time_points']
    time_points = np.linspace(0, max_time, n_time_points)  # fs
    print(f"  Running quantum dynamics simulation for {max_time} fs with {n_time_points} time points...")
    
    (time_points, density_matrices, populations, coherences, qfi_values, 
     entropy_values, purity_values, linear_entropy_values, 
     bipartite_ent_values, multipartite_ent_values, pairwise_concurrence_values) = quantum_sim.simulate_dynamics(
        time_points=time_points
    )
    print(f"  Quantum simulation completed successfully")
    print(f"    Final population in site 1: {populations[-1, 0]:.4f}")
    print(f"    Final l1-norm coherence: {coherences[-1]:.4f}")
    print(f"    Final QFI: {qfi_values[-1]:.4f}")
    
    # Part 2: Solar Spectrum and Agrivoltaic Coupling
    print("\n" + "="*50)
    print("PART 2: SOLAR SPECTRUM AND AGRIVOLTAIC COUPLING")
    print("="*50)
    
    wavelengths, solar_irradiance = create_solar_spectrum()
    print(f"  Solar spectrum created with {len(wavelengths)} wavelength points")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"  Solar irradiance range: {solar_irradiance.min():.3f} - {solar_irradiance.max():.3f} mW/cm²/nm")
    
    # Initialize agrivoltaic coupling model with parameters from JSON
    opv_bandgap = params['opv_params']['opv_bandgap']
    opv_absorption = params['opv_params']['opv_absorption_coeff']
    agrivoltaic_model = AgrivoltaicCouplingModel(
        fmo_hamiltonian, 
        solar_spectrum=(wavelengths, solar_irradiance),
        opv_bandgap=opv_bandgap,
        opv_absorption_coeff=opv_absorption
    )
    print(f"  Agrivoltaic coupling model initialized")
    print(f"    OPV bandgap: {agrivoltaic_model.opv_bandgap} eV")
    print(f"    OPV absorption coefficient: {agrivoltaic_model.opv_absorption_coeff}")
    
    # Part 3: Spectral Optimization
    print("\n" + "="*50)
    print("PART 3: SPECTRAL OPTIMIZATION")
    print("="*50)
    
    # Initialize spectral optimizer
    spec_opt = SpectralOptimizer(agrivoltaic_model, quantum_sim, n_processes=n_processes)
    print(f"  Spectral optimizer initialized with {spec_opt.n_filters} filters")
    print(f"  Optimization bounds: {len(spec_opt.bounds)} parameters")
    
    # Define initial transmission function (neutral - all light to OPV)
    initial_params = [0.01, 600, 50] * spec_opt.n_filters  # Minimal filtering
    T_initial = agrivoltaic_model.calculate_spectral_transmission(initial_params)
    
    # Calculate initial performance
    etr_initial, _ = calculate_etrs_for_transmission(
        T_initial, wavelengths, solar_irradiance, fmo_hamiltonian, max_time=500, num_time_points=500
    )
    pce_initial = spec_opt.calculate_pce(T_initial)
    spce_initial = 0.5 * pce_initial + 0.5 * etr_initial
    
    print(f"  Initial parameters set")
    print(f"    PCE (initial): {pce_initial:.4f}")
    print(f"    ETR (initial): {etr_initial:.4f}")
    print(f"    SPCE (initial): {spce_initial:.4f}")
    
    # Perform optimization
    try:
        # Use optimization parameters from JSON
        opt_params = params['optimization_params']
        spec_opt.maxiter = opt_params['maxiter']
        spec_opt.popsize = opt_params['popsize']
        spec_opt.strategy = opt_params['strategy']
        spec_opt.mutation = opt_params['mutation']
        spec_opt.recombination = opt_params['recombination']
        
        print(f"  Starting optimization with {spec_opt.maxiter} max iterations...")
        result = spec_opt._run_optimization(initial_params)
        
        if result.success:
            print(f"  Spectral optimization completed successfully")
            opt_results = {
                'success': True,
                'final_params': result.x,
                'final_transmission': agrivoltaic_model.calculate_spectral_transmission(result.x),
                'pce': spec_opt.calculate_pce(agrivoltaic_model.calculate_spectral_transmission(result.x)),
                'etr': calculate_etrs_for_transmission(
                    agrivoltaic_model.calculate_spectral_transmission(result.x),
                    wavelengths, solar_irradiance, fmo_hamiltonian, max_time=500, num_time_points=500
                )[0],
                'spce': 0  # Will calculate below
            }
            opt_results['spce'] = 0.5 * opt_results['pce'] + 0.5 * opt_results['etr']
            print(f"    PCE (optimized): {opt_results['pce']:.4f}")
            print(f"    ETR (optimized): {opt_results['etr']:.4f}")
            print(f"    SPCE (optimized): {opt_results['spce']:.4f}")
        else:
            print(f"  Spectral optimization failed: {result.message}")
            print(f"  Using initial parameters as fallback")
            opt_results = {
                'success': False,
                'final_params': initial_params,
                'final_transmission': T_initial,
                'pce': pce_initial,
                'etr': etr_initial,
                'spce': spce_initial
            }
    except Exception as e:
        print(f"  Spectral optimization encountered an error: {str(e)}, using initial parameters")
        # Calculate performance with initial parameters as fallback
        pce_initial = spec_opt.calculate_pce(T_initial)
        etr_initial = calculate_etrs_for_transmission(T_initial, wavelengths, solar_irradiance, fmo_hamiltonian, max_time=500, num_time_points=500)[0]
        spce_initial = 0.5 * pce_initial + 0.5 * etr_initial
        
        opt_results = {
            'success': False,
            'pce': pce_initial,
            'etr': etr_initial,
            'spce': spce_initial,
            'final_transmission': T_initial,
            'final_params': initial_params
        }
        print(f"    PCE (initial): {opt_results['pce']:.4f}")
        print(f"    ETR (initial): {opt_results['etr']:.4f}")
        print(f"    SPCE (initial): {opt_results['spce']:.4f}")
    
    # Part 4: Eco-Design Analysis
    print("\n" + "="*50)
    print("PART 4: ECO-DESIGN ANALYSIS")
    print("="*50)
    
    # Create eco-design analyzer
    eco_analyzer = EcoDesignAnalyzer()
    
    # Find eco-friendly candidates
    eco_candidates = eco_analyzer.find_eco_friendly_candidates(
        min_biodegradability=0.7, 
        min_pce_potential=0.12
    )
    
    print(f"  Eco-design analysis completed")
    print(f"    Number of eco-friendly candidates: {len(eco_candidates)}")
    if eco_candidates:
        print(f"    Top candidate: {eco_candidates[0]['name']}")
        print(f"      Biodegradability: {eco_candidates[0]['biodegradability']:.3f}")
        print(f"      PCE potential: {eco_candidates[0]['pce_potential']:.3f}")
        print(f"      Multi-objective score: {eco_candidates[0]['multi_objective_score']:.3f}")
    
    # Part 5: Data Storage
    print("\n" + "="*50)
    print("PART 5: DATA STORAGE")
    print("="*50)
    
    # Initialize CSV data storage
    data_storage = CSVDataStorage()
    
    # Save quantum dynamics
    data_storage.save_simulation_data_to_csv(
        time_points, 
        populations, 
        coherences, 
        qfi_values, 
        time_points  # Using time as proxy for ETR time series
    )
    
    # Save spectral optimization
    data_storage.save_optimization_results_to_csv(
        {'initial_params': initial_params}, 
        opt_results, 
        filename='spectral_optimization'
    )
    
    # Save eco-design analysis
    if eco_candidates:
        eco_df = pd.DataFrame(eco_candidates)
        data_storage.save_spectral_data_to_csv(
            wavelengths,
            solar_irradiance,
            [spec_opt.R_opv, spec_opt.R_psu],
            filename='eco_design_analysis'
        )
    
    # Part 6: Figure Generation
    print("\n" + "="*50)
    print("PART 6: FIGURE GENERATION")
    print("="*50)
    
    # Initialize unified figures class
    fig_gen = UnifiedFigures()
    
    # Generate quantum dynamics figure
    quantum_fig = fig_gen.plot_quantum_dynamics(
        time_points, 
        populations, 
        coherences, 
        qfi_values, 
        entropy_values, 
        purity_values
    )
    fig_gen.save_figure(quantum_fig, 'quantum_dynamics_analysis.png')
    
    # Generate spectral analysis figure
    spectral_fig = fig_gen.plot_spectral_analysis(
        wavelengths, 
        solar_irradiance, 
        [T_initial, opt_results['final_transmission']], 
        agrivoltaic_model.R_opv, 
        agrivoltaic_model.R_psu,
        title="Spectral Optimization Results"
    )
    fig_gen.save_figure(spectral_fig, 'spectral_analysis.png')
    
    # Generate eco-design analysis figure if there are candidates
    if eco_candidates:
        eco_df = pd.DataFrame(eco_candidates)
        eco_fig = fig_gen.plot_eco_design_analysis(eco_df, title="Eco-Design Analysis")
        fig_gen.save_figure(eco_fig, 'eco_design_analysis.png')
    
    # Generate optimization results figure
    optimization_fig = fig_gen.plot_optimization_results(
        [initial_params, opt_results['final_params']], 
        [[pce_initial, etr_initial, spce_initial], 
         [opt_results['pce'], opt_results['etr'], opt_results['spce']]], 
        title="Optimization Progress"
    )
    fig_gen.save_figure(optimization_fig, 'optimization_progress.png')
    
    # Generate quantum advantage analysis
    try:
        from quantum_advantage_calculator import analyze_robustness
        robustness_data, temperatures, disorder_strengths = analyze_robustness(
            fmo_hamiltonian, wavelengths, solar_irradiance
        )
        quantum_adv_fig = fig_gen.plot_quantum_advantage_analysis(
            temperatures, robustness_data['temperature_sensitivity'],
            disorder_strengths, robustness_data['disorder_sensitivity'],
            title="Quantum Advantage Analysis"
        )
        fig_gen.save_figure(quantum_adv_fig, 'quantum_advantage_analysis.png')
    except ImportError:
        print("  Quantum advantage calculator not available, skipping quantum advantage analysis")
    
    # Generate a comprehensive summary
    results_dict = {
        'time_points': time_points,
        'populations': populations,
        'coherences': coherences,
        'qfi': qfi_values,
        'entropy': entropy_values,
        'purity': purity_values,
        'wavelengths': wavelengths,
        'solar_irradiance': solar_irradiance,
        'transmission': opt_results['final_transmission'],
        'pce': opt_results['pce'],
        'etr': opt_results['etr'],
        'spce': opt_results['spce'],
        'eco_candidates': eco_candidates
    }
    
    summary_fig = fig_gen.plot_all_results_summary(results_dict, title="Comprehensive Results Summary")
    fig_gen.save_figure(summary_fig, 'comprehensive_summary.png')
    
    print(f"  All figures generated and saved to figures/ directory")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*60)
    print(f"  FMO sites: {fmo_hamiltonian.shape[0]}")
    print(f"  Simulation time: {max_time} fs")
    print(f"  Temperature: {temp} K")
    print(f"  Dephasing rate: {dephasing} cm⁻¹")
    print(f"  PCE (initial → optimized): {pce_initial:.4f} → {opt_results['pce']:.4f}")
    print(f"  ETR (initial → optimized): {etr_initial:.4f} → {opt_results['etr']:.4f}")
    print(f"  SPCE (initial → optimized): {spce_initial:.4f} → {opt_results['spce']:.4f}")
    print(f"  Eco-friendly candidates: {len(eco_candidates)}")
    if eco_candidates:
        print(f"  Best eco-score: {eco_candidates[0]['multi_objective_score']:.4f}")
    print("="*60)
    
    return {
        'quantum_sim': quantum_sim,
        'agrivoltaic_model': agrivoltaic_model,
        'spec_opt': spec_opt,
        'opt_results': opt_results,
        'eco_candidates': eco_candidates,
        'time_points': time_points,
        'populations': populations,
        'coherences': coherences,
        'qfi_values': qfi_values
    }


# Example usage and testing
if __name__ == "__main__":
    # Run the complete analysis
    run_complete_analysis()
