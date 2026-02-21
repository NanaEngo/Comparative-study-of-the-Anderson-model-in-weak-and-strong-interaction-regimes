"""
Quantum Advantage Calculation for Agrivoltaics

This module implements functions for calculating quantum advantage in agrivoltaic systems
as described in the quantum coherence analysis notebook.
"""

import numpy as np
from scipy.integrate import trapezoid


def calculate_quantum_advantage(fmo_hamiltonian, wavelengths, solar_irradiance, transmission_func):
    """
    Calculate quantum advantage by comparing non-Markovian and Markovian approaches.
    
    Mathematical Framework:
    Quantum advantage in photosynthetic energy transfer is quantified by comparing
    the efficiency of quantum coherent (non-Markovian) dynamics to classical
    incoherent (Markovian) dynamics. The quantum advantage factor is defined as:
    
    η_quantum = (ETR_quantum / ETR_classical) - 1
    
    where:
    - ETR_quantum is the Electron Transport Rate calculated with quantum coherence
    - ETR_classical is the ETR calculated in the absence of quantum coherence
    
    The Markovian (classical) limit is approximated by significantly increasing
    the dephasing rates, which destroys quantum coherence effects:
    
    Γ_dephasing >> H_ij / ℏ  (dephasing dominates Hamiltonian coupling)
    
    In this limit, the quantum master equation reduces to a classical rate equation
    without off-diagonal (coherence) terms, representing incoherent energy transfer.
    
    The quantum advantage quantifies how much more efficient energy transfer is
    when quantum coherence is preserved, which is particularly significant for
    understanding the role of quantum effects in natural photosynthetic systems
    and their potential applications in agrivoltaics.
    
    Parameters:
    fmo_hamiltonian: FMO Hamiltonian matrix
    wavelengths (array): Wavelengths in nm
    solar_irradiance (array): Solar irradiance values
    transmission_func (array): OPV transmission function
    
    Returns:
    eta_quantum (float): Quantum advantage as fractional improvement
    etr_quantum (float): ETR from quantum (non-Markovian) calculation
    etr_markovian (float): ETR from Markovian calculation
    """
    # Import needed functions locally to avoid circular imports
    from quantum_dynamics_simulator import QuantumDynamicsSimulator
    from quantum_agrivoltaics_simulations import calculate_etrs_for_transmission
    
    # Non-Markovian (quantum) calculation
    etr_quantum, _ = calculate_etrs_for_transmission(
        transmission_func, wavelengths, solar_irradiance, fmo_hamiltonian
    )
    
    # Markovian approximation: increase dephasing rate significantly
    fmo_markov = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=295, dephasing_rate=100)  # High dephasing
    
    # Calculate ETR with high dephasing (Markovian limit)
    initial_state = np.zeros((fmo_markov.n_sites, fmo_markov.n_sites), dtype=complex)
    initial_state[0, 0] = 1.0  # Start on site 1
    
    time_points = np.linspace(0, 500, 500)
    _, _, pops_markov, _, _, _, _, _, _, _, _ = fmo_markov.simulate_dynamics(
        initial_state=initial_state,
        time_points=time_points
    )
    
    _, _, etr_markovian = fmo_markov.calculate_etr(pops_markov, time_points)
    
    # Calculate quantum advantage
    if etr_markovian != 0:
        eta_quantum = (etr_quantum / etr_markovian) - 1
    else:
        eta_quantum = 0
    
    return eta_quantum, etr_quantum, etr_markovian


def analyze_robustness(fmo_hamiltonian, wavelengths, solar_irradiance, 
                      temperature_range=(273, 320), disorder_strengths=(0, 100)):
    """
    Comprehensive robustness analysis across temperature and disorder
    """
    from quantum_dynamics_simulator import QuantumDynamicsSimulator
    
    results = {'temperature_sensitivity': [], 'disorder_sensitivity': []}
    
    # Temperature sweep
    temperatures = np.linspace(*temperature_range, 10)
    for temp in temperatures:
        # Run quantum dynamics at this temperature
        simulator = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=temp)
        time_points, density_matrices, populations, coherences, qfi_values, _, _, _, _, _, _ = simulator.simulate_dynamics(
            time_points=np.linspace(0, 500, 500)
        )
        _, _, etr_per_photon = simulator.calculate_etr(populations, time_points)
        results['temperature_sensitivity'].append(etr_per_photon)
    
    # Disorder sweep
    disorder_vals = np.linspace(*disorder_strengths, 5)
    for disorder in disorder_vals:
        # Add static disorder to Hamiltonian
        ham_disordered = fmo_hamiltonian + np.diag(np.random.normal(0, disorder, 7))
        simulator = QuantumDynamicsSimulator(ham_disordered, temperature=295)
        time_points, density_matrices, populations, coherences, qfi_values, _, _, _, _, _, _ = simulator.simulate_dynamics(
            time_points=np.linspace(0, 500, 500)
        )
        _, _, etr_per_photon = simulator.calculate_etr(populations, time_points)
        results['disorder_sensitivity'].append(etr_per_photon)
    
    return results, temperatures, disorder_vals


def multi_objective_optimization(wavelengths, solar_irradiance, fmo_hamiltonian, n_processes=None):
    """
    Optimize for both ETR and PCE simultaneously
    """
    from scipy.optimize import differential_evolution
    import multiprocessing as mp
    
    # Use n_processes if provided, otherwise default to 1 to avoid multiprocessing issues
    n_workers = n_processes if n_processes is not None else 1
    
    def objective(params):
        # Extract parameters
        center_wls = params[:4]
        widths = params[4:8]
        peak_transmissions = params[8:12]
        base_transmission = params[12]
        
        # Create transmission function
        transmission_params = {
            'center_wls': center_wls,
            'widths': widths,
            'peak_transmissions': peak_transmissions,
            'base_transmission': base_transmission
        }
        from .agrivoltaic_coupling_model import AgrivoltaicCouplingModel
        model = AgrivoltaicCouplingModel(fmo_hamiltonian, solar_spectrum=(wavelengths, solar_irradiance))
        
        def create_transmission_func(wl):
            T = np.full_like(wl, transmission_params['base_transmission'])
            
            for center_wl, width, peak_trans in zip(
                transmission_params['center_wls'], 
                transmission_params['widths'], 
                transmission_params['peak_transmissions']):
                
                sigma = width / 2.355
                gaussian = peak_trans * np.exp(-((wl - center_wl)**2) / (2 * sigma**2))
                T = np.maximum(T, gaussian)
            
            return np.clip(T, 0, 1)
        
        transmission_func = create_transmission_func(wavelengths)
        
        # Calculate ETR (photosynthetic efficiency)
        from .quantum_agrivoltaics_simulations import calculate_etrs_for_transmission
        etr_per_photon, _ = calculate_etrs_for_transmission(
            transmission_func, wavelengths, solar_irradiance, fmo_hamiltonian
        )
        
        # Calculate PCE (photovoltaic efficiency) - simplified model
        # In practice, this would involve more complex material physics
        pce = 0.15 + 0.05 * np.mean(transmission_func)  # Simplified model
        
        # Combined objective: maximize ETR while maintaining PCE > 15%
        if pce < 0.15:
            return -etr_per_photon * 0.5  # Penalty for low PCE
        else:
            return -etr_per_photon * pce  # Reward for both high ETR and PCE
    
    # Define parameter bounds
    bounds = [(400, 500), (400, 500), (400, 500), (400, 500),  # center_wls
              (30, 100), (30, 100), (30, 100), (30, 100),      # widths
              (0.3, 1.0), (0.3, 1.0), (0.3, 1.0), (0.3, 1.0), # peak_transmissions
              (0.05, 0.3)]                                      # base_transmission
    
    # Run differential evolution with parallel workers
    result = differential_evolution(
        objective, bounds, workers=n_workers,
        maxiter=10, popsize=6
    )
    
    return result.x, -result.fun