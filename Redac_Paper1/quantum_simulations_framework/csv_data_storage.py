"""
CSV Data Storage for Quantum Agrivoltaics Simulations

This module implements data storage functionality for quantum agrivoltaics simulations,
saving results in CSV format for analysis and sharing.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime


class CSVDataStorage:
    """
    CSV data storage for quantum agrivoltaics simulation results.
    
    This class provides methods for saving simulation data to CSV files
    with proper metadata and formatting for scientific analysis.
    """
    
    def __init__(self, output_dir='data_output'):
        """
        Initialize the data storage system.
        
        Mathematical Framework:
        The data storage system organizes simulation results in a structured
        format that preserves the mathematical relationships and metadata
        of the quantum agrivoltaics simulations. Data is stored in CSV format
        for compatibility with analysis tools while maintaining precision
        and provenance information.
        
        Parameters:
        output_dir (str): Directory to save CSV files (default: 'data_output')
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def save_simulation_data_to_csv(self, time_points, populations, coherences, 
                                  qfi_values, etr_time, filename_prefix='quantum_dynamics'):
        """
        Save quantum dynamics simulation data to CSV.
        
        Mathematical Framework:
        The simulation data represents the time evolution of quantum states:
        
        ρ(t) = |ψ(t)⟩⟨ψ(t)| or ρ(t) for mixed states
        
        Populations: ρ_ii(t) = |⟨i|ψ(t)⟩|² (diagonal elements)
        Coherences: ρ_ij(t) = ⟨i|ψ(t)⟩⟨ψ(t)|j⟩ (off-diagonal elements)
        QFI: F_Q(t) (Quantum Fisher Information as function of time)
        
        Parameters:
        time_points (array): Time points in fs
        populations (2D array): Site populations over time
        coherences (3D array): Coherence matrices over time [time, i, j]
        qfi_values (array): QFI values over time
        etr_time (array): ETR values over time
        filename_prefix (str): Prefix for output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save populations
        pop_df = pd.DataFrame(populations)
        pop_df.columns = [f'Site_{i}_Population' for i in range(populations.shape[1])]
        pop_df.insert(0, 'Time_fs', time_points)
        
        pop_filename = os.path.join(self.output_dir, f'{filename_prefix}_populations_{timestamp}.csv')
        pop_df.to_csv(pop_filename, index=False, float_format='%.8e')
        
        # Save coherences (real and imaginary parts)
        n_sites = coherences.shape[1]
        
        # Create a DataFrame for real parts of coherences
        real_coherences = np.real(coherences)
        real_df = pd.DataFrame()
        real_df['Time_fs'] = time_points
        
        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:  # Only save off-diagonal elements
                    real_df[f'Coherence_Re_{i}_{j}'] = real_coherences[:, i, j]
        
        real_filename = os.path.join(self.output_dir, f'{filename_prefix}_coherences_real_{timestamp}.csv')
        real_df.to_csv(real_filename, index=False, float_format='%.8e')
        
        # Create a DataFrame for imaginary parts of coherences
        imag_coherences = np.imag(coherences)
        imag_df = pd.DataFrame()
        imag_df['Time_fs'] = time_points
        
        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:  # Only save off-diagonal elements
                    imag_df[f'Coherence_Im_{i}_{j}'] = imag_coherences[:, i, j]
        
        imag_filename = os.path.join(self.output_dir, f'{filename_prefix}_coherences_imag_{timestamp}.csv')
        imag_df.to_csv(imag_filename, index=False, float_format='%.8e')
        
        # Save QFI and ETR data
        metrics_df = pd.DataFrame({
            'Time_fs': time_points,
            'QFI': qfi_values,
            'ETR': etr_time
        })
        
        metrics_filename = os.path.join(self.output_dir, f'{filename_prefix}_quantum_metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_filename, index=False, float_format='%.8e')
        
        print(f"  Saved simulation data to:")
        print(f"    - Populations: {pop_filename}")
        print(f"    - Coherences (real): {real_filename}")
        print(f"    - Coherences (imag): {imag_filename}")
        print(f"    - Quantum metrics: {metrics_filename}")
    
    def save_optimization_results_to_csv(self, initial_params, opt_results, filename='optimization_results'):
        """
        Save spectral optimization results to CSV.
        
        Mathematical Framework:
        Optimization results include:
        
        - Optimal parameters: [A₁, λ₁, σ₁, A₂, λ₂, σ₂, ...] for filters
        - Performance metrics: PCE, ETR, SPCE
        - Objective function value
        - Convergence information
        
        The optimization problem is:
        max_T w₁*PCE(T) + w₂*ETR(T) + w₃*SPCE(T)
        
        Parameters:
        initial_params (dict): Initial parameters for optimization
        opt_results (dict): Optimization results
        filename (str): Base filename for output
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrame with optimization results
        data = {
            'timestamp': [timestamp],
            'success': [opt_results.get('success', False)],
            'message': [opt_results.get('message', 'No message')],
            'pce': [opt_results.get('pce', 0.0)],
            'etr': [opt_results.get('etr', 0.0)],
            'spce': [opt_results.get('spce', 0.0)],
            'optimal_performance': [opt_results.get('optimal_performance', 0.0)]
        }
        
        # Add initial parameters if they exist
        if initial_params and 'initial_params' in initial_params:
            for i, param in enumerate(initial_params['initial_params']):
                data[f'initial_param_{i}'] = [param]
        
        # Add optimal parameters if they exist
        if opt_results.get('optimal_params') is not None:
            for i, param in enumerate(opt_results['optimal_params']):
                data[f'optimal_param_{i}'] = [param]
        
        df = pd.DataFrame(data)
        
        filename_full = os.path.join(self.output_dir, f'{filename}_{timestamp}.csv')
        df.to_csv(filename_full, index=False, float_format='%.8e')
        
        print(f"  Saved optimization results to: {filename_full}")
    
    def save_spectral_data_to_csv(self, wavelengths, solar_spectrum, responses, filename='spectral_data'):
        """
        Save spectral data to CSV.
        
        Mathematical Framework:
        Spectral data includes:
        
        - Wavelength array (nm)
        - Solar spectrum: I_sun(λ) (mW/cm²/nm)
        - System responses: R_system(λ) (dimensionless, 0-1)
        
        For agrivoltaic systems:
        I_OPV(λ) = I_sun(λ) * T(λ)      (transmitted to OPV)
        I_PSU(λ) = I_sun(λ) * [1-T(λ)]  (absorbed by PSU)
        
        Parameters:
        wavelengths (array): Wavelength array in nm
        solar_spectrum (array): Solar spectrum values
        responses (list): List of response functions [R_OPV, R_PSU, ...]
        filename (str): Base filename for output
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Wavelength_nm': wavelengths,
            'Solar_Spectrum_mW_per_cm2_per_nm': solar_spectrum
        })
        
        # Add response functions
        for i, response in enumerate(responses):
            df[f'Response_{i}'] = response
        
        filename_full = os.path.join(self.output_dir, f'{filename}_{timestamp}.csv')
        df.to_csv(filename_full, index=False, float_format='%.8e')
        
        print(f"  Saved spectral data to: {filename_full}")
    
    def save_eco_analysis_to_csv(self, eco_results, filename='eco_analysis_results'):
        """
        Save eco-design analysis results to CSV.
        
        Mathematical Framework:
        Eco-design results include sustainability metrics:
        
        - Biodegradability: Based on Fukui functions
        - PCE potential: Power conversion efficiency potential
        - Toxicity: Potential for harmful environmental impact
        - Resource efficiency: Output per unit input
        - Multi-objective score: Weighted combination
        
        Sustainability = w₁*biodegradability + w₂*PCE_potential + w₃*(1-toxicity) + w₄*resource_efficiency
        
        Parameters:
        eco_results (list or dict): Eco-design analysis results
        filename (str): Base filename for output
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(eco_results, list) and len(eco_results) > 0:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(eco_results)
        elif isinstance(eco_results, dict):
            # Convert single dictionary to DataFrame
            df = pd.DataFrame([eco_results])
        else:
            print(f"  Warning: No eco-analysis data to save for {filename}")
            return
        
        # Add timestamp column
        df['timestamp'] = timestamp
        
        filename_full = os.path.join(self.output_dir, f'{filename}_{timestamp}.csv')
        df.to_csv(filename_full, index=False, float_format='%.8e')
        
        print(f"  Saved eco-analysis results to: {filename_full}")
    
    def save_robustness_analysis_to_csv(self, robustness_data, filename='robustness_analysis'):
        """
        Save robustness analysis results to CSV.
        
        Mathematical Framework:
        Robustness analysis quantifies sensitivity to parameters:
        
        S_T = ∂(performance_metric) / ∂(temperature)  (temperature sensitivity)
        S_D = ∂(performance_metric) / ∂(disorder)     (disorder sensitivity)
        
        Robustness is inversely related to sensitivity: R = 1/S
        
        Parameters:
        robustness_data (dict): Robustness analysis results
        filename (str): Base filename for output
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrames for each sensitivity analysis
        if 'temperature_sensitivity' in robustness_data and 'temperatures' in robustness_data:
            temp_df = pd.DataFrame({
                'Temperature_K': robustness_data['temperatures'],
                'Sensitivity': robustness_data['temperature_sensitivity']
            })
            temp_filename = os.path.join(self.output_dir, f'{filename}_temperature_{timestamp}.csv')
            temp_df.to_csv(temp_filename, index=False, float_format='%.8e')
            print(f"  Saved temperature robustness data to: {temp_filename}")
        
        if 'disorder_sensitivity' in robustness_data and 'disorder_strengths' in robustness_data:
            disorder_df = pd.DataFrame({
                'Disorder_Strength_cm-1': robustness_data['disorder_strengths'],
                'Sensitivity': robustness_data['disorder_sensitivity']
            })
            disorder_filename = os.path.join(self.output_dir, f'{filename}_disorder_{timestamp}.csv')
            disorder_df.to_csv(disorder_filename, index=False, float_format='%.8e')
            print(f"  Saved disorder robustness data to: {disorder_filename}")
    
    def save_quantum_metrics_to_csv(self, time_points, entropy_values, purity_values,
                                  linear_entropy_values, bipartite_ent_values,
                                  multipartite_ent_values, pairwise_concurrence_values,
                                  filename='quantum_metrics_extended'):
        """
        Save extended quantum metrics to CSV.
        
        Mathematical Framework:
        Extended quantum metrics include:
        
        - Von Neumann entropy: S = -Tr[ρ log ρ]
        - Purity: P = Tr[ρ²]
        - Linear entropy: S_L = (d/(d-1))(1-Tr[ρ²])
        - Bipartite entanglement: E_A = S(ρ_A)
        - Multipartite entanglement: Average of bipartite measures
        - Pairwise concurrence: C_ij for each pair
        
        Parameters:
        time_points (array): Time points in fs
        entropy_values (array): Von Neumann entropy over time
        purity_values (array): Purity over time
        linear_entropy_values (array): Linear entropy over time
        bipartite_ent_values (array): Bipartite entanglement over time
        multipartite_ent_values (array): Multipartite entanglement over time
        pairwise_concurrence_values (array): Pairwise concurrence over time
        filename (str): Base filename for output
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        df = pd.DataFrame({
            'Time_fs': time_points,
            'VonNeumann_Entropy': entropy_values,
            'Purity': purity_values,
            'Linear_Entropy': linear_entropy_values,
            'Bipartite_Entanglement': bipartite_ent_values,
            'Multipartite_Entanglement': multipartite_ent_values,
            'Pairwise_Concurrence': pairwise_concurrence_values
        })
        
        filename_full = os.path.join(self.output_dir, f'{filename}_{timestamp}.csv')
        df.to_csv(filename_full, index=False, float_format='%.8e')
        
        print(f"  Saved extended quantum metrics to: {filename_full}")
    
    def read_simulation_data_from_csv(self, filename):
        """
        Read simulation data from CSV file.
        
        Parameters:
        filename (str): Name of the CSV file to read
        
        Returns:
        data (DataFrame): Data from the CSV file
        """
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"  File not found: {filepath}")
            return None
    
    def create_metadata_file(self, simulation_params, filename='simulation_metadata'):
        """
        Create a metadata file for the simulation.
        
        Parameters:
        simulation_params (dict): Dictionary of simulation parameters
        filename (str): Base filename for metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a simple text file with metadata
        metadata_filepath = os.path.join(self.output_dir, f'{filename}_{timestamp}.txt')
        
        with open(metadata_filepath, 'w') as f:
            f.write(f"Simulation Metadata\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {self.output_dir}\n")
            f.write(f"Parameters:\n")
            for key, value in simulation_params.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"  Created metadata file: {metadata_filepath}")
    def save_environmental_data_to_csv(self, time_days, temperatures, humidity_values, 
                                     wind_speeds, pce_env, etr_env, dust_profile, 
                                     filename_prefix='environmental_effects'):
        """
        Save environmental effects data to CSV.
        
        Parameters:
        time_days (array): Time points in days
        temperatures (array): Temperature values in K
        humidity_values (array): Humidity values (0-1)
        wind_speeds (array): Wind speeds in m/s
        pce_env (array): PCE with environmental effects
        etr_env (array): ETR with environmental effects
        dust_profile (array): Dust thickness over time
        filename_prefix (str): Prefix for output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        env_df = pd.DataFrame({
            'Time_days': time_days,
            'Temperature_K': temperatures,
            'Humidity': humidity_values,
            'Wind_Speed_ms': wind_speeds,
            'Dust_Thickness': dust_profile,
            'PCE_Combined': pce_env,
            'ETR_Combined': etr_env
        })
        
        filename = os.path.join(self.output_dir, f'{filename_prefix}_{timestamp}.csv')
        env_df.to_csv(filename, index=False, float_format='%.6f')
        print(f"  Saved environmental data to: {filename}")
