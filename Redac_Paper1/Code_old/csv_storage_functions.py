import numpy as np
import pandas as pd
import os
from datetime import datetime

def save_simulation_data_to_csv(time_points, populations, coherences, qfi_values, etr_values, filename_prefix="simulation"):
    """
    Save quantum dynamics simulation data to CSV files.
    
    Parameters:
    time_points: array of time points
    populations: 2D array of site populations over time
    coherences: array of coherence values over time
    qfi_values: array of Quantum Fisher Information values over time
    etr_values: array of ETR values over time
    filename_prefix: prefix for output files
    """
    # Create directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Create a comprehensive DataFrame for all data
    data = {
        'time_fs': time_points
    }
    
    # Add populations for each site
    n_sites = populations.shape[1]
    for i in range(n_sites):
        data[f'population_site_{i+1}'] = populations[:, i]
    
    # Add other metrics
    data['coherence_l1_norm'] = coherences
    data['qfi'] = qfi_values
    data['etr'] = etr_values
    
    df = pd.DataFrame(data)
    df.to_csv(f"figures/{filename_prefix}_dynamics.csv", index=False)
    
    print(f"Saved simulation dynamics to figures/{filename_prefix}_dynamics.csv")
    
    # Also save summary statistics
    summary_data = {
        'metric': ['final_population_site_1', 'final_population_site_2', 'final_population_site_3', 
                   'final_coherence', 'final_qfi', 'average_etr', 'max_qfi', 'min_qfi'],
        'value': [populations[-1, 0] if n_sites > 0 else 0, 
                  populations[-1, 1] if n_sites > 1 else 0, 
                  populations[-1, 2] if n_sites > 2 else 0,
                  coherences[-1] if len(coherences) > 0 else 0,
                  qfi_values[-1] if len(qfi_values) > 0 else 0,
                  np.mean(etr_values) if len(etr_values) > 0 else 0,
                  np.max(qfi_values) if len(qfi_values) > 0 else 0,
                  np.min(qfi_values) if len(qfi_values) > 0 else 0]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"figures/{filename_prefix}_summary.csv", index=False)
    
    print(f"Saved simulation summary to figures/{filename_prefix}_summary.csv")
    
    return df, summary_df


def save_optimization_results_to_csv(optimization_params, optimization_results, filename="optimization_results"):
    """
    Save optimization results to CSV file.
    
    Parameters:
    optimization_params: Dictionary of optimization parameters
    optimization_results: Results from optimization
    filename: name for output file
    """
    os.makedirs("figures", exist_ok=True)
    
    # Flatten the parameters dictionary
    data = {}
    for key, value in optimization_params.items():
        if isinstance(value, (list, np.ndarray)):
            # If the value is a list/array, create multiple columns
            for i, v in enumerate(value):
                data[f"{key}_{i}"] = [v]
        else:
            data[key] = [value]
    
    # Add results
    if hasattr(optimization_results, '__iter__') and not isinstance(optimization_results, str):
        for i, result in enumerate(optimization_results):
            data[f"result_{i}"] = [result]
    else:
        data["result"] = [optimization_results]
    
    df = pd.DataFrame(data)
    df.to_csv(f"figures/{filename}.csv", index=False)
    
    print(f"Saved optimization results to figures/{filename}.csv")
    
    return df


def save_robustness_analysis_to_csv(robustness_data, temperatures, disorder_strengths, filename="robustness_analysis"):
    """
    Save robustness analysis results to CSV file.
    
    Parameters:
    robustness_data: dictionary with robustness data
    temperatures: array of temperatures tested
    disorder_strengths: array of disorder strengths tested
    filename: name for output file
    """
    os.makedirs("figures", exist_ok=True)
    
    # Create dataframes for temperature and disorder sensitivity
    temp_data = {
        'temperature_K': temperatures,
        'etr_per_photon': robustness_data.get('temperature_sensitivity', [])
    }
    
    disorder_data = {
        'disorder_strength_cm-1': disorder_strengths,
        'etr_per_photon': robustness_data.get('disorder_sensitivity', [])
    }
    
    temp_df = pd.DataFrame(temp_data)
    disorder_df = pd.DataFrame(disorder_data)
    
    temp_df.to_csv(f"figures/{filename}_temperature_sensitivity.csv", index=False)
    disorder_df.to_csv(f"figures/{filename}_disorder_sensitivity.csv", index=False)
    
    print(f"Saved robustness analysis to figures/{filename}_temperature_sensitivity.csv and figures/{filename}_disorder_sensitivity.csv")
    
    return temp_df, disorder_df


def save_spectral_data_to_csv(wavelengths, solar_irradiance, transmission_funcs, filename="spectral_data"):
    """
    Save spectral data to CSV file.
    
    Parameters:
    wavelengths: array of wavelengths
    solar_irradiance: array of solar irradiance values
    transmission_funcs: list of transmission functions
    filename: name for output file
    """
    os.makedirs("figures", exist_ok=True)
    
    data = {'wavelength_nm': wavelengths, 'solar_irradiance': solar_irradiance}
    
    # Add transmission functions if they exist
    if transmission_funcs is not None:
        for i, transmission in enumerate(transmission_funcs):
            data[f'transmission_func_{i}'] = transmission
    
    df = pd.DataFrame(data)
    df.to_csv(f"figures/{filename}.csv", index=False)
    
    print(f"Saved spectral data to figures/{filename}.csv")
    
    return df


def save_all_figures_to_folder(figures_dict, folder="figures"):
    """
    Save all matplotlib figures to specified folder.
    
    Parameters:
    figures_dict: Dictionary of matplotlib figure objects
    folder: folder to save figures to
    """
    os.makedirs(folder, exist_ok=True)
    
    if figures_dict:
        for fig_name, fig in figures_dict.items():
            if fig_name.startswith('fig'):
                # Handle different figure types
                fig.savefig(f"{folder}/{fig_name}.png", dpi=300, bbox_inches='tight')
                fig.savefig(f"{folder}/{fig_name}.pdf", bbox_inches='tight')  # Also save as PDF
                print(f"Saved figure {fig_name} to {folder}/{fig_name}.png")
            else:
                # For other figure names, ensure they are properly named
                clean_name = "".join(c for c in fig_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                fig.savefig(f"{folder}/{clean_name}.png", dpi=300, bbox_inches='tight')
                fig.savefig(f"{folder}/{clean_name}.pdf", bbox_inches='tight')
                print(f"Saved figure to {folder}/{clean_name}.png")
    else:
        print("No figures to save or figures_dict is None")