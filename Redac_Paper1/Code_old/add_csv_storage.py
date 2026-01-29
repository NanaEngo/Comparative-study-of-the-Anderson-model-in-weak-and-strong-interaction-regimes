#!/usr/bin/env python3
"""
Script to add CSV storage functionality to the quantum coherence agrivoltaics notebook.
This script adds functions to save simulation data to CSV files and save figures to the figures/ folder.
"""

import os
import sys
import json
import re

def add_csv_functions_to_notebook():
    """
    Add CSV storage functionality to the notebook.
    """
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    
    # CSV storage functions as a string
    csv_functions_code = '''# CSV Data Storage and Figure Saving Functions
# These functions save simulation results to CSV files and figures to the figures/ folder

import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

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

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)
print("CSV storage functions are ready and figures/ directory has been created.")
print("You can now save simulation data using the provided functions.")
'''

    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find the last code cell before the conclusion
    # Look for the last occurrence of a code cell before the conclusion
    last_code_cell_pattern = r'(\s*{\s*"cell_type": "code",.*?"source": \[.*?^\s*\],\s*"outputs":.*?)(\s*},\s*{\s*"cell_type": "markdown",\s*"metadata":.*?"## 9\. Summary and conclusions")'
    
    # Replace the pattern to insert our new cell
    def insert_csv_cell(match):
        existing_cell = match.group(1)
        conclusion = match.group(2)
        new_cell = f'''{existing_cell},
{{
 "cell_type": "code",
 "execution_count": null,
 "metadata": {{}},
 "outputs": [],
 "source": [
'''
        # Format the function code with proper escaping for JSON
        lines = csv_functions_code.split('\n')
        for i, line in enumerate(lines):
            escaped_line = line.replace('\\', '\\\\').replace('"', '\\"')
            new_cell += f'  "{escaped_line}\\\\n"'
            if i < len(lines) - 1:
                new_cell += ',\n'
        
        new_cell += f'''
 ]
}},
{{
 "cell_type": "markdown",
 "metadata": {{}},
 "source": [
'''
        
        return new_cell + conclusion
    
    # Try to find and replace the last code cell
    updated_content = re.sub(last_code_cell_pattern, insert_csv_cell, content, flags=re.DOTALL|re.MULTILINE)
    
    # If the pattern didn't match, we'll append the cell differently
    if updated_content == content:
        # Find the last occurrence of a code cell
        code_cell_pattern = r'(\s*{\s*"cell_type": "code",.*?"source": \[.*?^\s*\],\s*"outputs":.*?}\s*]),\s*{\s*"cell_type": "markdown",\s*"metadata":'
        if re.search(code_cell_pattern, content, re.DOTALL):
            # Insert our cell before the markdown conclusion
            updated_content = re.sub(
                code_cell_pattern, 
                lambda m: m.group(1) + f""",
{{
 "cell_type": "code",
 "execution_count": null,
 "metadata": {{}},
 "outputs": [],
 "source": [
""" + '\\\\n",\\\n'.join([f'  "{line.replace(chr(92), chr(92)+chr(92)).replace(chr(34), chr(92)+chr(34))}\\\\n"' for line in csv_functions_code.split('\n')]) + '''
 ]
}},
{
 "cell_type": "markdown",
 "metadata": {},
 "source": ['''
                , content, 1, re.DOTALL|re.MULTILINE)
    
    # If still no match, just append to the cells array before the metadata
    if updated_content == content:
        # Find the end of the cells array
        cells_end_pattern = r'(\s*\]\s*,\s*"metadata")'
        updated_content = re.sub(
            cells_end_pattern,
            lambda m: f""",
{{
 "cell_type": "code",
 "execution_count": null,
 "metadata": {{}},
 "outputs": [],
 "source": [
""" + '\\\\n",\\\n'.join([f'  "{line.replace(chr(92), chr(92)+chr(92)).replace(chr(34), chr(92)+chr(34))}\\\\n"' for line in csv_functions_code.split('\n')]) + '''
 ]
}]''' + m.group(1)
            , content, 1, re.DOTALL)
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("CSV storage functions have been added to the notebook.")
    print("The functions will save data to the 'figures/' folder as requested.")


if __name__ == "__main__":
    add_csv_functions_to_notebook()
