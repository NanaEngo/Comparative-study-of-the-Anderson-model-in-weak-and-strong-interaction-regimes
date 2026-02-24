"""
CSV Data Storage Module for Quantum Agrivoltaics Simulations.

This module provides tools for storing quantum simulation results to CSV files
with comprehensive metadata and proper formatting.
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class CSVDataStorage:
    """
    Class for storing simulation results to CSV files with metadata.
    """
    
    def __init__(self, output_dir: str = '../simulation_data/'):
        """
        Initialize CSV data storage.
        
        Parameters:
        -----------
        output_dir : str
            Directory to store CSV files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"CSV data storage initialized at {output_dir}")
    
    def save_quantum_dynamics_results(
        self,
        time_points: np.ndarray,
        populations: np.ndarray,
        coherences: np.ndarray,
        quantum_metrics: Dict[str, np.ndarray],
        filename_prefix: str = 'quantum_dynamics',
        **metadata
    ) -> str:
        """
        Save quantum dynamics simulation results to CSV.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time points for the simulation
        populations : np.ndarray
            Site populations over time
        coherences : np.ndarray
            Coherence measures over time
        quantum_metrics : dict
            Dictionary of quantum metrics (QFI, entropy, etc.)
        filename_prefix : str
            Prefix for the output filename
        **metadata : dict
            Additional metadata to include
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.output_dir, 
            f"{filename_prefix}_{timestamp}.csv"
        )
        
        # Create DataFrame with time series data
        data_dict = {'time_fs': time_points}
        
        # Add populations for each site
        n_sites = populations.shape[1] if len(populations.shape) > 1 else 1
        if n_sites > 1:
            for i in range(n_sites):
                data_dict[f'population_site_{i+1}'] = populations[:, i]
        else:
            data_dict['populations'] = populations
        
        # Add coherences
        data_dict['coherences'] = coherences
        
        # Add quantum metrics
        for metric_name, metric_values in quantum_metrics.items():
            if isinstance(metric_values, np.ndarray) and len(metric_values) == len(time_points):
                data_dict[metric_name] = metric_values
            else:
                # If single value, repeat for all time points
                data_dict[metric_name] = [metric_values] * len(time_points)
        
        df = pd.DataFrame(data_dict)
        
        # Add metadata as additional rows at the end
        metadata_row = {
            'time_fs': 'METADATA',
            'population_site_1': str(metadata) if metadata else 'None'
        }
        
        # Add a metadata row
        metadata_df = pd.DataFrame([metadata_row])
        df = pd.concat([df, metadata_df], ignore_index=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Quantum dynamics data saved to {filepath}")
        return filepath
    
    def save_spectral_optimization(
        self,
        optimization_results: Dict[str, Any],
        filename_prefix: str = 'spectral_optimization',
        **metadata
    ) -> str:
        """
        Save spectral optimization results to CSV.
        
        Parameters:
        -----------
        optimization_results : dict
            Results from spectral optimization
        filename_prefix : str
            Prefix for the output filename
        **metadata : dict
            Additional metadata to include
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.output_dir, 
            f"{filename_prefix}_{timestamp}.csv"
        )
        
        # Flatten the optimization results into a DataFrame
        data_dict = {}
        
        # Add parameters
        if 'optimal_params' in optimization_results:
            for i, param in enumerate(optimization_results['optimal_params']):
                data_dict[f'param_{i+1}'] = [param]
        
        # Add objectives
        if 'optimal_pce' in optimization_results:
            data_dict['optimal_pce'] = [optimization_results['optimal_pce']]
        if 'optimal_etr' in optimization_results:
            data_dict['optimal_etr'] = [optimization_results['optimal_etr']]
        
        # Add other metrics
        for key, value in optimization_results.items():
            if key not in ['optimal_params', 'optimal_pce', 'optimal_etr', 'transmission_func', 'success', 'message']:
                if not isinstance(value, (dict, list, np.ndarray)):
                    data_dict[key] = [value]
        
        # Add metadata
        for key, value in metadata.items():
            data_dict[f'meta_{key}'] = [str(value)]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Spectral optimization data saved to {filepath}")
        return filepath
    
    def save_agrivoltaic_results(
        self,
        pce: float,
        etr: float,
        spectral_data: Dict[str, np.ndarray],
        filename_prefix: str = 'agrivoltaic_results',
        **metadata
    ) -> str:
        """
        Save agrivoltaic coupling results to CSV.
        
        Parameters:
        -----------
        pce : float
            Power conversion efficiency
        etr : float
            Electron transport rate
        spectral_data : dict
            Spectral data (wavelengths, transmission, etc.)
        filename_prefix : str
            Prefix for the output filename
        **metadata : dict
            Additional metadata to include
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.output_dir, 
            f"{filename_prefix}_{timestamp}.csv"
        )
        
        # Determine the number of data points from the spectral data
        n_points = 0
        for key, value in spectral_data.items():
            if isinstance(value, np.ndarray):
                n_points = max(n_points, len(value))
        
        # Create DataFrame
        data_dict = {'pce': [pce], 'etr': [etr]}
        
        # Add spectral data as columns
        for key, value in spectral_data.items():
            if isinstance(value, np.ndarray):
                if len(value) == n_points:
                    data_dict[key] = value
                else:
                    # If it's a single value, repeat for all rows
                    data_dict[key] = [value[0]] * n_points
            else:
                data_dict[key] = [value] * n_points
        
        df = pd.DataFrame(data_dict)
        
        # Add metadata as additional rows at the end
        if metadata:
            metadata_row = {'pce': 'METADATA', 'etr': str(metadata)}
            metadata_df = pd.DataFrame([metadata_row])
            df = pd.concat([df, metadata_df], ignore_index=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Agrivoltaic results saved to {filepath}")
        return filepath
    
    def save_biodegradability_analysis(
        self,
        analysis_results: Dict[str, Any],
        filename_prefix: str = 'biodegradability_analysis',
        **metadata
    ) -> str:
        """
        Save biodegradability analysis results to CSV.
        
        Parameters:
        -----------
        analysis_results : dict
            Results from biodegradability analysis
        filename_prefix : str
            Prefix for the output filename
        **metadata : dict
            Additional metadata to include
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.output_dir, 
            f"{filename_prefix}_{timestamp}.csv"
        )
        
        # Create DataFrame from analysis results
        data_dict = {}
        
        # Add all scalar values from analysis results
        for key, value in analysis_results.items():
            if isinstance(value, (int, float, str, bool)):
                data_dict[key] = [value]
            elif isinstance(value, (list, np.ndarray)) and np.isscalar(value[0]):
                data_dict[key] = [str(value)]
            else:
                data_dict[key] = [str(value)]
        
        # Add metadata
        for key, value in metadata.items():
            data_dict[f'meta_{key}'] = [str(value)]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Biodegradability analysis saved to {filepath}")
        return filepath

    def save_lca_results(
        self,
        lca_results: Dict[str, Any],
        filename_prefix: str = 'lca_analysis',
        **metadata
    ) -> str:
        """
        Save Life Cycle Assessment results to CSV.
        
        Parameters:
        -----------
        lca_results : dict
            Results from LCA analysis
        filename_prefix : str
            Prefix for the output filename
        **metadata : dict
            Additional metadata to include
        
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.output_dir, 
            f"{filename_prefix}_{timestamp}.csv"
        )
        
        # Flatten nested LCA results dictionary
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_results = flatten_dict(lca_results)
        
        # Create DataFrame
        data_dict = {}
        for key, value in flat_results.items():
            if isinstance(value, (int, float, str, bool)):
                data_dict[key] = [value]
            else:
                data_dict[key] = [str(value)]
        
        # Add metadata
        for key, value in metadata.items():
            data_dict[f'meta_{key}'] = [str(value)]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        
        logger.info(f"LCA results saved to {filepath}")
        return filepath