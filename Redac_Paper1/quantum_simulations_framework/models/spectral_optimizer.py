"""
Spectral Optimizer for Quantum Agrivoltaic Systems.

This module implements optimization algorithms for spectral splitting
in agrivoltaic systems to maximize combined OPV and PSU performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import trapezoid
import logging
from typing import Dict, Any, Tuple, Callable, Optional
from datetime import datetime
import os


logger = logging.getLogger(__name__)


class SpectralOptimizer:
    """
    Spectral optimizer for agrivoltaic systems.
    
    Mathematical Framework:
    The optimization problem seeks to maximize a weighted objective function:
    
    max_{T(λ)} [w₁ * PCE(T) + w₂ * ETR(T)]
    
    subject to 0 ≤ T(λ) ≤ 1 for all wavelengths λ
    
    where PCE(T) is the power conversion efficiency of the OPV system,
    ETR(T) is the electron transport rate of the PSU system,
    and w₁, w₂ are weighting factors.
    """
    
    def __init__(self, 
                 solar_spectrum: Tuple[np.ndarray, np.ndarray],
                 opv_response: np.ndarray,
                 psu_response: np.ndarray,
                 weights: Tuple[float, float] = (0.5, 0.5)):
        """
        Initialize spectral optimizer.
        
        Parameters:
        -----------
        solar_spectrum : tuple
            Solar spectrum (wavelengths, irradiances)
        opv_response : np.ndarray
            OPV spectral response function
        psu_response : np.ndarray
            PSU spectral response function
        weights : tuple
            Weights for PCE and ETR in objective function (w_pce, w_etr)
        """
        self.lambda_range, self.solar_spec = solar_spectrum
        self.opv_response = opv_response
        self.psu_response = psu_response
        self.w_pce, self.w_etr = weights
        
        # Validate inputs
        assert len(self.solar_spec) == len(self.opv_response) == len(self.psu_response), \
            "All arrays must have the same length"
        
        logger.info(f"SpectralOptimizer initialized with {len(self.lambda_range)} wavelength points")
    
    def calculate_opv_efficiency(self, transmission: np.ndarray) -> float:
        """
        Calculate OPV efficiency for given transmission spectrum.
        
        Parameters:
        -----------
        transmission : np.ndarray
            Spectral transmission values
            
        Returns:
        --------
        float
            Power conversion efficiency
        """
        # Calculate absorbed irradiance for OPV
        absorbed_irradiance = self.solar_spec * transmission * self.opv_response
        
        # Integrate to get absorbed power
        absorbed_power = trapezoid(absorbed_irradiance, self.lambda_range)
        
        # Calculate total available power
        total_power = trapezoid(self.solar_spec * self.opv_response, self.lambda_range)
        
        # Calculate efficiency (realistic maximum of 20% for OPV)
        if total_power > 0:
            efficiency = min(0.20, 0.85 * absorbed_power / total_power)  # 85% utilization factor
        else:
            efficiency = 0.0
        
        return efficiency
    
    def calculate_psu_efficiency(self, transmission: np.ndarray) -> float:
        """
        Calculate PSU efficiency for given transmission spectrum.
        
        Parameters:
        -----------
        transmission : np.ndarray
            Spectral transmission values
            
        Returns:
        --------
        float
            Electron transport rate
        """
        # Calculate absorbed irradiance for PSU
        absorbed_irradiance = self.solar_spec * (1 - transmission) * self.psu_response
        
        # Integrate to get absorbed power
        absorbed_power = trapezoid(absorbed_irradiance, self.lambda_range)
        
        # Calculate total available power
        total_power = trapezoid(self.solar_spec * self.psu_response, self.lambda_range)
        
        # Calculate efficiency (realistic maximum of 95% for PSU)
        if total_power > 0:
            efficiency = min(0.95, 0.92 * absorbed_power / total_power)  # 92% utilization factor
        else:
            efficiency = 0.05  # Minimum efficiency
        
        return efficiency
    
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function to minimize (negative of weighted sum).
        
        Parameters:
        -----------
        params : np.ndarray
            Filter parameters [A1, λ1, σ1, A2, λ2, σ2, ...]
            
        Returns:
        --------
        float
            Negative of weighted objective to minimize
        """
        try:
            # Calculate transmission spectrum from parameters
            transmission = self.calculate_transmission_spectrum(params)
            
            # Calculate efficiencies
            pce = self.calculate_opv_efficiency(transmission)
            etr = self.calculate_psu_efficiency(transmission)
            
            # Calculate weighted objective (to maximize)
            objective = self.w_pce * pce + self.w_etr * etr
            
            # Return negative for minimization
            return -objective
            
        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return 10.0  # Large positive value to penalize invalid solutions
    
    def calculate_transmission_spectrum(self, params: np.ndarray) -> np.ndarray:
        """
        Calculate spectral transmission from filter parameters.
        
        Parameters:
        -----------
        params : np.ndarray
            Filter parameters [A1, λ1, σ1, A2, λ2, σ2, ...]
            
        Returns:
        --------
        np.ndarray
            Spectral transmission values
        """
        n_filters = len(params) // 3
        if n_filters == 0:
            return np.ones_like(self.lambda_range)
        
        transmission = np.ones(len(self.lambda_range))
        
        for i in range(n_filters):
            A = params[3*i]      # Amplitude (0-1)
            center = params[3*i+1]  # Center wavelength (nm)
            width = params[3*i+2]   # Width (nm)
            
            # Ensure physical bounds
            A = np.clip(A, 0.0, 1.0)
            width = max(width, 1.0)  # Minimum width
            
            # Add Gaussian filter contribution
            gaussian_filter = A * np.exp(-(self.lambda_range - center)**2 / (2 * width**2))
            # Multiple filters multiply transmission (more blocking)
            transmission = transmission * (1 - gaussian_filter)
        
        # Ensure transmission is between 0 and 1
        transmission = np.clip(transmission, 0.0, 1.0)
        
        return transmission

    def optimize_spectral_splitting(self, 
                                    n_filters: int = 3,
                                    popsize: int = 15,
                                    maxiter: int = 100,
                                    seed: int = 42) -> Dict[str, Any]:
        """
        Optimize spectral splitting using differential evolution.
        
        Parameters:
        -----------
        n_filters : int
            Number of spectral filters to optimize
        popsize : int
            Population size for differential evolution
        maxiter : int
            Maximum iterations
        seed : int
            Random seed
            
        Returns:
        --------
        dict
            Optimization results
        """
        logger.info(f"Starting optimization with {n_filters} filters...")
        
        # Define bounds for optimization
        bounds = []
        for i in range(n_filters):
            # Amplitude (0-1)
            bounds.append((0.0, 1.0))
            # Center wavelength (300-1100 nm)
            bounds.append((300.0, 1100.0))
            # Width (5-100 nm)
            bounds.append((5.0, 100.0))
        
        # Perform optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            disp=True,
            workers=1  # Avoid multiprocessing issues
        )
        
        # Calculate final values with optimal parameters
        optimal_transmission = self.calculate_transmission_spectrum(result.x)
        optimal_pce = self.calculate_opv_efficiency(optimal_transmission)
        optimal_etr = self.calculate_psu_efficiency(optimal_transmission)
        
        results = {
            'optimal_params': result.x,
            'optimal_pce': optimal_pce,
            'optimal_etr': optimal_etr,
            'optimal_transmission': optimal_transmission,
            'convergence_history': [],  # Could be added if needed
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'nit': result.nit
        }
        
        logger.info(f"Optimization completed: PCE={optimal_pce:.4f}, ETR={optimal_etr:.4f}")
        logger.info(f"Success: {result.success}, Message: {result.message}")
        
        return results

    def evaluate_single_transmission(self, transmission: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a given transmission spectrum.
        
        Parameters:
        -----------
        transmission : np.ndarray
            Spectral transmission values
            
        Returns:
        --------
        dict
            Evaluation results
        """
        pce = self.calculate_opv_efficiency(transmission)
        etr = self.calculate_psu_efficiency(transmission)
        objective = self.w_pce * pce + self.w_etr * etr
        
        return {
            'pce': pce,
            'etr': etr,
            'objective': objective
        }

    def analyze_tradeoff(self, n_points: int = 20) -> Dict[str, np.ndarray]:
        """
        Analyze the trade-off between PCE and ETR by varying weights.
        
        Parameters:
        -----------
        n_points : int
            Number of weight combinations to evaluate
            
        Returns:
        --------
        dict
            Trade-off analysis results
        """
        logger.info("Analyzing PCE-ETR trade-off...")
        
        w1_values = np.linspace(0.05, 0.95, n_points)
        w2_values = 1 - w1_values
        
        pce_values = []
        etr_values = []
        
        original_weights = (self.w_pce, self.w_etr)
        
        for w1, w2 in zip(w1_values, w2_values):
            self.w_pce = w1
            self.w_etr = w2
            
            # Simple optimization just using the objective function directly
            # For each weight combination, find the best transmission
            best_transmission = self._find_best_simple_transmission()
            eval_result = self.evaluate_single_transmission(best_transmission)
            
            pce_values.append(eval_result['pce'])
            etr_values.append(eval_result['etr'])
        
        # Restore original weights
        self.w_pce, self.w_etr = original_weights
        
        return {
            'pce_values': np.array(pce_values),
            'etr_values': np.array(etr_values),
            'w1_values': w1_values,
            'w2_values': w2_values
        }
    
    def _find_best_simple_transmission(self) -> np.ndarray:
        """
        Find a reasonable transmission function without full optimization.
        This is a simplified approach for the trade-off analysis.
        """
        # Create a transmission that favors either OPV or PSU based on weights
        if self.w_pce > self.w_etr:
            # Favor OPV - transmit more light to OPV
            transmission = np.ones_like(self.lambda_range)
            # Add some filtering in PSU peak regions to improve PSU efficiency
            psu_peak_regions = (self.lambda_range > 600) & (self.lambda_range < 700)
            transmission[psu_peak_regions] = 0.3  # Block some red light for PSU
        else:
            # Favor PSU - block more light from OPV
            transmission = np.zeros_like(self.lambda_range)
            # Transmit some light in OPV range
            opv_peak_regions = (self.lambda_range > 300) & (self.lambda_range < 500)
            transmission[opv_peak_regions] = 0.4  # Allow some blue light for OPV
        
        return np.clip(transmission, 0.0, 1.0)

    def save_optimization_results(self, 
                                  results: Dict[str, Any],
                                  filename_prefix: str = 'spectral_optimization',
                                  output_dir: str = '../simulation_data/') -> str:
        """
        Save optimization results to CSV.
        
        Parameters:
        -----------
        results : dict
            Optimization results
        filename_prefix : str
            Prefix for filename
        output_dir : str
            Output directory
            
        Returns:
        --------
        str
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filepath = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
        
        # Create DataFrame with results
        data_dict = {
            'metric': ['PCE', 'ETR', 'objective', 'success', 'nfev', 'nit'],
            'value': [
                results.get('optimal_pce', 0),
                results.get('optimal_etr', 0),
                results.get('w_pce', 0.5) * results.get('optimal_pce', 0) + 
                results.get('w_etr', 0.5) * results.get('optimal_etr', 0),
                results.get('success', False),
                results.get('nfev', 0),
                results.get('nit', 0)
            ]
        }
        
        # Add optimal parameters
        if 'optimal_params' in results:
            params = results['optimal_params']
            for i, param in enumerate(params):
                data_dict['metric'].append(f'param_{i}')
                data_dict['value'].append(param)
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Optimization results saved to {filepath}")
        return filepath

    def plot_optimization_results(self,
                                  results: Dict[str, Any],
                                  filename_prefix: str = 'spectral_optimization',
                                  figures_dir: str = '../Graphics/') -> str:
        """
        Plot optimization results.
        
        Parameters:
        -----------
        results : dict
            Optimization results
        filename_prefix : str
            Prefix for filename
        figures_dir : str
            Figures directory
            
        Returns:
        --------
        str
            Path to saved figure
        """
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pdf_path = os.path.join(figures_dir, f"{filename_prefix}_{timestamp}.pdf")
        png_path = os.path.join(figures_dir, f"{filename_prefix}_{timestamp}.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spectral Optimization Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance metrics
        ax1 = axes[0, 0]
        metrics = ['PCE', 'ETR']
        values = [results.get('optimal_pce', 0), results.get('optimal_etr', 0)]
        colors = ['#1f77b4', '#2ca02c']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Efficiency')
        ax1.set_title('Optimized Performance Metrics')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Add target lines
        ax1.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='OPV Target (20%)')
        ax1.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='PSU Target (90%)')
        
        # Plot 2: Optimal transmission spectrum
        ax2 = axes[0, 1]
        if 'optimal_transmission' in results:
            ax2.plot(self.lambda_range, results['optimal_transmission'], 'purple', linewidth=2)
            ax2.fill_between(self.lambda_range, results['optimal_transmission'], alpha=0.3, 
                           label='OPV Region', color='blue')
            ax2.fill_between(self.lambda_range, results['optimal_transmission'], 1, alpha=0.3, 
                           label='PSU Region', color='green')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Transmission')
            ax2.set_title('Optimized Spectral Transmission')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Transmission data not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('Optimized Spectral Transmission')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Solar spectrum and responses
        ax3 = axes[1, 0]
        ax3.plot(self.lambda_range, self.solar_spec, 'orange', label='Solar Spectrum', linewidth=2)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.lambda_range, self.opv_response, 'blue', label='OPV Response', 
                     linestyle='--', linewidth=2)
        ax3_twin.plot(self.lambda_range, self.psu_response, 'green', label='PSU Response', 
                     linestyle='--', linewidth=2)
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Solar Irradiance (a.u.)', color='orange')
        ax3_twin.set_ylabel('Response', color='blue')
        ax3.set_title('Solar Spectrum and System Responses')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter values
        ax4 = axes[1, 1]
        if 'optimal_params' in results:
            params = results['optimal_params']
            n_params = len(params)
            param_indices = range(n_params)
            ax4.bar(param_indices, params, alpha=0.7, color='red')
            ax4.set_xlabel('Parameter Index')
            ax4.set_ylabel('Parameter Value')
            ax4.set_title('Optimal Filter Parameters')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Parameter data not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title('Optimal Filter Parameters')
        
        plt.tight_layout()
        
        # Save figures
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization results plotted to {pdf_path} and {png_path}")
        return pdf_path


if __name__ == "__main__":
    # Example usage
    logger.info("Testing SpectralOptimizer...")
    
    # Create example solar spectrum
    lambda_range = np.linspace(300, 1100, 801)
    
    # Simplified solar irradiance
    solar_irradiance = np.zeros_like(lambda_range, dtype=float)
    for i, lam in enumerate(lambda_range):
        if 300 <= lam <= 400:  # UV-Violet
            solar_irradiance[i] = 0.5 + 1.2 * (lam - 300) / 100
        elif 400 <= lam <= 700:  # Visible
            solar_irradiance[i] = 1.7 - 0.3 * abs(lam - 550) / 150
        elif 700 <= lam <= 1100:  # NIR
            solar_irradiance[i] = 1.4 * np.exp(-0.002 * (lam - 700))
        else:
            solar_irradiance[i] = 0.0
    
    # Normalize
    integral = trapezoid(solar_irradiance, lambda_range)
    solar_irradiance = solar_irradiance * 100.0 / integral
    
    # Create example response functions
    opv_response = np.zeros_like(lambda_range, dtype=float)
    for i, lam in enumerate(lambda_range):
        if 300 <= lam <= 700:  # OPV active region
            opv_response[i] = 0.8 * np.exp(-(lam - 600)**2 / (2 * 100**2))
        else:
            opv_response[i] = 0.1  # Low response in NIR
    
    psu_response = np.zeros_like(lambda_range, dtype=float)
    for i, lam in enumerate(lambda_range):
        if 400 <= lam <= 500:  # Blue region
            psu_response[i] = 0.8 + 0.2 * np.sin(np.pi * (lam - 400) / 100)
        elif 600 <= lam <= 700:  # Red region
            psu_response[i] = 0.85 + 0.15 * np.cos(np.pi * (lam - 650) / 50)
        elif 500 < lam < 600:  # Green valley
            psu_response[i] = 0.2 + 0.1 * np.sin(np.pi * (lam - 500) / 100)
        elif lam < 400:  # UV region
            psu_response[i] = 0.1
        else:  # Beyond 700 nm
            psu_response[i] = 0.3 * np.exp(-0.01 * (lam - 700))
    
    # Normalize responses
    opv_response /= np.max(opv_response)
    psu_response /= np.max(psu_response)
    
    # Initialize optimizer
    solar_spectrum = (lambda_range, solar_irradiance)
    optimizer = SpectralOptimizer(
        solar_spectrum=solar_spectrum,
        opv_response=opv_response,
        psu_response=psu_response,
        weights=(0.5, 0.5)
    )
    
    # Run optimization
    results = optimizer.optimize_spectral_splitting(n_filters=2, maxiter=50)
    
    print(f"Optimization completed:")
    print(f"  PCE: {results['optimal_pce']:.4f}")
    print(f"  ETR: {results['optimal_etr']:.4f}")
    print(f"  Success: {results['success']}")
