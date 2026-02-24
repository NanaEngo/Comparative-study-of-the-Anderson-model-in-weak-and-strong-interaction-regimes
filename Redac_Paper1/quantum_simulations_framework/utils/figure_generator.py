
"""
Figure Generator Module for Quantum Agrivoltaics Simulations.

This module provides tools for generating publication-quality figures
from quantum simulation results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


logger = logging.getLogger(__name__)


class FigureGenerator:
    """
    Class for generating publication-quality figures from simulation results.
    """
    
    def __init__(self, figures_dir: str = '../Graphics/'):
        """
        Initialize figure generator.
        
        Parameters:
        -----------
        figures_dir : str
            Directory to save figures
        """
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        
        # Set publication-ready style (use available styles)
        try:
            import scienceplots
            plt.style.use(['science', 'nature'])
        except OSError:
            # Fallback to default if styles not available
            pass
        
        logger.info(f"Figure generator initialized at {figures_dir}")
    
    def plot_quantum_dynamics(
        self,
        time_points: np.ndarray,
        populations: np.ndarray,
        coherences: np.ndarray,
        quantum_metrics: Dict[str, np.ndarray],
        filename_prefix: str = 'quantum_dynamics',
        **kwargs
    ) -> str:
        """
        Plot quantum dynamics results.
        
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
        **kwargs : dict
            Additional plotting options
        
        Returns:
        --------
        str
            Path to the saved figure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.pdf"
        )
        png_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.png"
        )
        
        n_sites = populations.shape[1] if len(populations.shape) > 1 else 1
        n_metrics = len(quantum_metrics)
        
        # Create subplots
        n_cols = 2
        n_rows = max(2, n_metrics)  # At least 2 rows for populations and coherences
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        fig.suptitle('Quantum Dynamics Simulation Results', fontsize=16, fontweight='bold')
        
        # Plot populations over time
        if n_sites == 1:
            ax0 = axes[0, 0] if n_rows > 1 else axes[0]
            ax0.plot(time_points, populations, 'b-', linewidth=2)
            ax0.set_xlabel('Time (fs)')
            ax0.set_ylabel('Population')
            ax0.set_title('Population Dynamics')
            ax0.grid(True, alpha=0.3)
        else:
            ax0 = axes[0, 0] if n_rows > 1 else axes[0]
            for i in range(min(n_sites, 10)):  # Limit to first 10 sites for readability
                ax0.plot(time_points, populations[:, i], label=f'Site {i+1}', linewidth=1.5)
            ax0.set_xlabel('Time (fs)')
            ax0.set_ylabel('Population')
            ax0.set_title('Site Populations vs Time')
            ax0.legend(fontsize=8)
            ax0.grid(True, alpha=0.3)
        
        # Plot coherences
        ax1 = axes[0, 1] if n_rows > 1 else axes[1]
        ax1.plot(time_points, coherences, 'r-', linewidth=2)
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Coherence')
        ax1.set_title('Coherence Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot quantum metrics
        metric_idx = 1
        for i, (metric_name, metric_values) in enumerate(quantum_metrics.items()):
            if metric_idx < n_rows:
                ax = axes[metric_idx, 0] if i % 2 == 0 else axes[metric_idx, 1]
                if metric_idx >= n_rows and i % 2 == 0:
                    metric_idx += 1
                    if metric_idx >= n_rows:
                        break
                    ax = axes[metric_idx, 0]
                
                ax.plot(time_points, metric_values, linewidth=2)
                ax.set_xlabel('Time (fs)')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figures in multiple formats
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quantum dynamics figures saved to {pdf_path} and {png_path}")
        return pdf_path
    
    def plot_spectral_optimization(
        self,
        optimization_results: Dict[str, Any],
        solar_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        filename_prefix: str = 'spectral_optimization',
        **kwargs
    ) -> str:
        """
        Plot spectral optimization results.
        
        Parameters:
        -----------
        optimization_results : dict
            Results from spectral optimization
        solar_spectrum : tuple, optional
            Solar spectrum data (wavelengths, irradiance)
        filename_prefix : str
            Prefix for the output filename
        **kwargs : dict
            Additional plotting options
        
        Returns:
        --------
        str
            Path to the saved figure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.pdf"
        )
        png_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.png"
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Spectral Optimization Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Objective function evolution (if available)
        ax1 = axes[0, 0]
        if 'convergence_history' in optimization_results:
            history = optimization_results['convergence_history']
            ax1.plot(history, 'b-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Function')
            ax1.set_title('Optimization Convergence')
        else:
            # Just show the optimized values
            ax1.text(0.5, 0.5, f"PCE: {optimization_results.get('optimal_pce', 0):.3f}\nETR: {optimization_results.get('optimal_etr', 0):.3f}",
                    horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,
                    fontsize=14)
            ax1.set_title('Optimization Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spectral transmission (if available)
        ax2 = axes[0, 1]
        if 'transmission_func' in optimization_results and solar_spectrum:
            wavelengths, irradiances = solar_spectrum
            transmission_func = optimization_results['transmission_func']
            transmission_values = transmission_func(wavelengths)
            
            ax2.plot(wavelengths, irradiances, 'orange', label='Solar Spectrum', linewidth=2)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(wavelengths, transmission_values, 'blue', label='Transmission', linewidth=2)
            
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Solar Irradiance (a.u.)', color='orange')
            ax2_twin.set_ylabel('Transmission', color='blue')
            ax2.set_title('Optimized Spectral Transmission')
            
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'Transmission data not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('Spectral Transmission')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Parameter space (if available)
        ax3 = axes[1, 0]
        if 'optimal_params' in optimization_results:
            params = optimization_results['optimal_params']
            n_params = len(params)
            param_indices = range(n_params)
            ax3.bar(param_indices, params, alpha=0.7, color='green')
            ax3.set_xlabel('Parameter Index')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Optimal Parameters')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Parameter data not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_title('Optimal Parameters')
        
        # Plot 4: Performance metrics
        ax4 = axes[1, 1]
        pce = optimization_results.get('optimal_pce', 0)
        etr = optimization_results.get('optimal_etr', 0)
        
        metrics = ['PCE', 'ETR']
        values = [pce, etr]
        colors = ['#1f77b4', '#2ca02c']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Efficiency')
        ax4.set_title('Performance Metrics')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Add target lines for realistic values
        ax4.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='OPV Target (20%)')
        ax4.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='PSU Target (90%)')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save figures in multiple formats
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spectral optimization figures saved to {pdf_path} and {png_path}")
        return pdf_path
    
    def plot_agrivoltaic_performance(
        self,
        pce: float,
        etr: float,
        spectral_data: Dict[str, np.ndarray],
        filename_prefix: str = 'agrivoltaic_performance',
        **kwargs
    ) -> str:
        """
        Plot agrivoltaic system performance.
        
        Parameters:
        -----------
        pce : float
            Power conversion efficiency
        etr : float
            Electron transport rate
        spectral_data : dict
            Spectral data for plotting
        filename_prefix : str
            Prefix for the output filename
        **kwargs : dict
            Additional plotting options
        
        Returns:
        --------
        str
            Path to the saved figure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.pdf"
        )
        png_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.png"
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Agrivoltaic System Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance metrics
        ax1 = axes[0, 0]
        metrics = ['PCE', 'ETR']
        values = [pce, etr]
        colors = ['#1f77b4', '#2ca02c']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Efficiency')
        ax1.set_title('Performance Metrics')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Add target lines for realistic values
        ax1.axhline(y=0.20, color='red', linestyle='--', alpha=0.7, label='OPV Target (20%)')
        ax1.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='PSU Target (90%)')
        ax1.legend()
        
        # Plot 2: Spectral data if available
        ax2 = axes[0, 1]
        if 'wavelength' in spectral_data and 'transmission' in spectral_data:
            wavelengths = spectral_data['wavelength']
            transmission = spectral_data['transmission']
            ax2.plot(wavelengths, transmission, 'purple', linewidth=2, label='Transmission')
            ax2.fill_between(wavelengths, transmission, alpha=0.3, label='OPV Region', color='blue')
            ax2.fill_between(wavelengths, transmission, 1, alpha=0.3, label='PSU Region', color='green')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Transmission')
            ax2.set_title('Spectral Transmission Function')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Spectral data not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('Spectral Transmission')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Solar spectrum if available
        ax3 = axes[1, 0]
        if 'wavelength' in spectral_data and 'solar_irradiance' in spectral_data:
            wavelengths = spectral_data['wavelength']
            solar_irradiance = spectral_data['solar_irradiance']
            ax3.plot(wavelengths, solar_irradiance, 'orange', linewidth=2)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Solar Irradiance (a.u.)')
            ax3.set_title('Solar Spectrum')
        else:
            ax3.text(0.5, 0.5, 'Solar spectrum not available', 
                    horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_title('Solar Spectrum')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency trade-off
        ax4 = axes[1, 1]
        # This could be extended to show multiple operating points
        ax4.scatter([pce], [etr], color='red', s=100, label='Current Point', zorder=5)
        ax4.set_xlabel('PCE')
        ax4.set_ylabel('ETR')
        ax4.set_title('PCE vs ETR Trade-off')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add target region
        ax4.add_patch(plt.Rectangle((0.15, 0.85), 0.05, 0.1, fill=True, color='green', alpha=0.2, label='Target Region'))
        ax4.text(0.16, 0.87, 'Target\nRegion', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        # Save figures in multiple formats
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Agrivoltaic performance figures saved to {pdf_path} and {png_path}")
        return pdf_path
    
    def plot_quantum_metrics_evolution(
        self,
        time_points: np.ndarray,
        quantum_metrics: Dict[str, np.ndarray],
        filename_prefix: str = 'quantum_metrics_evolution',
        **kwargs
    ) -> str:
        """
        Plot evolution of various quantum metrics over time.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time points for the simulation
        quantum_metrics : dict
            Dictionary of quantum metrics
        filename_prefix : str
            Prefix for the output filename
        **kwargs : dict
            Additional plotting options
        
        Returns:
        --------
        str
            Path to the saved figure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.pdf"
        )
        png_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.png"
        )
        
        n_metrics = len(quantum_metrics)
        if n_metrics == 0:
            logger.warning("No quantum metrics to plot")
            return pdf_path
        
        # Determine subplot layout
        n_cols = 2
        n_rows = (n_metrics + 1) // 2  # Round up
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Quantum Metrics Evolution', fontsize=16, fontweight='bold')
        
        # Handle case where there's only one row
        if n_rows == 1:
            axes = [axes]
        
        metric_idx = 0
        for metric_name, metric_values in quantum_metrics.items():
            row = metric_idx // n_cols
            col = metric_idx % n_cols
            
            ax = axes[row] if n_rows == 1 else axes[row, col]
            
            ax.plot(time_points, metric_values, linewidth=2)
            ax.set_xlabel('Time (fs)')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution')
            ax.grid(True, alpha=0.3)
            
            metric_idx += 1
        
        # If there are empty subplots, remove them
        for idx in range(metric_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row] if n_rows == 1 else axes[row, col]
            fig.delaxes(ax)
        
        plt.tight_layout()
        
        # Save figures in multiple formats
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Quantum metrics figures saved to {pdf_path} and {png_path}")
        return pdf_path
    
    def plot_thermal_stability(
        self,
        temperatures: np.ndarray,
        pce_values: np.ndarray,
        etr_values: np.ndarray,
        filename_prefix: str = 'thermal_stability',
        **kwargs
    ) -> str:
        """
        Plot thermal stability of the agrivoltaic system.
        
        Parameters:
        -----------
        temperatures : np.ndarray
            Temperature values
        pce_values : np.ndarray
            PCE values at different temperatures
        etr_values : np.ndarray
            ETR values at different temperatures
        filename_prefix : str
            Prefix for the output filename
        **kwargs : dict
            Additional plotting options
        
        Returns:
        --------
        str
            Path to the saved figure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.pdf"
        )
        png_path = os.path.join(
            self.figures_dir, 
            f"{filename_prefix}_{timestamp}.png"
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Thermal Stability Analysis', fontsize=16, fontweight='bold')
        
        ax_twin = ax.twinx()
        
        # Plot PCE vs temperature
        ax.plot(temperatures, pce_values, 'b-', linewidth=2, label='PCE', marker='o')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('PCE', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot ETR vs temperature
        ax_twin.plot(temperatures, etr_values, 'r-', linewidth=2, label='ETR', marker='s')
        ax_twin.set_ylabel('ETR', color='r')
        ax_twin.tick_params(axis='y', labelcolor='r')
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        ax.grid(True, alpha=0.3)
        ax.set_title('Temperature Dependence of Performance Metrics')
        
        plt.tight_layout()
        
        # Save figures in multiple formats
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Thermal stability figures saved to {pdf_path} and {png_path}")
        return pdf_path
