"""
SensitivityAnalyzer: Comprehensive sensitivity analysis for quantum agrivoltaics.

This module provides tools for sensitivity analysis and uncertainty quantification
in quantum agrivoltaic simulations.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import required classes
try:
    from core.hops_simulator import HopsSimulator
except ImportError:
    HopsSimulator = None
    logging.warning("HopsSimulator not available for SensitivityAnalyzer")

try:
    from agrivoltaic_coupling_model import AgrivoltaicCouplingModel
except ImportError:
    AgrivoltaicCouplingModel = None

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis and uncertainty quantification for
    quantum agrivoltaics simulations.

    This class provides methods for:
    - Local sensitivity analysis (varying one parameter at a time)
    - Monte Carlo uncertainty quantification
    - Comprehensive sensitivity reporting

    Parameters
    ----------
    quantum_simulator : Any
        Quantum dynamics simulator instance
    agrivoltaic_model : Any
        Agrivoltaic coupling model instance

    Attributes
    ----------
    quantum_simulator : Any
        The quantum simulator instance
    agrivoltaic_model : Any
        The agrivoltaic model instance
    param_ranges : dict
        Dictionary of parameter ranges for sensitivity analysis
    """

    def __init__(self, quantum_simulator, agrivoltaic_model):
        """Initialize the sensitivity analyzer."""
        self.quantum_simulator = quantum_simulator
        self.agrivoltaic_model = agrivoltaic_model
        self.param_ranges = {
            'temperature': (273, 320),
            'dephasing_rate': (5, 50),
            'transmission_center': (400, 700),
            'transmission_width': (20, 150),
            'dust_thickness': (0, 5)
        }
        logger.info("SensitivityAnalyzer initialized")

    def local_sensitivity_analysis(
        self,
        base_params: Dict[str, float],
        param_name: str,
        n_points: int = 20
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Perform local sensitivity analysis by varying one parameter.

        Parameters
        ----------
        base_params : dict
            Base parameter values
        param_name : str
            Name of parameter to vary
        n_points : int
            Number of points to evaluate

        Returns
        -------
        tuple
            (param_values, pce_values, etr_values, coherence_values)
        """
        param_range = self.param_ranges.get(param_name, (0, 1))
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        pce_values, etr_values, coherence_values = [], [], []

        for val in param_values:
            current_params = base_params.copy()
            current_params[param_name] = val

            if param_name in ['temperature', 'dephasing_rate']:
                if HopsSimulator is not None:
                    sim = HopsSimulator(
                        self.quantum_simulator.hamiltonian,
                        temperature=current_params.get('temperature', 295),
                        dephasing_rate=current_params.get('dephasing_rate', 20)
                    )
                    result = sim.simulate_dynamics(
                        time_points=np.linspace(0, 100, 20)
                    )
                    # Extract coherences from result
                    cohers = result.get('coherences', np.array([0]))
                    coherence_values.append(np.mean(cohers))
                else:
                    coherence_values.append(0.0)

                # Calculate PCE and ETR
                if self.agrivoltaic_model is not None:
                    trans = self.agrivoltaic_model.calculate_spectral_transmission(
                        [0.5, 0.5, 0.5, 0.2]
                    )
                    pce_values.append(self.agrivoltaic_model.calculate_pce(trans))
                    etr_values.append(self.agrivoltaic_model.calculate_etr(trans))
                else:
                    pce_values.append(0.0)
                    etr_values.append(0.0)

            elif param_name == 'dust_thickness':
                if hasattr(self.agrivoltaic_model, 'update_environmental_conditions'):
                    self.agrivoltaic_model.update_environmental_conditions(
                        dust_thickness=val
                    )
                if self.agrivoltaic_model is not None:
                    trans = self.agrivoltaic_model.calculate_spectral_transmission(
                        [0.5, 0.5, 0.5, 0.2]
                    )
                    pce_values.append(self.agrivoltaic_model.calculate_pce(trans))
                    etr_values.append(self.agrivoltaic_model.calculate_etr(trans))
                else:
                    pce_values.append(0.0)
                    etr_values.append(0.0)
                coherence_values.append(0.0)
            else:
                pce_values.append(0.0)
                etr_values.append(0.0)
                coherence_values.append(0.0)

        return (
            param_values,
            np.array(pce_values),
            np.array(etr_values),
            np.array(coherence_values)
        )

    def monte_carlo_uncertainty(
        self,
        n_samples: int = 100,
        param_uncertainties: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo uncertainty quantification.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples
        param_uncertainties : dict, optional
            Relative uncertainties for each parameter

        Returns
        -------
        dict
            Dictionary with uncertainty statistics for PCE and ETR
        """
        if param_uncertainties is None:
            param_uncertainties = {key: 0.1 for key in self.param_ranges.keys()}

        pce_samples, etr_samples = [], []
        base_params = {
            key: (val[0] + val[1]) / 2
            for key, val in self.param_ranges.items()
        }

        for _ in range(n_samples):
            sampled_params = {}
            for key, base_val in base_params.items():
                uncertainty = param_uncertainties.get(key, 0.1)
                sampled_value = np.random.normal(
                    base_val, base_val * uncertainty
                )
                sampled_params[key] = np.clip(
                    sampled_value,
                    self.param_ranges[key][0],
                    self.param_ranges[key][1]
                )

            # Calculate PCE and ETR with sampled parameters
            if self.agrivoltaic_model is not None:
                trans = self.agrivoltaic_model.calculate_spectral_transmission(
                    [0.5, 0.5, 0.5, 0.2]
                )
                pce_samples.append(self.agrivoltaic_model.calculate_pce(trans))
                etr_samples.append(self.agrivoltaic_model.calculate_etr(trans))

        pce_samples = np.array(pce_samples)
        etr_samples = np.array(etr_samples)

        return {
            'pce': {
                'mean': np.mean(pce_samples),
                'std': np.std(pce_samples),
                'ci_95': (
                    np.percentile(pce_samples, 2.5),
                    np.percentile(pce_samples, 97.5)
                ),
                'samples': pce_samples
            },
            'etr': {
                'mean': np.mean(etr_samples),
                'std': np.std(etr_samples),
                'ci_95': (
                    np.percentile(etr_samples, 2.5),
                    np.percentile(etr_samples, 97.5)
                ),
                'samples': etr_samples
            }
        }

    def comprehensive_sensitivity_report(
        self,
        n_points: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate comprehensive sensitivity report for all parameters.

        Parameters
        ----------
        n_points : int
            Number of points for each parameter scan

        Returns
        -------
        dict
            Dictionary containing sensitivity analysis for each parameter
        """
        report = {}
        base_params = {
            key: (val[0] + val[1]) / 2
            for key, val in self.param_ranges.items()
        }

        param_names = [
            'temperature', 'dephasing_rate', 'transmission_center',
            'transmission_width', 'dust_thickness'
        ]

        for param_name in param_names:
            param_vals, pce_vals, etr_vals, coh_vals = \
                self.local_sensitivity_analysis(base_params, param_name, n_points)

            report[param_name] = {
                'param_values': param_vals,
                'pce_values': pce_vals,
                'etr_values': etr_vals,
                'coherence_values': coh_vals,
                'pce_sensitivity': (
                    (pce_vals.max() - pce_vals.min()) / pce_vals.mean()
                    if pce_vals.mean() > 0 else 0
                ),
                'etr_sensitivity': (
                    (etr_vals.max() - etr_vals.min()) / etr_vals.mean()
                    if etr_vals.mean() > 0 else 0
                )
            }

        logger.info("Comprehensive sensitivity report generated")
        return report

    def save_sensitivity_results_to_csv(
        self,
        report: Dict[str, Any],
        filename_prefix: str = 'sensitivity_analysis',
        output_dir: str = '../simulation_data/'
    ) -> str:
        """
        Save sensitivity analysis results to CSV.

        Parameters
        ----------
        report : dict
            Results from comprehensive_sensitivity_report()
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

        # Prepare data rows
        rows = []
        for param_name, data in report.items():
            for i, (param_val, pce_val, etr_val, coh_val) in enumerate(zip(
                data['param_values'],
                data['pce_values'],
                data['etr_values'],
                data['coherence_values']
            )):
                rows.append({
                    'parameter': param_name,
                    'param_value': param_val,
                    'pce': pce_val,
                    'etr': etr_val,
                    'coherence': coh_val,
                    'pce_sensitivity': data['pce_sensitivity'] if i == 0 else None,
                    'etr_sensitivity': data['etr_sensitivity'] if i == 0 else None
                })

        df = pd.DataFrame(rows)
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, float_format='%.6e')

        logger.info(f"Sensitivity analysis results saved to {filepath}")
        return filepath

    def plot_sensitivity_results(
        self,
        report: Dict[str, Any],
        filename_prefix: str = 'sensitivity_analysis',
        figures_dir: str = '../Graphics/'
    ) -> str:
        """
        Plot sensitivity analysis results.

        Parameters
        ----------
        report : dict
            Results from comprehensive_sensitivity_report()
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

        n_params = len(report)
        n_cols = 2
        n_rows = (n_params + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        fig.suptitle('Sensitivity Analysis Results', fontsize=16, fontweight='bold')

        for idx, (param_name, data) in enumerate(report.items()):
            ax = axes[idx]

            # Plot PCE and ETR on same axis
            ax2 = ax.twinx()

            line1 = ax.plot(data['param_values'], data['pce_values'],
                           'b-', marker='o', label='PCE', linewidth=2)
            line2 = ax2.plot(data['param_values'], data['etr_values'],
                            'r-', marker='s', label='ETR', linewidth=2)

            ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('PCE', color='b', fontsize=10)
            ax2.set_ylabel('ETR', color='r', fontsize=10)
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            ax.grid(alpha=0.3)

            # Add sensitivity info as text
            sens_text = f"PCE Sens: {data['pce_sensitivity']:.3f}\nETR Sens: {data['etr_sensitivity']:.3f}"
            ax.text(0.02, 0.98, sens_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        filename = f"{filename_prefix}_{timestamp}.pdf"
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        png_filename = f"{filename_prefix}_{timestamp}.png"
        png_filepath = os.path.join(figures_dir, png_filename)
        plt.savefig(png_filepath, dpi=150, bbox_inches='tight')

        plt.close()

        logger.info(f"Sensitivity plots saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    logger.info("SensitivityAnalyzer module - example usage")

    # Create dummy simulator and model for demonstration
    class DummySimulator:
        def __init__(self):
            self.hamiltonian = np.eye(7)

    class DummyModel:
        def calculate_spectral_transmission(self, params):
            return np.ones(100) * 0.5

        def calculate_pce(self, transmission):
            return 0.15

        def calculate_etr(self, transmission):
            return 0.8

    dummy_sim = DummySimulator()
    dummy_model = DummyModel()

    analyzer = SensitivityAnalyzer(dummy_sim, dummy_model)

    # Run Monte Carlo uncertainty
    mc_results = analyzer.monte_carlo_uncertainty(n_samples=50)
    print(f"PCE mean: {mc_results['pce']['mean']:.3f}")
    print(f"ETR mean: {mc_results['etr']['mean']:.3f}")
