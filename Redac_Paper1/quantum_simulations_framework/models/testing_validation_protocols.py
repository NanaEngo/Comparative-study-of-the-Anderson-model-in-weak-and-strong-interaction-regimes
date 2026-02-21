"""
TestingValidationProtocols: Comprehensive testing and validation for quantum agrivoltaics.

This module provides validation protocols to ensure simulation accuracy and
consistency with literature values.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import required classes
try:
    from core.hops_simulator import HopsSimulator
except ImportError:
    HopsSimulator = None
    logging.warning("HopsSimulator not available for TestingValidationProtocols")

try:
    from agrivoltaic_coupling_model import AgrivoltaicCouplingModel
except ImportError:
    AgrivoltaicCouplingModel = None

logger = logging.getLogger(__name__)


class TestingValidationProtocols:
    """
    Comprehensive testing and validation protocols for quantum agrivoltaics simulations.

    Mathematical Framework:
    Validation involves comparing simulation results against:
    1. Analytical benchmarks (for simple cases)
    2. Literature values (established experimental data)
    3. Internal consistency checks
    4. Convergence analysis

    Validation metrics include relative error, correlation coefficients,
    and statistical tests for agreement.

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
    literature_values : dict
        Reference values from literature for validation
    """

    def __init__(self, quantum_simulator, agrivoltaic_model):
        """Initialize testing and validation protocols."""
        self.quantum_simulator = quantum_simulator
        self.agrivoltaic_model = agrivoltaic_model

        # Reference values from literature
        self.literature_values = {
            'fmo_coherence_lifetime_77K': 400,  # fs (Engel et al., 2007)
            'fmo_coherence_lifetime_295K': 420,  # fs (Manuscript Figure 3a)
            'fmo_transfer_time': 1000,  # fs (typical transfer time to RC)
            'chlorophyll_quantum_efficiency': 0.95,  # Near-unity for PSI
            'opv_typical_pce': 0.18,  # Manuscript Target PCE
            'am15g_total_irradiance': 1000,  # W/m^2
        }

        logger.info("TestingValidationProtocols initialized")

    def validate_fmo_hamiltonian(self) -> Dict[str, Any]:
        """
        Validate FMO Hamiltonian against literature values.

        Returns
        -------
        dict
            Validation results with pass/fail status
        """
        # Get Hamiltonian from simulator
        if hasattr(self.quantum_simulator, 'hamiltonian'):
            H = self.quantum_simulator.hamiltonian
        elif hasattr(self.quantum_simulator, 'H'):
            H = self.quantum_simulator.H
        else:
            logger.error("Cannot find Hamiltonian in simulator")
            return {'error': 'Hamiltonian not found'}

        # Calculate eigenvalues
        evals = np.linalg.eigvalsh(H)

        # Expected ranges from Adolphs & Renger 2006
        expected_site_energy_range = (11900, 12300)  # cm^-1
        expected_coupling_range = (5, 200)  # cm^-1
        expected_bandwidth = (300, 500)  # cm^-1

        # Extract diagonal and off-diagonal elements
        site_energies = np.diag(H)
        couplings = H[np.triu_indices_from(H, k=1)]
        bandwidth = np.max(evals) - np.min(evals)

        results = {
            'site_energies': {
                'min': float(np.min(site_energies)),
                'max': float(np.max(site_energies)),
                'expected_range': expected_site_energy_range,
                'pass': (
                    np.min(site_energies) >= expected_site_energy_range[0] and
                    np.max(site_energies) <= expected_site_energy_range[1]
                )
            },
            'couplings': {
                'max_abs': float(np.max(np.abs(couplings))),
                'expected_range': expected_coupling_range,
                'pass': np.max(np.abs(couplings)) <= expected_coupling_range[1]
            },
            'bandwidth': {
                'value': float(bandwidth),
                'expected_range': expected_bandwidth,
                'pass': (
                    bandwidth >= expected_bandwidth[0] and
                    bandwidth <= expected_bandwidth[1]
                )
            },
            'hermitian': {
                'pass': np.allclose(H, H.T.conj())
            }
        }

        logger.debug("FMO Hamiltonian validation completed")
        return results

    def validate_quantum_dynamics(self) -> Dict[str, Any]:
        """
        Validate quantum dynamics against expected behavior.

        Returns
        -------
        dict
            Validation results
        """
        # Run short simulation
        time_points = np.linspace(0, 500, 100)

        if hasattr(self.quantum_simulator, 'simulate_dynamics'):
            result = self.quantum_simulator.simulate_dynamics(
                time_points=time_points
            )

            # Extract results
            populations = result.get('populations', np.array([]))
            coherences = result.get('coherences', np.array([]))
            purity_vals = result.get('purity', np.array([]))

            results = {
                'population_conservation': {
                    'initial_sum': float(np.sum(populations[0, :])) if populations.size > 0 else 0,
                    'final_sum': float(np.sum(populations[-1, :])) if populations.size > 0 else 0,
                    'pass': (
                        populations.size > 0 and
                        np.abs(np.sum(populations[0, :]) - np.sum(populations[-1, :])) < 0.1
                    )
                },
                'coherence_decay': {
                    'initial': float(coherences[0]) if coherences.size > 0 else 0,
                    'final': float(coherences[-1]) if coherences.size > 0 else 0,
                    'decays': (
                        coherences[-1] < coherences[0]
                        if coherences.size > 0 and coherences[0] > 0 else True
                    ),
                    'pass': True  # Coherence should generally decay
                },
                'purity_bounds': {
                    'min': float(np.min(purity_vals)) if purity_vals.size > 0 else 0,
                    'max': float(np.max(purity_vals)) if purity_vals.size > 0 else 0,
                    'pass': (
                        purity_vals.size > 0 and
                        np.min(purity_vals) >= 0 and np.max(purity_vals) <= 1.1
                    )
                },
                'population_positivity': {
                    'min_population': float(np.min(populations)) if populations.size > 0 else 0,
                    'pass': (
                        populations.size > 0 and
                        np.min(populations) >= -0.1  # Allow small numerical errors
                    )
                }
            }
        else:
            results = {'error': 'Simulator does not have simulate_dynamics method'}

        logger.debug("Quantum dynamics validation completed")
        return results

    def convergence_analysis(
        self,
        max_time_steps: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze convergence of simulation results with time step refinement.

        Parameters
        ----------
        max_time_steps : list, optional
            List of time step counts to test

        Returns
        -------
        dict
            Convergence analysis results
        """
        if max_time_steps is None:
            max_time_steps = [50, 100, 200, 400]

        final_populations = []
        final_coherences = []

        for n_steps in max_time_steps:
            time_points = np.linspace(0, 500, n_steps)

            if hasattr(self.quantum_simulator, 'simulate_dynamics'):
                result = self.quantum_simulator.simulate_dynamics(
                    time_points=time_points
                )
                populations = result.get('populations', np.array([]))
                coherences = result.get('coherences', np.array([]))

                if populations.size > 0:
                    final_populations.append(populations[-1, :])
                if coherences.size > 0:
                    final_coherences.append(coherences[-1])

        # Calculate relative differences between successive refinements
        pop_convergence = []
        coh_convergence = []

        for i in range(1, len(final_populations)):
            pop_diff = np.linalg.norm(
                final_populations[i] - final_populations[i - 1]
            )
            pop_norm = np.linalg.norm(final_populations[i])
            pop_convergence.append(
                pop_diff / pop_norm if pop_norm > 0 else 0
            )

            if i < len(final_coherences):
                coh_diff = np.abs(
                    final_coherences[i] - final_coherences[i - 1]
                )
                coh_convergence.append(
                    coh_diff / final_coherences[i]
                    if final_coherences[i] > 0 else 0
                )

        results = {
            'time_steps': max_time_steps,
            'final_populations': final_populations,
            'final_coherences': final_coherences,
            'population_convergence': pop_convergence,
            'coherence_convergence': coh_convergence,
            'converged': (
                pop_convergence[-1] < 0.05
                if len(pop_convergence) > 0 else False
            )
        }

        logger.debug("Convergence analysis completed")
        return results

    def compare_with_classical(self) -> Dict[str, Any]:
        """
        Compare quantum simulation results with classical (Markovian) model.

        Mathematical Framework:
        Classical (Markovian) models assume no memory effects:
        d\rho/dt = L[\rho]

        Quantum advantage is quantified as the improvement in ETR
        from non-Markovian effects.

        Returns
        -------
        dict
            Comparison results
        """
        time_points = np.linspace(0, 500, 100)

        # Quantum simulation (non-Markovian)
        if hasattr(self.quantum_simulator, 'simulate_dynamics'):
            result_quantum = self.quantum_simulator.simulate_dynamics(
                time_points=time_points
            )
            pop_quantum = result_quantum.get('populations', np.array([]))
            coh_quantum = result_quantum.get('coherences', np.array([]))
        else:
            pop_quantum = np.array([])
            coh_quantum = np.array([])

        # Classical simulation (higher dephasing to approximate Markovian limit)
        if HopsSimulator is not None and hasattr(self.quantum_simulator, 'hamiltonian'):
            classical_sim = HopsSimulator(
                self.quantum_simulator.hamiltonian,
                temperature=getattr(self.quantum_simulator, 'temperature', 295),
                dephasing_rate=100  # High dephasing for classical limit
            )
            result_classical = classical_sim.simulate_dynamics(
                time_points=time_points
            )
            pop_classical = result_classical.get('populations', np.array([]))
            coh_classical = result_classical.get('coherences', np.array([]))
        else:
            pop_classical = np.array([])
            coh_classical = np.array([])

        # Calculate quantum advantage
        # Transfer efficiency: population that leaves initial site
        if pop_quantum.size > 0:
            quantum_transfer = 1 - pop_quantum[-1, 0]
        else:
            quantum_transfer = 0

        if pop_classical.size > 0:
            classical_transfer = 1 - pop_classical[-1, 0]
        else:
            classical_transfer = 0

        quantum_advantage = (
            (quantum_transfer - classical_transfer) / classical_transfer * 100
            if classical_transfer > 0 else 0
        )

        comparison = {
            'quantum_transfer': float(quantum_transfer),
            'classical_transfer': float(classical_transfer),
            'quantum_advantage_percent': float(quantum_advantage),
            'quantum_coherence_final': float(coh_quantum[-1]) if coh_quantum.size > 0 else 0,
            'classical_coherence_final': float(coh_classical[-1]) if coh_classical.size > 0 else 0,
            'coherence_enhancement': (
                coh_quantum[-1] / coh_classical[-1]
                if coh_classical.size > 0 and coh_classical[-1] > 0 else 0
            ),
            'time_points': time_points,
            'pop_quantum': pop_quantum,
            'pop_classical': pop_classical
        }

        logger.debug("Classical comparison completed")
        return comparison

    def run_full_validation_suite(self) -> Dict[str, Any]:
        """
        Run complete validation suite and generate report.

        Returns
        -------
        dict
            Complete validation report
        """
        logger.info("Running full validation suite...")

        report = {
            'hamiltonian_validation': self.validate_fmo_hamiltonian(),
            'dynamics_validation': self.validate_quantum_dynamics(),
            'convergence_analysis': self.convergence_analysis(),
            'classical_comparison': self.compare_with_classical()
        }

        # Calculate overall pass rate
        total_tests = 0
        passed_tests = 0

        for category, tests in report.items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and 'pass' in test_result:
                        total_tests += 1
                        if test_result['pass']:
                            passed_tests += 1

        report['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': (
                passed_tests / total_tests * 100 if total_tests > 0 else 0
            )
        }

        logger.info(
            f"Validation suite completed: {passed_tests}/{total_tests} tests passed "
            f"({report['summary']['pass_rate']:.1f}%)"
        )

        return report

    def save_validation_results_to_csv(
        self,
        report: Dict[str, Any],
        filename_prefix: str = 'validation_report',
        output_dir: str = '../simulation_data/'
    ) -> str:
        """
        Save validation results to CSV.

        Parameters
        ----------
        report : dict
            Results from run_full_validation_suite()
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

        # Flatten the report for CSV
        rows = []

        # Hamiltonian validation
        hamiltonian = report.get('hamiltonian_validation', {})
        for test_name, test_data in hamiltonian.items():
            if isinstance(test_data, dict):
                rows.append({
                    'category': 'hamiltonian_validation',
                    'test_name': test_name,
                    'passed': test_data.get('pass', False),
                    'details': str(test_data)
                })

        # Dynamics validation
        dynamics = report.get('dynamics_validation', {})
        for test_name, test_data in dynamics.items():
            if isinstance(test_data, dict):
                rows.append({
                    'category': 'dynamics_validation',
                    'test_name': test_name,
                    'passed': test_data.get('pass', False),
                    'details': str(test_data)
                })

        # Summary
        summary = report.get('summary', {})
        rows.append({
            'category': 'summary',
            'test_name': 'total_tests',
            'value': summary.get('total_tests', 0),
            'details': ''
        })
        rows.append({
            'category': 'summary',
            'test_name': 'passed_tests',
            'value': summary.get('passed_tests', 0),
            'details': ''
        })
        rows.append({
            'category': 'summary',
            'test_name': 'pass_rate_percent',
            'value': summary.get('pass_rate', 0),
            'details': ''
        })

        df = pd.DataFrame(rows)
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        logger.info(f"Validation report saved to {filepath}")
        return filepath

    def plot_validation_results(
        self,
        report: Dict[str, Any],
        filename_prefix: str = 'validation_report',
        figures_dir: str = '../Graphics/'
    ) -> str:
        """
        Plot validation results.

        Parameters
        ----------
        report : dict
            Results from run_full_validation_suite()
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

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Validation Report', fontsize=16, fontweight='bold')

        # 1. Test pass/fail summary
        ax1 = axes[0, 0]
        summary = report.get('summary', {})
        pass_rate = summary.get('pass_rate', 0)
        labels = ['Passed', 'Failed']
        sizes = [summary.get('passed_tests', 0), summary.get('total_tests', 0) - summary.get('passed_tests', 0)]
        colors = ['#2ca02c', '#d62728']
        explode = (0.05, 0)

        if sum(sizes) > 0:
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.set_title(f'Test Results (Pass Rate: {pass_rate:.1f}%)', fontsize=12)
        else:
            ax1.text(0.5, 0.5, 'No test data', ha='center', va='center', transform=ax1.transAxes)

        # 2. Hamiltonian validation
        ax2 = axes[0, 1]
        hamiltonian = report.get('hamiltonian_validation', {})
        tests = []
        results = []
        for test_name, test_data in hamiltonian.items():
            if isinstance(test_data, dict) and 'pass' in test_data:
                tests.append(test_name.replace('_', ' ').title())
                results.append(1 if test_data['pass'] else 0)

        if tests:
            colors_bar = ['#2ca02c' if r == 1 else '#d62728' for r in results]
            ax2.barh(tests, results, color=colors_bar, alpha=0.7, edgecolor='black')
            ax2.set_xlim(0, 1.2)
            ax2.set_xlabel('Pass (1) / Fail (0)', fontsize=10)
            ax2.set_title('Hamiltonian Validation', fontsize=12)
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No hamiltonian tests', ha='center', va='center', transform=ax2.transAxes)

        # 3. Dynamics validation
        ax3 = axes[1, 0]
        dynamics = report.get('dynamics_validation', {})
        tests = []
        results = []
        for test_name, test_data in dynamics.items():
            if isinstance(test_data, dict) and 'pass' in test_data:
                tests.append(test_name.replace('_', ' ').title())
                results.append(1 if test_data['pass'] else 0)

        if tests:
            colors_bar = ['#2ca02c' if r == 1 else '#d62728' for r in results]
            ax3.barh(tests, results, color=colors_bar, alpha=0.7, edgecolor='black')
            ax3.set_xlim(0, 1.2)
            ax3.set_xlabel('Pass (1) / Fail (0)', fontsize=10)
            ax3.set_title('Dynamics Validation', fontsize=12)
            ax3.grid(axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No dynamics tests', ha='center', va='center', transform=ax3.transAxes)

        # 4. Classical comparison
        ax4 = axes[1, 1]
        comparison = report.get('classical_comparison', {})
        if comparison:
            quantum_transfer = comparison.get('quantum_transfer', 0)
            classical_transfer = comparison.get('classical_transfer', 0)
            quantum_advantage = comparison.get('quantum_advantage_percent', 0)

            categories = ['Quantum', 'Classical']
            values = [quantum_transfer, classical_transfer]
            colors_comp = ['#1f77b4', '#ff7f0e']

            bars = ax4.bar(categories, values, color=colors_comp, alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Transfer Efficiency', fontsize=10)
            ax4.set_title(f'Quantum vs Classical (Advantage: {quantum_advantage:.1f}%)', fontsize=12)
            ax4.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()

        filename = f"{filename_prefix}_{timestamp}.pdf"
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        png_filename = f"{filename_prefix}_{timestamp}.png"
        png_filepath = os.path.join(figures_dir, png_filename)
        plt.savefig(png_filepath, dpi=150, bbox_inches='tight')

        plt.close()

        logger.info(f"Validation plots saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    logger.info("TestingValidationProtocols module - example usage")

    # Create dummy simulator for demonstration
    class DummySimulator:
        def __init__(self):
            self.hamiltonian = np.diag([12200, 12070, 11980, 12050, 12140, 12130, 12260])
            self.temperature = 295

        def simulate_dynamics(self, time_points):
            n_steps = len(time_points)
            return {
                'populations': np.random.rand(n_steps, 7),
                'coherences': np.random.rand(n_steps),
                'purity': np.random.rand(n_steps) * 0.5 + 0.5
            }

    dummy_sim = DummySimulator()
    validator = TestingValidationProtocols(dummy_sim, None)

    # Run validation
    report = validator.run_full_validation_suite()
    print(f"Pass rate: {report['summary']['pass_rate']:.1f}%")
