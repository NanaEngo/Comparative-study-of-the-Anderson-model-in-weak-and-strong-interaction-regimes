"""
Unified Figures Class for Quantum Agrivoltaics Simulations

This module implements a unified class for generating all figures
related to quantum agrivoltaics simulations, including quantum dynamics,
spectral analysis, optimization results, and eco-design validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # For publication-quality plots
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os


class UnifiedFigures:
    """
    Unified class for generating all figures in quantum agrivoltaics simulations.
    
    This class consolidates all visualization functionality into a single
    comprehensive interface, providing publication-quality plots for
    quantum dynamics, spectral analysis, optimization results, and eco-design.
    """
    
    def __init__(self, style='science', figsize_base=(7, 5), data_dir='data_output', figures_dir='figures'):
        """
        Initialize the unified figures class.
        
        Parameters:
        style (str): Matplotlib style to use ('science', 'default', etc.)
        figsize_base (tuple): Base figure size (width, height)
        data_dir (str): Directory to save data files (default: 'data_output')
        figures_dir (str): Directory to save figures (default: 'figures')
        """
        plt.style.use([style, 'notebook']) if style else plt.style.use('default')
        self.figsize_base = figsize_base
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        import os
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        self.colors = {
            'opv': '#1f77b4',      # blue
            'psu': '#ff7f0e',      # orange  
            'etr': '#2ca02c',      # green
            'pce': '#d62728',      # red
            'coherence': '#9467bd', # purple
            'qfi': '#8c564b',      # brown
            'entropy': '#e377c2',   # pink
            'background': '#ffffff' # white
        }
    
    def plot_quantum_dynamics(self, time_points, populations, coherences, 
                             qfi_values, entropy_values, purity_values,
                             title="Quantum Dynamics Analysis"):
        """
        Plot comprehensive quantum dynamics results.
        
        Mathematical Framework:
        The quantum dynamics visualization displays the time evolution of
        key quantum observables that characterize the system's behavior:
        
        - Site populations: ρ_ii(t) representing the probability of finding
          the excitation at site i at time t
        - Quantum coherence: l1-norm of coherence ||ρ||_l1 = Σᵢⱼ |ρᵢⱼ| (i≠j)
          quantifying quantum superposition effects
        - Quantum Fisher Information: F_Q(ρ,H) = 2 Σᵢⱼ (λᵢ-λⱼ)²/(λᵢ+λⱼ) |⟨ψᵢ|H|ψⱼ⟩|²
          measuring the system's sensitivity to parameter changes
        - Von Neumann entropy: S(ρ) = -Tr[ρ log ρ] quantifying mixedness
        - Purity: Tr[ρ²] measuring the degree of quantum coherence
        
        Parameters:
        time_points (array): Time points in fs
        populations (2D array): Site populations over time
        coherences (array): Coherence values over time
        qfi_values (array): QFI values over time
        entropy_values (array): Entropy values over time
        purity_values (array): Purity values over time
        title (str): Plot title
        """
        n_sites = populations.shape[1]
        n_plots = 2 + min(n_sites, 4)  # Total population, coherences, and up to 4 site pops
        
        fig = plt.figure(figsize=(self.figsize_base[0], self.figsize_base[1] * 1.5))
        gs = GridSpec(3, 3, figure=fig)
        
        # Total population decay
        ax1 = fig.add_subplot(gs[0, :2])
        total_pop = np.sum(populations, axis=1)
        ax1.plot(time_points, total_pop, 'k-', linewidth=2, label='Total Population')
        for i in range(min(n_sites, 4)):
            ax1.plot(time_points, populations[:, i], 
                    label=f'Site {i+1} Population', 
                    linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Population')
        ax1.set_title('Population Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Coherences over time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(time_points, coherences, color=self.colors['coherence'], linewidth=2)
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Coherence (l1-norm)')
        ax2.set_title('Quantum Coherence')
        ax2.grid(True, alpha=0.3)
        
        # QFI over time
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_points, qfi_values, color=self.colors['qfi'], linewidth=2)
        ax3.set_xlabel('Time (fs)')
        ax3.set_ylabel('QFI')
        ax3.set_title('Quantum Fisher Information')
        ax3.grid(True, alpha=0.3)
        
        # Entropy over time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_points, entropy_values, color=self.colors['entropy'], linewidth=2)
        ax4.set_xlabel('Time (fs)')
        ax4.set_ylabel('Von Neumann Entropy')
        ax4.set_title('Quantum Entropy')
        ax4.grid(True, alpha=0.3)
        
        # Purity over time
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(time_points, purity_values, 'purple', linewidth=2)
        ax5.set_xlabel('Time (fs)')
        ax5.set_ylabel('Purity (Tr[ρ²])')
        ax5.set_title('State Purity')
        ax5.grid(True, alpha=0.3)
        
        # Final state analysis
        ax6 = fig.add_subplot(gs[2, :])
        final_pops = populations[-1, :] if len(populations) > 0 else np.zeros(n_sites)
        sites = np.arange(1, len(final_pops) + 1)
        bars = ax6.bar(sites, final_pops, color=self.colors['psu'], alpha=0.7)
        ax6.set_xlabel('Site Index')
        ax6.set_ylabel('Final Population')
        ax6.set_title('Final State Distribution')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_pops):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig
    
    def plot_spectral_analysis(self, wavelengths, solar_irradiance, 
                              transmission_funcs, R_opv, R_psu,
                              title="Spectral Analysis and Filtering"):
        """
        Plot spectral analysis including solar spectrum, transmission, and responses.
        
        Mathematical Framework:
        The spectral analysis displays the key components of the agrivoltaic system:
        
        - Solar irradiance: S₀(λ) representing the standard AM1.5G solar spectrum
        - OPV transmission: T(λ) describing the spectrally selective filtering
          function of the organic photovoltaic layer
        - PSU response: R_PSU(λ) representing the photosynthetic unit's spectral sensitivity
        - OPV response: R_OPV(λ) representing the organic photovoltaic's spectral sensitivity
        
        The transmitted spectrum reaching the photosynthetic units is:
        S_transmitted(λ) = S₀(λ) * T(λ)
        
        The absorbed spectra determine the efficiency of both systems:
        S_absorbed_PSU(λ) = S_transmitted(λ) * R_PSU(λ)
        S_absorbed_OPV(λ) = S₀(λ) * [1 - T(λ)] * R_OPV(λ)
        
        Parameters:
        wavelengths (array): Wavelengths in nm
        solar_irradiance (array): Solar irradiance values
        transmission_funcs (list): List of transmission functions
        R_opv (array): OPV spectral response
        R_psu (array): PSU spectral response
        title (str): Plot title
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figsize_base[0], self.figsize_base[1]))
        
        # Solar spectrum and transmission
        ax1.plot(wavelengths, solar_irradiance, 'orange', linewidth=2, label='Solar Spectrum (AM1.5G)', alpha=0.7)
        for i, T in enumerate(transmission_funcs):
            label = f'Transmission (Optimized)' if i == len(transmission_funcs) - 1 else f'Transmission (Initial)'
            ax1.plot(wavelengths, T * np.max(solar_irradiance), linewidth=2, label=label, alpha=0.8)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title('Solar Spectrum and OPV Transmission')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spectral responses
        ax2.plot(wavelengths, R_psu, label='PSU Response', linewidth=2, color=self.colors['psu'])
        ax2.plot(wavelengths, R_opv, label='OPV Response', linewidth=2, color=self.colors['opv'])
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Spectral Response')
        ax2.set_title('Photosynthetic and Photovoltaic Responses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Transmitted vs absorbed spectra
        if len(transmission_funcs) > 0:
            T_final = transmission_funcs[-1]  # Use last (optimized) transmission
            transmitted = solar_irradiance * T_final
            absorbed_psu = transmitted * R_psu
            absorbed_opv = solar_irradiance * (1 - T_final) * R_opv
            
            ax3.plot(wavelengths, solar_irradiance, label='Incident Solar', linewidth=2, alpha=0.5)
            ax3.plot(wavelengths, transmitted, label='Transmitted to PSU', linewidth=2)
            ax3.plot(wavelengths, absorbed_psu, label='Absorbed by PSU', linewidth=2, linestyle='--')
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Spectral Intensity')
            ax3.set_title('Spectral Flow Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Quantum advantage regions
        if len(transmission_funcs) > 0:
            T_final = transmission_funcs[-1]
            # Calculate regions of enhanced quantum effects
            quantum_advantage = np.abs(np.gradient(T_final, wavelengths))  # High gradient = potential quantum advantage
            ax4.plot(wavelengths, quantum_advantage, linewidth=2, color='red')
            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Quantum Advantage Indicator')
            ax4.set_title('Spectral Regions for Quantum Enhancement')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig
    
    def plot_optimization_results(self, param_history, performance_history,
                                  title="Optimization Progress"):
        """
        Plot optimization results and convergence.
        
        Mathematical Framework:
        The optimization visualization shows the evolution of key parameters
        and performance metrics during the differential evolution optimization:
        
        Objective function: f(params) = α·PCE(params) + β·ETR(params) + γ·SPCE(params)
        
        where PCE is the Power Conversion Efficiency, ETR is the Electron
        Transport Rate, and SPCE is the Symbiotic Power Conversion Efficiency.
        
        The optimization seeks to maximize the combined objective while
        satisfying constraints on biodegradability, toxicity, and agricultural
        performance metrics.
        
        Parameters:
        param_history (list): History of parameter vectors
        performance_history (list): History of performance metrics
        title (str): Plot title
        """
        if not param_history or not performance_history:
            # Create an empty plot if no history
            fig, ax = plt.subplots(figsize=self.figsize_base)
            ax.text(0.5, 0.5, 'No optimization history available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Convert to numpy arrays
        params_array = np.array(param_history)
        perf_array = np.array(performance_history)
        
        n_params = params_array.shape[1] if len(params_array.shape) > 1 else 1
        n_metrics = perf_array.shape[1] if len(perf_array.shape) > 1 else 1
        
        fig, axes = plt.subplots(2, max(1, n_metrics or 1), 
                                figsize=(self.figsize_base[0], self.figsize_base[1]))
        if n_metrics == 1:
            axes = axes[:, np.newaxis] if n_metrics == 1 else axes
        
        # Plot parameter evolution
        generations = range(len(param_history))
        for i in range(min(n_params, 4)):  # Limit to first 4 parameters
            if n_params == 1:
                axes[0, 0].plot(generations, params_array, label=f'Parameter {i+1}', linewidth=2)
            else:
                axes[0, 0].plot(generations, params_array[:, i], label=f'Parameter {i+1}', linewidth=2)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Parameter Value')
        axes[0, 0].set_title('Parameter Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot performance metrics
        for i in range(n_metrics):
            metric_idx = i if len(perf_array.shape) > 1 else 0
            metric_data = perf_array[:, i] if len(perf_array.shape) > 1 else perf_array
            axes[1, i].plot(generations, metric_data, linewidth=2, color=self.colors.get(list(self.colors.keys())[i % len(self.colors)], 'blue'))
            axes[1, i].set_xlabel('Generation')
            axes[1, i].set_ylabel('Performance')
            axes[1, i].set_title(f'Metric {i+1}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig
    
    def plot_eco_design_analysis(self, eco_df, title="Eco-Design Analysis"):
        """
        Plot eco-design analysis results.
        
        Mathematical Framework:
        The eco-design analysis visualizes the multi-objective optimization
        of materials properties balancing performance with sustainability:
        
        Biodegradability index: B = Σᵢ f⁺(rᵢ) · wᵢ for enzymatic attack sites
        where f⁺(r) is the Fukui function for electrophilic attack.
        
        The eco-design objective combines:
        - Power conversion efficiency: PCE > 20%
        - Biodegradability: B > 80%
        - Low toxicity: LC50 > 400 mg/L
        - Agricultural performance: ETR_rel > 90%
        
        Parameters:
        eco_df (DataFrame): Eco-design results DataFrame
        title (str): Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize_base[0], self.figsize_base[1]))
        
        # Biodegradability vs PCE
        if 'biodegradability' in eco_df.columns and 'pce_potential' in eco_df.columns:
            scatter = axes[0, 0].scatter(eco_df['biodegradability'], eco_df['pce_potential'],
                                       c=eco_df.get('multi_objective_score', range(len(eco_df))),
                                       cmap='viridis', alpha=0.7, s=60)
            axes[0, 0].set_xlabel('Biodegradability')
            axes[0, 0].set_ylabel('PCE Potential')
            axes[0, 0].set_title('Biodegradability vs PCE')
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0])
        
        # Distribution of eco-friendly candidates
        if 'biodegradability' in eco_df.columns:
            axes[0, 1].hist(eco_df['biodegradability'], bins=20, color=self.colors['psu'], alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Biodegradability')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Biodegradability Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Multi-objective scores
        if 'multi_objective_score' in eco_df.columns:
            scores = eco_df['multi_objective_score']
            axes[1, 0].hist(scores, bins=20, color=self.colors['opv'], alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Multi-Objective Score')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Multi-Objective Optimization Scores')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Eco-friendly candidates identification
        if all(col in eco_df.columns for col in ['biodegradability', 'pce_potential']):
            eco_threshold = eco_df[(eco_df['biodegradability'] > 0.7) & (eco_df['pce_potential'] > 0.12)]
            n_eco = len(eco_threshold)
            total = len(eco_df)
            
            labels = ['Eco-friendly', 'Standard']
            sizes = [n_eco, total - n_eco]
            colors_pie = [self.colors['psu'], 'lightgray']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Eco-Design Success Rate')
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig
    
    def plot_quantum_advantage_analysis(self, temperatures, temp_sensitivity, 
                                       disorder_strengths, disorder_sensitivity,
                                       title="Quantum Advantage Analysis"):
        """
        Plot quantum advantage analysis with robustness to parameters.
        
        Mathematical Framework:
        Quantum advantage quantification compares coherent (non-Markovian)
        and incoherent (Markovian) dynamics:
        
        η_quantum = (ETR_coherent / ETR_incoherent) - 1
        
        Robustness analysis examines sensitivity to:
        - Temperature: dETR/dT measuring thermal stability
        - Disorder: dETR/dσ measuring structural stability
        - Dephasing: dETR/dΓ measuring coherence preservation
        
        The quantum advantage factor indicates the enhancement due to
        quantum coherence effects in the energy transfer process.
        
        Parameters:
        temperatures (array): Temperature values in Kelvin
        temp_sensitivity (array): ETR values at different temperatures
        disorder_strengths (array): Disorder strength values
        disorder_sensitivity (array): ETR values at different disorder strengths
        title (str): Plot title
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figsize_base[0], self.figsize_base[1]))
        
        # Temperature sensitivity
        ax1.plot(temperatures, temp_sensitivity, 'r.-', linewidth=2, markersize=8, label='Temperature Sensitivity')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('ETR per Photon')
        ax1.set_title('Temperature Sensitivity Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Disorder sensitivity
        ax2.plot(disorder_strengths, disorder_sensitivity, 'b.--', linewidth=2, markersize=8, label='Disorder Sensitivity')
        ax2.set_xlabel('Disorder Strength (cm⁻¹)')
        ax2.set_ylabel('ETR per Photon')
        ax2.set_title('Disorder Sensitivity Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Quantum advantage heatmap (conceptual)
        # Create a synthetic heatmap showing quantum advantage over parameter space
        temp_grid, dis_grid = np.meshgrid(temperatures[::max(1, len(temperatures)//10)], 
                                         disorder_strengths[::max(1, len(disorder_strengths)//10)])
        # Simulated quantum advantage - higher at moderate temperature and low disorder
        qa_grid = np.exp(-((temp_grid - 295)**2)/5000) * np.exp(-dis_grid/20)
        
        im = ax3.contourf(temp_grid, dis_grid, qa_grid, levels=20, cmap='plasma')
        ax3.set_xlabel('Temperature (K)')
        ax3.set_ylabel('Disorder Strength (cm⁻¹)')
        ax3.set_title('Quantum Advantage Heatmap')
        plt.colorbar(im, ax=ax3)
        
        # Statistical summary
        if len(temp_sensitivity) > 1 and len(disorder_sensitivity) > 1:
            ax4.boxplot([temp_sensitivity, disorder_sensitivity], labels=['Temperature', 'Disorder'])
            ax4.set_ylabel('ETR per Photon')
            ax4.set_title('Sensitivity Distribution Comparison')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        return fig
    
    def plot_all_results_summary(self, results_dict, title="Comprehensive Results Summary"):
        """
        Create a comprehensive summary plot of all results.
        
        Parameters:
        results_dict (dict): Dictionary containing all simulation results
        title (str): Plot title
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Quantum dynamics summary
        if all(k in results_dict for k in ['time_points', 'populations', 'coherences']):
            ax1 = fig.add_subplot(gs[0, 0])
            time_points = results_dict['time_points']
            pops = results_dict['populations']
            coherences = results_dict['coherences']
            
            total_pop = np.sum(pops, axis=1) if pops.ndim > 1 else pops
            ax1.plot(time_points, total_pop, 'k-', linewidth=2, label='Total Population')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_points, coherences, color=self.colors['coherence'], linewidth=2, label='Coherence')
            
            ax1.set_xlabel('Time (fs)')
            ax1.set_ylabel('Population', color='k')
            ax1_twin.set_ylabel('Coherence', color=self.colors['coherence'])
            ax1.set_title('Dynamics Summary')
            ax1.grid(True, alpha=0.3)
        
        # Performance metrics
        if 'pce' in results_dict and 'etr' in results_dict:
            ax2 = fig.add_subplot(gs[0, 1])
            metrics = ['PCE', 'ETR', 'SPCE']
            values = [results_dict.get('pce', 0), 
                     results_dict.get('etr', 0), 
                     results_dict.get('spce', 0)]
            bars = ax2.bar(metrics, values, color=[self.colors['opv'], self.colors['psu'], self.colors['etr']])
            ax2.set_ylabel('Efficiency')
            ax2.set_title('Performance Metrics')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                         f'{value:.3f}', ha='center', va='bottom')
        
        # Spectral analysis
        if all(k in results_dict for k in ['wavelengths', 'solar_irradiance', 'transmission']):
            ax3 = fig.add_subplot(gs[0, 2])
            wavelengths = results_dict['wavelengths']
            solar = results_dict['solar_irradiance']
            T = results_dict['transmission']
            
            ax3.plot(wavelengths, solar, label='Solar', linewidth=2, alpha=0.7)
            ax3.plot(wavelengths, solar * T, label='Transmitted', linewidth=2)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Intensity')
            ax3.set_title('Spectral Summary')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Quantum metrics
        quantum_metrics = {}
        for key in ['qfi', 'entropy', 'purity', 'concurrence']:
            if key in results_dict:
                if hasattr(results_dict[key], '__len__') and len(results_dict[key]) > 0:
                    quantum_metrics[key] = np.mean(results_dict[key][-10:])  # Last 10 values average
                else:
                    quantum_metrics[key] = results_dict[key]
        
        if quantum_metrics:
            ax4 = fig.add_subplot(gs[1, :2])
            metrics_names = list(quantum_metrics.keys())
            metrics_values = list(quantum_metrics.values())
            bars = ax4.bar(metrics_names, metrics_values, 
                          color=[self.colors.get(m, 'gray') for m in metrics_names])
            ax4.set_ylabel('Average Value')
            ax4.set_title('Quantum Metrics Summary')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, metrics_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                         f'{value:.3f}', ha='center', va='bottom')
        
        # Eco-design results
        if 'eco_candidates' in results_dict and results_dict['eco_candidates']:
            eco_data = results_dict['eco_candidates']
            if isinstance(eco_data, list) and len(eco_data) > 0:
                ax5 = fig.add_subplot(gs[1, 2])
                biodegradability = [c.get('biodegradability', 0) for c in eco_data]
                pce_potential = [c.get('pce_potential', 0) for c in eco_data]
                
                ax5.scatter(biodegradability, pce_potential, alpha=0.7)
                ax5.set_xlabel('Biodegradability')
                ax5.set_ylabel('PCE Potential')
                ax5.set_title('Eco-Design Candidates')
                ax5.grid(True, alpha=0.3)
        
        # Parameter sensitivity
        if 'robustness' in results_dict:
            robustness = results_dict['robustness']
            if 'temperature_sensitivity' in robustness and 'disorder_sensitivity' in robustness:
                ax6 = fig.add_subplot(gs[2, 0])
                temp_data = robustness['temperature_sensitivity']
                dis_data = robustness['disorder_sensitivity']
                
                ax6.plot(range(len(temp_data)), temp_data, label='Temperature', linewidth=2)
                ax6.plot(range(len(dis_data)), dis_data, label='Disorder', linewidth=2)
                ax6.set_xlabel('Parameter Index')
                ax6.set_ylabel('Sensitivity')
                ax6.set_title('Robustness Analysis')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        # Final performance summary
        ax7 = fig.add_subplot(gs[2, 1:])
        performance_items = []
        performance_values = []
        
        for key, name in [('pce', 'Power Conversion Efficiency'), 
                         ('etr', 'Electron Transport Rate'),
                         ('spce', 'Symbiotic Efficiency'),
                         ('quantum_advantage', 'Quantum Advantage')]:
            if key in results_dict:
                performance_items.append(name)
                performance_values.append(results_dict[key])
        
        if performance_items:
            bars = ax7.bar(performance_items, performance_values, 
                          color=[self.colors.get('opv'), self.colors.get('psu'), 
                                self.colors.get('etr'), 'purple'])
            ax7.set_ylabel('Performance')
            ax7.set_title('Final Performance Summary')
            ax7.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax7.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add value labels
            for bar, value in zip(bars, performance_values):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                         f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        return fig

    def save_data_to_csv(self, data, filename, subdir=None):
        """
        Save data to CSV file in the data directory.
        
        Parameters:
        data (DataFrame or array-like): Data to save
        filename (str): Name of the file to save
        subdir (str): Subdirectory within data_dir to save to (optional)
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        # Create the full path
        if subdir:
            full_dir = os.path.join(self.data_dir, subdir)
        else:
            full_dir = self.data_dir
        os.makedirs(full_dir, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(filename)
        if ext.lower() != '.csv':
            filename = base_name + '.csv'
        else:
            base_name = base_name  # Remove .csv from base_name to add timestamp before it
            filename = f"{base_name}_{timestamp}.csv"
        
        filepath = os.path.join(full_dir, filename)
        
        # Convert data to DataFrame if needed and save
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            # Convert array-like data to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        
        print(f"Saved data to {filepath}")
        return filepath

    def save_figure(self, fig, filename, folder=None, dpi=300, bbox_inches='tight'):
        """
        Save figure to specified folder.
        
        Parameters:
        fig (matplotlib.figure): Figure object to save
        filename (str): Name of the file to save
        folder (str): Folder to save the figure in (if None, uses self.figures_dir)
        dpi (int): Resolution of the saved figure
        bbox_inches (str): Bounding box setting for tight layout
        """
        import os
        # Use default figures directory if no folder specified
        if folder is None:
            folder = self.figures_dir
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure to {filepath}")
        
        # Also save as PDF if needed
        pdf_path = os.path.splitext(filepath)[0] + '.pdf'
        fig.savefig(pdf_path, bbox_inches=bbox_inches)
        print(f"Saved figure as PDF to {pdf_path}")