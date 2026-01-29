"""
Figure Generator for Quantum Agrivoltaics Simulations

This module implements visualization tools for quantum agrivoltaics simulation results,
creating publication-quality figures for analysis and reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # For publication-quality plots
from matplotlib.colors import LinearSegmentedColormap


class FigureGenerator:
    """
    Figure generator for quantum agrivoltaics simulation visualizations.
    
    This class provides methods for creating publication-quality figures
    from quantum agrivoltaics simulation data, including quantum dynamics,
    spectral optimization results, and eco-design analysis.
    """
    
    def __init__(self, figures_dir='figures'):
        """
        Initialize the figure generator.
        
        Mathematical Framework:
        The figure generator creates visualizations that represent the
        mathematical relationships in quantum agrivoltaics systems. Plots
        are designed to clearly show quantum effects, efficiency metrics,
        and optimization results while maintaining scientific accuracy.
        
        Parameters:
        figures_dir (str): Directory to save figures (default: 'figures')
        """
        # Set publication style plots
        plt.style.use(['science', 'notebook'])
        self.figures_dir = figures_dir
        import os
        os.makedirs(figures_dir, exist_ok=True)
    
    def plot_quantum_dynamics(self, time_points, populations, coherences, 
                            qfi_values, etr_time, title="Quantum Dynamics in FMO Complex"):
        """
        Plot quantum dynamics results.
        
        Mathematical Framework:
        The plot shows the time evolution of quantum states:
        
        - Populations: ρ_ii(t) representing site occupations
        - Coherences: |ρ_ij(t)| representing quantum coherence between sites
        - QFI: F_Q(t) representing parameter sensitivity
        - ETR: Electron transport rate over time
        
        Parameters:
        time_points (array): Time points in fs
        populations (2D array): Site populations over time
        coherences (3D array): Coherence matrices over time
        qfi_values (array): QFI values over time
        etr_time (array): ETR values over time
        title (str): Title for the plot
        """
        n_sites = populations.shape[1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Populations over time
        ax1 = axes[0, 0]
        for i in range(min(n_sites, 5)):  # Limit to first 5 sites for clarity
            ax1.plot(time_points, populations[:, i], label=f'Site {i}', linewidth=2)
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Population')
        ax1.set_title('Site Populations vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coherences over time (magnitude of off-diagonal elements)
        ax2 = axes[0, 1]
        if coherences.size > 0 and n_sites > 1:
            # Calculate average coherence magnitude
            coherence_mags = np.zeros(len(time_points))
            count = 0
            for i in range(n_sites):
                for j in range(i+1, n_sites):
                    coherence_mags += np.abs(coherences[:, i, j])
                    count += 1
            if count > 0:
                coherence_mags /= count
                ax2.plot(time_points, coherence_mags, 'r-', linewidth=2)
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Average Coherence Magnitude')
        ax2.set_title('Quantum Coherences vs Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: QFI over time
        ax3 = axes[1, 0]
        ax3.plot(time_points, qfi_values, 'g-', linewidth=2)
        ax3.set_xlabel('Time (fs)')
        ax3.set_ylabel('Quantum Fisher Information')
        ax3.set_title('QFI vs Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: ETR over time
        ax4 = axes[1, 1]
        ax4.plot(time_points, etr_time, 'm-', linewidth=2)
        ax4.set_xlabel('Time (fs)')
        ax4.set_ylabel('ETR (relative units)')
        ax4.set_title('Electron Transport Rate vs Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_spectral_optimization(self, wavelengths, initial_transmission, optimized_transmission,
                                 opv_response, psu_response, title="Spectral Optimization Results"):
        """
        Plot spectral optimization results.
        
        Mathematical Framework:
        The plot shows the spectral transmission function that optimizes
        the split between OPV and PSU:
        
        I_sun(λ) → T_opt(λ) → I_OPV(λ) = I_sun(λ)*T_opt(λ)
                    ↓
              [1-T_opt(λ)]
                    ↓
              I_PSU(λ) = I_sun(λ)*[1-T_opt(λ)]
        
        Parameters:
        wavelengths (array): Wavelength array in nm
        initial_transmission (array): Initial transmission function
        optimized_transmission (array): Optimized transmission function
        opv_response (array): OPV response function
        psu_response (array): PSU response function
        title (str): Title for the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Transmission functions
        ax1 = axes[0]
        ax1.plot(wavelengths, initial_transmission, 'b--', label='Initial Transmission', linewidth=2)
        ax1.plot(wavelengths, optimized_transmission, 'r-', label='Optimized Transmission', linewidth=2)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Transmission')
        ax1.set_title('Spectral Transmission Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: System responses
        ax2 = axes[1]
        ax2.plot(wavelengths, opv_response, 'orange', label='OPV Response', linewidth=2)
        ax2.plot(wavelengths, psu_response, 'green', label='PSU Response', linewidth=2)
        ax2.fill_between(wavelengths, 0, 1, 
                         where=optimized_transmission >= 0.5, 
                         alpha=0.3, color='orange', label='OPV Dominant Region')
        ax2.fill_between(wavelengths, 0, 1, 
                         where=optimized_transmission < 0.5, 
                         alpha=0.3, color='green', label='PSU Dominant Region')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Response')
        ax2.set_title('System Responses and Spectral Regions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_eco_design_analysis(self, eco_df, title="Eco-Design Analysis"):
        """
        Plot eco-design analysis results.
        
        Mathematical Framework:
        The plot shows sustainability metrics:
        
        Sustainability = w₁*biodegradability + w₂*PCE_potential + w₃*(1-toxicity) + w₄*resource_efficiency
        
        Parameters:
        eco_df (DataFrame): DataFrame with eco-design results
        title (str): Title for the plot
        """
        if eco_df.empty:
            print("No eco-design data to plot")
            return None
        
        n_materials = len(eco_df)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Biodegradability vs PCE Potential
        ax1 = axes[0, 0]
        if 'biodegradability' in eco_df.columns and 'pce_potential' in eco_df.columns:
            ax1.scatter(eco_df['biodegradability'], eco_df['pce_potential'], 
                       c=eco_df.get('multi_objective_score', range(n_materials)), 
                       cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('Biodegradability')
            ax1.set_ylabel('PCE Potential')
            ax1.set_title('Biodegradability vs PCE Potential')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sustainability metrics comparison
        ax2 = axes[0, 1]
        if all(col in eco_df.columns for col in ['biodegradability', 'pce_potential', 'toxicity', 'resource_efficiency']):
            x = np.arange(n_materials)
            width = 0.2
            ax2.bar(x - 1.5*width, eco_df['biodegradability'], width, label='Biodegradability', alpha=0.8)
            ax2.bar(x - 0.5*width, eco_df['pce_potential'], width, label='PCE Potential', alpha=0.8)
            ax2.bar(x + 0.5*width, 1-eco_df['toxicity'], width, label='(1-Toxicity)', alpha=0.8)  # Invert toxicity
            ax2.bar(x + 1.5*width, eco_df['resource_efficiency'], width, label='Resource Efficiency', alpha=0.8)
            ax2.set_xlabel('Material Index')
            ax2.set_ylabel('Metric Value')
            ax2.set_title('Sustainability Metrics Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Multi-objective scores
        ax3 = axes[1, 0]
        if 'multi_objective_score' in eco_df.columns:
            ax3.bar(range(n_materials), eco_df['multi_objective_score'], alpha=0.7)
            ax3.set_xlabel('Material Index')
            ax3.set_ylabel('Multi-Objective Score')
            ax3.set_title('Multi-Objective Sustainability Scores')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Material names vs scores (if names available)
        ax4 = axes[1, 1]
        if 'material_name' in eco_df.columns and 'multi_objective_score' in eco_df.columns:
            ax4.barh(range(n_materials), eco_df['multi_objective_score'])
            ax4.set_yticks(range(n_materials))
            ax4.set_yticklabels(eco_df['material_name'], fontsize=8)
            ax4.set_xlabel('Multi-Objective Score')
            ax4.set_title('Materials Ranked by Sustainability')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_quantum_metrics_extended(self, time_points, entropy_values, purity_values,
                                    linear_entropy_values, bipartite_ent_values,
                                    multipartite_ent_values, pairwise_concurrence_values,
                                    title="Extended Quantum Metrics"):
        """
        Plot extended quantum metrics.
        
        Mathematical Framework:
        The plot shows various quantum information measures:
        
        - Von Neumann entropy: S = -Tr[ρ log ρ] (information content)
        - Purity: P = Tr[ρ²] (mixedness measure)
        - Linear entropy: S_L = (d/(d-1))(1-Tr[ρ²]) (approximation)
        - Bipartite entanglement: E = S(ρ_A) (subsystem entropy)
        - Multipartite entanglement: Average bipartite measures
        - Pairwise concurrence: C_ij (pairwise entanglement)
        
        Parameters:
        time_points (array): Time points in fs
        entropy_values (array): Von Neumann entropy over time
        purity_values (array): Purity over time
        linear_entropy_values (array): Linear entropy over time
        bipartite_ent_values (array): Bipartite entanglement over time
        multipartite_ent_values (array): Multipartite entanglement over time
        pairwise_concurrence_values (array): Pairwise concurrence over time
        title (str): Title for the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Von Neumann Entropy
        ax1 = axes[0, 0]
        ax1.plot(time_points, entropy_values, 'b-', linewidth=2)
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Von Neumann Entropy')
        ax1.set_title('Entropy vs Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Purity
        ax2 = axes[0, 1]
        ax2.plot(time_points, purity_values, 'r-', linewidth=2)
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Purity')
        ax2.set_title('Purity vs Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Linear Entropy
        ax3 = axes[0, 2]
        ax3.plot(time_points, linear_entropy_values, 'g-', linewidth=2)
        ax3.set_xlabel('Time (fs)')
        ax3.set_ylabel('Linear Entropy')
        ax3.set_title('Linear Entropy vs Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Bipartite Entanglement
        ax4 = axes[1, 0]
        ax4.plot(time_points, bipartite_ent_values, 'm-', linewidth=2)
        ax4.set_xlabel('Time (fs)')
        ax4.set_ylabel('Bipartite Entanglement')
        ax4.set_title('Bipartite Entanglement vs Time')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Multipartite Entanglement
        ax5 = axes[1, 1]
        ax5.plot(time_points, multipartite_ent_values, 'c-', linewidth=2)
        ax5.set_xlabel('Time (fs)')
        ax5.set_ylabel('Multipartite Entanglement')
        ax5.set_title('Multipartite Entanglement vs Time')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Pairwise Concurrence
        ax6 = axes[1, 2]
        ax6.plot(time_points, pairwise_concurrence_values, 'y-', linewidth=2)
        ax6.set_xlabel('Time (fs)')
        ax6.set_ylabel('Pairwise Concurrence')
        ax6.set_title('Pairwise Concurrence vs Time')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_robustness_analysis(self, robustness_data, title="Robustness Analysis"):
        """
        Plot robustness analysis results.
        
        Mathematical Framework:
        The plot shows sensitivity to environmental parameters:
        
        S_T = |∂(performance)/∂(temperature)| (temperature sensitivity)
        S_D = |∂(performance)/∂(disorder)|   (disorder sensitivity)
        
        Robust systems show low sensitivity to parameter variations.
        
        Parameters:
        robustness_data (dict): Robustness analysis results
        title (str): Title for the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Temperature sensitivity
        ax1 = axes[0]
        if 'temperature_sensitivity' in robustness_data and 'temperatures' in robustness_data:
            ax1.plot(robustness_data['temperatures'], robustness_data['temperature_sensitivity'], 
                    'r.-', linewidth=2, markersize=8)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Performance Metric')
            ax1.set_title('Temperature Sensitivity Analysis')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Disorder sensitivity
        ax2 = axes[1]
        if 'disorder_sensitivity' in robustness_data and 'disorder_strengths' in robustness_data:
            ax2.plot(robustness_data['disorder_strengths'], robustness_data['disorder_sensitivity'], 
                    'b.--', linewidth=2, markersize=8)
            ax2.set_xlabel('Disorder Strength (cm⁻¹)')
            ax2.set_ylabel('Performance Metric')
            ax2.set_title('Disorder Sensitivity Analysis')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_spectral_transmission_heatmap(self, lambda_range, transmission_profiles, 
                                         title="Spectral Transmission Heatmap"):
        """
        Plot a heatmap of transmission profiles.
        
        Mathematical Framework:
        The heatmap visualizes how different transmission functions
        affect various wavelength regions. Each row represents a
        different transmission profile, and each column a wavelength.
        
        Parameters:
        lambda_range (array): Wavelength range in nm
        transmission_profiles (list): List of transmission functions
        title (str): Title for the plot
        """
        if not transmission_profiles:
            print("No transmission profiles to plot")
            return None
        
        # Create 2D array for heatmap
        n_profiles = len(transmission_profiles)
        n_wavelengths = len(lambda_range)
        
        transmission_matrix = np.zeros((n_profiles, n_wavelengths))
        for i, profile in enumerate(transmission_profiles):
            if hasattr(profile, '__call__'):
                transmission_matrix[i, :] = profile(lambda_range)
            else:
                transmission_matrix[i, :] = profile
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(transmission_matrix, aspect='auto', cmap='viridis',
                       extent=[lambda_range[0], lambda_range[-1], n_profiles, 0])
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transmission Profile Index')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transmission')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_figure(self, fig, filename, dpi=300, bbox_inches='tight'):
        """
        Save a figure to file.
        
        Parameters:
        fig (Figure): Matplotlib figure object
        filename (str): Output filename
        dpi (int): Resolution in dots per inch
        bbox_inches (str): Bounding box setting
        """
        import os
        if fig is not None:
            # If filename doesn't contain a directory, save to default figures directory
            if os.path.dirname(filename) == '':
                filepath = os.path.join(self.figures_dir, filename)
            else:
                filepath = filename
            
            fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
            print(f"  Saved figure to: {filepath}")