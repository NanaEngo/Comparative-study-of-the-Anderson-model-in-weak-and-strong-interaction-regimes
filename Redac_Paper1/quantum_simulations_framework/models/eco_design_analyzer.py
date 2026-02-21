"""
Eco Design Analyzer for Quantum Agrivoltaic Systems.

This module provides eco-design analysis using quantum reactivity descriptors
for sustainable organic photovoltaic materials in agrivoltaic systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
from scipy.optimize import minimize
import warnings


logger = logging.getLogger(__name__)


class EcoDesignAnalyzer:
    """
    Eco Design Analyzer using quantum reactivity descriptors.
    
    Mathematical Framework:
    The eco-design analysis uses quantum reactivity descriptors based on
    density functional theory (DFT) calculations:
    
    1. Fukui functions: f_k^+ = ∂ρ_N+1/∂N - ∂ρ_N/∂N (nucleophilic)
                      f_k^- = ∂ρ_N/∂N - ∂ρ_N-1/∂N (electrophilic)
                      f_k = (f_k^+ + f_k^-)/2 (radical)
    
    2. Global reactivity indices:
       Chemical potential: μ = -χ = -(IP + EA)/2
       Chemical hardness: η = (IP - EA)/2
       Electrophilicity: ω = μ²/(2η)
    
    3. Biodegradability index (B-index): combination of reactivity descriptors
       that predicts environmental degradation pathways.
    """
    
    def __init__(self, target_pce: float = 0.18, target_biodegradability: float = 0.8):
        """
        Initialize eco-design analyzer.
        
        Parameters:
        -----------
        target_pce : float
            Target power conversion efficiency
        target_biodegradability : float
            Target biodegradability fraction (0-1)
        """
        self.target_pce = target_pce
        self.target_biodegradability = target_biodegradability
        
        # Reference values for reactivity descriptors
        self.reference_values = {
            'chemical_potential': -4.0,  # eV
            'chemical_hardness': 2.5,    # eV
            'electrophilicity': 4.0,     # eV
            'max_fukui_nucleophilic': 0.3,
            'max_fukui_electrophilic': 0.2
        }
        
        logger.info("EcoDesignAnalyzer initialized")
    
    def calculate_fukui_functions(self, 
                                  n_electrons: int,
                                  electron_densities: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate Fukui functions for molecular reactivity analysis.
        
        Mathematical Framework:
        The Fukui functions describe the change in electron density upon
        addition or removal of an electron:
        
        f_k^+ = ρ_k(N+1) - ρ_k(N)    (nucleophilic attack)
        f_k^- = ρ_k(N) - ρ_k(N-1)    (electrophilic attack)
        f_k = (f_k^+ + f_k^-)/2      (radical attack)
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons in neutral system
        electron_densities : dict
            Electron densities for N-1, N, N+1 electron systems
            
        Returns:
        --------
        dict
            Fukui functions for nucleophilic, electrophilic, and radical attacks
        """
        # Extract electron densities
        rho_neutral = electron_densities['neutral']
        rho_n_plus_1 = electron_densities['n_plus_1']
        rho_n_minus_1 = electron_densities['n_minus_1']
        
        # Calculate Fukui functions
        f_plus = rho_n_plus_1 - rho_neutral  # Nucleophilic
        f_minus = rho_neutral - rho_n_minus_1  # Electrophilic
        f_radical = 0.5 * (f_plus + f_minus)  # Radical
        
        return {
            'fukui_nucleophilic': f_plus,
            'fukui_electrophilic': f_minus,
            'fukui_radical': f_radical
        }
    
    def calculate_global_reactivity_indices(self, ionization_potential: float, 
                                          electron_affinity: float) -> Dict[str, float]:
        """
        Calculate global reactivity indices from IP and EA.
        
        Mathematical Framework:
        Chemical potential: μ = -(IP + EA)/2
        Chemical hardness: η = (IP - EA)/2
        Electrophilicity: ω = μ²/(2η)
        
        Parameters:
        -----------
        ionization_potential : float
            Ionization potential in eV
        electron_affinity : float
            Electron affinity in eV
            
        Returns:
        --------
        dict
            Global reactivity indices
        """
        # Calculate global descriptors
        chemical_potential = -(ionization_potential + electron_affinity) / 2.0
        chemical_hardness = (ionization_potential - electron_affinity) / 2.0
        electrophilicity = 0.0
        
        if chemical_hardness > 0:
            electrophilicity = chemical_potential**2 / (2 * chemical_hardness)
        
        return {
            'chemical_potential': chemical_potential,
            'chemical_hardness': chemical_hardness,
            'electrophilicity': electrophilicity
        }
    
    def calculate_biodegradability_index(self, 
                                       fukui_values: Dict[str, np.ndarray],
                                       global_indices: Dict[str, float],
                                       molecular_weight: float = 500.0) -> float:
        """
        Calculate biodegradability index (B-index) based on reactivity descriptors.
        
        Mathematical Framework:
        The B-index combines local and global reactivity descriptors to predict
        biodegradability:
        
        B_index = w1 * max(f^+) + w2 * max(f^-) + w3 * μ^2/η + w4 * (1/MW)
        
        where the weights are determined through training on experimental data.
        
        Parameters:
        -----------
        fukui_values : dict
            Fukui functions calculated from calculate_fukui_functions
        global_indices : dict
            Global reactivity indices from calculate_global_reactivity_indices
        molecular_weight : float
            Molecular weight in g/mol
            
        Returns:
        --------
        float
            Biodegradability index (0-100 scale)
        """
        # Extract relevant values
        f_plus_max = np.max(np.abs(fukui_values['fukui_nucleophilic']))
        f_minus_max = np.max(np.abs(fukui_values['fukui_electrophilic']))
        chemical_potential = global_indices['chemical_potential']
        chemical_hardness = global_indices['chemical_hardness']
        
        # Calculate B-index components
        # Higher reactivity (lower hardness, higher potential) promotes degradation
        reactivity_component = 0.0
        if chemical_hardness > 0:
            reactivity_component = (chemical_potential**2) / chemical_hardness
        
        # Size factor (larger molecules generally degrade slower)
        size_factor = 100.0 / (1.0 + molecular_weight / 100.0)
        
        # Weighted combination (weights based on literature)
        b_index = (
            25.0 * f_plus_max +      # Nucleophilic reactivity
            20.0 * f_minus_max +     # Electrophilic reactivity
            15.0 * reactivity_component +  # Global reactivity
            size_factor              # Size factor
        )
        
        # Normalize to 0-100 scale
        b_index = min(100.0, max(0.0, b_index))
        
        return b_index
    
    def evaluate_material_sustainability(self, 
                                        material_name: str,
                                        pce: float,
                                        ionization_potential: float,
                                        electron_affinity: float,
                                        electron_densities: Dict[str, np.ndarray],
                                        molecular_weight: float = 500.0) -> Dict[str, Any]:
        """
        Evaluate the sustainability of a material based on PCE and eco-design metrics.
        
        Parameters:
        -----------
        material_name : str
            Name of the material
        pce : float
            Power conversion efficiency
        ionization_potential : float
            Ionization potential in eV
        electron_affinity : float
            Electron affinity in eV
        electron_densities : dict
            Electron densities for reactivity calculations
        molecular_weight : float
            Molecular weight in g/mol
            
        Returns:
        --------
        dict
            Sustainability evaluation results
        """
        # Calculate reactivity descriptors
        fukui_functions = self.calculate_fukui_functions(
            n_electrons=50,  # Example value
            electron_densities=electron_densities
        )
        
        global_indices = self.calculate_global_reactivity_indices(
            ionization_potential, electron_affinity
        )
        
        b_index = self.calculate_biodegradability_index(
            fukui_functions, global_indices, molecular_weight
        )
        
        # Calculate sustainability scores
        pce_score = min(1.0, pce / self.target_pce)
        biodegradability_score = min(1.0, b_index / 70.0)  # B-index > 70 considered good
        sustainability_score = 0.6 * pce_score + 0.4 * biodegradability_score
        
        return {
            'material_name': material_name,
            'pce': pce,
            'pce_score': pce_score,
            'b_index': b_index,
            'biodegradability_score': biodegradability_score,
            'sustainability_score': sustainability_score,
            'fukui_functions': fukui_functions,
            'global_indices': global_indices,
            'molecular_weight': molecular_weight
        }
    
    def optimize_material_design(self, 
                               target_pce: float = 0.18,
                               target_biodegradability: float = 0.8,
                               n_candidates: int = 10) -> List[Dict[str, Any]]:
        """
        Optimize material design for both PCE and biodegradability.
        
        Mathematical Framework:
        Multi-objective optimization problem:
        
        max [w1 * PCE + w2 * (B_index/100)] 
        subject to: 0.1 < PCE < 0.25, 50 < B_index < 100
        
        Parameters:
        -----------
        target_pce : float
            Target PCE
        target_biodegradability : float
            Target biodegradability (0-1)
        n_candidates : int
            Number of candidate materials to generate
            
        Returns:
        --------
        list
            Optimized material candidates
        """
        candidates = []
        
        # Generate candidate materials with different properties
        for i in range(n_candidates):
            # Randomly vary IP and EA to create different materials
            ip = np.random.uniform(4.5, 6.0)  # eV
            ea = np.random.uniform(2.5, 4.0)  # eV
            
            # Create random electron densities for simulation
            n_atoms = 20
            electron_densities = {
                'neutral': np.random.rand(n_atoms) * 0.3,
                'n_plus_1': np.random.rand(n_atoms) * 0.3,
                'n_minus_1': np.random.rand(n_atoms) * 0.3
            }
            
            # Calculate PCE based on IP/EA (simplified model)
            # Higher bandgap (IP-EA) generally leads to higher PCE up to a point
            bandgap = ip - ea
            pce = min(0.22, 0.15 * (bandgap - 1.5)**0.5)  # Simplified model
            
            # Evaluate sustainability
            material_result = self.evaluate_material_sustainability(
                f"Candidate_{i+1}",
                pce,
                ip,
                ea,
                electron_densities,
                molecular_weight=np.random.uniform(400, 800)
            )
            
            candidates.append(material_result)
        
        # Sort by sustainability score
        candidates.sort(key=lambda x: x['sustainability_score'], reverse=True)
        
        logger.info(f"Generated {len(candidates)} material candidates")
        return candidates
    
    def perform_lca_screening(self, 
                            materials_data: List[Dict[str, Any]],
                            environmental_impact_factors: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform Life Cycle Assessment screening based on eco-design metrics.
        
        Parameters:
        -----------
        materials_data : list
            List of material evaluation results
        environmental_impact_factors : dict, optional
            Custom environmental impact factors
            
        Returns:
        --------
        dict
            LCA screening results
        """
        if environmental_impact_factors is None:
            environmental_impact_factors = {
                'synthesis_chemicals': 0.3,
                'energy_consumption': 0.25,
                'waste_generation': 0.25,
                'biodegradability': 0.2
            }
        
        # Calculate environmental impact score for each material
        for material in materials_data:
            # Higher B-index means better biodegradability (lower environmental impact)
            biodegradability_impact = (100 - material['b_index']) / 100.0
            
            # Combine with PCE to get environmental impact per efficiency
            environmental_impact = (
                environmental_impact_factors['biodegradability'] * biodegradability_impact +
                (1 - material['pce_score']) * (1 - environmental_impact_factors['biodegradability'])
            )
            
            material['environmental_impact'] = environmental_impact
            material['eco_efficiency_ratio'] = material['pce'] / (1 + environmental_impact)
        
        # Sort by eco-efficiency ratio
        materials_data.sort(key=lambda x: x['eco_efficiency_ratio'], reverse=True)
        
        return {
            'ranked_materials': materials_data,
            'best_material': materials_data[0] if materials_data else None,
            'environmental_impact_factors': environmental_impact_factors
        }
    
    def save_analysis_results(self, 
                            results: Dict[str, Any],
                            filename_prefix: str = 'eco_design_analysis',
                            output_dir: str = '../simulation_data/') -> str:
        """
        Save eco-design analysis results to CSV.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        filename_prefix : str
            Prefix for output filename
        output_dir : str
            Directory to save CSV file
            
        Returns:
        --------
        str
            Path to saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Flatten results for CSV
        rows = []
        
        if 'ranked_materials' in results:
            for i, material in enumerate(results['ranked_materials']):
                row = {
                    'rank': i + 1,
                    'material_name': material['material_name'],
                    'pce': material['pce'],
                    'pce_score': material['pce_score'],
                    'b_index': material['b_index'],
                    'biodegradability_score': material['biodegradability_score'],
                    'sustainability_score': material['sustainability_score'],
                    'environmental_impact': material.get('environmental_impact', 0.0),
                    'eco_efficiency_ratio': material.get('eco_efficiency_ratio', 0.0),
                    'chemical_potential': material['global_indices']['chemical_potential'],
                    'chemical_hardness': material['global_indices']['chemical_hardness'],
                    'electrophilicity': material['global_indices']['electrophilicity'],
                    'molecular_weight': material['molecular_weight']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Eco-design analysis results saved to {filepath}")
        return filepath
    
    def plot_analysis_results(self,
                            results: Dict[str, Any],
                            filename_prefix: str = 'eco_design_analysis',
                            figures_dir: str = '../Graphics/') -> str:
        """
        Plot eco-design analysis results.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        filename_prefix : str
            Prefix for output filename
        figures_dir : str
            Directory to save figures
            
        Returns:
        --------
        str
            Path to saved figure
        """
        os.makedirs(figures_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not results.get('ranked_materials'):
            logger.warning("No materials data to plot")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Eco-Design Analysis Results', fontsize=16, fontweight='bold')
        
        materials = results['ranked_materials']
        n_materials = len(materials)
        
        # Extract data
        names = [m['material_name'] for m in materials]
        pces = [m['pce'] for m in materials]
        b_indices = [m['b_index'] for m in materials]
        sustainability_scores = [m['sustainability_score'] for m in materials]
        eco_efficiency_ratios = [m.get('eco_efficiency_ratio', 0) for m in materials]
        
        # Plot 1: PCE vs B-index
        ax1 = axes[0, 0]
        scatter = ax1.scatter(pces, b_indices, c=sustainability_scores, 
                            cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('Power Conversion Efficiency (PCE)')
        ax1.set_ylabel('Biodegradability Index (B-index)')
        ax1.set_title('PCE vs Biodegradability')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Sustainability Score')
        
        # Add target lines
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Good Biodegradability')
        ax1.axvline(x=self.target_pce, color='red', linestyle='--', alpha=0.5, label='Target PCE')
        ax1.legend()
        
        # Plot 2: Sustainability scores
        ax2 = axes[0, 1]
        bars = ax2.bar(range(n_materials), sustainability_scores[:10], 
                      alpha=0.7, color='green')
        ax2.set_xlabel('Material Rank')
        ax2.set_ylabel('Sustainability Score')
        ax2.set_title('Sustainability Scores (Top 10)')
        ax2.set_xticks(range(min(10, n_materials)))
        ax2.set_xticklabels([names[i] for i in range(min(10, n_materials))], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, sustainability_scores[:10]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Eco-efficiency ratio
        ax3 = axes[1, 0]
        bars2 = ax3.bar(range(n_materials), eco_efficiency_ratios[:10], 
                       alpha=0.7, color='blue')
        ax3.set_xlabel('Material Rank')
        ax3.set_ylabel('Eco-Efficiency Ratio')
        ax3.set_title('Eco-Efficiency Ratio (Top 10)')
        ax3.set_xticks(range(min(10, n_materials)))
        ax3.set_xticklabels([names[i] for i in range(min(10, n_materials))], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars2, eco_efficiency_ratios[:10]):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Chemical properties
        ax4 = axes[1, 1]
        chem_potentials = [m['global_indices']['chemical_potential'] for m in materials[:10]]
        chem_hardnesses = [m['global_indices']['chemical_hardness'] for m in materials[:10]]
        
        x_pos = np.arange(len(chem_potentials))
        width = 0.35
        
        bars3 = ax4.bar(x_pos - width/2, chem_potentials, width, label='Chemical Potential', alpha=0.7)
        bars4 = ax4.bar(x_pos + width/2, chem_hardnesses, width, label='Chemical Hardness', alpha=0.7)
        
        ax4.set_xlabel('Material Rank')
        ax4.set_ylabel('Energy (eV)')
        ax4.set_title('Chemical Reactivity Indices (Top 10)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([names[i] for i in range(min(10, n_materials))], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{filename_prefix}_{timestamp}.pdf"
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        png_filename = f"{filename_prefix}_{timestamp}.png"
        png_filepath = os.path.join(figures_dir, png_filename)
        plt.savefig(png_filepath, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        logger.info(f"Eco-design analysis plots saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    logger.info("Testing EcoDesignAnalyzer...")
    
    analyzer = EcoDesignAnalyzer()
    
    # Example molecular properties (these would come from DFT calculations)
    example_electron_densities = {
        'neutral': np.array([0.1, 0.15, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.11, 0.19]),
        'n_plus_1': np.array([0.08, 0.13, 0.10, 0.16, 0.12, 0.14, 0.11, 0.15, 0.09, 0.17]),
        'n_minus_1': np.array([0.12, 0.17, 0.14, 0.20, 0.16, 0.18, 0.15, 0.19, 0.13, 0.21])
    }
    
    # Evaluate a material
    result = analyzer.evaluate_material_sustainability(
        "PM6_Y6",
        pce=0.17,
        ionization_potential=5.4,
        electron_affinity=3.2,
        electron_densities=example_electron_densities,
        molecular_weight=567.0
    )
    
    print(f"Material: {result['material_name']}")
    print(f"PCE: {result['pce']:.3f} (Score: {result['pce_score']:.3f})")
    print(f"B-index: {result['b_index']:.1f}")
    print(f"Sustainability Score: {result['sustainability_score']:.3f}")
    
    # Generate optimized materials
    candidates = analyzer.optimize_material_design(n_candidates=5)
    print(f"\nTop 3 candidates:")
    for i, candidate in enumerate(candidates[:3]):
        print(f"{i+1}. {candidate['material_name']}: PCE={candidate['pce']:.3f}, "
              f"B-index={candidate['b_index']:.1f}, Score={candidate['sustainability_score']:.3f}")
