"""
LCAAnalyzer: Life Cycle Assessment analyzer for sustainable agrivoltaics.

This module provides comprehensive Life Cycle Assessment (LCA) tools for analyzing
the environmental impact of quantum agrivoltaic systems, including carbon footprint,
energy payback time, and sustainability metrics.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class LCAAnalyzer:
    """
    Life Cycle Assessment analyzer for quantum agrivoltaic systems.

    This class implements LCA methodologies to evaluate the environmental impact
    of organic photovoltaic (OPV) materials and agrivoltaic systems throughout
    their entire lifecycle, from raw material extraction to end-of-life disposal.

    Mathematical Framework:
    The LCA impact is calculated as:

    I_total = Σ_i (I_manufacturing + I_operation + I_end_of_life)

    where each phase includes:
    - Energy consumption (MJ)
    - Carbon emissions (kg CO2eq)
    - Material toxicity (dimensionless score)
    - Resource depletion (kg scarce materials)

    Key Metrics:
    1. Carbon Footprint: gCO2eq/kWh of electricity generated
    2. Energy Payback Time (EPBT): Years to recover manufacturing energy
    3. Energy Return on Investment (EROI): Ratio of energy output to input
    4. Material Toxicity: Cumulative toxicity score
    5. Resource Efficiency: Material utilization effectiveness

    Parameters
    ----------
    system_lifetime : float
        Expected system lifetime in years (default: 20)
    annual_irradiance : float
        Annual solar irradiance in kWh/m² (default: 1700)
    system_efficiency : float
        System power conversion efficiency (default: 0.15)

    Attributes
    ----------
    system_lifetime : float
        System lifetime in years
    annual_irradiance : float
        Annual solar irradiance kWh/m²
    system_efficiency : float
        System efficiency (0-1)
    lca_parameters : dict
        LCA calculation parameters

    Examples
    --------
    >>> from models.lca_analyzer import LCAAnalyzer
    >>> lca = LCAAnalyzer(system_lifetime=25, system_efficiency=0.18)
    >>> results = lca.calculate_lca_impact(
    ...     manufacturing_energy=1500,  # MJ/m²
    ...     operational_time=25,         # years
    ...     carbon_intensity=0.5,        # kg CO2eq/kWh
    ...     material_mass=0.3            # kg/m²
    ... )
    >>> print(f"Carbon footprint: {results['carbon_footprint_gco2eq_per_kwh']:.1f} "
    ...       "gCO2eq/kWh")
    """

    def __init__(
        self,
        system_lifetime: float = 20.0,
        annual_irradiance: float = 1700.0,
        system_efficiency: float = 0.15
    ):
        """Initialize the LCA analyzer."""
        self.system_lifetime = system_lifetime
        self.annual_irradiance = annual_irradiance
        self.system_efficiency = system_efficiency

        # LCA parameters based on literature values for OPV systems
        self.lca_parameters = {
            # Manufacturing phase
            'manufacturing_carbon_intensity': 0.05,  # kg CO2eq/MJ
            'material_toxicity_factor': 0.1,  # dimensionless
            'resource_depletion_factor': 0.05,  # kg scarce materials/kg OPV

            # Operational phase
            'maintenance_energy_factor': 0.02,  # fraction of manufacturing energy/year
            'cleaning_energy_factor': 0.01,  # fraction of manufacturing energy/year

            # End-of-life phase
            'recycling_rate': 0.8,  # fraction of materials recycled
            'landfill_impact': 0.5,  # dimensionless score
            'recycling_energy': 0.3,  # fraction of manufacturing energy

            # Reference values
            'reference_silicon_carbon': 40.0,  # gCO2eq/kWh for silicon PV
            'reference_silicon_epbt': 2.0,  # years for silicon PV
        }

        logger.info(
            f"LCAAnalyzer initialized (lifetime={system_lifetime}yr, "
            f"efficiency={system_efficiency:.1%})"
        )

    def calculate_manufacturing_impact(
        self,
        manufacturing_energy: float,
        material_mass: float,
        carbon_intensity: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate manufacturing phase environmental impact.

        Parameters
        ----------
        manufacturing_energy : float
            Energy required for manufacturing in MJ/m²
        material_mass : float
            Mass of materials used in kg/m²
        carbon_intensity : float, optional
            Carbon intensity of electricity in kg CO2eq/kWh
            If None, uses default from lca_parameters

        Returns
        -------
        dict
            Manufacturing impact metrics:
            - energy_mj: Energy consumption (MJ/m²)
            - carbon_kg_co2eq: Carbon emissions (kg CO2eq/m²)
            - toxicity_score: Material toxicity score
            - resource_depletion: Resource depletion (kg scarce materials/m²)
        """
        if carbon_intensity is None:
            carbon_intensity = self.lca_parameters['manufacturing_carbon_intensity']

        # Convert MJ to kWh for carbon calculation (1 MJ = 0.2778 kWh)
        energy_kwh = manufacturing_energy * 0.2778

        carbon_emissions = energy_kwh * carbon_intensity

        toxicity = (
            material_mass * self.lca_parameters['material_toxicity_factor']
        )

        resource_depletion = (
            material_mass * self.lca_parameters['resource_depletion_factor']
        )

        impact = {
            'energy_mj': manufacturing_energy,
            'carbon_kg_co2eq': carbon_emissions,
            'toxicity_score': toxicity,
            'resource_depletion': resource_depletion
        }

        logger.debug(
            f"Manufacturing impact: {carbon_emissions:.2f} kg CO2eq/m², "
            f"{manufacturing_energy:.1f} MJ/m²"
        )

        return impact

    def calculate_operational_impact(
        self,
        operational_time: float,
        maintenance_frequency: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate operational phase environmental impact.

        Parameters
        ----------
        operational_time : float
            Operational lifetime in years
        maintenance_frequency : float
            Maintenance events per year (default: 1.0)

        Returns
        -------
        dict
            Operational impact metrics:
            - energy_mj: Maintenance energy (MJ/m²)
            - carbon_kg_co2eq: Carbon emissions (kg CO2eq/m²)
            - maintenance_events: Total maintenance events
        """
        # Assume manufacturing energy as baseline
        baseline_energy = 1500.0  # MJ/m² (typical OPV manufacturing)

        maintenance_energy = (
            baseline_energy *
            self.lca_parameters['maintenance_energy_factor'] *
            operational_time *
            maintenance_frequency
        )

        cleaning_energy = (
            baseline_energy *
            self.lca_parameters['cleaning_energy_factor'] *
            operational_time
        )

        total_energy = maintenance_energy + cleaning_energy

        # Convert to carbon (using grid average)
        grid_carbon_intensity = 0.5  # kg CO2eq/kWh
        energy_kwh = total_energy * 0.2778
        carbon_emissions = energy_kwh * grid_carbon_intensity

        impact = {
            'energy_mj': total_energy,
            'carbon_kg_co2eq': carbon_emissions,
            'maintenance_events': operational_time * maintenance_frequency
        }

        logger.debug(
            f"Operational impact: {carbon_emissions:.2f} kg CO2eq/m², "
            f"{total_energy:.1f} MJ/m² over {operational_time} years"
        )

        return impact

    def calculate_end_of_life_impact(
        self,
        material_mass: float,
        recycling_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate end-of-life phase environmental impact.

        Parameters
        ----------
        material_mass : float
            Mass of materials in kg/m²
        recycling_rate : float, optional
            Fraction of materials recycled (0-1)
            If None, uses default from lca_parameters

        Returns
        -------
        dict
            End-of-life impact metrics:
            - recycled_mass: Mass recycled (kg/m²)
            - landfill_mass: Mass to landfill (kg/m²)
            - recycling_energy_mj: Energy for recycling (MJ/m²)
            - carbon_kg_co2eq: Net carbon impact (kg CO2eq/m²)
            - landfill_impact: Landfill toxicity score
        """
        if recycling_rate is None:
            recycling_rate = self.lca_parameters['recycling_rate']

        recycled_mass = material_mass * recycling_rate
        landfill_mass = material_mass * (1 - recycling_rate)

        # Energy for recycling (typically 30% of manufacturing energy)
        baseline_energy = 1500.0  # MJ/m²
        recycling_energy = (
            baseline_energy *
            self.lca_parameters['recycling_energy'] *
            recycling_rate
        )

        # Carbon credit for recycling (avoided virgin material production)
        carbon_credit = recycled_mass * 2.0  # kg CO2eq/kg recycled

        # Landfill impact
        landfill_impact = (
            landfill_mass *
            self.lca_parameters['landfill_impact']
        )

        # Net carbon (recycling energy - carbon credit)
        recycling_carbon = recycling_energy * 0.2778 * 0.5  # kg CO2eq
        net_carbon = recycling_carbon - carbon_credit

        impact = {
            'recycled_mass': recycled_mass,
            'landfill_mass': landfill_mass,
            'recycling_energy_mj': recycling_energy,
            'carbon_kg_co2eq': net_carbon,
            'landfill_impact': landfill_impact
        }

        logger.debug(
            f"End-of-life impact: {recycled_mass:.2f} kg recycled, "
            f"{landfill_mass:.2f} kg landfill"
        )

        return impact

    def calculate_lca_impact(
        self,
        manufacturing_energy: float = 1500.0,
        operational_time: Optional[float] = None,
        carbon_intensity: Optional[float] = None,
        material_mass: float = 0.3,
        maintenance_frequency: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive Life Cycle Assessment impact.

        This method combines manufacturing, operational, and end-of-life phases
        to provide a complete LCA of the agrivoltaic system.

        Parameters
        ----------
        manufacturing_energy : float
            Manufacturing energy in MJ/m² (default: 1500)
        operational_time : float, optional
            Operational lifetime in years
            If None, uses system_lifetime
        carbon_intensity : float, optional
            Carbon intensity of electricity in kg CO2eq/kWh
        material_mass : float
            Mass of OPV materials in kg/m² (default: 0.3)
        maintenance_frequency : float
            Maintenance events per year (default: 1.0)

        Returns
        -------
        dict
            Comprehensive LCA results:
            - manufacturing: Manufacturing phase impact
            - operational: Operational phase impact
            - end_of_life: End-of-life phase impact
            - total_carbon_kg_co2eq: Total carbon emissions (kg CO2eq/m²)
            - total_energy_mj: Total energy consumption (MJ/m²)
            - carbon_footprint_gco2eq_per_kwh: Carbon footprint (gCO2eq/kWh)
            - energy_payback_time_years: Energy payback time (years)
            - eroi: Energy return on investment (dimensionless)
            - comparison_to_silicon: Ratio vs silicon PV
        """
        if operational_time is None:
            operational_time = self.system_lifetime

        # Calculate phase impacts
        manufacturing = self.calculate_manufacturing_impact(
            manufacturing_energy, material_mass, carbon_intensity
        )

        operational = self.calculate_operational_impact(
            operational_time, maintenance_frequency
        )

        end_of_life = self.calculate_end_of_life_impact(material_mass)

        # Total impacts
        total_carbon = (
            manufacturing['carbon_kg_co2eq'] +
            operational['carbon_kg_co2eq'] +
            end_of_life['carbon_kg_co2eq']
        )

        total_energy = (
            manufacturing['energy_mj'] +
            operational['energy_mj'] +
            end_of_life['recycling_energy_mj']
        )

        # Calculate energy generation over lifetime
        # E = irradiance * efficiency * area * time
        annual_energy = (
            self.annual_irradiance *
            self.system_efficiency
        )  # kWh/m²/year

        lifetime_energy = annual_energy * operational_time  # kWh/m²

        # Key metrics
        carbon_footprint = (
            (total_carbon * 1000) / lifetime_energy
            if lifetime_energy > 0 else 0
        )  # gCO2eq/kWh

        energy_payback_time = (
            manufacturing['energy_mj'] / (annual_energy * 3.6)
            if annual_energy > 0 else 0
        )  # years (convert kWh to MJ: 1 kWh = 3.6 MJ)

        eroi = (
            (lifetime_energy * 3.6) / total_energy
            if total_energy > 0 else 0
        )  # dimensionless

        # Comparison to silicon PV
        comparison = {
            'carbon_ratio': (
                carbon_footprint /
                self.lca_parameters['reference_silicon_carbon']
            ),
            'epbt_ratio': (
                energy_payback_time /
                self.lca_parameters['reference_silicon_epbt']
            )
        }

        results = {
            'manufacturing': manufacturing,
            'operational': operational,
            'end_of_life': end_of_life,
            'total_carbon_kg_co2eq': total_carbon,
            'total_energy_mj': total_energy,
            'lifetime_energy_kwh_per_m2': lifetime_energy,
            'carbon_footprint_gco2eq_per_kwh': carbon_footprint,
            'energy_payback_time_years': energy_payback_time,
            'eroi': eroi,
            'comparison_to_silicon': comparison
        }

        logger.info(
            f"LCA completed: Carbon footprint={carbon_footprint:.1f} gCO2eq/kWh, "
            f"EPBT={energy_payback_time:.2f} years, EROI={eroi:.1f}"
        )

        return results

    def calculate_sustainability_score(
        self,
        lca_results: Dict[str, Any],
        biodegradability_score: float = 0.5,
        pce: float = 0.15
    ) -> Dict[str, float]:
        """
        Calculate overall sustainability score combining LCA and performance.

        Parameters
        ----------
        lca_results : dict
            Results from calculate_lca_impact()
        biodegradability_score : float
            Biodegradability score (0-1, default: 0.5)
        pce : float
            Power conversion efficiency (0-1, default: 0.15)

        Returns
        -------
        dict
            Sustainability metrics:
            - overall_score: Weighted sustainability score (0-1)
            - lca_score: Normalized LCA score
            - biodegradability_score: Biodegradability component
            - performance_score: Performance component
        """
        # Normalize LCA metrics (lower is better, so invert)
        max_carbon = 100.0  # gCO2eq/kWh
        max_epbt = 5.0  # years

        carbon_score = max(
            0, 1 - lca_results['carbon_footprint_gco2eq_per_kwh'] / max_carbon
        )
        epbt_score = max(0, 1 - lca_results['energy_payback_time_years'] / max_epbt)
        eroi_score = min(1, lca_results['eroi'] / 10.0)  # Normalize to 10

        lca_score = (carbon_score + epbt_score + eroi_score) / 3.0

        # Performance score (normalized to 20% PCE)
        performance_score = min(1.0, pce / 0.20)

        # Weighted overall score
        # 40% biodegradability, 30% LCA, 30% performance
        overall_score = (
            0.4 * biodegradability_score +
            0.3 * lca_score +
            0.3 * performance_score
        )

        scores = {
            'overall_score': overall_score,
            'lca_score': lca_score,
            'carbon_score': carbon_score,
            'epbt_score': epbt_score,
            'eroi_score': eroi_score,
            'biodegradability_score': biodegradability_score,
            'performance_score': performance_score
        }

        logger.info(f"Sustainability score: {overall_score:.3f}")

        return scores

    def compare_materials(
        self,
        materials: list
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare LCA impacts of multiple materials.

        Parameters
        ----------
        materials : list
            List of dicts with keys:
            - name: Material name
            - manufacturing_energy: MJ/m²
            - material_mass: kg/m²
            - efficiency: PCE (0-1)
            - biodegradability: Score (0-1)
            - lifetime: Years

        Returns
        -------
        dict
            Comparison results for each material
        """
        comparison = {}

        for material in materials:
            name = material['name']

            # Calculate LCA
            lca = self.calculate_lca_impact(
                manufacturing_energy=material.get('manufacturing_energy', 1500),
                operational_time=material.get('lifetime', self.system_lifetime),
                material_mass=material.get('material_mass', 0.3)
            )

            # Calculate sustainability score
            scores = self.calculate_sustainability_score(
                lca,
                biodegradability_score=material.get('biodegradability', 0.5),
                pce=material.get('efficiency', 0.15)
            )

            comparison[name] = {
                'lca': lca,
                'scores': scores
            }

        # Rank materials by overall sustainability score
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1]['scores']['overall_score'],
            reverse=True
        )

        logger.info(
            f"Material comparison completed. Best: {ranked[0][0]} "
            f"(score={ranked[0][1]['scores']['overall_score']:.3f})"
        )

        return comparison

    def save_lca_results_to_csv(
        self,
        lca_results: Dict[str, Any],
        filename_prefix: str = 'lca_analysis',
        output_dir: str = '../simulation_data/'
    ) -> str:
        """
        Save LCA results to CSV file.

        Parameters
        ----------
        lca_results : dict
            Results from calculate_lca_impact()
        filename_prefix : str
            Prefix for output filename
        output_dir : str
            Directory to save CSV file

        Returns
        -------
        str
            Path to saved CSV file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Flatten nested dictionaries
        flat_data = {
            'timestamp': timestamp,
            'system_lifetime_years': self.system_lifetime,
            'system_efficiency': self.system_efficiency,
            'annual_irradiance_kwh_per_m2': self.annual_irradiance,
            'total_carbon_kg_co2eq': lca_results.get('total_carbon_kg_co2eq', 0),
            'total_energy_mj': lca_results.get('total_energy_mj', 0),
            'lifetime_energy_kwh_per_m2': lca_results.get(
                'lifetime_energy_kwh_per_m2', 0
            ),
            'carbon_footprint_gco2eq_per_kwh': lca_results.get(
                'carbon_footprint_gco2eq_per_kwh', 0
            ),
            'energy_payback_time_years': lca_results.get(
                'energy_payback_time_years', 0
            ),
            'eroi': lca_results.get('eroi', 0),
            'manufacturing_carbon_kg_co2eq': lca_results.get('manufacturing', {}).get(
                'carbon_kg_co2eq', 0
            ),
            'manufacturing_energy_mj': lca_results.get('manufacturing', {}).get(
                'energy_mj', 0
            ),
            'operational_carbon_kg_co2eq': lca_results.get('operational', {}).get(
                'carbon_kg_co2eq', 0
            ),
            'operational_energy_mj': lca_results.get('operational', {}).get(
                'energy_mj', 0
            ),
            'eol_carbon_kg_co2eq': lca_results.get('end_of_life', {}).get(
                'carbon_kg_co2eq', 0
            ),
            'eol_recycled_mass': lca_results.get('end_of_life', {}).get(
                'recycled_mass', 0
            ),
            'eol_landfill_mass': lca_results.get('end_of_life', {}).get(
                'landfill_mass', 0
            ),
            'comparison_to_silicon_carbon': lca_results.get(
                'comparison_to_silicon', {}
            ).get('carbon_ratio', 0),
            'comparison_to_silicon_epbt': lca_results.get(
                'comparison_to_silicon', {}
            ).get('epbt_ratio', 0)
        }

        # Create DataFrame and save
        df = pd.DataFrame([flat_data])
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, float_format='%.6e')

        logger.info(f"LCA results saved to {filepath}")
        return filepath

    def plot_lca_results(
        self,
        lca_results: Dict[str, Any],
        filename_prefix: str = 'lca_analysis',
        figures_dir: str = '../Graphics/'
    ) -> str:
        """
        Plot LCA results and save to file.

        Parameters
        ----------
        lca_results : dict
            Results from calculate_lca_impact()
        filename_prefix : str
            Prefix for output filename
        figures_dir : str
            Directory to save figures

        Returns
        -------
        str
            Path to saved figure
        """
        # Create figures directory if it doesn't exist
        os.makedirs(figures_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Life Cycle Assessment Results', fontsize=16, fontweight='bold')

        # 1. Carbon footprint by phase
        ax1 = axes[0, 0]
        phases = ['Manufacturing', 'Operational', 'End-of-Life']
        carbon_values = [
            lca_results.get('manufacturing', {}).get('carbon_kg_co2eq', 0),
            lca_results.get('operational', {}).get('carbon_kg_co2eq', 0),
            lca_results.get('end_of_life', {}).get('carbon_kg_co2eq', 0)
        ]
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        ax1.bar(phases, carbon_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Carbon Emissions (kg CO2eq/m²)', fontsize=11)
        ax1.set_title('Carbon Footprint by Lifecycle Phase', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Energy metrics
        ax2 = axes[0, 1]
        metrics = ['EPBT\n(years)', 'EROI', 'Carbon\n(gCO2eq/kWh)']
        values = [
            lca_results.get('energy_payback_time_years', 0),
            lca_results.get('eroi', 0),
            lca_results.get('carbon_footprint_gco2eq_per_kwh', 0) / 10  # Scale for vis.
        ]
        bars = ax2.bar(
            metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            alpha=0.7, edgecolor='black'
        )
        ax2.set_ylabel('Value (normalized)', fontsize=11)
        ax2.set_title('Key Sustainability Metrics', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 3. Comparison to silicon PV
        ax3 = axes[1, 0]
        comparison_labels = ['Carbon Footprint\nRatio', 'EPBT Ratio']
        comparison_values = [
            lca_results.get('comparison_to_silicon', {}).get('carbon_ratio', 0),
            lca_results.get('comparison_to_silicon', {}).get('epbt_ratio', 0)
        ]
        bars = ax3.bar(
            comparison_labels, comparison_values, color=['#9467bd', '#8c564b'],
            alpha=0.7, edgecolor='black'
        )
        ax3.axhline(
            y=1.0, color='red', linestyle='--', linewidth=2,
            label='Silicon PV Reference'
        )
        ax3.set_ylabel('Ratio (OPV / Silicon)', fontsize=11)
        ax3.set_title('Comparison to Silicon PV', fontsize=12)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, comparison_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9)

        # 4. End-of-life material flow
        ax4 = axes[1, 1]
        eol_data = lca_results.get('end_of_life', {})
        recycled = eol_data.get('recycled_mass', 0)
        landfill = eol_data.get('landfill_mass', 0)
        if recycled + landfill > 0:
            labels = ['Recycled', 'Landfill']
            sizes = [recycled, landfill]
            colors_pie = ['#2ca02c', '#d62728']
            explode = (0.05, 0)
            ax4.pie(
                sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90
            )
            ax4.set_title('End-of-Life Material Flow', fontsize=12)
        else:
            ax4.text(
                0.5, 0.5, 'No EOL data available', ha='center',
                va='center', transform=ax4.transAxes
            )
            ax4.set_title('End-of-Life Material Flow', fontsize=12)

        plt.tight_layout()

        # Save figure
        filename = f"{filename_prefix}_{timestamp}.pdf"
        filepath = os.path.join(figures_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        # Also save as PNG
        png_filename = f"{filename_prefix}_{timestamp}.png"
        png_filepath = os.path.join(figures_dir, png_filename)
        plt.savefig(png_filepath, dpi=150, bbox_inches='tight')

        plt.close()

        logger.info(f"LCA plots saved to {filepath} and {png_filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    logger.info("LCAAnalyzer module - example usage")

    # Initialize analyzer
    lca = LCAAnalyzer(system_lifetime=20, system_efficiency=0.18)

    # Calculate LCA for a typical OPV system
    results = lca.calculate_lca_impact(
        manufacturing_energy=1200,  # Lower than silicon
        operational_time=20,
        material_mass=0.3
    )

    print("\n=== LCA Results ===")
    print(
        f"Carbon footprint: {results['carbon_footprint_gco2eq_per_kwh']:.1f} gCO2eq/kWh"
    )
    print(
        f"Energy payback time: {results['energy_payback_time_years']:.2f} years"
    )
    print(f"Energy return on investment: {results['eroi']:.1f}")
    carbon_ratio = results['comparison_to_silicon']['carbon_ratio']
    print(f"vs Silicon PV - Carbon: {carbon_ratio:.2f}x")

    # Calculate sustainability score
    scores = lca.calculate_sustainability_score(
        results,
        biodegradability_score=0.7,
        pce=0.18
    )

    print(f"\nOverall sustainability score: {scores['overall_score']:.3f}")

    # Compare multiple materials
    materials = [
        {
            'name': 'PM6:Y6-BO', 'manufacturing_energy': 1200, 'efficiency': 0.17,
            'biodegradability': 0.6, 'lifetime': 15
        },
        {
            'name': 'PM6:IT-4F', 'manufacturing_energy': 1100, 'efficiency': 0.14,
            'biodegradability': 0.7, 'lifetime': 12
        },
        {
            'name': 'Silicon Ref', 'manufacturing_energy': 5000, 'efficiency': 0.22,
            'biodegradability': 0.0, 'lifetime': 25
        }
    ]

    comparison = lca.compare_materials(materials)
    print("\n=== Material Comparison ===")
    for name, data in comparison.items():
        print(f"{name}: Score={data['scores']['overall_score']:.3f}, "
              f"Carbon={data['lca']['carbon_footprint_gco2eq_per_kwh']:.1f} gCO2eq/kWh")
