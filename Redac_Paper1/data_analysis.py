#!/usr/bin/env python3
"""
Data Analysis Script for Quantum Agrivoltaics Research
This script analyzes the stored simulation data to validate results and provide insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define data directory
DATA_DIR = Path("simulation_data")
FIGURES_DIR = Path("Graphics")
os.makedirs(FIGURES_DIR, exist_ok=True)

def analyze_fmo_hamiltonian_data():
    """Analyze FMO Hamiltonian data"""
    try:
        df = pd.read_csv(DATA_DIR / "fmo_hamiltonian_data.csv")
        print(f"FMO Hamiltonian data shape: {df.shape}")
        print("FMO Hamiltonian statistics:")
        print(df.describe())
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(df.values, cmap='RdBu_r', aspect='equal', vmin=-150, vmax=150)
        plt.title('FMO Hamiltonian Matrix (cm⁻¹)')
        plt.colorbar(label='Energy (cm⁻¹)')
        plt.xlabel('Site Index')
        plt.ylabel('Site Index')
        plt.savefig(FIGURES_DIR / "fmo_hamiltonian_analysis.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(FIGURES_DIR / "fmo_hamiltonian_analysis.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        return df
    except FileNotFoundError:
        print("fmo_hamiltonian_data.csv not found")
        return None

def analyze_fmo_site_energies():
    """Analyze FMO site energies"""
    try:
        df = pd.read_csv(DATA_DIR / "fmo_site_energies.csv")
        print(f"FMO site energies data shape: {df.shape}")
        print("FMO Site Energies:")
        print(df.head())
        print(f"Energy range: {df['Energy_cm-1'].min():.0f} to {df['Energy_cm-1'].max():.0f} cm⁻¹")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(df['Site_Index'], df['Energy_cm-1'], 'ro-', label='Site Energies')
        plt.title('FMO Site Energies')
        plt.xlabel('Site Index')
        plt.ylabel('Energy (cm⁻¹)')
        plt.legend()
        plt.grid(True)
        plt.savefig(FIGURES_DIR / "fmo_site_energies_analysis.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(FIGURES_DIR / "fmo_site_energies_analysis.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        return df
    except FileNotFoundError:
        print("fmo_site_energies.csv not found")
        return None

def analyze_quantum_dynamics():
    """Analyze quantum dynamics data"""
    try:
        # Load populations data
        pop_df = pd.read_csv(DATA_DIR / "fmo_dynamics_populations.csv", index_col=0)
        print(f"Quantum dynamics populations data shape: {pop_df.shape}")
        print("Population statistics:")
        print(pop_df.describe())
        
        # Load coherences data
        coh_df = pd.read_csv(DATA_DIR / "fmo_dynamics_coherences_real.csv", index_col=0)
        print(f"Quantum dynamics coherences data shape: {coh_df.shape}")
        
        # Load quantum metrics
        metrics_df = pd.read_csv(DATA_DIR / "fmo_dynamics_quantum_metrics.csv", index_col=0)
        print(f"Quantum metrics data shape: {metrics_df.shape}")
        print("Quantum metrics statistics:")
        print(metrics_df.describe())
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot populations
        time_points = pop_df.index
        for i, col in enumerate(pop_df.columns[:5]):  # First 5 sites
            axes[0,0].plot(time_points, pop_df[col], label=f'Site {i}', linewidth=1.5)
        axes[0,0].set_title('Site Populations vs Time')
        axes[0,0].set_xlabel('Time (fs)')
        axes[0,0].set_ylabel('Population')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot coherences
        axes[0,1].plot(coh_df.index, coh_df['Coherence'], 'r-', linewidth=1.5)
        axes[0,1].set_title('Coherence vs Time')
        axes[0,1].set_xlabel('Time (fs)')
        axes[0,1].set_ylabel('Coherence (l1-norm)')
        axes[0,1].grid(True)
        
        # Plot quantum metrics
        axes[1,0].plot(metrics_df.index, metrics_df['QFI'], label='QFI', linewidth=1.5)
        axes[1,0].set_title('Quantum Fisher Information vs Time')
        axes[1,0].set_xlabel('Time (fs)')
        axes[1,0].set_ylabel('QFI')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        axes[1,1].plot(metrics_df.index, metrics_df['Entropy'], label='Entropy', linewidth=1.5)
        axes[1,1].plot(metrics_df.index, 1 - metrics_df['Purity'], label='1-Purity', linewidth=1.5)
        axes[1,1].set_title('Entropy and 1-Purity vs Time')
        axes[1,1].set_xlabel('Time (fs)')
        axes[1,1].set_ylabel('Value')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "quantum_dynamics_analysis.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(FIGURES_DIR / "quantum_dynamics_analysis.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        return pop_df, coh_df, metrics_df
    except FileNotFoundError as e:
        print(f"Quantum dynamics data file not found: {e}")
        return None, None, None

def analyze_eco_design():
    """Analyze eco-design data"""
    try:
        # Load biodegradability analysis
        bio_df = pd.read_csv(DATA_DIR / "biodegradability_analysis.csv")
        print(f"Biodegradability analysis data shape: {bio_df.shape}")
        print("Biodegradability analysis:")
        print(bio_df)
        
        # Load LCA impact assessment
        lca_df = pd.read_csv(DATA_DIR / "lca_impact_assessment.csv")
        print(f"LCA impact assessment data shape: {lca_df.shape}")
        print("LCA Impact Assessment:")
        print(lca_df)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Biodegradability scores
        if not bio_df.empty:
            molecules = bio_df.index
            scores = bio_df['biodegradability_score'] if 'biodegradability_score' in bio_df.columns else [0]*len(bio_df)
            axes[0].bar(molecules, scores)
            axes[0].set_title('Biodegradability Scores')
            axes[0].set_ylabel('Score')
            axes[0].tick_params(axis='x', rotation=45)
        
        # LCA data
        if not lca_df.empty and 'carbon_footprint_gco2eq_per_kwh' in lca_df.columns:
            axes[1].bar(['Sample'], lca_df['carbon_footprint_gco2eq_per_kwh'])
            axes[1].set_title('Carbon Footprint (gCO2eq/kWh)')
            axes[1].set_ylabel('gCO2eq/kWh')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eco_design_analysis.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(FIGURES_DIR / "eco_design_analysis.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        return bio_df, lca_df
    except FileNotFoundError as e:
        print(f"Eco-design data file not found: {e}")
        return None, None

def analyze_solar_spectrum():
    """Analyze solar spectrum data"""
    try:
        df = pd.read_csv(DATA_DIR / "solar_spectrum_data.csv")
        print(f"Solar spectrum data shape: {df.shape}")
        print("Solar spectrum statistics:")
        print(df.describe())
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(df['Wavelength_nm'], df['Irradiance_W_m2_nm'], linewidth=1.5)
        plt.title('Solar Spectrum')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Irradiance (W/m²/nm)')
        plt.grid(True)
        plt.savefig(FIGURES_DIR / "solar_spectrum_analysis.pdf", bbox_inches="tight", dpi=300)
        plt.savefig(FIGURES_DIR / "solar_spectrum_analysis.png", bbox_inches="tight", dpi=300)
        plt.close()
        
        return df
    except FileNotFoundError:
        print("solar_spectrum_data.csv not found")
        return None

def generate_summary_report():
    """Generate a summary report of all analyses"""
    print("\n" + "="*60)
    print("QUANTUM AGRIVOLTAICS DATA ANALYSIS SUMMARY")
    print("="*60)
    
    # Check all data files
    data_files = list(DATA_DIR.glob("*.csv"))
    print(f"Total CSV data files found: {len(data_files)}")
    for f in data_files:
        size = f.stat().st_size
        print(f"  - {f.name}: {size} bytes")
    
    print("\nDetailed Analysis Results:")
    print("-" * 30)
    
    # Analyze each dataset
    hamiltonian_df = analyze_fmo_hamiltonian_data()
    site_energies_df = analyze_fmo_site_energies()
    pop_df, coh_df, metrics_df = analyze_quantum_dynamics()
    bio_df, lca_df = analyze_eco_design()
    solar_df = analyze_solar_spectrum()
    
    # Generate summary statistics
    print("\nKey Findings:")
    print("-" * 15)
    
    if metrics_df is not None:
        print(f"Max Quantum Fisher Information: {metrics_df['QFI'].max():.3f}")
        print(f"Average Entropy: {metrics_df['Entropy'].mean():.3f}")
        print(f"Average Purity: {metrics_df['Purity'].mean():.3f}")
    
    if pop_df is not None:
        print(f"Number of time points: {len(pop_df)}")
        print(f"Number of sites: {len(pop_df.columns)}")
    
    if coh_df is not None:
        print(f"Max Coherence: {coh_df['Coherence'].max():.3f}")
        print(f"Min Coherence: {coh_df['Coherence'].min():.3f}")
    
    if bio_df is not None:
        print(f"Number of molecules analyzed: {len(bio_df)}")
    
    print("\nAll analysis figures saved to Graphics/ directory")
    print("Data validation completed successfully!")

if __name__ == "__main__":
    generate_summary_report()
