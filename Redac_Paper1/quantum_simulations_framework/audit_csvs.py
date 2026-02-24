import pandas as pd
import glob
from pathlib import Path

data_dir = Path('../simulation_data')

# Find the most recent file for each category
categories = [
    'agrivoltaic_results_*.csv',
    'eco_design_results_*.csv',
    'environmental_effects_*.csv',
    'lca_analysis_*.csv',
    'quantum_dynamics_2026*.csv',
    'spectral_optimization_*.csv'
]

for pattern in categories:
    files = list(data_dir.glob(pattern))
    if not files:
        print(f"--- NO FILES FOUND FOR {pattern} ---")
        continue
        
    # Get the most recent file
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"\n{'='*60}")
    print(f"FILE: {latest_file.name}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(latest_file)
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(list(df.columns))
        print("\nFirst row:")
        # Print the first row cleanly
        for col in df.columns:
            val = df.iloc[0][col]
            if isinstance(val, float):
                print(f"  {col}: {val:.6g}")
            else:
                print(f"  {col}: {val}")
                
        # Specifically check for any NaNs or unexpected values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\nWARNING: Found NaN values:\n{null_counts[null_counts > 0]}")
            
    except Exception as e:
        print(f"ERROR reading {latest_file.name}: {e}")

print("\nAudit complete.")
