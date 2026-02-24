import pandas as pd
from pathlib import Path

data_dir = Path('../simulation_data')
qd_files = list(data_dir.glob('quantum_dynamics_2026*.csv'))
latest_qd = max(qd_files, key=lambda f: f.stat().st_mtime)

df = pd.read_csv(latest_qd)
print(f"Shape: {df.shape}")
print("Rows with NaNs:")
print(df[df.isnull().any(axis=1)])
