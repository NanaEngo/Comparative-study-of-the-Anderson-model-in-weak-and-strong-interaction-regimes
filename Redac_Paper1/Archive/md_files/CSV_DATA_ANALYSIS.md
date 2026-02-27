# CSV Data Analysis

Purpose: provide a concise, reproducible guide for inspecting and summarizing CSV outputs produced by this repository's simulations and data pipelines.

## Data sources
- `simulation_data/fmo_hamiltonian_data.csv`
- `simulation_data/fmo_site_energies.csv`
- `simulation_data/quantum_metrics_evolution.csv`
- `quantum_simulations_framework/data_input/ASTMG173.csv`
- `quantum_simulations_framework/data_output/` (timestamped CSV outputs such as `environmental_effects_*.csv`, `lca_analysis_*.csv`, `sensitivity_analysis_*.csv`)

If you add or generate other CSVs, place them under `simulation_data/` or `quantum_simulations_framework/data_output/` so this guide's examples work unchanged.

## Quick inspection (shell)
Show header and first rows:

```bash
head -n 20 simulation_data/quantum_metrics_evolution.csv
```

Count rows (large files):

```bash
wc -l simulation_data/quantum_metrics_evolution.csv
```

## Summary statistics (Python / Pandas)
Example: compute summary statistics and save them.

```python
import pandas as pd

# load a representative CSV
df = pd.read_csv('simulation_data/quantum_metrics_evolution.csv')

# quick overview
print(df.info())
print(df.describe(include='all'))

# save numeric summary to CSV
df.describe().to_csv('analysis_summary_quantum_metrics.csv')

# group-based summary example (if columns like 'site' or 'run' exist)
if 'site' in df.columns:
    df.groupby('site').mean().to_csv('per_site_mean_quantum_metrics.csv')

# simple plot (requires matplotlib)
import matplotlib.pyplot as plt
if 'time' in df.columns and 'QFI' in df.columns:
    plt.plot(df['time'], df['QFI'])
    plt.xlabel('time')
    plt.ylabel('QFI')
    plt.tight_layout()
    plt.savefig('fig_qfi_vs_time.png', dpi=200)

```

## One-line analysis examples
Overall mean for a column:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('simulation_data/quantum_metrics_evolution.csv')
print(df['QFI'].mean())
PY
```

Export selected columns and filter rows:

```python
import pandas as pd
df = pd.read_csv('simulation_data/fmo_site_energies.csv')
# keep sites with energy > threshold
df[df['energy'] > 12500][['site','energy']].to_csv('high_energy_sites.csv', index=False)
```

## Suggested workflow
1. Inspect file headers with `head` or `csvcut` (from `csvkit`).
2. Use Pandas for richer summaries and grouping (`groupby`, `pivot_table`).
3. Save intermediate CSVs with timestamps to preserve provenance.
4. Produce plots with `matplotlib` or `seaborn` and save to `figures/`.

## Notes and conventions
- Timestamps: prefer filenames like `analysis_YYYYMMDD_HHMMSS.csv` to avoid overwrites.
- Missing data: use `df.isna().sum()` to locate nulls before numeric reductions.
- Large files: for CSVs >100MB, consider `dask.dataframe` or chunked `pd.read_csv(..., chunksize=...)`.

If you want, I can add a runnable Jupyter notebook that executes these examples and produces the figures and CSV summaries automatically.
