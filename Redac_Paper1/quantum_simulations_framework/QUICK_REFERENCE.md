# Quick Reference: Updated Notebook Usage

## Running the Notebook

### 1. Activate Environment
```bash
conda activate MesoHOP-sim
cd /home/taamangtchu/Documents/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework
jupyter notebook quantum_coherence_agrivoltaics_mesohops.ipynb
```

### 2. Run Cells Sequentially
**Important**: Must run from beginning (Cell 1 → Cell N)

Key cells:
- **Cell 1-4**: Setup and imports
- **Cell 5**: HopsSimulator (MesoHOPS integration) ✓ UPDATED
- **Cell 6-13**: FMO Hamiltonian and helper functions
- **Cell 14**: QuantumDynamicsSimulator wrapper
- **Cell 15+**: Analysis and simulations

### 3. Expected Behavior

#### Cell 5 Output (HopsSimulator)
```
✓ MesoHOPS direct integration available
  MesoHOPS version: 1.6.1
Default simulator: MesoHOPS
Use HopsSimulator class for unified interface
```

#### First Simulation Output
```
  ✓ MesoHOPS simulator initialized
Integration from 0 to 500.0
Noise Model initialized with SEED = 0
```

## Key Parameters

### HopsSimulator Initialization
```python
sim = HopsSimulator(
    hamiltonian,              # np.array, shape (n, n)
    temperature=295,          # K
    reorganization_energy=35, # cm^-1
    drude_cutoff=50          # cm^-1
)
```

### Simulation Parameters
```python
results = sim.simulate_dynamics(
    time_points,              # np.array of time values (fs)
    initial_state=None,       # Default: site 1 excited
    max_hierarchy=4,          # Hierarchy truncation level
    seed=0                    # Random seed for noise
)
```

### Return Values
```python
(t_axis,           # Time points
 populations,      # Site populations [n_times, n_sites]
 populations,      # (duplicate)
 coherences,       # L1-norm coherence
 qfi,             # Quantum Fisher Information
 entropy,         # von Neumann entropy
 purity,          # State purity
 linear_entropy,  # Linear entropy (placeholder)
 bipartite_ent,   # Bipartite entanglement (placeholder)
 multipartite_ent,# Multipartite entanglement (placeholder)
 pairwise_conc)   # Pairwise concurrence (placeholder)
```

## Troubleshooting

### Issue: "MesoHOPS initialization failed"
**Solution**: Check that MesoHOPS is properly installed
```bash
conda activate MesoHOP-sim
python -c "from mesohops import HopsSystem; print('OK')"
```

### Issue: "Using fallback simulator"
**Cause**: MesoHOPS failed, using mock data  
**Solution**: Check error message above, may need to adjust parameters

### Issue: "Trajectory times longer than noise.param['TLEN']"
**Cause**: Simulation time > noise length  
**Solution**: Automatically handled (TLEN = t_max * 2.0)

### Issue: Simulation very slow
**Cause**: High hierarchy level or long simulation time  
**Solutions**:
- Reduce `max_hierarchy` (try 2-4)
- Reduce time range or increase time step
- Use fewer time points

## Performance Tips

### Fast Testing (seconds)
```python
time_points = np.linspace(0, 100, 11)  # 0-100 fs, 11 points
max_hierarchy = 2
```

### Standard Run (minutes)
```python
time_points = np.linspace(0, 500, 51)  # 0-500 fs, 51 points
max_hierarchy = 4
```

### Production Run (hours)
```python
time_points = np.linspace(0, 1000, 101)  # 0-1 ps, 101 points
max_hierarchy = 6
```

## Validation Checks

### After Running Simulation
```python
# Check population conservation
assert np.allclose(np.sum(populations, axis=1), 1.0, atol=0.01)

# Check energy transfer occurred
assert populations[-1, 0] < populations[0, 0]

# Check purity bounds
assert np.all((purity >= 0) & (purity <= 1))

# Check time resolution
assert len(t_axis) >= 50
```

## CSV Output Files

### Expected Files (after full run)
```
simulation_data/
├── quantum_dynamics_hops_results.csv          # 500+ rows ✓
├── quantum_dynamics_populations_*.csv         # 500+ rows ✓
├── quantum_dynamics_coherences_real_*.csv     # 500+ rows ✓
├── quantum_dynamics_quantum_metrics_*.csv     # 500+ rows ✓
├── agrivoltaic_coupling_results.csv
├── biodegradability_analysis.csv
└── ...
```

### Verify CSV Quality
```bash
# Check row counts
wc -l simulation_data/*.csv

# Should see 500+ rows in quantum dynamics files
```

## Common Workflows

### 1. Quick Test
```python
# Run cells 1-14
# Then run a single simulation cell with reduced parameters
```

### 2. Full Analysis
```python
# Run all cells sequentially
# Wait for completion (~30-60 minutes)
# Check CSV outputs
```

### 3. Parameter Sweep
```python
# Modify parameters in simulation cells
# Re-run specific analysis sections
# Compare results
```

## Getting Help

### Check Logs
- Notebook cell outputs show detailed progress
- Error messages indicate specific issues
- Traceback shows where failures occur

### Validation Scripts
```bash
# Test MesoHOPS directly
python test_mesohops_working.py

# Validate notebook integration
python validate_notebook_mesohops.py
```

### Documentation
- `MESOHOPS_SUCCESS.md` - API reference
- `MESOHOPS_IMPLEMENTATION_COMPLETE.md` - Full details
- `PACKAGE_CHECK_REPORT.md` - Package status

## Success Indicators

✓ MesoHOPS initializes without errors  
✓ Simulations complete in reasonable time  
✓ Population conservation maintained  
✓ Energy transfer occurs  
✓ CSV files have 500+ time points  
✓ Coherence decays smoothly  
✓ Purity stays in [0, 1] range  

## Next Steps

After verifying MesoHOPS works:
1. Fix PCE calculation (target: 0.15-0.20)
2. Fix ETR calculation (target: >0.90)
3. Add validation checks
4. Benchmark against literature
5. Run production simulations
