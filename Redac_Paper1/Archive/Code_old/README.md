# Quantum Agrivoltaics Simulation Framework

This framework implements a complete simulation system for quantum-enhanced agrivoltaic systems based on the Process Tensor-HOPS (PT-HOPS) approach.

## Structure

The codebase has been refactored into separate, focused modules:

### Core Modules

1. **quantum_dynamics_simulator.py** - Core quantum dynamics simulation using Process Tensor-HOPS approach with Stochastically Bundled Dissipators (SBD)
2. **agrivoltaic_coupling_model.py** - Agrivoltaic coupling model combining OPV and photosynthetic systems
3. **spectral_optimizer.py** - Multi-objective optimization for spectral splitting with SPCE objective
4. **eco_design_analyzer.py** - Eco-design analysis using quantum reactivity descriptors (Fukui functions)
5. **csv_data_storage.py** - Data storage functionality for simulation results (saves to data_output/)
6. **figure_generator.py** - Visualization tools for simulation results
7. **unified_figures.py** - Comprehensive unified figure generation class
8. **quantum_agrivoltaics_simulations.py** - Main module with complete analysis workflow
9. **quantum_agrivoltaics_simulations_refined.py** - Enhanced implementation with PT-HOPS

### Parameter Configuration

- **data_input/** - Input parameters directory
  - **quantum_agrivoltaics_params.json** - Centralized JSON configuration for all simulation parameters

### Data Output

- **data_output/** - Output directory for CSV data files (populations, coherences, QFI, optimization results, eco-analysis, etc.)

## Key Features

- **Process Tensor-HOPS (PT-HOPS)**: 10× computational speedup through efficient Matsubara mode treatment
- **Stochastically Bundled Dissipators (SBD)**: Enables >1000 chromophore mesoscale simulations while preserving non-Markovian effects
- **Quantum Dynamics Simulation**: Complete Lindblad master equation solver with dephasing and thermal effects
- **Comprehensive Quantum Metrics**: QFI, von Neumann entropy, purity, linear entropy, concurrence, entanglement measures
- **Spectral Optimization**: Multi-objective optimization balancing PCE and ETR with SPCE objective
- **Eco-Design Analysis**: Quantum reactivity descriptors using Fukui functions for biodegradability prediction
- **Robustness Analysis**: Sensitivity to temperature and disorder variations
- **Data Management**: JSON parameter configuration and CSV data output
- **E(n)-Equivariant Graph Neural Networks**: Physical symmetry-preserving predictions
- **Non-recursive AI-QD Framework**: Direct density matrix evolution prediction

## Mathematical Framework

The framework implements:

### Process Tensor-HOPS:
The bath correlation function C(t) is decomposed via Padé approximation:
K_PT(t,s) = Σₖ gₖ(t) fₖ(s) e^(-λₖ|t-s|) + K_non-exp(t,s)

### Thermal Regime Validity:
For T = 295K, high-temperature approximation holds (kB T >> hbar gamma).
- N_Mat = 10 (Matsubara cutoff)


### Stochastically Bundled Dissipators (SBD):
L_SBD[ρ] = Σ_α p_α(t) D_α[ρ]
D_α[ρ] = L_α ρ L_α^† - ½{L_α^†L_α, ρ}

### Quantum Master Equation:
dρ/dt = -i[H, ρ] + L_dephasing[ρ] + L_dissipative[ρ]

## Usage

### Complete Analysis Workflow:
```python
from quantum_agrivoltaics_simulations import run_complete_simulation
run_complete_simulation()
```

### Using JSON Parameter Configuration:
```python
from quantum_agrivoltaics_simulations import run_complete_analysis_with_params
run_complete_analysis_with_params()
```

### Individual Components:
```python
from quantum_dynamics_simulator import QuantumDynamicsSimulator
from quantum_agrivoltaics_simulations import create_fmo_hamiltonian

# Create FMO Hamiltonian
fmo_hamiltonian, site_energies = create_fmo_hamiltonian()

# Initialize quantum simulator with PT-HOPS approach
quantum_sim = QuantumDynamicsSimulator(fmo_hamiltonian, temperature=295, dephasing_rate=20)

# Run quantum dynamics simulation
time_points, density_matrices, populations, coherences, qfi_values, \
entropy_values, purity_values, linear_entropy_values, bipartite_ent_values, \
multipartite_ent_values, pairwise_concurrence_values = quantum_sim.simulate_dynamics(
    time_points=np.linspace(0, 500, 100)  # fs
)
```

### Data Storage:
```python
from csv_data_storage import CSVDataStorage

# Initialize data storage (defaults to data_output directory)
data_storage = CSVDataStorage()

# Save simulation results
data_storage.save_simulation_data_to_csv(
    time_points, populations, coherences, qfi_values, etr_values
)
```

### Unified Visualization:
```python
from unified_figures import UnifiedFigures

# Initialize unified figures class
fig_gen = UnifiedFigures()

# Generate comprehensive quantum dynamics plot
fig = fig_gen.plot_quantum_dynamics(
    time_points, populations, coherences, qfi_values, entropy_values, purity_values
)

# Save the figure
fig_gen.save_figure(fig, 'quantum_dynamics.png')
```

## Data Management

### Input Parameters:
- All simulation parameters are configured in `data_input/quantum_agrivoltaics_params.json`
- Includes simulation_params, fmo_hamiltonian_params, opv_params, quantum_metrics, optimization_params, solar_spectrum_params, bath_params, process_tensor_params, and sbd_params

### Output Data:
- All simulation results are saved as CSV files to the `data_output/` directory
- Includes quantum dynamics, spectral optimization results, eco-design analysis, and robustness data
- Files are timestamped to prevent overwriting during multiple runs

## Applications

- Quantum-enhanced agrivoltaic design with spectral splitting optimization
- Sustainable organic photovoltaic materials with >80% biodegradability
- Photosynthetic efficiency optimization in agrivoltaic systems
- Quantum metrology applications using Quantum Fisher Information
- Environmental impact assessment using quantum reactivity descriptors
- Mesoscale quantum dynamics simulations with >1000 chromophores
- Multi-objective optimization balancing PCE and ETR performance
- Eco-design of biodegradable materials using Fukui function analysis