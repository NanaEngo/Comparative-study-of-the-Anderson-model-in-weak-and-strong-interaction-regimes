# Quantum Agrivoltaics Research Framework

## Project Overview

This is a comprehensive research framework for quantum-enhanced agrivoltaic systems, focusing on the comparative study of the Anderson model in weak and strong interaction regimes. The project implements advanced quantum simulation techniques to model photosynthetic energy transfer in the presence of spectrally filtered light from organic photovoltaic (OPV) panels.

The core research combines quantum dynamics simulation with agrivoltaic system design, specifically implementing:
- Process Tensor-HOPS with Low-Temperature Correction (PT-HOPS+LTC) for efficient quantum dynamics
- Stochastically Bundled Dissipators (SBD) for mesoscale systems (>1000 chromophores)
- Quantum coherence analysis using metrics like Quantum Fisher Information (QFI)
- Fenna-Matthews-Olsen (FMO) complex modeling for photosynthetic energy transfer
- Agrivoltaic coupling models that simulate the interaction between OPV panels and photosynthetic systems

## Key Technologies and Methods

### Quantum Simulation Framework
- **Process Tensor-HOPS+LTC**: Non-recursive framework achieving 10× computational speedup at low temperatures (T<150K)
- **Stochastically Bundled Dissipators (SBD)**: Enables simulations of systems with >1000 chromophores while preserving non-Markovian effects
- **Liouvillian Superoperator**: Mathematical framework for Lindblad master equation in Liouville space
- **FMO Complex Modeling**: 7-site model for photosynthetic energy transfer

### Mathematical Framework
The implementation follows the theoretical foundation with key equations:
- Process Tensor Decomposition: K_PT(t,s) = Σₖ gₖ(t) fₖ(s) e^(-λₖ|t-s|) + K_non-exp(t,s)
- Low-Temperature Correction: N_Mat = 10 (Matsubara cutoff), eta_LTC = 10 (Time step enhancement factor)
- Stochastically Bundled Dissipators: L_SBD[ρ] = Σ_α p_α(t) D_α[ρ]

### Code Architecture
The main simulation code is organized in the `quantum_simulations_framework/` directory with modules:
- `quantum_dynamics_simulator.py`: Core quantum dynamics simulation with PT-HOPS+LTC
- `quantum_agrivoltaics_simulations.py`: Main simulation workflow
- `agrivoltaic_coupling_model.py`: OPV-PSU coupling implementation
- `spectral_optimizer.py`: Optimization algorithms for spectral filtering
- `eco_design_analyzer.py`: Sustainability analysis using quantum reactivity descriptors
- `unified_figures.py`: Visualization tools for publication-quality figures
- `csv_data_storage.py`: Data management and storage to CSV format

## Building and Running

### Prerequisites
- Python 3.7+
- NumPy, SciPy, Pandas, Matplotlib
- SciencePlots (for publication-quality plots)
- Additional dependencies for quantum simulation

### Running Simulations
The main simulation can be run using:
```python
from quantum_simulations_framework.quantum_agrivoltaics_simulations import run_complete_analysis
run_complete_analysis()
```

Or run the complete analysis with parameters from JSON file:
```python
from quantum_simulations_framework.quantum_agrivoltaics_simulations import run_complete_analysis_with_params
run_complete_analysis_with_params()
```

### Configuration
Simulation parameters are configured in `quantum_simulations_framework/data_input/quantum_agrivoltaics_params.json` which includes:
- Simulation parameters (temperature, dephasing rate, time points)
- FMO Hamiltonian parameters
- OPV parameters (bandgap, absorption coefficient)
- Quantum metrics calculation flags
- Optimization parameters
- Solar spectrum parameters
- Bath parameters for environmental coupling
- Process Tensor parameters for LTC
- SBD parameters for mesoscale systems
- Parallelization parameters

## Research Focus Areas

### Quantum Dynamics Framework
1. **Process Tensor-HOPS with Low-Temperature Correction (PT-HOPS+LTC)**: Non-recursive framework achieving 10× computational speedup at T<150K
2. **Stochastically Bundled Dissipators (SBD)**: Enables simulations of systems with >1000 chromophores while preserving non-Markovian effects
3. **Quantum Coherence Analysis**: Advanced metrics including Quantum Fisher Information (QFI) for parameter estimation sensitivity

### Simulation Components
- **Fenna-Matthews-Olsen (FMO) Complex Modeling**: 7-site model for photosynthetic energy transfer
- **Agrivoltaic Coupling Model**: Quantum-coherent spectral splitting between OPV and PSU systems
- **Spectral Optimization**: Multi-objective optimization balancing PCE and ETR performance
- **Eco-Design Analysis**: Quantum reactivity descriptors using Fukui functions for biodegradability prediction

## Key Research Contributions

1. **Process Tensor-HOPS+LTC Framework**: Efficient treatment of Matsubara modes with Low-Temperature Correction for enhanced computational performance
2. **Mesoscale SBD Implementation**: Scalable approach for simulating large chromophore systems
3. **Quantum Reactivity Descriptors**: Fukui function-based eco-design for biodegradable OPV materials
4. **Multi-Objective Optimization**: Simultaneous optimization of PCE and biodegradability with ETR preservation
5. **E(n)-Equivariant Graph Neural Networks**: Physics-informed machine learning for molecular property prediction
6. **Agricultural Quality Enhancement**: Advanced metrics for crop productivity and quality

## Applications

This research framework enables the design of next-generation quantum-enhanced agrivoltaic systems that:
- Achieve high power conversion efficiency (>20%)
- Maintain agricultural productivity (ETR_rel >90%)
- Utilize biodegradable materials (>80% biodegradability)
- Implement quantum coherence effects for enhanced performance
- Support sustainable agriculture through symbiotic design
- Enable circular economy principles through eco-friendly materials

## Data Management

- **JSON Parameter Configuration**: Centralized parameter management in `data_input/quantum_agrivoltaics_params.json`
- **CSV Data Output**: Comprehensive results saved to `data_output/` directory
- **Unified Visualization**: All figures generated through unified class in `unified_figures.py`

## Development Conventions

The codebase follows scientific computing best practices with emphasis on:
- Reproducible research workflows
- Comprehensive parameter configuration
- Proper documentation of mathematical frameworks
- Modular code organization for maintainability
- Extensive quantum metrics computation
- Parallel processing capabilities for large-scale simulations