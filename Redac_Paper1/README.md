# Quantum Agrivoltaics Research Framework

This directory contains the comprehensive research framework for quantum-enhanced agrivoltaic systems, incorporating advanced methodologies for simulating and optimizing symbiotic energy-agriculture systems.

## Core Research Focus

### Quantum Dynamics Framework
1. **Process Tensor-HOPS with Low-Temperature Correction (PT-HOPS+LTC)**: Non-recursive framework achieving 10× computational speedup at T<150K
2. **Stochastically Bundled Dissipators (SBD)**: Enables simulations of systems with >1000 chromophores while preserving non-Markovian effects
3. **Quantum Coherence Analysis**: Advanced metrics including Quantum Fisher Information (QFI) for parameter estimation sensitivity

### Simulation Components
- **Fenna-Matthews-Olsen (FMO) Complex Modeling**: 7-site model for photosynthetic energy transfer
- **Agrivoltaic Coupling Model**: Quantum-coherent spectral splitting between OPV and PSU systems
- **Spectral Optimization**: Multi-objective optimization balancing PCE and ETR performance
- **Eco-Design Analysis**: Quantum reactivity descriptors using Fukui functions for biodegradability prediction

## Directory Structure

```
Redac_Paper1/
├── README.md                    # This file
├── Goumai_Paper1_Draft_2601.pdf   # Research paper
├── Goumai_Paper1_Draft_2601.tex   # LaTeX source
├── Ref_HOPS.bib                 # Bibliography
├── Suggest.md                   # Research suggestions
├── quantum_coherence_agrivoltaics_analysis.ipynb  # Main analysis notebook
├── Code_old/                    # Legacy code
├── Graphics/                    # Visual materials
├── quantum_simulations_framework/ # Main simulation codebase
│   ├── quantum_agrivoltaics_simulations.py    # Main simulation module
│   ├── quantum_agrivoltaics_simulations_refined.py  # Enhanced implementation
│   ├── quantum_dynamics_simulator.py          # PT-HOPS+LTC simulator
│   ├── agrivoltaic_coupling_model.py         # Coupling model implementation
│   ├── spectral_optimizer.py                 # Optimization algorithms
│   ├── eco_design_analyzer.py                # Eco-design analysis
│   ├── csv_data_storage.py                   # Data storage to CSV
│   ├── unified_figures.py                    # Visualization tools
│   └── data_input/                           # Input parameters
│       └── quantum_agrivoltaics_params.json  # JSON parameter file
└── data_output/                 # Output data (CSV files)
```

## Key Research Contributions

1. **Process Tensor-HOPS+LTC Framework**: Efficient treatment of Matsubara modes with Low-Temperature Correction for enhanced computational performance
2. **Mesoscale SBD Implementation**: Scalable approach for simulating large chromophore systems
3. **Quantum Reactivity Descriptors**: Fukui function-based eco-design for biodegradable OPV materials
4. **Multi-Objective Optimization**: Simultaneous optimization of PCE and biodegradability with ETR preservation
5. **E(n)-Equivariant Graph Neural Networks**: Physics-informed machine learning for molecular property prediction
6. **Agricultural Quality Enhancement**: Advanced metrics for crop productivity and quality

## Simulation Framework

### Quantum Dynamics Simulation
- **Liouvillian Superoperator**: Mathematical framework for Lindblad master equation in Liouville space
- **PT-HOPS+LTC Implementation**: Advanced non-Markovian dynamics with efficient low-temperature treatment
- **Stochastically Bundled Dissipators**: Method for handling large systems while preserving quantum effects

### Agrivoltaic Coupling Model
- **Hamiltonian Construction**: Tensor product approach for OPV-PSU coupling
- **Spectral Filtering**: Quantum transmission operators for optimal light distribution
- **Energy Transfer Dynamics**: Coupled system evolution simulation

### Data Management
- **JSON Parameter Configuration**: Centralized parameter management in `data_input/quantum_agrivoltaics_params.json`
- **CSV Data Output**: Comprehensive results saved to `data_output/` directory
- **Unified Visualization**: All figures generated through unified class in `unified_figures.py`

## Mathematical Framework

The implementation follows the theoretical foundation outlined in the thesis document, with the following key equations:

### Process Tensor Decomposition:
The bath correlation function C(t) is decomposed via Padé approximation:
K_PT(t,s) = Σₖ gₖ(t) fₖ(s) e^(-λₖ|t-s|) + K_non-exp(t,s)

### Low-Temperature Correction:
For T < 150K, Matsubara modes are efficiently treated with:
- N_Mat = 10 (Matsubara cutoff)
- eta_LTC = 10 (Time step enhancement factor)
- epsilon_LTC = 1e-8 (Convergence tolerance)

### Stochastically Bundled Dissipators:
L_SBD[ρ] = Σ_α p_α(t) D_α[ρ]
D_α[ρ] = L_α ρ L_α^† - ½{L_α^†L_α, ρ}

## Usage

The simulation framework can be run using the main module:
```python
from quantum_simulations_framework.quantum_agrivoltaics_simulations import run_complete_simulation
run_complete_simulation()
```

## Applications

This research framework enables the design of next-generation quantum-enhanced agrivoltaic systems that:
- Achieve high power conversion efficiency (>20%)
- Maintain agricultural productivity (ETR_rel >90%)
- Utilize biodegradable materials (>80% biodegradability)
- Implement quantum coherence effects for enhanced performance
- Support sustainable agriculture through symbiotic design
- Enable circular economy principles through eco-friendly materials