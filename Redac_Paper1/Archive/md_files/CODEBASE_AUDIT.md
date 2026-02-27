# Codebase Audit Report

Date: 2026-02-26
Repository: Quantum_Agrivoltaic_HOPS/Redac_Paper1

## Scope

- Map repository against documented architecture (AGENTS.md)
- Inspect packaging, dependencies, tests, and runtime entry points
- Identify risks, inconsistencies, and missing elements
- Provide prioritized, actionable fixes

## Summary

The repository is well-structured and aligns closely with the documented architecture. Core simulation components, models, utilities, and extensions are present and functional. A unified simulator (HopsSimulator) provides a façade over MesoHOPS with sensible fallbacks. Tests exist across domains, and data/figures are timestamped and organized. Key gaps include packaging metadata for installability, import-path robustness, MesoHOPS API coupling risks, configuration validation, CI/coverage, and minor documentation drift.

## Findings

### Architecture alignment
- Matches AGENTS.md: core/, models/, simulations/, utils/, extensions/ are present and populated.
- quantum_simulations_framework/__init__.py exports HopsSimulator and TestingValidationProtocols, plus models and logging utilities.
- Data I/O and figures are organized under simulation_data/ and Graphics/ with timestamps.
- Manuscripts and LaTeX structure are complete (Q_Agrivoltaics_EES_Main.tex, Supporting_Info_EES.tex, files2/ sections).

### Packaging and distribution
- pyproject.toml lacks [project] metadata (name, version, description, authors, dependencies) and setuptools config (packages discovery). Build system block exists but minimal.
- requirements.txt and requirements-mesohops.txt exist; no constraints/lock for reproducible builds.
- No explicit Python version declaration.

### Import hygiene and robustness
- core/hops_simulator.py uses layered absolute + relative imports via try/except. This can be fragile across contexts (notebook runs vs installed package).
- Absolute-first imports (e.g., from extensions..., from models...) may fail if not installed as a package; relative imports should be standard inside the package.

### MesoHOPS integration risks
- Assumes mesohops.util.bath_corr_functions.bcf_convert_dl_to_exp and mesohops.trajectory.exp_noise.bcf_exp exist with certain signatures; version drift may break.
- Trajectory storage keys ("t_axis", "psi_traj") are assumed stable.
- Noise engine reassignment (trajectory.noise vs noise_engine) is version-dependent; partial guards exist.
- Drude-Lorentz correlation fallback is simplified; ensure unit consistency and document approximations.

### Fallback simulator API coupling
- Fallback relies on models.SimpleQuantumDynamicsSimulator and QuantumDynamicsSimulator. _init_fallback filters kwargs via a hardcoded allowlist; additions require updates to keep in sync.

### Tests and CI
- Tests exist in quantum_simulations_framework/tests. No CI workflow present (no GitHub Actions config).
- No coverage reporting or thresholds to protect critical paths.

### Configuration and reproducibility
- data_input/ contains CSV/JSON, but there is no schema-based validation (e.g., pydantic/jsonschema) for parameters.
- Random seed is hardcoded (42) in noise generation; global reproducibility across pipeline not enforced.

### Logging and error handling
- Central logging utils exist. Some code prints tracebacks directly; prefer structured logging with logger.exception.
- Mixed behavior between logging and raising errors; standardize for clarity.

### Documentation drift
- AGENTS.md mentions quantum_coherence_agrivoltaics_mesohops.py as consolidated entry; repository has quantum_coherence_agrivoltaics_mesohops_complete.py and a .ipynb instead.
- AGENTS.md places spectral_optimizer under simulations/; current SpectralOptimizer resides in models/ while simulations/ focuses on validation/standardization. Functionally fine but mismatched with docs.

### ORCA/quantum chemistry integration
- utils/orca_wrapper.py and tests/test_orca_runner.py imply external ORCA dependency. INSTALLATION.md lacks ORCA setup instructions and tests may fail without it.

### Manuscript build automation
- No script/Makefile to compile LaTeX, verify figure references, and check bibliography consistency. Legacy guides exist in Archive/md_files but not a current streamlined build.

## Strengths
- Clear modular design with unified simulator façade and sensible fallbacks.
- Extensive figure/CSV generation with timestamping for provenance.
- Tests span core, models, utils, and MesoHOPS integration.
- Good separation of manuscript assets and simulation outputs.

## Prioritized action plan

1) Packaging and installability (highest ROI)
- Add [project] metadata to pyproject.toml:
  - name, version (single source of truth; optionally read from package), description, authors, license, requires-python, dependencies (sync from requirements.txt).
  - Optional extras: "mesohops" to include mesohops and performance extras.
- Add [tool.setuptools] / [tool.setuptools.packages.find] to discover packages and include package data if needed.
- Document editable install in README: pip install -e .

2) Import hygiene and stability
- Standardize relative imports within package modules:
  - In core/hops_simulator.py, use from ..extensions... and from ..models... consistently; remove double try/except import ladders.
- Encourage user code to import only through quantum_simulations_framework public API.

3) MesoHOPS compatibility layer
- Create quantum_simulations_framework/extensions/mesohops_api.py to centralize:
  - Version detection and safe imports for bcf_convert_dl_to_exp and bcf_exp.
  - System parameter construction (H/L operators, GW_SYSBATH) with unit handling.
  - Trajectory data extraction helpers tolerant of key changes.
- Refactor HopsSimulator to use this adapter, reducing coupling to mesohops internals.

4) Configuration validation
- Introduce a config schema using pydantic or jsonschema to validate quantum_agrivoltaics_params.json and runtime kwargs.
- Provide defaults, ranges, and human-readable validation errors.

5) Tests and CI
- Add a GitHub Actions workflow:
  - Matrix: Ubuntu, Python 3.9–3.12.
  - Steps: pip install -e .; pip install -r requirements.txt; pytest -q.
  - Separate optional job for mesohops-enabled tests, guarded by a pytest marker and conditional install of requirements-mesohops.txt.
- Add pytest-cov with a baseline coverage threshold (e.g., 70%) for core and models.

6) Documentation alignment and entry points
- Update AGENTS.md and README.md to either:
  - Point to quantum_coherence_agrivoltaics_mesohops_complete.py as the consolidated script, or
  - Create quantum_coherence_agrivoltaics_mesohops.py as documented, importing/forwarding the complete module API.
- Align module location descriptions (simulations/ vs models/) or update diagrams to current reality.

7) ORCA dependency handling
- Extend INSTALLATION.md with ORCA setup instructions, environment variables, and license acceptance notes.
- Mark ORCA-dependent tests with a pytest marker (e.g., @pytest.mark.orca) and skip if ORCA is not detected.
- Provide runtime checks to gracefully disable ORCA features when unavailable.

8) Logging and error handling
- Expose setup_logging(level=...) in the package’s public API and document usage.
- Replace print/traceback with logger.exception where appropriate; avoid stdout pollution.
- Normalize error messages with actionable hints (e.g., "Install mesohops or set use_mesohops=False").

9) Reproducibility
- Add a seed management helper to set numpy, random, and library seeds consistently; log seed usage.
- Optionally add a constraints.txt to pin versions and reduce breakages.

10) Manuscript build automation
- Add a Makefile or script to:
  - Compile LaTeX with latexmk
  - Verify that referenced figures exist in Graphics/
  - Run bibliography tool and link checking
  - Produce a summary artifact list ready for submission

## Quick wins (suggested next changes)
- Create mesohops_api adapter and refactor HopsSimulator to use it.
- Add pyproject [project] metadata and extras for mesohops.
- Add pytest markers and skips for mesohops and ORCA dependent tests.
- Update README with minimal install and quick-start using the public API.

## Conventions to adopt
- Document and centralize units (Hamiltonian in cm^-1, time in fs/ps, temperature in K) in constants or a units module. Ensure consistent conversions.
- Centralize quantum metric calculations in a single engine to avoid duplication and numeric drift. Keep safeguards (e.g., eigenvalue clipping) standardized.
- Prefer dataclasses or pydantic models for simulation parameters passed through multiple layers.

## Verification checklist
- pip install -e .
- python -c "import quantum_simulations_framework as qsf; print(qsf.__version__)"
- pytest -q
- Minimal runtime sanity test:
  ```python
  import numpy as np
  from quantum_simulations_framework import HopsSimulator
  H = np.array([[12450., 100.],[100., 12500.]])
  sim = HopsSimulator(H, temperature=295, use_mesohops=False)
  t = np.linspace(0,100,50)
  res = sim.simulate_dynamics(t)
  assert res['populations'].shape[0] == len(t)
  ```

## Outcome

The codebase is close to production quality with strong modularity and functionality. Addressing packaging, API stability around MesoHOPS, configuration validation, CI/coverage, ORCA handling, and documentation alignment will harden the project for installation, testing, reproducibility, and submission workflows.