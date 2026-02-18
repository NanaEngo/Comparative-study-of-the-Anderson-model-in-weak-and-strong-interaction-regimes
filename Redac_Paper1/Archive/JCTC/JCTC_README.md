# JCTC Manuscript - Modular Structure

## Overview
This directory contains the manuscript "Non-Markovian Quantum Dynamics for Spectral Optimization in Photosynthetic Systems" prepared for submission to the Journal of Chemical Theory and Computation (JCTC).

## File Structure

### Main Files
- **Goumai_JCTC_Main.tex** - Main manuscript file (compile this)
- **Supporting_Information.tex** - Supporting Information document
- **references.bib** - Bibliography (to be created/updated)

### Section Files (files/ directory)
- **introduction.tex** - Introduction section
- **theory_methods.tex** - Theory and computational methods
- **results.tex** - Results and validation
- **discussion.tex** - Discussion and implications
- **conclusion.tex** - Conclusion

### Graphics
- **Graphics/** - Directory containing figures (4 for main text, 6 for SI)

## Compilation

To compile the main manuscript:
```bash
pdflatex Goumai_JCTC_Main.tex
bibtex Goumai_JCTC_Main
pdflatex Goumai_JCTC_Main.tex
pdflatex Goumai_JCTC_Main.tex
```

To compile Supporting Information:
```bash
pdflatex Supporting_Information.tex
```

## Key Changes for JCTC

### Title
- **New**: "Non-Markovian Quantum Dynamics for Spectral Optimization in Photosynthetic Systems" (10 words)
- **Old**: "Process Tensor-HOPS with Low-Temperature Correction: A non-recursive framework for quantum-enhanced agrivoltaic design"

### Abstract
- Condensed from 5 paragraphs to 4 sentences
- Emphasizes: adHOPS methodology, 25% ETR enhancement, 12/12 validation success
- De-emphasizes: agricultural applications, market analysis

### Keywords
- **New order**: Non-Markovian Dynamics, Quantum Coherence, Adaptive HOPS, Photosynthetic Energy Transfer, Spectral Density Engineering, Open Quantum Systems, FMO Complex, Vibronic Coupling
- **Removed**: Agrivoltaics, Biodegradability, Environmental Factors (moved to SI)

### Content Focus
- **Main Text**: Quantum dynamics methodology, HOPS validation, coherence mechanisms
- **Supporting Info**: Environmental models, biodegradability details, agricultural context

## Validation Highlights

**12/12 Tests Passed (100% Success)**:
- 4 Convergence tests (HEOM benchmark, Matsubara, time step, hierarchy)
- 4 Physical consistency tests (trace, positivity, energy, detailed balance)
- 4 Environmental robustness tests (temperature, disorder, bath parameters, Markovian limit)

## Key Results

- **ETR Enhancement**: 25% improvement via spectral filtering
- **Coherence Lifetime**: 20-50% extension under optimal filtering
- **Delocalization**: 3-5 → 8-10 chromophores
- **Temperature Robustness**: Effects persist at 295 K
- **Disorder Tolerance**: Significant enhancement with σ = 50 cm⁻¹

## Figures

### Main Text (4 figures)
1. Quantum advantage vs spectral filtering parameters
2. Coherence dynamics under optimal filtering
3. Temperature dependence of quantum advantage
4. Pareto frontier (PCE vs biological ETR)

### Supporting Information (6 figures)
1. Spectral density components
2. Global reactivity indices
3. PAR transmission (clean vs dusty)
4. Response functions (OPV vs PSU)
5. Latitude/month climatic maps
6. ETR uncertainty distributions

## Status
- [x] Title revised for JCTC
- [x] Abstract condensed to 4 sentences
- [x] Keywords reordered
- [x] Modular structure created
- [x] Supporting Information template created
- [ ] Content migration to SI (in progress)
- [ ] Figure selection and placement
- [ ] Bibliography formatting
- [ ] Graphical abstract
- [ ] Cover letter

## Notes

The modular structure follows LaTeX best practices and makes the manuscript easier to:
- Revise individual sections independently
- Collaborate with multiple authors
- Track changes via version control
- Reorganize content for different journals if needed
- Maintain consistency across sections
