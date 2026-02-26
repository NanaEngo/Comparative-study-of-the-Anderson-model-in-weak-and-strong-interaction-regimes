# Submission TODO (BMAD)

**Goal**: Finalize EES submission package  
**Priority**: P0  
**Status**: Final Polish

---

## Manuscript
- [ ] Replace `[REPOSITORY_URL]` in `Q_Agrivoltaics_EES_Main.tex:111` with actual URL
- [ ] Standardize macros: use `physics` package for $\ket{\psi}$ and `siunitx` for \SI{750}{\nano\meter} throughout `files2/`
- [ ] Remove all `[TBD]` and `[TEMP]` placeholders from `results.tex` and `discussion.tex`

## Graphics (Critical)
- [ ] Generate/export Figures 1-7 as individual high-res PDF/PNG files to `Graphics/`
- [ ] Verify `Graphics/Graphical_Abstract_EES.png` exists at 600dpi, 5cm x 5cm
- [ ] Check sequential Fig 1-7 citations in main text match file order
- [ ] Verify Figures S1-S8 in `Supporting_Info_EES.tex` have corresponding files

## Bibliography
- [ ] Add DOIs to all `references.bib` entries; confirm `unsrt` style is active
- [ ] Delete uncited entries from `references.bib`
- [ ] Integrate SI-specific references into `Supporting_Info_EES.tex`

## Supporting Information
- [ ] Audit 12-test validation table values in SI against `tests/test_*.py` outputs
- [ ] Confirm $B_{\rm index}$ values in SI match `BiodegradabilityAnalyzer` final output
- [ ] Zip `quantum_simulations_framework/` and `quantum_coherence_agrivoltaics_mesohops_complete.ipynb` for repository upload

## Final Steps
- [ ] Add 3-5 reviewer suggestions to `Cover_Letter_EES.tex`
- [ ] Compile: `pdflatex` → `bibtex` → `pdflatex` (x2); fix any errors
- [ ] Upload to RSC portal

---
**BMAD**: Brief. Meaningful. Actionable. Direct.
