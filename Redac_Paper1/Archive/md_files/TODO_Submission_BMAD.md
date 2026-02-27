# Submission TODO (BMAD)

**Goal**: Finalize EES submission package — data-aligned, citation-clean, publication-ready  
**Priority**: P0  
**Status**: Manuscript Refinement Phase  
**Source of truth**: [`CSV_Data_Analysis.md`](CSV_Data_Analysis.md)

---

## A — Bibliography (BLOCKING)

- [x] **Add 6 missing BibTeX keys** — remapped to existing keys (completed)
- [x] **Confirm bibliography style** — switched `unsrt` → `rsc` in both main and SI tex files

## B — Main Manuscript → CSV Data Alignment

### Abstract & Broader Context (`Q_Agrivoltaics_EES_Main.tex`)
- [x] Reconcile PCE range: 16–18 % → 15–19 % (covers CSV 18.83 %)
- [x] Reconcile B-index: 72 → 101.5 (PM6)
- [x] Reconcile biodegradability score: 0.72 → 1.02 (PM6)
- [x] Added <0.2 %/yr stability finding to broader context

### Results (`files2/results.tex`)
- [x] Insert 30 fs population transfer time from CSV
- [x] Insert peak l₁-norm coherence = 0.988
- [x] Insert QFI initial value (32 348)
- [x] Add 365-day environmental stability: <0.2 %/yr PCE and ETR degradation
- [x] Add chemical reactivity descriptors (μ = −4.30 eV, η = 1.10 eV, ω = 8.40 eV)
- [x] Update B-index 72 → 101.5, score 0.72 → 1.02, eco-design score 0.78 → 1.12

### Discussion (`files2/discussion.tex`)
- [x] Add operational stability data (365-day, dust accumulation, degradation)

### Conclusion (`files2/conclusion.tex`)
- [x] Update B-index 72 → 101.5, score 0.72 → 1.02
- [x] Insert <0.2 % annual degradation finding

## C — Supporting Information
- [x] Insert environmental effects summary table (365 days) from CSV into SI
- [x] Update eco-design descriptor values (μ, η, ω, B-index 101.5, score 1.12) in SI
- [ ] Cross-check 12-test validation table vs `tests/test_*.py` outputs
- [x] B-index values match CSV output

## D — Graphics
- [x] All 10 referenced figure files present in `Graphics/`
- [x] 4 main figures copied from latest timestamped outputs
- [x] 6 SI placeholder PDFs generated
- [x] Graphical abstract present (`Graphical_Abstract_EES.pdf`)

## E — Final Steps
- [x] Fixed `sec:discussion` → `sec:Discussion` label mismatch
- [x] Fixed `Asaa2024` BibTeX key (removed apostrophe)
- [x] Switched `unsrt` → `rsc` bibliography style
- [x] LaTeX compilation — zero errors, zero undefined citations
- [x] Main: 20 pages, SI: 19 pages

---

## F — Audit Recommendations
- [x] Add a short “Limitations & Future Work” paragraph explicitly acknowledging the FMO‑centric focus.
- [x] Insert a quantitative comparison table between the quantum‑optimized OPV design and a state‑of‑the‑art classical design.
- [x] Verify all numerical values against `CSV_Data_Analysis.md` and correct any mismatches.
- [x] Perform a final spell‑check and ensure the cover letter mirrors the manuscript’s key numbers.
- [x] Run a LaTeX compilation check to confirm no missing references or package conflicts.

**BMAD**: Brief. Meaningful. Actionable. Direct.
