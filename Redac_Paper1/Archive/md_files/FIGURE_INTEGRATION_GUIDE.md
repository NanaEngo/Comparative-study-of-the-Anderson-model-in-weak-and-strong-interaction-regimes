% Figure Integration Guide for EES Manuscript

## Main Text Figures - Placement Recommendations

### Figure 1: Quantum Advantage in Energy Transfer
**Location**: Results section, after describing ETR enhancement calculations  
**File**: `Graphics/Quantum_Advantage_in_Energy_Transfer.pdf`  
**Caption**: Quantum enhancement of electron transport rate (ETR) through strategic spectral filtering. (a) Comparison of ETR between quantum-optimized filtered spectrum (750/820 nm dual-band, orange) and classical full-spectrum illumination (blue), demonstrating 25\% enhancement. (b) Spectral filtering profile showing OPV transmission windows aligned with vibronic resonances in photosynthetic complex. Temperature: 295 K.

**LaTeX code**:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{Graphics/Quantum_Advantage_in_Energy_Transfer.pdf}
\caption{Quantum enhancement of electron transport rate (ETR) through strategic spectral filtering. (a) Comparison of ETR between quantum-optimized filtered spectrum (750/820 nm dual-band, orange) and classical full-spectrum illumination (blue), demonstrating 25\% enhancement. (b) Spectral filtering profile showing OPV transmission windows aligned with vibronic resonances. Temperature: 295 K.}
\label{fig:quantum_advantage}
\end{figure}
```

---

### Figure 2: Pareto Frontier (PCE vs ETR Trade-off)
**Location**: Results section, after Pareto optimization discussion  
**File**: `Graphics/Pareto_Front__PCE_vs_ETR_Trade_off.pdf`  
**Caption**: Multi-objective optimization results showing trade-off between OPV power conversion efficiency (PCE) and photosynthetic energy transfer enhancement. Three optimal configurations identified: (1) Balanced (16-18\% PCE, 15-20\% ETR, blue star), (2) Energy-focused (19-21\% PCE, 8-12\% ETR, red circle), (3) Agriculture-focused (13-15\% PCE, 22-25\% ETR, green square). Pareto frontier (black line) represents non-dominated solutions.

**LaTeX code**:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.75\textwidth]{Graphics/Pareto_Front__PCE_vs_ETR_Trade_off.pdf}
\caption{Multi-objective optimization showing trade-off between OPV power conversion efficiency (PCE) and photosynthetic ETR enhancement. Three optimal configurations: Balanced (16-18\% PCE, 15-20\% ETR), Energy-focused (19-21\% PCE, 8-12\% ETR), Agriculture-focused (13-15\% PCE, 22-25\% ETR). Pareto frontier (black line) represents non-dominated solutions.}
\label{fig:pareto}
\end{figure}
```

---

### Figure 3: Spectral Optimization Concept
**Location**: Introduction or Theory section, to illustrate concept  
**File**: `Graphics/spectral_optimization.png`  
**Caption**: Spectral bath engineering concept for agrivoltaic optimization. (a) System schematic showing solar spectrum filtered through semi-transparent OPV panel before reaching crops. (b) Optimized transmission spectrum (750/820 nm dual-band, FWHM 70 nm) maximizes vibronic resonance overlap. (c) Enhanced quantum coherence in photosynthetic complex enables efficient energy transfer.

**LaTeX code**:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{Graphics/spectral_optimization.png}
\caption{Spectral bath engineering for agrivoltaic optimization. (a) System schematic: solar spectrum filtered through semi-transparent OPV. (b) Optimized transmission (750/820 nm, FWHM 70 nm) maximizes vibronic resonance. (c) Enhanced quantum coherence enables efficient energy transfer.}
\label{fig:concept}
\end{figure}
```

---

### Figure 4: Coherence Dynamics
**Location**: Results section, after coherence discussion  
**File**: `Graphics/quantum_dynamics.png`  
**Caption**: Quantum coherence dynamics in FMO complex under optimized spectral filtering. (a) Time evolution of electronic coherences showing 20-50\% lifetime extension compared to classical illumination. (b) Exciton delocalization increasing from 3-5 sites (classical) to 8-10 sites (quantum-optimized). (c) Population transfer efficiency as function of time, demonstrating accelerated energy transfer through coherent pathways. Shaded regions indicate standard deviation over 100 disorder realizations.

**LaTeX code**:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{Graphics/quantum_dynamics.png}
\caption{Quantum coherence dynamics in FMO complex under optimized illumination. (a) Electronic coherence evolution showing 20-50\% lifetime extension. (b) Exciton delocalization: 3-5 sites (classical) to 8-10 sites (quantum). (c) Population transfer efficiency vs time. Shaded regions: standard deviation (100 disorder realizations).}
\label{fig:dynamics}
\end{figure}
```

---

### Figure 5: Environmental Robustness
**Location**: Results section, after validation discussion  
**File**: `Graphics/ETR_Under_Environmental_Effects.pdf`  
**Caption**: Quantum advantage robustness under realistic environmental variations. (a) Temperature dependence (280-320 K) showing 18-26\% ETR enhancement maintained across physiological range. (b) Static disorder tolerance: quantum advantage persists despite 20\% reduction at σ = 50 cm$^{-1}$ (typical protein disorder). (c) Geographic applicability across climate zones: temperate (22-26\%), subtropical (24-28\%), tropical (24-26\%), desert (18-24\%). Error bars represent 95\% confidence intervals.

**LaTeX code**:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{Graphics/ETR_Under_Environmental_Effects.pdf}
\caption{Environmental robustness of quantum advantage. (a) Temperature (280-320 K): 18-26\% ETR enhancement. (b) Disorder tolerance: advantage persists at σ = 50 cm$^{-1}$. (c) Geographic applicability: temperate (22-26\%), subtropical (24-28\%), tropical (24-26\%), desert (18-24\%). Error bars: 95\% CI.}
\label{fig:robustness}
\end{figure}
```

---

### Figure 6 (Optional): Economic Analysis
**Location**: Discussion section, with economic impact  
**File**: `Graphics/eco_design_analysis.png`  
**Caption**: Economic viability analysis for quantum-optimized agrivoltaic systems. (a) Revenue comparison: classical (35\% PV coverage) vs quantum (40\% coverage) configurations per hectare. (b) Net present value over 20-year lifetime showing \$9,400/ha additional value. (c) Sensitivity analysis: economic benefit ranges from \$470/ha/year (commodity crops) to \$3,000/ha/year (high-value specialty crops). Installation cost amortized over system lifetime (25 years).

**LaTeX code**:
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{Graphics/eco_design_analysis.png}
\caption{Economic viability of quantum-optimized agrivoltaics. (a) Revenue comparison: classical vs quantum per hectare. (b) 20-year NPV: \$9,400/ha additional. (c) Sensitivity: \$470-3,000/ha/year depending on crop value.}
\label{fig:economics}
\end{figure}
```

---

## Supporting Information Figures

### SI Figure S1: Spectral Density
**File**: `Graphics/Spectral_Density_Components_for_FMO_Environment.pdf`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{Graphics/Spectral_Density_Components_for_FMO_Environment.pdf}
\caption{Spectral density components for FMO environment. Overdamped (Drude) contribution (blue, λ = 35 cm$^{-1}$, γ = 50 cm$^{-1}$) and underdamped vibronic modes (orange peaks at 150, 200, 575, 1185 cm$^{-1}$). Total spectral density J(ω) shown in black.}
\label{fig:spectral_density}
\end{figure}
```

### SI Figure S2: Biodegradability
**File**: `Graphics/Global_Reactivity_Indices.pdf`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{Graphics/Global_Reactivity_Indices.pdf}
\caption{Global reactivity indices for biodegradable OPV candidates. Fukui functions f$^+$ (electrophilic, red) and f$^-$ (nucleophilic, blue) map reactive sites. Chemical hardness η and biodegradability index B shown for Molecule A (highly biodegradable, \textless 6 months) and Molecule B (moderately biodegradable, 6-18 months).}
\label{fig:biodegradability}
\end{figure}
```

### SI Figure S3: PAR Transmission
**File**: `Graphics/PAR_Transmission__Clean_vs_Dusty_Conditions.pdf`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{Graphics/PAR_Transmission__Clean_vs_Dusty_Conditions.pdf}
\caption{Photosynthetically active radiation (PAR) transmission under varying dust accumulation. Clean surface (black), 30-day accumulation (blue), 90-day accumulation (red). Quantum resonance windows (750/820 nm) remain effective despite 10-15\% transmission reduction.}
\label{fig:par_transmission}
\end{figure}
```

### SI Figure S4: Response Functions
**File**: `Graphics/Response_Functions__OPV_vs_PSU.pdf`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{Graphics/Response_Functions__OPV_vs_PSU.pdf}
\caption{Spectral response functions for OPV (organic photovoltaic, blue) and PSU (photosynthetic unit, orange). Optimal design (750/820 nm dual-band) minimizes overlap for energy harvesting while maximizing vibronic resonance enhancement for photosynthesis.}
\label{fig:response}
\end{figure}
```

### SI Figure S5: Climate Map
**File**: `Graphics/fLatitude__lat__u00b0__Month__month.pdf`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{Graphics/fLatitude__lat__u00b0__Month__month.pdf}
\caption{Geographic and seasonal variation of quantum advantage. Contour map showing ETR enhancement (\%) as function of latitude and month. Year-round viability confirmed across temperate (40-60°N), subtropical (15-35°N), tropical (0-15°), and desert regions. Color scale: 18-28\% enhancement.}
\label{fig:climate}
\end{figure}
```

### SI Figure S6: ETR Uncertainty
**File**: `Graphics/ETR_Uncertainty_Distribution.pdf`
```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{Graphics/ETR_Uncertainty_Distribution.pdf}
\caption{Statistical distribution of ETR enhancement from disorder ensemble (N = 100 realizations, σ = 50 cm$^{-1}$). Mean: 20\%, standard deviation: 4\%. Histogram (blue bars) with Gaussian fit (red line). Inset: quantile-quantile plot confirming near-normal distribution.}
\label{fig:uncertainty}
\end{figure}
```

---

## Quick Reference for Manuscript Integration

**In results.tex, add after relevant paragraphs**:
- Line ~50-60: Figure 1 (Quantum Advantage) after ETR discussion
- Line ~120-140: Figure 2 (Pareto) after optimization results
- Line ~180-200: Figure 4 (Dynamics) after coherence analysis  
- Line ~250-270: Figure 5 (Robustness) after validation

**In introduction.tex or theory.tex**:
- Early pages: Figure 3 (Conceptual framework)

**In discussion.tex**:
- With economic analysis: Figure 6 (Economics) optional

**All figures ready to insert - no creation needed!**
