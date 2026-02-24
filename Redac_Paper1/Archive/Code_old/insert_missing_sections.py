import json

file_path = '/home/taamangtchu/Documents/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops_complete.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Insert Key Numerical Results in Introduction Cell
intro_cell = nb['cells'][0]
new_intro_lines = []
for line in intro_cell['source']:
    new_intro_lines.append(line)
    if "Environmental impact assessment under realistic conditions" in line:  # End of Research Objectives
        new_intro_lines.append("\n")
        new_intro_lines.append("### Key Numerical Results Integrated\n")
        new_intro_lines.append("- **ETR enhancement**: Up to 25% under optimal spectral filtering\n")
        new_intro_lines.append("- **Validation success**: 12/12 tests passed (100%)\n")
        new_intro_lines.append("- **Biodegradability score**: 0.133 (normalized)\n")
        new_intro_lines.append("- **Maximum Fukui f+**: 0.311 at site 4\n")
        new_intro_lines.append("- **PM6 derivative (Molecule A)**: $B_{\\rm index} = 72$, >15% PCE\n")
        new_intro_lines.append("- **Y6-BO derivative (Molecule B)**: $B_{\\rm index} = 58$, >15% PCE\n")

intro_cell['source'] = new_intro_lines

# 2. Insert Spectral Density Models after FMO Hamiltonian Implementation
fmo_markdown_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        if any("FMO Hamiltonian Implementation" in line for line in cell['source']):
            fmo_markdown_idx = i

spectral_density_cell = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "### Spectral Density Models\n",
    "\n",
    "The spectral density $J(\\omega)$ characterizes the interaction between the quantum system and its environment, describing how environmental modes couple to the system at different frequencies.\n",
    "\n",
    "#### Drude-Lorentz Model\n",
    "The Drude-Lorentz model describes the continuous part of the spectral density with a Lorentzian shape, typical for protein environments:\n",
    "$$J(\\omega) = \\frac{2\\lambda_{reorg} \\gamma \\omega}{\\omega^2 + \\gamma^2} \\coth\\left(\\frac{\\hbar\\omega}{2kT}\\right)$$\n",
    "\n",
    "#### Vibronic Modes\n",
    "Discrete vibrational modes in the photosynthetic environment are modeled as underdamped oscillators:\n",
    "$$J_{vib}(\\omega) = \\sum_i \\frac{2\\pi S_i \\omega \\omega_i \\Gamma_i}{(\\omega^2 - \\omega_i^2)^2 + \\omega^2 \\Gamma_i^2}$$\n",
    "\n",
    "#### Total Spectral Density\n",
    "The total spectral density combines both continuous and discrete environmental effects:\n",
    "$$J_{total}(\\omega) = J_{drude}(\\omega) + J_{vib}(\\omega)$$\n",
    "\n",
    "This combined model captures the full environmental interaction relevant for photosynthetic quantum dynamics in agrivoltaic systems."
  ]
}

if fmo_markdown_idx != -1:
    # Insert after the FMO Hamiltonian Implementation markdown cell, or its corresponding code cell if present
    # Usually FMO markdown is followed by a code cell for FMO
    insert_at = fmo_markdown_idx + 1
    if insert_at < len(nb['cells']) and nb['cells'][insert_at]['cell_type'] == 'code':
        insert_at += 1
    
    nb['cells'].insert(insert_at, spectral_density_cell)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    f.write('\n')

print("Notebook updated with missing sections.")
