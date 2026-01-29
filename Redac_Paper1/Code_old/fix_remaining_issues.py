#!/usr/bin/env python3
"""
Script to fix remaining issues in the quantum coherence agrivoltaics notebook:
1. Update QFI calculation with more conservative threshold
2. Fix robustness analysis overlapping y-axis issue
"""

import re
import json

def fix_remaining_issues():
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Update QFI calculation to use more conservative threshold and add upper bound
    qfi_pattern = r'(\s*def calculate_qfi\(self, rho, H\):\n.*?for i in range\(n\):\n.*?for j in range\(n\):\n.*?if eigenvals\[i\] \+ eigenvals\[j\] > )1e-12(\s*:.*?hij = eigenvecs\[:, i\]\.conj\(\)\.T @ H @ eigenvecs\[:, j\].*?qfi \+= np\.abs\(hij\)\*\*2 / \(eigenvals\[i\] \+ eigenvals\[j\]\).*?\n\s*return 2 \* qfi)'
    
    def update_qfi(match):
        start = match.group(1)
        end = match.group(2)
        # Replace with more conservative threshold and add upper bound
        updated = start + "1e-8" + end + "\n        # Apply upper bound to prevent extremely large values\n        qfi = 2 * qfi\n        if qfi > 1e10:  # Cap extremely large values\n            qfi = 1e10\n        \n        return qfi"
        return updated
    
    # Apply the QFI fix
    updated_content = re.sub(qfi_pattern, update_qfi, content, flags=re.DOTALL)
    
    # Fix 2: Update the robustness analysis to fix overlapping y-axes
    robustness_pattern = r'(\s*plt\.twinx\(\)\.plot\(dephasing_rates, etr_at_dephasing,.*?\n\s*)(plt\.twinx\(\)\.set_ylabel.*?color=\'blue\'.*?\n\s*)(plt\.twinx\(\)\.legend\(loc=\'upper right\'.*?\n)'
    
    robustness_fix = '''\1
# Create twin axis object once and reuse it
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(dephasing_rates, etr_at_dephasing, 'b.--', label='Dephasing Sensitivity', 
         linewidth=2, markersize=8)
ax2.set_ylabel('ETR per Photon (Dephasing)', color='blue')
ax2.legend(loc='upper right')

'''
    
    updated_content = re.sub(robustness_pattern, robustness_fix, updated_content, flags=re.DOTALL)
    
    # Write the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Fixed QFI calculation and robustness analysis in the notebook.")


if __name__ == "__main__":
    fix_remaining_issues()