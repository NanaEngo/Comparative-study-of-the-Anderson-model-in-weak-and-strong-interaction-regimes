#!/usr/bin/env python3
"""
Script to diagnose JSON issues in the notebook
"""

import json

def diagnose_json(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        print("JSON is valid!")
        return notebook
    except json.JSONDecodeError as e:
        print(f"JSON Error at line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"Character position: {e.pos}")
        
        # Print lines around the error
        with open(notebook_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start = max(0, e.lineno - 5)
        end = min(len(lines), e.lineno + 5)
        
        print("\nLines around error:")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"{marker}{i+1:3d}: {lines[i].rstrip()}")
        
        return None

if __name__ == "__main__":
    notebook_path = "/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb"
    notebook = diagnose_json(notebook_path)
