#!/usr/bin/env python3
"""
Use nbformat to fix the notebook.
"""

import nbformat
import json

def try_nbformat_fix():
    """Try to fix the notebook using nbformat."""
    
    try:
        # Try to read the notebook with nbformat
        nb = nbformat.read('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', as_version=4)
        
        print("Successfully read notebook with nbformat!")
        
        # Write it back out
        nbformat.write(nb, '/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb')
        
        print("Fixed notebook written successfully!")
        return True
        
    except Exception as e:
        print(f"nbformat failed: {e}")
        return False

def try_json_fix_with_validation():
    """Try to fix JSON with validation."""
    
    # Read the file
    with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', 'r') as f:
        content = f.read()
    
    # Try to find valid JSON structure by looking for the main components
    
    # Find the cells array
    cells_start = content.find('"cells": [')
    if cells_start == -1:
        print("Could not find cells array")
        return False
    
    # Find the metadata section
    metadata_start = content.find('"metadata": {', cells_start)
    if metadata_start == -1:
        print("Could not find metadata section")
        return False
    
    # Find the end of the notebook
    nbformat_start = content.find('"nbformat":', metadata_start)
    if nbformat_start == -1:
        print("Could not find nbformat section")
        return False
    
    # Find the closing braces
    nbformat_end = content.find('}', nbformat_start)
    final_end = content.find('}', nbformat_end + 1)
    
    if final_end == -1:
        print("Could not find final closing brace")
        return False
    
    # Extract the main notebook structure
    notebook_json = content[:final_end + 1]
    
    print(f"Extracted JSON structure of length: {len(notebook_json)}")
    
    # Try to parse this
    try:
        data = json.loads(notebook_json)
        print("Successfully parsed extracted notebook structure!")
        
        # Validate it's a proper notebook
        if 'cells' in data and 'metadata' in data and 'nbformat' in data:
            print("Valid notebook structure found!")
            
            # Write the fixed notebook
            with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb', 'w') as f:
                json.dump(data, f, indent=1)
            
            print("Fixed notebook written successfully!")
            return True
        else:
            print("Missing required notebook fields")
            return False
            
    except json.JSONDecodeError as e:
        print(f"Extracted JSON still invalid: {e}")
        print(f"Error at position: {e.pos}")
        
        # Show context around error
        error_pos = e.pos
        start = max(0, error_pos - 100)
        end = min(len(notebook_json), error_pos + 100)
        
        print("Context around error:")
        print(notebook_json[start:end])
        
        return False

if __name__ == "__main__":
    print("Trying nbformat fix...")
    success1 = try_nbformat_fix()
    
    if not success1:
        print("\nTrying JSON extraction fix...")
        success2 = try_json_fix_with_validation()
        
        if success2:
            print("JSON extraction fix successful!")
        else:
            print("All fixes failed.")
    else:
        print("nbformat fix successful!")