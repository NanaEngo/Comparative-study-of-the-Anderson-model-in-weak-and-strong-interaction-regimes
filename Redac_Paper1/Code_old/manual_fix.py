#!/usr/bin/env python3
"""
Manual fix for the quantum_coherence_agrivoltaics_analysis.ipynb file.
"""

import json
import re

def manual_fix_notebook():
    """Manually fix the notebook JSON structure."""
    
    # Read the corrupted file
    with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', 'r') as f:
        content = f.read()
    
    print(f"Original file length: {len(content)}")
    
    # Find the problematic area and fix it
    # The issue is around the ending of the cells array
    
    # Look for the pattern that's causing the issue
    problematic_pattern = r'\n\]\s*\}\s*\]\s*,\s*"metadata"'
    
    # Replace with proper structure
    fixed_content = re.sub(
        problematic_pattern, 
        '\n  ]\n },\n "metadata"', 
        content
    )
    
    # Also fix any escaped quotes
    fixed_content = re.sub(r'\\"', '"', fixed_content)
    
    print(f"Fixed file length: {len(fixed_content)}")
    
    # Try to parse the fixed content
    try:
        data = json.loads(fixed_content)
        print("Successfully fixed JSON!")
        
        # Write the fixed content
        with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb', 'w') as f:
            json.dump(data, f, indent=1)
        
        print("Fixed notebook written to quantum_coherence_agrivoltaics_analysis_fixed.ipynb")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Still invalid JSON: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        
        # Show the context around the error
        lines = fixed_content.split('\n')
        start = max(0, e.lineno - 5)
        end = min(len(lines), e.lineno + 5)
        
        print("Context around error:")
        for i in range(start, end):
            marker = ">>> " if i == e.lineno - 1 else "    "
            print(f"{marker}{i+1}: {lines[i]}")
        
        return False

if __name__ == "__main__":
    success = manual_fix_notebook()
    if success:
        print("Manual fix successful!")
    else:
        print("Manual fix failed. Need to try different approach.")