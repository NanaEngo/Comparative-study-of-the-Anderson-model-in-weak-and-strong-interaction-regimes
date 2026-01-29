#!/usr/bin/env python3
"""
Line-by-line JSON repair for the notebook.
"""

import json

def line_by_line_fix():
    """Fix the notebook by processing it line by line."""
    
    # Read the file
    with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', 'r') as f:
        lines = f.readlines()
    
    print(f"File has {len(lines)} lines")
    
    # Join lines and fix common issues
    content = ''.join(lines)
    
    # Fix specific issues found in the file
    
    # 1. Fix escaped quotes in markdown content
    content = content.replace('\\"', '"')
    
    # 2. Fix the problematic structure around the cells ending
    # The issue is: \n ]\n  }\n  ],\n "metadata"
    # Should be: \n  ]\n },\n "metadata"
    content = content.replace('\n ]\n  }\n  ],\n "metadata"', '\n  ]\n },\n "metadata"')
    
    # 3. Fix any double commas
    content = content.replace(',,', ',')
    
    # 4. Remove any trailing commas before closing braces/brackets
    content = content.replace(', }', ' }')
    content = content.replace(', ]', ' ]')
    
    # Try to parse the fixed content
    try:
        data = json.loads(content)
        print("Successfully fixed JSON!")
        
        # Write the fixed notebook
        with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb', 'w') as f:
            json.dump(data, f, indent=1)
        
        print("Fixed notebook written successfully!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Still invalid: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        
        # Show context around error
        error_pos = e.pos
        start = max(0, error_pos - 200)
        end = min(len(content), error_pos + 200)
        
        print("Context around error:")
        print(content[start:end])
        
        return False

if __name__ == "__main__":
    success = line_by_line_fix()
    if success:
        print("Line-by-line fix completed successfully!")
    else:
        print("Line-by-line fix failed.")