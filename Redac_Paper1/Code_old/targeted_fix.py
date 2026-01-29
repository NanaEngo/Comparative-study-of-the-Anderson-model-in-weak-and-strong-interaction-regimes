#!/usr/bin/env python3
"""
Targeted fix for specific JSON issues in the notebook.
"""

import json
import re

def fix_unescaped_quotes():
    """Fix unescaped quotes in the notebook content."""
    
    # Read the file
    with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', 'r') as f:
        content = f.read()
    
    # Fix the specific unescaped quotes in the markdown content
    # The issue is: described in the paper "Process Tensor-HOPS with Low-Temperature Correction: A non-recursive framework for quantum-enhanced agrivoltaic design"
    # The quotes around the paper title need to be escaped
    
    # Pattern to find unescaped quotes in markdown strings
    # Look for quotes that are inside JSON string values but not escaped
    content = re.sub(
        r'("source":\s*\[\s*"[^"]*")"([^"]*"[^"]*")"([^"]*")',
        r'\1\\"\2\\"\3',
        content
    )
    
    # More specific fix for the known problematic line
    content = content.replace(
        'described in the paper "Process Tensor-HOPS with Low-Temperature Correction: A non-recursive framework for quantum-enhanced agrivoltaic design"',
        'described in the paper \\"Process Tensor-HOPS with Low-Temperature Correction: A non-recursive framework for quantum-enhanced agrivoltaic design\\"'
    )
    
    # Fix any other obvious unescaped quotes in markdown
    # Look for patterns where quotes appear between other quotes in source arrays
    content = re.sub(
        r'("source":\s*\[.*?)([^\\])"([^"]*?)"([^"]*?\])',
        r'\1\2\\"\3\\"\4',
        content,
        flags=re.DOTALL
    )
    
    # Fix the structure issue at the end of cells
    content = content.replace('\n ]\n  }\n  ],\n "metadata"', '\n  ]\n },\n "metadata"')
    
    # Remove any double commas
    content = content.replace(',,', ',')
    
    # Remove trailing commas before closing braces/brackets
    content = re.sub(r',\s*\}', ' }', content)
    content = re.sub(r',\s*\]', ' ]', content)
    
    return content

def targeted_fix():
    """Apply targeted fixes to the notebook."""
    
    # Apply the quote fixing
    fixed_content = fix_unescaped_quotes()
    
    # Try to parse the fixed content
    try:
        data = json.loads(fixed_content)
        print("Successfully fixed JSON!")
        
        # Write the fixed notebook
        with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb', 'w') as f:
            json.dump(data, f, indent=1)
        
        print("Fixed notebook written successfully!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Still invalid: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        
        # Save the attempted fix for inspection
        with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/fixed_content_attempt.json', 'w') as f:
            f.write(fixed_content)
        
        print("Attempted fix saved to fixed_content_attempt.json")
        return False

if __name__ == "__main__":
    success = targeted_fix()
    if success:
        print("Targeted fix completed successfully!")
    else:
        print("Targeted fix failed.")