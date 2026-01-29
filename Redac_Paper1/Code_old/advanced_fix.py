#!/usr/bin/env python3
"""
Advanced JSON repair for Jupyter notebooks.
"""

import json
import re

def extract_json_objects(text):
    """Extract JSON objects from text, handling malformed JSON."""
    
    # Pattern to match JSON-like structures
    # This is a more robust approach for extracting valid JSON
    
    # First, let's try to find the main notebook structure
    notebook_start = text.find('{"cells":')
    if notebook_start == -1:
        return None
    
    # Extract from the start to the end
    remaining_text = text[notebook_start:]
    
    # Try to balance braces and brackets
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    end_pos = 0
    
    for i, char in enumerate(remaining_text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            
        # Check if we've reached the end of the main JSON object
        if brace_count == 0 and i > 100:  # i > 100 to avoid matching empty objects
            end_pos = i + 1
            break
    
    if end_pos > 0:
        return remaining_text[:end_pos]
    
    return None

def fix_notebook_json():
    """Fix the notebook JSON using extraction and repair techniques."""
    
    # Read the corrupted file
    with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Try to extract valid JSON
    extracted_json = extract_json_objects(content)
    
    if extracted_json:
        print(f"Extracted JSON of size: {len(extracted_json)} characters")
        
        # Try to parse the extracted JSON
        try:
            data = json.loads(extracted_json)
            print("Successfully parsed extracted JSON!")
            
            # Write the fixed notebook
            with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb', 'w') as f:
                json.dump(data, f, indent=1)
            
            print("Fixed notebook written successfully!")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Extracted JSON still invalid: {e}")
            
            # Save the extracted content for manual inspection
            with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/extracted_json.json', 'w') as f:
                f.write(extracted_json)
            
            print("Extracted content saved to extracted_json.json for manual inspection")
            return False
    else:
        print("Could not extract valid JSON structure")
        return False

if __name__ == "__main__":
    success = fix_notebook_json()
    if success:
        print("Notebook fixing completed successfully!")
    else:
        print("Notebook fixing failed.")