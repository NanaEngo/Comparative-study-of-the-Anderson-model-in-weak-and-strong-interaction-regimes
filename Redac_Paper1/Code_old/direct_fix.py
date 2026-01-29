#!/usr/bin/env python3
"""
Direct manual fix for the notebook JSON issues.
"""

import json
import re

def direct_fix():
    """Apply direct fixes to known issues."""
    
    # Read the file
    with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis.ipynb', 'r') as f:
        content = f.read()
    
    print(f"Original content length: {len(content)}")
    
    # Fix 1: Escape quotes in markdown content
    # Find all source arrays and fix quotes within them
    def fix_source_quotes(match):
        """Fix quotes within source array content."""
        source_content = match.group(0)
        
        # Replace unescaped quotes with escaped ones, but be careful about JSON structure
        # Look for quotes that are part of the markdown text, not JSON structure
        lines = source_content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip JSON structure lines
            if '"source": [' in line or line.strip() == ']' or line.strip() == '],':
                fixed_lines.append(line)
                continue
                
            # For content lines, escape quotes that aren't already escaped
            if line.strip().startswith('"') and line.strip().endswith('",'):
                # This is a content line with quotes
                content_part = line.strip()[1:-2]  # Remove surrounding quotes and comma
                
                # Escape any unescaped quotes in the content
                content_part = content_part.replace('"', '\"')
                
                # Reconstruct the line
                fixed_line = f'    "{content_part}",'
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    # Apply the source quote fixing
    content = re.sub(r'"source": \[.*?\](?:,)?', fix_source_quotes, content, flags=re.DOTALL)
    
    # Fix 2: Fix the structural issue at the end
    # Replace the problematic pattern
    content = content.replace(
        '\n" ]\n  }\n  ],\n "metadata"',
        '\n  ]\n },\n "metadata"'
    )
    
    # Fix 3: Clean up any remaining structural issues
    content = re.sub(r'\n\s*"\s*\]\s*\}\s*\]\s*,\s*"metadata"', '\n  ]\n },\n "metadata"', content)
    
    # Fix 4: Remove any double commas
    content = re.sub(r',\s*,', ',', content)
    
    # Fix 5: Fix any remaining unescaped quotes in a more targeted way
    # Look for quotes that appear to be in the middle of string content
    content = re.sub(r'([^\\])"([^",\]}\s])', r'\1\"\2', content)
    content = re.sub(r'([^\\])"([^",\]}\s])', r'\1\"\2', content)  # Do it twice for overlapping matches
    
    print(f"Fixed content length: {len(content)}")
    
    # Try to parse the fixed content
    try:
        data = json.loads(content)
        print("Successfully fixed JSON!")
        
        # Validate basic notebook structure
        required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
        if all(field in data for field in required_fields):
            print("Valid notebook structure!")
            
            # Write the fixed notebook
            with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/quantum_coherence_agrivoltaics_analysis_fixed.ipynb', 'w') as f:
                json.dump(data, f, indent=1)
            
            print("Fixed notebook written successfully!")
            return True
        else:
            print("Missing required notebook fields")
            return False
            
    except json.JSONDecodeError as e:
        print(f"Still invalid: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        
        # Save the attempted fix
        with open('/media/taamangtchu/MYDATA/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Redac_Paper1/direct_fix_attempt.json', 'w') as f:
            f.write(content)
        
        print("Attempted fix saved to direct_fix_attempt.json")
        return False

if __name__ == "__main__":
    success = direct_fix()
    if success:
        print("Direct fix completed successfully!")
    else:
        print("Direct fix failed.")