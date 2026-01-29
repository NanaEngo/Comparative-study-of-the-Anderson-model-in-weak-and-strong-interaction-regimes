#!/usr/bin/env python3
"""
Fix malformed JSON in Jupyter notebook files.
This script attempts to repair common JSON syntax errors in notebook files.
"""

import json
import re
import sys

def fix_json_content(content):
    """Attempt to fix common JSON syntax errors in notebook content."""
    
    # Fix escaped quotes in strings
    content = re.sub(r'\\"', '"', content)
    
    # Fix malformed string endings
    content = re.sub(r'"\n"\n"\n"\n"\n"', '"', content)
    
    # Fix missing commas between object properties
    content = re.sub(r'"(\w+)":\s*"([^"]*)"\s*"(\w+)":', r'"\1": "\2",\n"\3":', content)
    
    # Fix missing closing braces/brackets
    # Count opening and closing braces
    open_braces = content.count('{')
    close_braces = content.count('}')
    
    # Add missing closing braces if needed
    if open_braces > close_braces:
        content += '}' * (open_braces - close_braces)
    
    return content

def repair_notebook(input_file, output_file):
    """Repair a malformed notebook JSON file."""
    
    try:
        # First, try to read the file as-is
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Original file size: {len(content)} characters")
        
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            print("File is already valid JSON!")
            
            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
            
            print(f"Successfully wrote cleaned JSON to {output_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print("Attempting to fix common issues...")
            
            # Try to fix the content
            fixed_content = fix_json_content(content)
            
            # Try parsing again
            try:
                data = json.loads(fixed_content)
                print("Successfully fixed JSON!")
                
                # Write to output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=1)
                
                print(f"Successfully wrote repaired JSON to {output_file}")
                return True
                
            except json.JSONDecodeError as e2:
                print(f"Still unable to parse JSON after basic fixes: {e2}")
                
                # Try a more aggressive approach - extract valid JSON portions
                print("Attempting to extract valid JSON portions...")
                
                # Look for the main notebook structure
                notebook_pattern = r'\{\s*"cells":\s*\[.*?\]\s*,\s*"metadata":\s*\{.*?\}\s*,\s*"nbformat"\s*:\s*\d+\s*,\s*"nbformat_minor"\s*:\s*\d+\s*\}'
                match = re.search(notebook_pattern, content, re.DOTALL)
                
                if match:
                    try:
                        data = json.loads(match.group(0))
                        print("Successfully extracted valid notebook structure!")
                        
                        # Write to output file
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=1)
                        
                        print(f"Successfully wrote extracted JSON to {output_file}")
                        return True
                        
                    except json.JSONDecodeError as e3:
                        print(f"Even extracted structure is invalid: {e3}")
                        return False
                else:
                    print("Could not find valid notebook structure")
                    return False
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_notebook_json.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = repair_notebook(input_file, output_file)
    
    if success:
        print("Notebook repair completed successfully!")
    else:
        print("Notebook repair failed.")
        sys.exit(1)