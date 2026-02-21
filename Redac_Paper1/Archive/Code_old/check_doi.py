#!/usr/bin/env python3

import re
import subprocess
import sys

def check_doi(doi):
    """Check if a DOI is valid by testing the URL"""
    url = f"https://doi.org/{doi}"
    try:
        # Use curl to check if the DOI resolves
        result = subprocess.run(['curl', '-s', '-I', url], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'HTTP' in result.stdout:
            # Extract the HTTP status code
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('HTTP/'):
                    status = line.split()[1]
                    if status.startswith('2') or status.startswith('3'):
                        return True, f"Valid (Status: {status})"
                    else:
                        return False, f"Invalid (Status: {status})"
        return False, "Could not verify"
    except Exception as e:
        return False, f"Error: {str(e)}"

def extract_dois_from_file(filepath):
    """Extract all DOIs from a BibTeX file"""
    dois = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Pattern to match DOI lines in BibTeX entries
    pattern = r'doi\s*=\s*\{([^}]+)\}'
    matches = re.findall(pattern, content)
    
    for match in matches:
        doi = match.strip()
        if doi:
            dois.append(doi)
    
    return dois

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_doi.py <bib_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Checking DOIs in {filepath}...")
    
    dois = extract_dois_from_file(filepath)
    print(f"Found {len(dois)} DOIs to check\n")
    
    valid_count = 0
    invalid_count = 0
    
    for i, doi in enumerate(dois, 1):
        print(f"{i:3d}. DOI: {doi}")
        is_valid, status = check_doi(doi)
        if is_valid:
            print(f"      Status: ✓ {status}")
            valid_count += 1
        else:
            print(f"      Status: ✗ {status}")
            invalid_count += 1
        print()
    
    print(f"Summary: {valid_count} valid, {invalid_count} invalid DOIs")
    
    # Write invalid DOIs to a file for review
    if invalid_count > 0:
        print("\nChecking invalid DOIs for potential corrections...")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find invalid DOIs and their context
        for i, doi in enumerate(dois):
            is_valid, _ = check_doi(doi)
            if not is_valid:
                # Find the context of this DOI in the file
                pattern = rf'doi\s*=\s*\{{\s*{re.escape(doi)}\s*\}}'
                match = re.search(pattern, content)
                if match:
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    print(f"\nInvalid DOI context:\n{context}")

if __name__ == "__main__":
    main()