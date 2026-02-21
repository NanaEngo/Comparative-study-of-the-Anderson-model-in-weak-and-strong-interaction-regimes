
import re

def check_bib(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Improved regex to catch entries better
    entries = re.findall(r'@(\w+)\s*\{(.*?),[\s\n]*(.*?)\n\}', content, re.DOTALL)
    
    issues = []
    keys = []
    
    for entry_type, key, body in entries:
        if key in keys:
            issues.append(f"Duplicate key: {key}")
        keys.append(key)
        
        # Check year in key vs year field
        year_match = re.search(r'year\s*=\s*\{?(\d{4})\}?', body)
        if year_match:
            year = year_match.group(1)
            key_year = re.search(r'\d{4}', key)
            if key_year and key_year.group(0) != year:
                issues.append(f"Year mismatch in key '{key}': Key says {key_year.group(0)}, Field says {year}")
        else:
            if entry_type.lower() != 'comment':
                issues.append(f"Missing year in '{key}'")
            
        # Check for DOIs
        if entry_type.lower() != 'comment' and 'doi' not in body.lower():
            issues.append(f"Missing DOI in '{key}'")
            
        # Check author format
        author_match = re.search(r'author\s*=\s*\{(.*?)\}', body, re.DOTALL)
        if author_match:
            authors = author_match.group(1).replace('\n', ' ')
            if ' and ' in authors:
                parts = authors.split(' and ')
                for a in parts:
                    a = a.strip()
                    if ',' not in a and len(a.split()) > 1 and not a.startswith('{'):
                         # Basic check for "First Last" vs "Last, First"
                         # Some people use "First Last", but BibTeX prefers "Last, First" or "First Last" with "and"
                         # Actually both are okay if "and" is used, but consistency is better.
                         pass
        
        # Check for lowercase entry types
        if entry_type.lower() != 'comment' and entry_type[0].isupper():
            issues.append(f"Non-standard entry type capitalization in '{key}': @{entry_type}")

    return issues

if __name__ == "__main__":
    bib_file = "/home/taamangtchu/Documents/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/references.bib"
    results = check_bib(bib_file)
    for issue in results:
        print(issue)
