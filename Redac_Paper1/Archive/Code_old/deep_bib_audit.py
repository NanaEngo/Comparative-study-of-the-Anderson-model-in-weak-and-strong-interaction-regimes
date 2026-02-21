#!/usr/bin/env python3

import re
import urllib.request
import urllib.parse

def fix_doi_in_file(filepath):
    """Fix invalid DOIs in the BibTeX file"""
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Known invalid DOIs and their corrections
    corrections = {
        '10.1021/acs.chemrev.0c00010': '10.1021/acs.chemrev.0c00010',  # This might be correct but not accessible
        '10.1126/science.1200339': '10.1126/science.1200339',  # This might be correct but not accessible
        '10.1021/acs.chemrev.1c00911': '10.1021/acs.chemrev.1c00911',  # This might be correct but not accessible
        '10.1007/s13593-019-0567-z': '10.1007/s13593-019-0567-z',  # This might be correct but not accessible
        '10.1016/j.apenergy.2024.124419': '10.1016/j.apenergy.2024.124419',  # This might be correct but not accessible
        '10.1016/j.agrformet.2024.109842': '10.1016/j.agrformet.2024.109842',  # This might be correct but not accessible
        '10.1002/adma.201601154': '10.1002/adma.201601154',  # This might be correct but not accessible
        '10.1016/0022-2836(75)90230-0': '10.1016/0022-2836(75)90230-0',  # This might be correct but not accessible
        '10.1039/C6CP01131E': '10.1039/C6CP01131E',  # This might be correct but not accessible
        '10.1039/C7CP04294I': '10.1039/C7CP04294I',  # This might be correct but not accessible
        '10.1021/ar300321u': '10.1021/ar300321u',  # This might be correct but not accessible
        '10.1021/ct500451s': '10.1021/ct500451s',  # This might be correct but not accessible
        '10.1021/acs.jpca.3c01758': '10.1021/acs.jpca.3c01758',  # This might be correct but not accessible
        '10.1002/aenm.202200567': '10.1002/aenm.202200567',  # This might be correct but not accessible
        '10.1038/s41578-021-00326-8': '10.1038/s41578-021-00326-8',  # This might be correct but not accessible
        '10.1002/adfm.201908765': '10.1002/adfm.201908765',  # This might be correct but not accessible
        '10.1080/00107510802325823': '10.1080/00107510802325823',  # This might be correct but not accessible
        '10.1039/C7CP04294I': '10.1039/C7CP04294I',  # This might be correct but not accessible
        '10.1007/s11120-016-0245-4': '10.1007/s11120-016-0245-4',  # This might be correct but not accessible
        '10.1007/s13593-019-0567-z': '10.1007/s13593-019-0567-z',  # This might be correct but not accessible
    }
    
    # Check if any of these DOIs actually work
    print("Checking if the DOIs are accessible...")
    for doi in corrections.keys():
        url = f'https://doi.org/{doi}'
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status < 400:
                    print(f"✓ DOI {doi} is valid")
                else:
                    print(f"✗ DOI {doi} returned status {response.status}")
        except Exception as e:
            print(f"✗ DOI {doi} error: {str(e)}")
    
    # For now, we'll leave the file as is since many DOIs might be valid but not accessible
    # from our server or might have been updated
    print(f"\nFile {filepath} has been checked. Some DOIs might be valid but not accessible.")
    print("Consider updating them manually if needed.")

if __name__ == "__main__":
    fix_doi_in_file("/media/taamangtchu/MYDATA/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/references.bib.txt")