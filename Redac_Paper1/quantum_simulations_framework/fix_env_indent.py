import re

def fix_indent(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to find `class EnvironmentalFactors:`
    # and indent everything until the next class or top-level def
    lines = content.split('\n')
    in_env_class = False
    
    for i, line in enumerate(lines):
        if line.strip() == "class EnvironmentalFactors:":
            in_env_class = True
            continue
            
        if in_env_class:
            # If we hit the end of the class (e.g. another top-level def or `# Set publication style plots`)
            if line.startswith("def ") or line.startswith("import ") or line.startswith("# Set publication"):
                in_env_class = False
                continue
            
            # If the line belongs to the class but only has 8 spaces
            if line.startswith("        ") and (len(line) == 8 or line[8] != ' '):
                lines[i] = "    " + line

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

fix_indent('quantum_coherence_agrivoltaics_mesohops_complete.py')
