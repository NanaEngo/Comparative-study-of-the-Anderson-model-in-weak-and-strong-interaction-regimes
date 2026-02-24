import nbformat

nb = nbformat.read("quantum_coherence_agrivoltaics_mesohops_complete.ipynb", as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        source = cell.source
        if "from pathlib import Path" in source:
            print(f"Cell {i}:")
            print(repr(source[:100]))
            print("---")
            for line in source.split("\n"):
                if "pathlib" in line:
                    print(repr(line))
