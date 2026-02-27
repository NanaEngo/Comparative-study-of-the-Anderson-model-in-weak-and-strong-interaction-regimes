import logging
import re
import uuid
from pathlib import Path

import nbformat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def standardize_notebook():
    nb_path = Path(
        "/media/taamangtchu/MYDATA/Github/Quantum_Agrivoltaic_HOPS/Redac_Paper1/quantum_simulations_framework/quantum_coherence_agrivoltaics_mesohops_complete.ipynb"
    )
    if not nb_path.exists():
        logger.error("Notebook not found: %s", nb_path)
        return

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except (FileNotFoundError, OSError, ValueError) as e:
        logger.error("Failed to read notebook (file/format error): %s", e)
        return

    for idx, cell in enumerate(nb.cells, start=1):
        # Ensure metadata exists
        meta = cell.get("metadata", {}) or {}

        # Ensure language metadata
        if "language" not in meta:
            meta["language"] = "python" if cell.get("cell_type") == "code" else "markdown"

        # Ensure a stable unique id exists for each existing cell
        if "id" not in meta or not meta["id"]:
            # Use uuid4 hex to avoid exposing larger UUID formats
            meta["id"] = uuid.uuid4().hex

        cell["metadata"] = meta

        # Only operate on code cells for source transformations
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")

        # 1. Clean up imports in the initial import cell
        if "Import required libraries" in str(source) and "TechnoEconomicModel" in str(source):
            source = source.replace(
                "from models.techno_economic_model import TechnoEconomicModel\n", ""
            )
            source = source.replace("from models.spectroscopy_2des import Spectroscopy2DES\n", "")
            if (
                "import pandas as pd" in source
                and "from models import TechnoEconomicModel" not in source
            ):
                source = source.replace(
                    "import pandas as pd\n",
                    "import pandas as pd\nfrom models import TechnoEconomicModel, Spectroscopy2DES\n",
                )

        # 2. Standardize framework import paths in import-heavy cells
        if "from core.constants import" in str(source) and "FMO_SITE_ENERGIES_7" in str(source):
            source = source.replace("from .core.constants", "from core.constants")
            source = source.replace("from .core.hops_simulator", "from core.hops_simulator")
            source = source.replace("from .models.", "from models.")
            source = source.replace("from .simulations.", "from simulations.")
            source = source.replace("from .utils.", "from utils.")

            if "from models.biodegradability_analyzer" in source:
                model_import = (
                    "# Import models\n"
                    "from models import (\n"
                    "    BiodegradabilityAnalyzer,\n"
                    "    SensitivityAnalyzer,\n"
                    "    LCAAnalyzer,\n"
                    "    TechnoEconomicModel,\n"
                    "    Spectroscopy2DES,\n"
                    "    MultiScaleTransformer,\n"
                    "    QuantumDynamicsSimulator,\n"
                    "    AgrivoltaicCouplingModel,\n"
                    "    SpectralOptimizer,\n"
                    "    EcoDesignAnalyzer,\n"
                    "    EnvironmentalFactors\n"
                    ")\n"
                    "from simulations import TestingValidationProtocols\n"
                    "from utils import CSVDataStorage\n"
                    "from utils.figure_generator import FigureGenerator\n"
                )
                pattern = r"# Import models.*?(?=print\()"
                try:
                    source = re.sub(pattern, model_import + "\n", source, flags=re.DOTALL)
                except re.error:
                    logger.debug("Import consolidation pattern did not match for cell %d", idx)

        # 3. Fix specific broken import occurrences
        if "from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian" in source:
            source = source.replace(
                "from quantum_coherence_agrivoltaics_mesohops import create_fmo_hamiltonian",
                "from core.hamiltonian_factory import create_fmo_hamiltonian",
            )

        cell["source"] = source

    try:
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        logger.info("Notebook standardized successfully: %s", nb_path)
    except (OSError, TypeError, ValueError) as e:
        logger.error("Failed to write notebook (file/format error): %s", e)


if __name__ == "__main__":
    standardize_notebook()
