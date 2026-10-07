"""Common ETL helpers shared across FDA FAERS ETLs."""

from typing import Any
import logging
from pathlib import Path

from tgedr_dataops_abs.etl4gh import Etl4GH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("great_expectations._docs_decorators").setLevel(logging.WARNING)


class CommonEtl(Etl4GH):
    """Base ETL class with common helpers shared across FDA FAERS ETLs."""

    __ODCS_DIR = Path(__file__).resolve().parents[1] / "odcs"

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)

    def contract_path(self, name: str) -> Path:
        """Return the path to the ODCS data contract file for ``name``."""
        return self.__ODCS_DIR / f"{name}.odcs.yaml"
