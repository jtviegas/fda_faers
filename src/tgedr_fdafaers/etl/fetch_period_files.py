"""Fetch and expose FAERS source files for requested reporting periods."""

from typing import Any
import logging

from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_fdafaers.faers_file_source import FaersFileSource
from tgedr_fdafaers.utils.utils_io import UtilsIO


logger = logging.getLogger(__name__)


class FetchPeriodFiles(Etl4GH):
    """Fetch and expose FAERS files for the requested reporting periods."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._tmp_dir: str = UtilsIO.tmp_dir()
        self._files_fetched: dict[str, str] = {}

    @Etl4GH.inject_configuration
    def extract(self, periods: str, max_periods: int) -> Any:
        """Fetch source files for the requested periods."""
        logger.info(f"[extract|in] (periods={periods}, max_periods={max_periods})")
        periods_to_fetch: list[str] = [p.strip() for p in periods.split(",")][:max_periods]
        source: FaersFileSource = FaersFileSource()
        for period in periods_to_fetch:
            self._files_fetched[period] = source.get(context={FaersFileSource.CONTEXT_KEY_OUTPUT_URL: self._tmp_dir})
        logger.info(f"[extract|out] files fetched: {self._files_fetched}")

    def transform(self) -> Any:
        """Transform nothing."""
        logger.info("[transform|in]")
        logger.info("[transform|out]")

    def load(self) -> str:
        """Return the fetched file periods as a sorted, comma-separated string."""
        logger.info("[load|in]")
        result: str = ",".join(sorted(self._files_fetched)) if self._files_fetched else ""
        logger.info(f"[load|out] => {result}")
        return result
