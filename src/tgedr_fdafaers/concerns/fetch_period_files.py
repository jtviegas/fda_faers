"""Fetch and expose FAERS source files for requested reporting periods."""

from typing import Any
import logging

from tgedr_dataops_abs.processor import Processor
from tgedr_fdafaers.faers_file_source import FaersFileSource
from tgedr_fdafaers.utils.utils_io import UtilsIO


logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class FetchPeriodFiles(Processor):
    """Fetch and expose FAERS files for the requested reporting periods."""

    CONTEXT_KEY_PERIODS = "periods"
    CONTEXT_KEY_MAX_PERIODS = "max_periods"
    __DEFAULT_MAX_PERIODS = 5

    def process(self, context: dict[str, Any] | None = None) -> list[str] | None:
        """Fetch and expose FAERS source files for the requested reporting periods."""
        logger.info(f"[process|in] ({context})")

        if self.CONTEXT_KEY_PERIODS not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_PERIODS}")
        periods: list[str] = context[self.CONTEXT_KEY_PERIODS]
        max_periods: int = context.get(self.CONTEXT_KEY_MAX_PERIODS, self.__DEFAULT_MAX_PERIODS)
        result: list[str] | None = None

        if 0 < len(periods):
            tmp_dir: str = UtilsIO.tmp_dir()
            periods_to_fetch: list[str] = periods[:max_periods]
            source: FaersFileSource = FaersFileSource()
            files_fetched: list[str] = [
                source.get(
                    context={
                        FaersFileSource.CONTEXT_KEY_OUTPUT_URL: tmp_dir,
                        FaersFileSource.CONTEXT_KEY_PERIOD: period,
                    }
                )
                for period in periods_to_fetch
            ]
            logger.info(f"[process] number of files fetched: {len(files_fetched)}")
            if 0 < len(files_fetched):
                result = files_fetched

        logger.info(f"[process|out] => {result}")
        return result
