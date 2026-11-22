"""ETL that determines which FAERS quarterly periods are missing from the bronze dataset."""

from typing import Any
import logging

from tgedr_dataops.store.hf_dataset import NoStoreException
from tgedr_fdafaers.constants import Constants
from tgedr_dataops_abs.processor import Processor
from tgedr_dataops_abs.store import Store

from tgedr_fdafaers.utils.faers_period import UtilsFaersPeriod


logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class FindPeriodsToProcess(Processor):
    """Processor that determines which FAERS quarterly periods are not processed."""

    CONFIG_KEY_STORE = "store"
    CONTEXT_KEY_DB_SCHEMA = "db_schema"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the FindPeriodsToProcess processor with the given configuration."""
        logger.info(f"[init|in] ({config})")
        super().__init__(config=config)
        if self.CONFIG_KEY_STORE not in config:
            raise ValueError(f"[init] Missing required context key: {self.CONFIG_KEY_STORE}")
        self._store: Store = config[self.CONFIG_KEY_STORE]
        logger.info("[init|out]")

    def process(self, context: dict[str, Any] | None = None) -> list[str] | None:
        """Determine which FAERS quarterly periods are not processed."""
        logger.info(f"[process|in] ({context})")

        if self.CONTEXT_KEY_DB_SCHEMA not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_DB_SCHEMA}")
        schema = context[self.CONTEXT_KEY_DB_SCHEMA]

        result: list[str] | None = None

        processed_periods: list[str] = self.find_processed_periods(schema)
        logger.info(f"[process] number of processed periods: {len(processed_periods)}")
        all_faers_periods = UtilsFaersPeriod.get_all_faers_periods()
        logger.info(f"[process] number of all FAERS periods: {len(all_faers_periods)}")
        periods_missing = [str(x) for x in all_faers_periods if str(x) not in processed_periods]
        logger.info(f"[process] number of unprocessed FAERS periods: {len(periods_missing)}")
        if 0 < len(periods_missing):
            result = periods_missing

        logger.info(f"[process|out] => {result}")
        return result

    def find_processed_periods(self, schema: str) -> list[str]:
        """Find the FAERS quarterly periods that have been processed in the given schema."""
        logger.info(f"[find_processed_periods|in] ({schema})")

        constants: Constants = Constants()
        periods: list[str] = []

        for table in constants.TABLES:
            dataset_table = f"{schema}.{table}"
            try:
                table_periods = [row[0] for row in self._store.get(dataset_table).select("period").distinct().collect()]
                data_periods: list[str] = sorted({p for p in table_periods if p is not None})
                if 0 < len(data_periods):
                    periods = data_periods if len(periods) == 0 else list(set(periods).intersection(data_periods))
            except NoStoreException as e:
                logger.warning(f"[find_processed_periods] failed to get bronze dataset {dataset_table}: {e}")

        logger.info(f"[find_processed_periods|out] => {periods}")
        return periods
