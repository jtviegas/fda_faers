"""ETL that determines which FAERS quarterly periods are missing from the bronze dataset."""

from typing import Any
import logging
import pandas as pd

from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.hf_dataset import HuggingFaceDatasetStore, NoStoreException

from tgedr_fdafaers.utils.faers_period import UtilsFaersPeriod


logger = logging.getLogger(__name__)


class Periods2Process(Etl4GH):
    """Determine which FAERS quarterly periods are missing from bronze data."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._existing_periods: list[str] = []
        self._periods_missing: list[str] = []

    def __find_periods_in_bronze_dataset(self, bronze_dataset: str) -> list[str]:
        """List the periods available in the bronze dataset."""
        logger.info(f"[__find_periods_in_bronze_dataset|in] ({bronze_dataset})")
        periods: list[str] = []

        store: HuggingFaceDatasetStore = HuggingFaceDatasetStore()

        df_data: pd.DataFrame | None = None
        try:
            df_data = store.get(key=bronze_dataset).train
        except NoStoreException as e:
            logger.warning(f"[__find_periods_in_bronze_dataset] failed to get bronze dataset: {bronze_dataset} - {e}")

        if df_data is not None and not df_data.empty:
            periods = df_data["period"].dropna().unique().tolist()

        logger.info(f"[__find_periods_in_bronze_dataset|out] => {periods}")
        return periods

    @Etl4GH.inject_configuration
    def extract(self, bronze_dataset: str) -> Any:
        """Fetch the list of periods already present in the bronze HuggingFace dataset."""
        logger.info(f"[extract|in] ({bronze_dataset})")
        self._existing_periods = self.__find_periods_in_bronze_dataset(bronze_dataset)
        logger.info(f"[extract|out] existing periods: {self._existing_periods}")

    def transform(self) -> Any:
        """Compute which FAERS periods are missing by comparing all known periods to existing ones."""
        logger.info(f"[transform|in] existing periods: {self._existing_periods}")

        self._periods_missing = [str(x) for x in UtilsFaersPeriod.get_all_faers_periods() if str(x) not in self._existing_periods]

        logger.info(f"[transform|out] periods missing: {self._periods_missing}")

    def load(self) -> str:
        """Return the missing periods as a sorted, comma-separated string."""
        logger.info("[load|in]")
        result: str = ",".join(sorted(self._periods_missing)) if self._periods_missing else ""
        logger.info(f"[load|out] => {result}")
        return result
