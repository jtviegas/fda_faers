"""ETL that determines which FAERS quarterly periods are missing from the bronze dataset."""

from typing import Any
import logging

from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.hf_dataset import HuggingFaceDatasetStore, NoStoreException

from tgedr_fdafaers.constants import Constants
from tgedr_fdafaers.utils.faers_period import UtilsFaersPeriod


logger = logging.getLogger(__name__)


class Periods2Process(Etl4GH):
    """Determine which FAERS quarterly periods are missing from bronze data."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._existing_periods: list[str] = []
        self._periods_missing: list[str] = []

    def __find_periods_in_bronze_dataset(self, dataset_prefix: str) -> list[str]:
        """List the periods available in the bronze dataset."""
        logger.info(f"[__find_periods_in_bronze_dataset|in] ({dataset_prefix})")

        constants: Constants = Constants()
        store: HuggingFaceDatasetStore = HuggingFaceDatasetStore()
        periods: list[str] = []

        for table in constants.TABLES:
            dataset_table = f"{dataset_prefix}{table}"
            try:
                df_data = store.get(key=dataset_table).train
                if df_data is not None and not df_data.empty:
                    data_periods = df_data["period"].dropna().unique().tolist()
                    periods = data_periods if len(periods) == 0 else list(set(periods).intersection(data_periods))
            except NoStoreException as e:
                logger.warning(
                    f"[__find_periods_in_bronze_dataset] failed to get bronze dataset: {dataset_prefix} - {e}"
                )

        logger.info(f"[__find_periods_in_bronze_dataset|out] => {periods}")
        return periods

    @Etl4GH.inject_configuration
    def extract(self, dataset_prefix: str) -> Any:
        """Fetch the list of periods already present in the bronze HuggingFace dataset."""
        logger.info(f"[extract|in] ({dataset_prefix})")
        self._existing_periods = self.__find_periods_in_bronze_dataset(dataset_prefix)
        logger.info(f"[extract|out] existing periods: {self._existing_periods}")

    def transform(self) -> Any:
        """Compute which FAERS periods are missing by comparing all known periods to existing ones."""
        logger.info(f"[transform|in] existing periods: {self._existing_periods}")

        self._periods_missing = [
            str(x) for x in UtilsFaersPeriod.get_all_faers_periods() if str(x) not in self._existing_periods
        ]

        logger.info(f"[transform|out] periods missing: {self._periods_missing}")

    def load(self) -> str:
        """Return the missing periods as a sorted, comma-separated string."""
        logger.info("[load|in]")
        result: str = ",".join(sorted(self._periods_missing)) if self._periods_missing else ""
        logger.info(f"[load|out] => {result}")
        return result
