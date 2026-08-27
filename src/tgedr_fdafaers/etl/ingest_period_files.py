"""Process and load corrected FAERS period files."""

from typing import Any
import pandas as pd
import logging
from pathlib import Path

from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_fdafaers.constants import Constants
from tgedr_dataops.store.hf_dataset import DataFrameSplits, HuggingFaceDatasetStore, NoStoreException
from tgedr_fdafaers.raw_data_ingestion import RawDataIngestion
from tgedr_observability.metrics import Metrics


logger = logging.getLogger(__name__)


class IngestPeriodFiles(Etl4GH):
    """ETL workflow for extracting, correcting, and loading FAERS period files."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._data: dict[str, pd.DataFrame] = {}
        self._constants = Constants()

    def __handle_file(self, filepath: str) -> None:
        """Handle the ingestion of a single FAERS text file."""
        logger.info(f"[__handle_file|in] ({filepath})")

        file = Path(filepath).name
        table = file[0:4]
        df: pd.DataFrame = pd.read_csv(
            filepath, delimiter=self._constants.CSV_DELIMITER, index_col=False, low_memory=False
        )
        result: pd.DataFrame = RawDataIngestion().process(
             context={RawDataIngestion.CONTEXT_KEY_TABLE: table,
                      RawDataIngestion.CONTEXT_KEY_DATAFRAME: df})

        if table not in self._data:
            self._data[table] = result
        else:
            self._data[table] = pd.concat([self._data[table], result], ignore_index=True)
        logger.info(f"[__handle_file|out] new data | table: {table} | df.shape: {result.shape}")

    @Etl4GH.inject_configuration
    def extract(self, files: str) -> Any:
        """Extract relevant FAERS text files from the specified archives."""
        logger.info(f"[extract|in] ({files})")

        for file in files.split(","):
            self.__handle_file(file.strip())

        logger.info("[extract|out]")

    def transform(self) -> Any:
        """Transform step for file status processing workflow."""
        logger.info("[transform|in]")
        logger.info("[transform|out]")

    @Etl4GH.inject_configuration
    def load(self, dataset_prefix: str) -> str:
        """Return the corrected file paths as a comma-separated string."""
        logger.info(f"[load|in] ({dataset_prefix})")

        periods: set[str] = set()
        store: HuggingFaceDatasetStore = HuggingFaceDatasetStore()
        for table, df in self._data.items():
            periods.update(df["period"].unique().tolist())
            dataset_name = f"{dataset_prefix}{table}"
            dfs: DataFrameSplits = DataFrameSplits(train=df)
            try:
                store.update(
                    df=dfs,
                    key=dataset_name,
                    append=True,
                )
            except NoStoreException as e:
                logger.warning(f"could not update dataset {dataset_name}: {e}. Attempting to create new dataset.")   #nosec B608
                store.save(
                    df=dfs,
                    key=dataset_name
                )
            Metrics.instance().add_to_gauge("fda_faers.ingest_period_files.new_rows", df.shape[0], {"table": table}) # pyright: ignore[reportOptionalMemberAccess]

        result = ",".join(sorted(periods)) if periods else ""
        logger.info(f"[load|out] => {result}")
        return result

