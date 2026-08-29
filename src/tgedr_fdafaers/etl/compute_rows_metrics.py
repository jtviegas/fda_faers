"""Compute and report the total rows per period and table from the bronze datasets."""

from typing import Any
import logging
import pandas as pd

from tgedr_dataops_abs.etl import EtlException
from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.hf_dataset import HuggingFaceDatasetStore, NoStoreException
from tgedr_observability.metrics import Metrics
from tgedr_fdafaers.constants import Constants


logger = logging.getLogger(__name__)


class ComputeRowsMetrics(Etl4GH):
    """ETL that reads the bronze datasets back and reports rows per period and table as a gauge metric."""

    METRIC_NAME = "fda_faers.ingest_period_files.rows"

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._data: dict[str, pd.DataFrame] = {}
        self._rows: dict[str, dict[str, int]] = {}
        self._constants = Constants()

    @Etl4GH.inject_configuration
    def extract(self, dataset_prefix: str) -> Any:
        """Read each table's bronze dataset from HuggingFace."""
        logger.info(f"[extract|in] (dataset_prefix={dataset_prefix})")

        store: HuggingFaceDatasetStore = HuggingFaceDatasetStore()
        for table in self._constants.TABLES:
            dataset_name = f"{dataset_prefix}{table}"
            try:
                splits = store.get(key=dataset_name)
            except NoStoreException as e:
                msg = f"[extract] required dataset not found: {dataset_name}"
                raise EtlException(msg) from e
            df: pd.DataFrame | None = splits.train
            if df is None or df.empty:
                msg = f"[extract] required dataset is empty: {dataset_name}"
                raise EtlException(msg)
            self._data[table] = df

        logger.info(f"[extract|out] tables loaded: {sorted(self._data.keys())}")

    def transform(self) -> Any:
        """Compute the row count per period for each table."""
        logger.info(f"[transform|in] (tables={sorted(self._data.keys())})")

        for table, df in self._data.items():
            counts: pd.Series = df.groupby("period").size()
            self._rows[table] = {str(period): int(rows) for period, rows in counts.items()}

        logger.info(f"[transform|out] rows computed: {self._rows}")

    @Etl4GH.inject_configuration
    def load(self) -> str:
        """Record the rows-per-(table, period) gauge metric and return a summary string."""
        logger.info(f"[load|in] (rows={self._rows})")

        points: int = 0
        metrics = Metrics.instance()
        for table, period_counts in self._rows.items():
            for period, rows in period_counts.items():
                metrics.add_to_gauge(
                    self.METRIC_NAME,
                    rows,
                    {"table": table, "period": period},
                )
                points += 1

        result: str = f"{points} metric points recorded"
        logger.info(f"[load|out] => {result}")
        return result
