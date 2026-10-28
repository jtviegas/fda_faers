"""Enrich bronze FAERS datasets with ATC mappings and save them to silver."""

import atexit
import logging
from typing import Any

from datasets import Dataset, concatenate_datasets
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore, NoStoreException
from tgedr_fdafaers.utils.common_etl import CommonEtl
from tgedr_observability.metrics import Metrics


logger = logging.getLogger(__name__)


class SilverLayer(CommonEtl):
    """Process and load corrected FAERS period files into the silver layer."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._data: dict[str, Dataset] = {}
        self._result: dict[str, Dataset] = {}
        self._store = HuggingFaceDatasetFileArrowBasedStore()
        self.__TABLES_PROCESSING: dict[str, Any] = {"drug": self._do_drug}
        atexit.register(Metrics.app_shutdown)

    @CommonEtl.inject_configuration
    def extract(self, bronze_dataset_prefix: str) -> Any:
        """Load each configured bronze dataset into the in-memory input mapping.

        Args:
            bronze_dataset_prefix: Prefix used to resolve the input dataset keys.

        Returns:
            None. Loaded datasets are stored in ``self._data``.
        """
        logger.info(f"[extract|in] ({bronze_dataset_prefix})")
        for table in self.__TABLES_PROCESSING:
            dataset_name = f"{bronze_dataset_prefix}{table}"
            self._data[table] = self._store.get(dataset_name).train
        logger.info("[extract|out]")

    @CommonEtl.inject_configuration
    def transform(self, silver_dataset_prefix: str) -> Any:
        """Apply the configured transformation to every loaded dataset.

        Args:
            silver_dataset_prefix: Prefix for auxiliary silver datasets used
                during enrichment.

        Returns:
            None. Transformed datasets are stored in ``self._result``.
        """
        logger.info(f"[transform|in] ({silver_dataset_prefix})")
        for table, ds in self._data.items():
            self._result[table] = self.__TABLES_PROCESSING.get(table)(
                ds=ds, silver_dataset_prefix=silver_dataset_prefix
            )
        logger.info("[transform|out]")

    def _do_drug(self, ds: Dataset, silver_dataset_prefix: str) -> Dataset:
        """Attach term-level ATC codes to drug rows, preserving unmapped rows.

        The auxiliary ``term_atc`` dataset is optional. Terms with mapping rows
        receive those codes; rows without mappings retain a null ``atc_codes``
        value. Metrics are recorded for total, mapped, and unmapped rows.

        Args:
            ds: Drug-term dataset to enrich. It must not already contain an
                ``atc_codes`` column.
            silver_dataset_prefix: Prefix used to locate the ``term_atc`` data.

        Returns:
            A dataset containing the original rows and an ``atc_codes`` column.
        """
        logger.info(f"[_do_drug|in] ({ds}, {silver_dataset_prefix})")

        ds_term_atc_url = f"{silver_dataset_prefix}term_atc"
        atc_map: dict[str, list[str]] = {}
        try:
            ds_term_atc = self._store.get(ds_term_atc_url).train
            for row in ds_term_atc:
                atc_map[row["term"]] = row["atc_codes"]
        except NoStoreException as nse:
            logger.warning(f"[_do_drug] Failed to get dataset {ds_term_atc_url}: {nse}")

        ds = ds.add_column("atc_codes", [None] * ds.num_rows)
        Metrics.instance().add_to_gauge(
            name="fda_faers.silver_layer.rows",
            value=ds.num_rows,
            attributes={"type": "silver_term"},
        )
        ds_with_mapping = ds.filter(lambda row: row["term"] in atc_map)

        ds_with_no_mapping = ds.filter(lambda row: row["term"] not in atc_map)
        Metrics.instance().add_to_gauge(
            name="fda_faers.silver_layer.rows",
            value=ds_with_no_mapping.num_rows,
            attributes={"type": "silver_term_unmapped"},
        )
        ds_with_mapping = ds_with_mapping.map(lambda row: {**row, "atc_codes": atc_map.get(row["term"], None)})
        Metrics.instance().add_to_gauge(
            name="fda_faers.silver_layer.rows",
            value=ds_with_mapping.num_rows,
            attributes={"type": "silver_term_mapped"},
        )
        result = concatenate_datasets([ds_with_mapping, ds_with_no_mapping])
        logger.info(f"[_do_drug|out] ({result})")
        return result

    @CommonEtl.inject_configuration
    def load(self, silver_dataset_prefix: str) -> str:
        """Save transformed datasets under the silver prefix.

        Args:
            silver_dataset_prefix: Prefix used to construct destination keys.

        Returns:
            A comma-separated list of saved dataset keys. The current
            implementation returns an empty string because it does not add
            keys to the result list.
        """
        logger.info(f"[load|in] ({silver_dataset_prefix})")
        result = []
        for table, ds in self._result.items():
            dataset_table = f"{silver_dataset_prefix}{table}"
            self._store.save(df=ds, key=dataset_table, split="train")
            result.append(dataset_table)
        result = ",".join(sorted(result)) if result else ""
        logger.info(f"[load|out] => {result}")
        return result
