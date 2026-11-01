"""Enrich bronze FAERS datasets with ATC mappings and save them to silver."""

import logging
from typing import Any

from datasets import Dataset
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore, NoStoreException
from tgedr_fdafaers.utils.common_etl import CommonEtl


logger = logging.getLogger(__name__)


class Mapping(CommonEtl):
    """Process and load corrected FAERS period files into the silver layer."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._data: dict[str, Dataset] = {}
        self._result: dict[str, Dataset] = {}
        self._store = HuggingFaceDatasetFileArrowBasedStore()
        self.__TABLES_PROCESSING: dict[str, Any] = {
            "drug": {"data_function": self._get_drug, "processing_function": self._do_drug}
        }

    @CommonEtl.inject_configuration
    def extract(self, bronze_dataset_prefix: str, silver_dataset_prefix: str, batch_size: int = 100_000) -> Any:
        """Load each configured bronze dataset into the in-memory input mapping.

        Args:
            bronze_dataset_prefix: Prefix used to resolve the input dataset keys.
            silver_dataset_prefix: Prefix used to resolve the auxiliary silver datasets.
            batch_size: Number of rows to load from the bronze dataset in each batch.

        Returns:
            None. Loaded datasets are stored in ``self._data``.
        """
        logger.info(f"[extract|in] ({bronze_dataset_prefix}, {silver_dataset_prefix}, {batch_size})")
        for table, entry in self.__TABLES_PROCESSING.items():
            self._data[table] = entry.get("data_function")(
                bronze_dataset_prefix=bronze_dataset_prefix,
                silver_dataset_prefix=silver_dataset_prefix,
                batch_size=batch_size,
            )

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
            self._result[table] = self.__TABLES_PROCESSING.get(table).get("processing_function")(
                ds=ds, silver_dataset_prefix=silver_dataset_prefix
            )
        logger.info("[transform|out]")

    def _get_drug(self, bronze_dataset_prefix: str, silver_dataset_prefix: str, batch_size: int) -> Dataset:
        logger.info(f"[_get_drug|in] ({bronze_dataset_prefix}, {silver_dataset_prefix}, {batch_size})")

        ids: list[int] = []
        try:
            ids = self._store.get(f"{silver_dataset_prefix}drug").train.unique("primaryid")
        except NoStoreException:
            logger.warning("[_get_drug] no silver dataset for drug")

        dataset_stream = self._store.get(f"{bronze_dataset_prefix}drug", streaming=True).train
        subset = dataset_stream.filter(lambda row: row["primaryid"] not in ids).take(batch_size)
        result: Dataset = Dataset.from_list(subset)
        logger.info(f"[_get_drug|out] Retrieved {len(subset)} rows from bronze dataset")
        return result

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
        term_atc_map: dict[str, list[str]] = {}
        try:
            ds_term_atc = self._store.get(ds_term_atc_url).train
            for row in ds_term_atc:
                term_atc_map[row["term"]] = row["atc_codes"]
        except NoStoreException as nse:
            logger.warning(f"[_do_drug] Failed to get dataset {ds_term_atc_url}: {nse}")

        ds = ds.add_column("atc_codes", [None] * ds.num_rows)
        result = ds.map(lambda row: {**row, "atc_codes": term_atc_map.get(row["term"], None)})
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
            self._store.save(df=ds, key=dataset_table, split="train", append=True)
            result.append(dataset_table)
        result = ",".join(sorted(result)) if result else ""
        logger.info(f"[load|out] => {result}")
        return result
