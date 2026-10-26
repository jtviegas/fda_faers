"""Process and load corrected FAERS period files."""

import atexit
from collections import defaultdict
from typing import Any
from datasets import Dataset
import logging

from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore
from tgedr_dataops_abs.etl4gh import Etl4GH
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

    def _do_drug(self, ds: Dataset) -> Dataset:
        logger.info(f"[_do_drug|in] ({ds})")
        silver_ds_prefix = self._configuration["silver_dataset_prefix"]
        ds_term_entity_url = f"{silver_ds_prefix}term_entity"
        ds_entity_atc_url = f"{silver_ds_prefix}entity_atc"

        ds_term_entity = self._store.get(ds_term_entity_url).train
        term_entity_map: dict[str, list[str]] = defaultdict(list)
        for row in ds_term_entity:
            term_entity_map[row["term"]].append(row["entity"])

        ds_entity_atc = self._store.get(ds_entity_atc_url).train
        entity_atc_map: dict[str, list[str]] = defaultdict(list)
        for row in ds_entity_atc:
            entity_atc_map[row["entity"]].extend(row["atc_codes"])

        term_atc_map: dict[str, list[Any]] = {}
        for term, entities in term_entity_map.items():
            atc_codes = []
            for entity in entities:
                atc_codes.extend(entity_atc_map.get(entity, []))
            term_atc_map[term] = atc_codes

        def map_term_to_atc(row):  # noqa: ANN202
            if not row.get("atc_codes"):
                new_row = dict(row)
                new_row["atc_codes"] = term_atc_map.get(row["term"], [])
                return new_row
            return row

        result = ds.map(map_term_to_atc)
        mapped_rows_count = sum(1 for row in result if row.get("atc_codes"))
        logger.info(f"[_do_drug|mapped_rows_count] {mapped_rows_count}")
        Metrics.instance().add_to_gauge(
            name="fda_faers.silver_layer.rows",
            value=mapped_rows_count,
            attributes={"type": "drug_mapped"},
        )

        logger.info(f"[_do_drug|out] ({result})")
        return result

    @Etl4GH.inject_configuration
    def extract(self, bronze_dataset_prefix: str) -> Any:
        """Extract relevant FAERS text files from the specified archives."""
        logger.info(f"[extract|in] ({bronze_dataset_prefix})")
        for table in self.__TABLES_PROCESSING:
            dataset_name = f"{bronze_dataset_prefix}{table}"
            self._data[table] = self._store.get(dataset_name).train
        logger.info("[extract|out]")

    def transform(self) -> Any:
        """Transform step for file status processing workflow."""
        logger.info("[transform|in]")
        for table, df in self._data.items():
            self._result[table] = self.__TABLES_PROCESSING.get(table)(df)
        logger.info("[transform|out]")

    @Etl4GH.inject_configuration
    def load(self, silver_dataset_prefix: str) -> str:
        """Return the corrected file paths as a comma-separated string."""
        logger.info(f"[load|in] ({silver_dataset_prefix})")
        result = []
        for table, ds in self._result.items():
            dataset_table = f"{silver_dataset_prefix}{table}"
            self._store.save(df=ds, key=dataset_table, split="train")
            result.append(dataset_table)
        result = ",".join(sorted(result)) if result else ""
        logger.info(f"[load|out] => {result}")
        return result
