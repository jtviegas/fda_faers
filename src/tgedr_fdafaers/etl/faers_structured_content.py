"""Enrich bronze FAERS datasets with ATC mappings and save them to silver."""

import atexit
from collections.abc import Iterator
import logging
from typing import Any

from datasets import Dataset, concatenate_datasets
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore, NoStoreException
from tgedr_fdafaers.constants import Constants
from tgedr_fdafaers.faers_narrative import (
    FaersCaseAssembler,
    FaersCaseBatcher,
    FaersNarrativeRenderer,
    FaersTables,
    LatestCaseVersions,
)
from tgedr_fdafaers.utils.common_etl import CommonEtl
from tgedr_observability.metrics import Metrics


logger = logging.getLogger(__name__)


class FaersStructuredContent(CommonEtl):
    """Process and load corrected FAERS period files into the silver layer."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._data: dict[str, Dataset] = {}
        self._result: Iterator[FaersTables] | None = None
        self._store = HuggingFaceDatasetFileArrowBasedStore()
        # atexit.register(Metrics.app_shutdown)

    @CommonEtl.inject_configuration
    def extract(self, bronze_dataset_prefix: str, silver_dataset_prefix: str) -> Any:
        logger.info(f"[extract|in] ({bronze_dataset_prefix}, {silver_dataset_prefix})")
        for table in Constants().TABLES:
            dataset_name = f"{bronze_dataset_prefix}{table}" if table != "drug" else f"{bronze_dataset_prefix}{table}"
            self._data[table] = self._store.get(dataset_name).train
        logger.info("[extract|out]")

    def transform(self) -> Any:
        logger.info("[transform|in]")
        tables_with_latest_cases: FaersTables = LatestCaseVersions().process(
            context={"tables": FaersTables.from_dict(self._data)}
        )
        self._data.clear()
        self._result = FaersCaseBatcher().process(context={"tables": tables_with_latest_cases})
        logger.info("[transform|out]")

    @CommonEtl.inject_configuration
    def load(self, silver_dataset_prefix: str) -> str:
        logger.info(f"[load|in] ({silver_dataset_prefix})")

        if self._result is not None:
            renderer: FaersNarrativeRenderer = FaersNarrativeRenderer()
            assembler: FaersCaseAssembler = FaersCaseAssembler()

            for tables in self._result:
                cases_structure: list[dict[str, Any]] = assembler.process(context={"tables": tables})
                cases_content: list[dict[str, Any]] = renderer.process(context={"cases": cases_structure})
                print(cases_content)
                del tables, cases_structure, cases_content
                break

            #     for table, ds in tables.items():
            #         dataset_table = f"{silver_dataset_prefix}{table}"
            #         self._store.save(df=ds, key=dataset_table, split="train")
            #         result.append(dataset_table)
            # result = ",".join(sorted(result)) if result else ""
        logger.info("[load|out]")
