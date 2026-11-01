"""ETL that processes corrected FAERS period files into structured content for the silver layer.

This module provides:
- FaersStructuredContent: an ETL that extracts the latest case versions from bronze
  FAERS tables, assembles them into cases, renders structured narratives, and loads
  the result into the silver layer.
"""

import logging
from typing import Any, ClassVar
from datasets import Dataset

from tgedr_fdafaers.faers_tables.faers_tables import FaersTables
from tgedr_fdafaers.structured_content.faers_case_assembler import FaersCaseAssembler
from tgedr_fdafaers.structured_content.faers_narrative_renderer import FaersNarrativeRenderer
from tgedr_fdafaers.utils.common_etl import CommonEtl
from tgedr_dataops_abs.store import Store, NoStoreException
from tgedr_fdafaers.faers_tables.latest_case_versions import LatestCaseVersions
from tgedr_fdafaers.constants import Constants
from tgedr_pycommons.utils.reflection import UtilsReflection


logger = logging.getLogger(__name__)


class FaersStructuredContent(CommonEtl):
    """Process and load corrected FAERS period files into the silver layer."""

    # Columns required by FaersCaseAssembler and FaersNarrativeRenderer.
    __TABLE_PROJECTION: ClassVar[dict[str, list[str]]] = {
        "demo": ["primaryid", "caseid", "caseversion", "age", "age_cod", "sex"],
        "drug": ["primaryid", "drug_seq", "drugname", "prod_ai", "role_cod"],
        "indi": ["primaryid", "indi_drug_seq", "indi_pt"],
        "ther": ["primaryid", "dsg_drug_seq"],
        "reac": ["primaryid", "pt"],
        "outc": ["primaryid", "outc_cod"],
        "rpsr": ["primaryid"],
    }

    __DEFAULT_STORE: str = "tgedr_dataops.store.hf_dataset_file_arrow_based.HuggingFaceDatasetFileArrowBasedStore"
    __CONFIGURATION_KEY_STORE: str = "store"

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        store_impl: str = (
            configuration.get(self.__CONFIGURATION_KEY_STORE, self.__DEFAULT_STORE)
            if configuration
            else self.__DEFAULT_STORE
        )
        self._store = UtilsReflection.load_class(store_impl, Store)()
        self._tables: FaersTables | None = None
        self._result: list[dict[str, Any]] | None = None

    @CommonEtl.inject_configuration
    def extract(self, bronze_dataset_prefix: str, silver_dataset_prefix: str, batch_size: int = 500_000) -> Any:
        """Extract the latest case versions from bronze FAERS tables.

        Args:
            bronze_dataset_prefix: Prefix of the bronze datasets to read from.
            silver_dataset_prefix: Prefix of the silver datasets to read from.
            batch_size: Maximum number of primary IDs to process per batch.

        Returns:
            None
        """
        logger.info(f"[extract|in] ({bronze_dataset_prefix}, {silver_dataset_prefix}, {batch_size})")

        processed_ids = []
        try:
            processed_ids = self._store.get(
                key=f"{silver_dataset_prefix}_structured_content", columns=["primaryid"], streaming=False
            ).train.unique("primaryid")
        except NoStoreException:
            logger.warning(f"[extract] no dataset: {f'{silver_dataset_prefix}_structured_content'}")
        logger.info(f"[extract] processed IDs number: {len(processed_ids)}")

        ds_demo = self._store.get(
            key=f"{bronze_dataset_prefix}demo", columns=["primaryid", "caseid", "caseversion"], streaming=False
        ).train
        primaryids: set[int] = LatestCaseVersions().process({"demo_dataset": ds_demo})
        logger.info(f"[extract] primary IDs number: {len(primaryids)}")

        ids: set[int] = set(sorted(primaryids - set(processed_ids))[:batch_size])
        logger.info(f"[extract] IDs to process number: {len(ids)}")

        if 0 < len(ids):
            table_keys = {table: f"{bronze_dataset_prefix}{table}" for table in Constants().TABLES}
            tables = {}
            for table, table_key in table_keys.items():
                tables[table] = self._store.get(
                    key=table_key, columns=self.__TABLE_PROJECTION[table], streaming=False
                ).train.filter(lambda row, ids=ids: row["primaryid"] in ids)

            self._tables = FaersTables.from_dict(tables)

        logger.info("[extract|out]")

    def transform(self) -> Any:
        """Assemble the extracted tables into cases and render structured narratives.

        Returns:
            None
        """
        logger.info("[transform|in]")

        if self._tables is not None:
            assembler: FaersCaseAssembler = FaersCaseAssembler()
            renderer: FaersNarrativeRenderer = FaersNarrativeRenderer()
            content: list[dict[str, Any]] = assembler.process(context={"tables": self._tables})
            self._result = renderer.process(context={"cases": content})

        logger.info("[transform|out]")

    @CommonEtl.inject_configuration
    def load(self, silver_dataset_prefix: str) -> Any:
        """Load the rendered structured content into the silver layer.

        Args:
            silver_dataset_prefix: Prefix of the silver datasets to write to.

        Returns:
            None
        """
        logger.info(f"[load|in] ({silver_dataset_prefix})")

        if self._result is not None:
            output_dataset: str = f"{silver_dataset_prefix}structured_content"
            self._store.save(key=output_dataset, df=Dataset.from_list(self._result), append=True)

        logger.info("[load|out]")
