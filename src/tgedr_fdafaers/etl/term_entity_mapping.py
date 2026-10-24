"""ETL step that maps drug terms to entities.

This module provides:
- TermEntityMapping: ETL step that extracts unmapped drug terms from bronze,
  maps them to entities, and loads the resulting mappings into silver.
"""

import atexit
from typing import Any
import pandas as pd
import logging
import random
from datasets import Dataset
from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore
from tgedr_dataops.store.contracted_store import NoStoreException
from tgedr_fdafaers.entity_mapping.term_entity import TermEntity
from tgedr_fdafaers.utils.common_etl import CommonEtl
from tgedr_observability.metrics import Metrics

logger = logging.getLogger(__name__)


class TermEntityMapping(CommonEtl):
    """ETL step that maps drug terms to entities."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initializes the TermEntityMapping ETL step.

        Args:
            configuration: Optional configuration dictionary passed to the parent ETL class.
        """
        super().__init__(configuration=configuration)
        self._data: dict[str, list[str]] = {}
        self._result: pd.DataFrame | None = None
        self._store: HuggingFaceDatasetFileArrowBasedStore = HuggingFaceDatasetFileArrowBasedStore(
            config={"visibility": "public"}
        )
        atexit.register(Metrics.app_shutdown)

    @Etl4GH.inject_configuration
    def extract(self, drug_data_url: str, term_entity_data_url: str, sample_size: int = 50000) -> Any:
        """Extracts terms to map from the bronze dataset, excluding already mapped terms.

        Args:
            drug_data_url: Prefix of the bronze dataset tables to read from.
            term_entity_data_url: Prefix of the silver dataset tables to read from.
            sample_size: Maximum number of terms to sample for mapping (default is 50000).

        Returns:
            None.
        """
        logger.info(f"[extract|in] ({drug_data_url}, {term_entity_data_url}, {sample_size})")

        terms: list[str] = (self._store.get(key=drug_data_url, columns=["term"]).train).unique("term")
        logger.info(f"[extract] bronze terms count: {len(terms)}")
        Metrics.instance().add_to_gauge(
            name="fda_faers.term_entity_mapping.rows", value=len(terms), attributes={"type": "term"}
        )

        mapped_terms: list[str] = []
        try:
            mapped_terms = (self._store.get(key=term_entity_data_url, columns=["term"]).train).unique("term")
            Metrics.instance().add_to_gauge(
                name="fda_faers.term_entity_mapping.rows",
                value=len(mapped_terms),
                attributes={"type": "term_mapped"},
            )
            logger.info(f"[extract] mapped terms count: {len(mapped_terms)}")
        except NoStoreException as nse:
            logger.warning(f"[extract] silver table {term_entity_data_url} not found: {nse}")

        terms_to_map: list[str] = list(set(terms) - set(mapped_terms))

        # sample up to sample_size elements (without replacement)
        sample_terms: list[str] = random.sample(terms_to_map, min(sample_size, len(terms_to_map)))

        if sample_terms:
            self._data["terms_to_map"] = sample_terms
        logger.info(f"[extract|out] terms to map count: {len(sample_terms) if sample_terms is not None else 0}")

    def transform(self) -> Any:
        """Transforms the terms to map into term-entity mappings.

        Returns:
            None.
        """
        logger.info("[transform|in]")
        entity_mapper = TermEntity()
        if "terms_to_map" in self._data and 0 < len(self._data["terms_to_map"]):
            terms_to_map = self._data["terms_to_map"]
            df_terms_to_map = pd.DataFrame({"term": terms_to_map})

            df_mapped = entity_mapper.process(context={"dataframe": df_terms_to_map})
            df_mapped = df_mapped[df_mapped["entity"].notna()]
            if not df_mapped.empty:
                self._result = Dataset.from_pandas(df_mapped)

        logger.info(f"[transform|out] result shape: {self._result.shape if self._result is not None else (0, 0)}")

    @Etl4GH.inject_configuration
    def load(self, term_entity_data_url: str) -> str:
        """Loads the transformed term-entity mapping into the silver dataset store.

        Args:
            term_entity_data_url: Prefix of the silver dataset table to save to.

        Returns:
            None.
        """
        logger.info(f"[load|in] ({term_entity_data_url})")

        if (self._result is not None) and (0 < self._result.num_rows):
            self._store.save(
                df=self._result,
                key=term_entity_data_url,
                split="train",
                append=True,
                data_contract=self.contract_path("term_entity"),
            )
            Metrics.instance().add_to_gauge(
                name="fda_faers.term_entity_mapping.rows",
                value=self._result.shape[0],
                attributes={"type": "term_mapped_new"},
            )
        logger.info("[load|out]")
