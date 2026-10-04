"""ETL step that maps drug terms to entities.

This module provides:
- TermEntityMapping: ETL step that extracts unmapped drug terms from bronze,
  maps them to entities, and loads the resulting mappings into silver.
"""

from typing import Any
import pandas as pd
import logging
from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.contracted_store import ContractedHFDatasetFileBasedStore, NoStoreException
from tgedr_fdafaers.entity_mapping.term_entity import TermEntity
from tgedr_observability.metrics import Metrics

logger = logging.getLogger(__name__)


class TermEntityMapping(Etl4GH):
    """ETL step that maps drug terms to entities, extracting unmapped terms from bronze and loading mappings into silver."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initializes the TermEntityMapping ETL step.

        Args:
            configuration: Optional configuration dictionary passed to the parent ETL class.
        """
        super().__init__(configuration=configuration)
        self._data: dict[str, pd.DataFrame] = {}
        self._result: pd.DataFrame | None = None
        self._store: ContractedHFDatasetFileBasedStore = ContractedHFDatasetFileBasedStore(
            config={"visibility": "public"}
        )

    @Etl4GH.inject_configuration
    def extract(self, bronze_dataset_prefix: str, silver_dataset_prefix: str, sample_size: int = 300000) -> Any:
        """Extracts terms to map from the bronze dataset, excluding already mapped terms.

        Args:
            bronze_dataset_prefix: Prefix of the bronze dataset tables to read from.
            silver_dataset_prefix: Prefix of the silver dataset tables to read from.
            sample_size: Maximum number of terms to sample for mapping.

        Returns:
            None.
        """
        logger.info(f"[extract|in] ({bronze_dataset_prefix}, {silver_dataset_prefix}, {sample_size})")

        drug_table = f"{bronze_dataset_prefix}drug"
        df_terms = (self._store.get(key=drug_table).train)[["term"]].drop_duplicates()
        logger.info(f"[extract] bronze terms shape: {df_terms.shape}")
        Metrics.instance().add_to_gauge(
            name="fda_faers.drug_term_entity_mapping.rows", value=df_terms.shape[0], attributes={"type": "term"}
        )

        term_entity_table = f"{silver_dataset_prefix}term_entity"
        df_mapped_terms: pd.DataFrame | None = None
        try:
            ds_term_entity = self._store.get(key=term_entity_table)
            df_mapped_terms = (ds_term_entity.train)[["term"]].drop_duplicates()
            Metrics.instance().add_to_gauge(
                name="fda_faers.drug_term_entity_mapping.rows",
                value=df_mapped_terms.shape[0],
                attributes={"type": "term_entity"},
            )
            logger.info(f"[extract] mapped terms shape: {df_mapped_terms.shape}")
        except NoStoreException as nse:
            logger.warning(f"[extract] silver table {term_entity_table} not found: {nse}")

        df_terms_to_map: pd.DataFrame = df_terms
        if df_mapped_terms is not None:
            df_terms_to_map = df_terms_to_map.merge(
                df_mapped_terms,
                on="term",
                how="left",
                indicator=True,
            )
            df_terms_to_map = (df_terms_to_map[df_terms_to_map["_merge"] == "left_only"])[["term"]].drop_duplicates()

        sample_size = min(sample_size, df_terms_to_map.shape[0])
        self._data["terms_to_map"] = df_terms_to_map.sample(n=sample_size)
        logger.info(f"[extract|out] terms to map shape: {self._data['terms_to_map'].shape}")

    def transform(self) -> Any:
        """Transforms the terms to map into term-entity mappings.

        Returns:
            The transformed dataframe with non-null entity mappings.
        """
        logger.info("[transform|in]")
        entity_mapper = TermEntity()
        if not (self._data["terms_to_map"]).empty:
            df_mapped = entity_mapper.process(context={"dataframe": self._data["terms_to_map"]})
            df_mapped = df_mapped[df_mapped["entity"].notna()]
            if not df_mapped.empty:
                self._result = df_mapped

        logger.info(f"[transform|out] result shape: {self._result.shape if self._result is not None else (0, 0)}")

    @Etl4GH.inject_configuration
    def load(self, silver_dataset_prefix: str) -> str:
        """Loads the transformed term-entity mapping into the silver dataset store.

        Args:
            silver_dataset_prefix: Prefix of the silver dataset table to save to.

        Returns:
            The name of the table where the data was saved.
        """
        logger.info(f"[load|in] ({silver_dataset_prefix})")
        term_entity_table = f"{silver_dataset_prefix}term_entity"
        if (self._result is not None) and (not self._result.empty):
            self._store.save(df=self._result, key=term_entity_table, split="train", append=True)
            Metrics.instance().add_to_gauge(
                name="fda_faers.drug_term_entity_mapping.rows",
                value=self._result.shape[0],
                attributes={"type": "term_entity_new"},
            )
        logger.info("[load|out]")
