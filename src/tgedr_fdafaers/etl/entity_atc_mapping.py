"""ETL step that maps entities to ATC codes.

This module provides:
- EntityAtcMapping: an ETL step that extracts unmapped entities from the silver
  dataset, maps them to ATC codes using the NLM mapping, and loads the results
  back into the silver entity_atc table.
"""

from typing import Any
import pandas as pd
import logging
from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_dataops.store.contracted_store import ContractedHFDatasetFileBasedStore, NoStoreException
from tgedr_observability.metrics import Metrics
from tgedr_fdafaers.atc_mapping.nlm_mapping import NLMMapping

logger = logging.getLogger(__name__)


class EntityAtcMapping(Etl4GH):
    """ETL step that maps entities to ATC codes.

    Extracts unmapped entities from the silver dataset, maps them to ATC codes
    using the NLM mapping, and loads the results back into the silver
    entity_atc table.
    """

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initializes the EntityAtcMapping ETL step.

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
    def extract(self, silver_dataset_prefix: str, sample_size: int = 50000) -> Any:
        """Extracts unmapped entities from the silver dataset.

        Reads the silver ``term_entity`` table, removes entities already present
        in the ``entity_atc`` table, and samples up to ``sample_size`` entities
        to be mapped. The resulting entities are stored in ``self._data`` for
        the transform step.

        Args:
            silver_dataset_prefix: Prefix of the silver dataset tables.
            sample_size: Maximum number of entities to sample for mapping.
        """
        logger.info(f"[extract|in] ({silver_dataset_prefix}, {sample_size})")

        term_entity_table = f"{silver_dataset_prefix}term_entity"
        df_entity = (self._store.get(key=term_entity_table).train)[["entity"]].drop_duplicates()
        logger.info(f"[extract] silver entities shape: {df_entity.shape}")
        Metrics.instance().add_to_gauge(
            name="fda_faers.entity_atc_mapping.rows", value=df_entity.shape[0], attributes={"type": "entity"}
        )

        entity_atc_table = f"{silver_dataset_prefix}entity_atc"
        df_mapped_entities: pd.DataFrame | None = None
        try:
            ds_entity_atc = self._store.get(key=entity_atc_table)
            df_mapped_entities = (ds_entity_atc.train)[["entity"]].drop_duplicates()
            logger.info(f"[extract] mapped entities shape: {df_mapped_entities.shape}")
            Metrics.instance().add_to_gauge(
                name="fda_faers.entity_atc_mapping.rows",
                value=df_mapped_entities.shape[0],
                attributes={"type": "entity_mapped"},
            )
        except NoStoreException as nse:
            logger.warning(f"[extract] silver table {entity_atc_table} not found: {nse}")

        df_entities_to_map: pd.DataFrame = df_entity
        if df_mapped_entities is not None:
            df_entities_to_map = df_entities_to_map.merge(
                df_mapped_entities,
                on="entity",
                how="left",
                indicator=True,
            )
            df_entities_to_map = (df_entities_to_map[df_entities_to_map["_merge"] == "left_only"])[
                ["entity"]
            ].drop_duplicates()

        if not df_entities_to_map.empty:
            sample_size = min(sample_size, df_entities_to_map.shape[0])
            self._data["df_entities_to_map"] = df_entities_to_map.sample(n=sample_size)
        logger.info(f"[extract|out] entities to map shape: {self._data['df_entities_to_map'].shape}")

    def transform(self) -> Any:
        """Maps the extracted entities to ATC codes using the NLM mapping.

        The mapped entities are stored in ``self._result`` for the load step.
        """
        logger.info("[transform|in]")

        if "df_entities_to_map" in self._data:
            atc_mapper = NLMMapping(term_col="entity")
            df_mapped = atc_mapper.decorate(df=self._data["df_entities_to_map"])
            df_mapped = df_mapped[df_mapped["atc_codes"].notna()]
            if not df_mapped.empty:
                self._result = df_mapped

        logger.info(f"[transform|out] result shape: {self._result.shape if self._result is not None else (0, 0)}")

    @Etl4GH.inject_configuration
    def load(self, silver_dataset_prefix: str) -> str:
        """Loads the mapped entities back into the silver entity_atc table.

        Args:
            silver_dataset_prefix: Prefix of the silver dataset tables.

        Returns:
            The name of the entity_atc table that was loaded.
        """
        logger.info(f"[load|in] ({silver_dataset_prefix})")
        entity_atc_table = f"{silver_dataset_prefix}entity_atc"
        if (self._result is not None) and (not self._result.empty):
            self._store.save(df=self._result, key=entity_atc_table, split="train", append=True)
            Metrics.instance().add_to_gauge(
                name="fda_faers.entity_atc_mapping.rows",
                value=self._result.shape[0],
                attributes={"type": "entity_mapped_new"},
            )
        logger.info("[load|out]")
