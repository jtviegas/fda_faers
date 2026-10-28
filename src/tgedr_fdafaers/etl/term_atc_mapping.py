"""Build and persist mappings from FAERS terms to ATC codes."""

from typing import Any
import logging
import atexit
from datasets import Dataset
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore, NoStoreException
from tgedr_observability.metrics import Metrics
from tgedr_fdafaers.utils.common_etl import CommonEtl

logger = logging.getLogger(__name__)


class TermAtcMapping(CommonEtl):
    """Map FAERS terms to ATC codes using term-entity and entity-ATC datasets."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initializes the TermAtcMapping ETL step.

        Args:
            configuration: Optional configuration dictionary passed to the parent ETL class.
        """
        super().__init__(configuration=configuration)
        self._data: dict[str, Dataset] = {}
        self._result: Dataset | None = None
        self._store: HuggingFaceDatasetFileArrowBasedStore = HuggingFaceDatasetFileArrowBasedStore()
        atexit.register(Metrics.app_shutdown)

    @CommonEtl.inject_configuration
    def extract(self, silver_dataset_prefix: str) -> Any:
        """Load unmapped terms and their entity-to-ATC data from the silver layer.

        Terms already present in ``term_atc`` are excluded from processing. If
        that mapping dataset is absent, all terms are considered unmapped.

        Args:
            silver_dataset_prefix: Prefix used to locate the silver datasets.

        Returns:
            None. Filtered datasets are stored in ``self._data``.
        """
        logger.info(f"[extract|in] ({silver_dataset_prefix})")

        term_atc_dataset = f"{silver_dataset_prefix}term_atc"
        try:
            mapped_terms: set[str] = set(self._store.get(key=term_atc_dataset, streaming=False).train.unique("term"))
        except NoStoreException as nse:
            logger.warning(f"[extract] dataset '{term_atc_dataset}' not found on Hugging Face Hub: {nse}")
            mapped_terms = set()

        Metrics.instance().add_to_gauge(
            name="fda_faers.term_atc_mapping.rows",
            value=len(mapped_terms),
            attributes={"type": "term_atc_mapped"},
        )

        term_entity_table = f"{silver_dataset_prefix}term_entity"
        ds_term_entity = self._store.get(key=term_entity_table, streaming=False).train.filter(
            lambda row: row["term"] not in mapped_terms
        )
        self._data["term_entity"] = ds_term_entity
        Metrics.instance().add_to_gauge(
            name="fda_faers.term_atc_mapping.rows",
            value=len(ds_term_entity),
            attributes={"type": "term_atc_unmapped"},
        )
        entities: set[str] = set(ds_term_entity.unique("entity"))
        entity_atc_table = f"{silver_dataset_prefix}entity_atc"
        self._data["entity_atc"] = self._store.get(key=entity_atc_table, streaming=False).train.filter(
            lambda row: row["entity"] in entities
        )

        logger.info("[extract|out]")

    def transform(self) -> Any:
        """Map each unmapped term to the ATC codes of its linked entities.

        ATC codes are looked up per entity, then accumulated for each term.
        Terms without any matching ATC codes are omitted. The resulting
        Hugging Face dataset is stored in ``self._result`` and a metric records
        the number of newly mapped terms.

        Returns:
            None. The mapped dataset is stored in ``self._result`` for ``load``.
        """
        logger.info("[transform|in]")

        entity_atc_lookup: dict[str, list[Any]] = {}
        for row in self._data["entity_atc"]:
            entity_atc_lookup.setdefault(row["entity"], []).extend(row["atc_codes"])

        term_atc_map: dict[str, list[Any]] = {}
        for te in self._data["term_entity"]:
            term = te["term"]
            entity = te["entity"]
            atc_codes = entity_atc_lookup.get(entity, [])
            if term not in term_atc_map:
                term_atc_map[term] = []
            term_atc_map[term].extend(atc_codes)

        new_rows = [
            {"term": term, "atc_codes": atc_codes} for term, atc_codes in term_atc_map.items() if 0 < len(atc_codes)
        ]
        if 0 < len(new_rows):
            self._result = Dataset.from_list(new_rows)
            Metrics.instance().add_to_gauge(
                name="fda_faers.term_atc_mapping.rows",
                value=self._result.num_rows,
                attributes={"type": "term_atc_mapped_new"},
            )

        logger.info(f"[transform|out] => {self._result}")

    @CommonEtl.inject_configuration
    def load(self, silver_dataset_prefix: str) -> None:
        """Save the generated term-to-ATC dataset to the silver layer.

        Empty or missing results are not written. Nonempty results are
        appended to the existing ``term_atc`` dataset.

        Args:
            silver_dataset_prefix: Prefix used to name the output dataset.

        Returns:
            None.
        """
        logger.info(f"[load|in] ({silver_dataset_prefix})")
        term_atc_dataset = f"{silver_dataset_prefix}term_atc"
        if (self._result is not None) and self._result.num_rows > 0:
            self._store.save(df=self._result, key=term_atc_dataset, split="train", append=True)
        logger.info("[load|out]")
