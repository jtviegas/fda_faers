"""Module for mapping drug terms to ATC classes using the NLM RxNav API.

Provides both a term-level helper (``map_atc_classes``) and a DataFrame-level
decorator (``NLMMapping.decorate``) that resolves unique terms via a PySpark UDF.
"""

from typing import Any
import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pvprototypes_faers.data_mapping.nlm_rxnav_api.facade import NLMRxNavApiFacade
from pvprototypes_faers.data_mapping.atc_mapping.mapping import AtcClass, AtcMapping

logger = logging.getLogger(__name__)


def map_atc_classes(term: str) -> list[AtcClass]:
    """Map ATC classes for a given drug term using the NLM RxNav API.

    Args:
        term: The drug term to map ATC classes for.

    Returns:
        A list of AtcClass objects.
    """
    logger.info(f"[map_atc_classes|in] ({term})")
    facade = NLMRxNavApiFacade()
    result: list[AtcClass] = []

    rxnormid: str | None = facade.get_rxnormid(term)
    if rxnormid is None:
        logger.warning(f"[map_atc_classes] No RxNorm ID found for term: {term}")
        return []

    ingredients: list[Any] = facade.get_concepts(rxnormid, tty_filter=["IN", "PIN"])
    if len(ingredients) == 0:
        logger.warning(f"[map_atc_classes] No active RxNorm ID ({rxnormid}) ingredients found, will try history")
        ingredients = facade.get_ingredients_from_history(rxnormid)

    if len(ingredients) == 0:
        logger.warning(f"[map_atc_classes] No ingredients found for RxNorm ID: {rxnormid}")
        return []

    for ingredient in ingredients:
        rxcui = ingredient["rxcui"]
        for atc_class in facade.get_atc_classes(rxcui):
            result.append(  # noqa: PERF401
                AtcClass(id=atc_class["classId"], name=atc_class["className"], class_type=atc_class["classType"])
            )

    logger.info(f"[map_atc_classes|out] => {result}")
    return result


class NLMMapping(AtcMapping):
    """DataFrame-level ATC mapping using the NLM RxNav API.

    Unique terms are resolved first (deduplication) so that each distinct drug
    name hits the NLM API exactly once. The mapped results are then joined
    back to the original DataFrame.

    Example::

        mapping = NLMMapping(term_col="term", output_col="atc_codes")
        decorated_df = mapping.decorate(drug_df)
    """

    def __init__(self, term_col: str = "term", output_col: str = "atc_codes") -> None:
        """Initialise with configurable column names.

        Args:
            term_col: DataFrame column holding drug/ingredient term strings.
            output_col: Name of the column added with the resolved ATC codes.
        """
        super().__init__(term_col=term_col, output_col=output_col)

    def decorate(self, df: DataFrame) -> DataFrame:
        """Decorate *df* with ATC codes resolved via the NLM RxNav API.

        Steps:
        1. Select distinct values of ``term_col`` to avoid duplicate API calls.
        2. Apply a UDF that calls the NLM API for each unique term.
        3. Left-join the results back to the original DataFrame.

        Args:
            df: Input DataFrame. Must contain a string column named ``self._term_col``.

        Returns:
            Original DataFrame with ``self._output_col`` appended.
        """
        logger.info(f"[decorate|in] term_col={self._term_col!r}, output_col={self._output_col!r}")

        @F.udf(returnType=AtcMapping.ATC_CODE_SCHEMA)
        def _resolve_atc(term: str) -> list[dict] | None:
            if not term:
                return None
            classes = map_atc_classes(term=term)
            return [c.to_dict() for c in classes] if classes else None

        unique_terms = df.select(self._term_col).distinct()
        mapped = unique_terms.withColumn(self._output_col, _resolve_atc(F.col(self._term_col))).withColumn(
            "strategy", F.lit(self.__class__.__name__)
        )
        result = df.join(mapped, on=self._term_col, how="left")

        logger.info(f"[decorate|out] => added column {self._output_col!r}")
        return result
