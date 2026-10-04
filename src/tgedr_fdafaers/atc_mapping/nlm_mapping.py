"""Module for mapping drug terms to ATC classes using the NLM RxNav API.

Provides a DataFrame-level decorator (``NLMMapping.decorate``) that resolves
unique terms via pandas, backed by a term-level method (``NLMMapping.map_atc_classes``)
that calls the NLM RxNav API.
"""

from typing import Any
import logging

import pandas as pd

from tgedr_fdafaers.atc_mapping.nlm_rxnav_api.facade import NLMRxNavApiFacade
from tgedr_fdafaers.atc_mapping.mapping import AtcClass, AtcMapping

logger = logging.getLogger(__name__)


class NLMMapping(AtcMapping):
    """DataFrame-level ATC mapping using the NLM RxNav API.

    Unique terms are resolved first (deduplication) so that each distinct drug
    name hits the NLM API exactly once. The mapped results are then joined
    back to the original DataFrame.

    Example::

        mapping = NLMMapping(term_col="term", output_col="atc_codes")
        decorated_df = mapping.decorate(drug_df)
    """

    def __init__(
        self,
        term_col: str = "term",
        output_col: str = "atc_codes",
        facade: NLMRxNavApiFacade | None = None,
    ) -> None:
        """Initialise with configurable column names and an optional facade.

        Args:
            term_col: DataFrame column holding drug/ingredient term strings.
            output_col: Name of the column added with the resolved ATC codes.
            facade: Optional NLMRxNavApiFacade instance; a new one is created
                when not provided (useful for injecting mocks in tests).
        """
        super().__init__(term_col=term_col, output_col=output_col)
        logger.info(
            f"[__init__|in] term_col={term_col!r}, output_col={output_col!r}, facade_provided={facade is not None}"
        )
        self._facade = facade if facade is not None else NLMRxNavApiFacade()
        logger.info("[__init__|out]")

    def map_atc_classes(self, term: str) -> list[AtcClass]:
        """Map ATC classes for a given drug term using the NLM RxNav API.

        Args:
            term: The drug term to map ATC classes for.

        Returns:
            A list of AtcClass objects.
        """
        logger.debug(f"[map_atc_classes|in] ({term})")
        facade = self._facade
        result: list[AtcClass] = []

        rxnormid: str | None = facade.get_rxnormid(term)
        if rxnormid is None:
            logger.debug(f"[map_atc_classes] No RxNorm ID found for term: {term}")
            return []

        ingredients: list[Any] = facade.get_concepts(rxnormid, tty_filter=["IN", "PIN"])
        if len(ingredients) == 0:
            logger.debug(f"[map_atc_classes] No active RxNorm ID ({rxnormid}) ingredients found, will try history")
            ingredients = facade.get_ingredients_from_history(rxnormid)

        if len(ingredients) == 0:
            logger.debug(f"[map_atc_classes] No ingredients found for RxNorm ID: {rxnormid}")
            return []

        for ingredient in ingredients:
            rxcui = ingredient["rxcui"]
            for atc_class in facade.get_atc_classes(rxcui):
                result.append(  # noqa: PERF401
                    AtcClass(id=atc_class["classId"], name=atc_class["className"], class_type=atc_class["classType"])
                )

        logger.debug(f"[map_atc_classes|out] => {result}")
        return result

    def decorate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decorate *df* with ATC codes resolved via the NLM RxNav API.

        Steps:
        1. Select distinct values of ``term_col`` to avoid duplicate API calls.
        2. Apply ``map_atc_classes`` to each unique term.
        3. Left-join the results back to the original DataFrame.

        Args:
            df: Input DataFrame. Must contain a string column named ``self._term_col``.

        Returns:
            Original DataFrame with ``self._output_col`` appended.
        """
        logger.info(f"[decorate|in] ({df.shape})")

        def _resolve_atc(term: str) -> list[dict] | None:
            if not term:
                return None
            classes = self.map_atc_classes(term=term)
            return [c.to_dict() for c in classes] if classes else None

        unique_terms = df[[self._term_col]].drop_duplicates()
        mapped = unique_terms.copy()
        mapped[self._output_col] = mapped[self._term_col].apply(_resolve_atc)
        mapped["strategy"] = self.__class__.__name__
        result = df.merge(mapped, on=self._term_col, how="left")
        n_mapped = mapped[self._output_col].notna().sum()
        logger.info(f"[decorate] number of terms successfully mapped: {n_mapped}")

        logger.info(f"[decorate|out] => {result.shape}")
        return result
