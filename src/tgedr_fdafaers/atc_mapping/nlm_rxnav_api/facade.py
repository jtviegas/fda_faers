"""NLM RxNav API facade for mapping drugs to RxNorm Ingredient IDs and its ATC classifications.

References:
    - RxNav API documentation: https://rxnav.nlm.nih.gov/RxNormAPIs.html
    - RxNav main site: https://lhncbc.nlm.nih.gov/RxNav/
    - https://lhncbc.nlm.nih.gov/RxNav/applications/RxClassIntro.html
    - https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getRxcuiHistoryStatus.html

"""

import logging
from typing import Any, Final, TypedDict

import requests


logger = logging.getLogger(__name__)


class AtcMappingError(Exception):
    """Raised when the NLM RxNav API cannot be reached or returns a server error."""


class Concept(TypedDict):
    """A related RxNorm concept."""

    rxcui: str
    name: str
    tty: str


class ConceptWithActive(Concept):
    """A related RxNorm concept that also carries an active flag."""

    active: str


class Ingredient(TypedDict):
    """An RxNorm ingredient."""

    rxcui: str
    name: str


class HistoryStatus(TypedDict):
    """Historical status for an RxNorm ID."""

    status: str | None
    concepts: dict[str, Concept]
    ingredients: dict[str, Ingredient]


class AtcClass(TypedDict):
    """An ATC class concept returned by the RxClass API."""

    classId: str
    className: str
    classType: str
    rela: str
    relaSource: str


class NLMRxNavApiFacade:
    """Facade for NLM RxNav API to map drugs to RxNorm Ingredient IDs and its ATC classifications."""

    TERM_TO_RXNORMID_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    RXNORMID_TO_CLASS_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
    RELATED_CONCEPTS_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    DRUG_NAMES_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    HISTORY_STATUS_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/historystatus.json"
    TIMEOUT: Final[int] = 30
    INGREDIENT_TTY_FILTER: Final[list[str]] = ["PIN", "IN"]
    CONCEPT_TTY_FILTER: Final[list[str]] = ["SCD", "QD"]

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """GET a URL and return the parsed JSON payload.

        Client errors (4xx) return None so callers can fall back to a default value;
        server errors (5xx), network failures and invalid payloads raise AtcMappingError.

        Args:
            url: The URL to request.
            params: Optional query parameters.

        Returns:
            The parsed JSON payload, or None for 4xx responses.

        Raises:
            AtcMappingError: on network failures, 5xx responses or invalid JSON.
        """
        try:
            response = requests.get(url, params=params, timeout=self.TIMEOUT)
        except requests.RequestException as ex:
            raise AtcMappingError(f"[_get_json] request failed for url: {url}") from ex
        if response.status_code >= 500:
            raise AtcMappingError(f"[_get_json] server error {response.status_code} for url: {url}")
        if response.status_code >= 400:
            logger.warning(f"[_get_json] client error {response.status_code} for url: {url}")
            return None
        try:
            return response.json()
        except ValueError as ex:
            raise AtcMappingError(f"[_get_json] invalid JSON for url: {url}") from ex

    def get_rxnormid(self, term: str) -> str | None:
        """Maps a string to the most likely RxNorm ID."""
        logger.debug(f"[get_rxnormid|in] ({term})")

        params = {"term": term, "maxEntries": 1}
        result: str | None = None
        response = self._get_json(self.TERM_TO_RXNORMID_URL, params=params)
        if response is not None:
            candidates = response.get("approximateGroup", {}).get("candidate", [])
            result = candidates[0]["rxcui"] if candidates else None
        logger.debug(f"[get_rxnormid|out] => {result}")
        return result

    def get_concepts(self, rxnormid: str, tty_filter: list[str] | None = None) -> list[Concept]:
        """Get related concepts for a given RxNorm ID.

        Args:
            rxnormid: The RxNorm ID to retrieve concepts for.
            tty_filter: Optional list of TTY (term type) codes to filter concepts by.

        Returns:
            A list of concept properties (rxcui, name and tty).
        """
        logger.debug(f"[get_concepts|in] ({rxnormid})")
        result: list[Concept] = []
        response = self._get_json(self.RELATED_CONCEPTS_URL.format(rxcui=rxnormid))
        if response is not None:
            entries: dict[str, Concept] = {}
            related_concepts = response.get("allRelatedGroup", {}).get("conceptGroup", [])
            for entry in related_concepts:
                if tty_filter is None or entry.get("tty") in tty_filter:
                    for concept_property in entry.get("conceptProperties", []):
                        entries[concept_property.get("rxcui")] = {
                            "rxcui": concept_property.get("rxcui"),
                            "name": concept_property.get("name"),
                            "tty": concept_property.get("tty"),
                        }
            result = list(entries.values())
        logger.debug(f"[get_concepts|out] => {result}")
        return result

    def get_history(self, rxnormid: str) -> HistoryStatus:
        """Get historical status and related concepts for a given RxNorm ID.

        Args:
            rxnormid: The RxNorm ID to retrieve history for.

        Returns:
            A dictionary containing status, concepts, and ingredients information.

        Logs:
            Debug information about the input and output of the method.
        """
        logger.debug(f"[get_history|in] ({rxnormid})")
        concepts: dict[str, Concept] = {}
        ingredients: dict[str, Ingredient] = {}
        result: HistoryStatus = {"status": None, "concepts": concepts, "ingredients": ingredients}
        response = self._get_json(self.HISTORY_STATUS_URL.format(rxcui=rxnormid))
        if response is not None:
            status_history = response.get("rxcuiStatusHistory")
            if status_history is not None:
                metadata = status_history.get("metaData")
                result["status"] = metadata.get("status", "").lower()

                for feature in status_history.get("definitionalFeatures", {}).get("ingredientAndStrength", []):
                    ingredients[feature.get("activeIngredientRxcui")] = {
                        "name": feature.get("activeIngredientName"),
                        "rxcui": feature.get("activeIngredientRxcui"),
                    }
                derived_concepts = status_history.get("derivedConcepts", None)
                if derived_concepts is not None:
                    for ingredient_concept in derived_concepts.get("ingredientConcept", []):
                        ingredients[ingredient_concept.get("ingredientRxcui")] = {
                            "name": ingredient_concept.get("ingredientName"),
                            "rxcui": ingredient_concept.get("ingredientRxcui"),
                        }
                    for quantified_concept in derived_concepts.get("quantifiedConcept", []):
                        concepts[quantified_concept.get("quantifiedRxcui")] = {
                            "name": quantified_concept.get("quantifiedName"),
                            "tty": quantified_concept.get("quantifiedTTY"),
                            "active": quantified_concept.get("quantifiedActive"),
                            "rxcui": quantified_concept.get("quantifiedRxcui"),
                        }

                    for remapped_concept in derived_concepts.get("remappedConcept", []):
                        concepts[remapped_concept.get("remappedRxCui")] = {
                            "name": remapped_concept.get("remappedName"),
                            "tty": remapped_concept.get("remappedTTY"),
                            "active": remapped_concept.get("remappedActive"),
                            "rxcui": remapped_concept.get("remappedRxCui"),
                        }

                    if "scdConcept" in derived_concepts:
                        scd_concept = derived_concepts.get("scdConcept")
                        # semantic clinical drugs
                        # you must perform a new API call to /rxcui/{SCD_RxCUI}/allrelated.json or /rxcui/{SCD_RxCUI}/property?propName=INGREDIENT
                        concepts[scd_concept.get("scdConceptRxcui")] = {
                            "name": scd_concept.get("scdConceptName"),
                            "tty": "SCD",
                            "rxcui": scd_concept.get("scdConceptRxcui"),
                        }
                    if "qdFreeConcept" in derived_concepts:
                        qd_free_concept = derived_concepts.get("qdFreeConcept")
                        # A QD (Quantified Clinical Drug) is simply an SCD that has been "wrapped" in a specific volume or count.
                        # Take the remappedRxCui of the QD and call the /rxcui/{rxcui}/allrelated.json endpoint, which returns the IN (Ingredient) directly.
                        concepts[qd_free_concept.get("qdFreeRxcui")] = {
                            "name": qd_free_concept.get("qdFreeName"),
                            "tty": "QD",
                            "rxcui": qd_free_concept.get("qdFreeRxcui"),
                        }
        logger.debug(f"[get_history|out] => {result}")
        return result

    def get_atc_classes(self, rxnormid: str) -> list[AtcClass]:
        """Get ATC classifications for a given RxNorm ID.

        Args:
            rxnormid: The RxNorm ID to retrieve ATC classifications for.

        Returns:
            A list of ATC class information.
        """
        logger.debug(f"[get_atc_classes|in] ({rxnormid})")
        params = {"rxcui": rxnormid, "relaSource": "ATC"}
        result: list[AtcClass] = []
        response = self._get_json(self.RXNORMID_TO_CLASS_URL, params=params)
        if response is not None:
            for clazz in response.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
                rxclass = clazz.get("rxclassMinConceptItem", None)
                if rxclass:
                    result.append(rxclass)
        logger.debug(f"[get_atc_classes|out] => {result}")
        return result

    def find_concept_ingredients(self, concepts: dict[str, Concept]) -> list[Concept]:
        """Find ingredient concepts from a given set of concepts.

        Args:
            concepts: A dictionary of concepts with rxcui as key.

        Returns:
            A list of ingredient concepts found.
        """
        logger.debug(f"[find_concept_ingredients|in] ({concepts})")
        ingredients: dict[str, Concept] = {}
        for rxnormid, concept in concepts.items():
            if concept.get("tty") in self.CONCEPT_TTY_FILTER:
                related_concepts = self.get_concepts(rxnormid, tty_filter=self.INGREDIENT_TTY_FILTER)
                for related_concept in related_concepts:
                    rxcui = related_concept["rxcui"]
                    ingredients[rxcui] = {
                        "rxcui": rxcui,
                        "name": related_concept["name"],
                        "tty": related_concept["tty"],
                    }
            # preference to the ingredients found directly
            if concept.get("tty") in self.INGREDIENT_TTY_FILTER:
                ingredients[rxnormid] = {
                    "rxcui": rxnormid,
                    "name": concept["name"],
                    "tty": concept["tty"],
                }
        result = list(ingredients.values())
        logger.debug(f"[find_concept_ingredients|out] => {result}")
        return result

    def get_ingredients_from_history(self, rxnormid: str) -> list[Concept]:
        """Get all ingredients for a given RxNorm ID from its history.

        Args:
            rxnormid: The RxNorm ID to retrieve ingredients for.

        Returns:
            A list of ingredients found in the history, combining concept-derived and direct ingredients.
        """
        logger.debug(f"[get_ingredients_from_history|in] ({rxnormid})")
        history_status: HistoryStatus = self.get_history(rxnormid)
        concept_ingredients: list[Concept] = self.find_concept_ingredients(history_status.get("concepts", {}))
        seen: set[str] = {ingredient["rxcui"] for ingredient in concept_ingredients}
        result = concept_ingredients + [
            ingredient
            for ingredient in history_status.get("ingredients", {}).values()
            if ingredient["rxcui"] not in seen
        ]
        logger.debug(f"[get_ingredients_from_history|out] => {result}")
        return result
