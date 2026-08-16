"""NLM RxNav API facade for mapping drugs to RxNorm Ingredient IDs and its ATC classifications.

References:
    - RxNav API documentation: https://rxnav.nlm.nih.gov/RxNormAPIs.html
    - RxNav main site: https://lhncbc.nlm.nih.gov/RxNav/
    - https://lhncbc.nlm.nih.gov/RxNav/applications/RxClassIntro.html
    - https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getRxcuiHistoryStatus.html

"""

from typing import Any, Final
import logging
import requests


logger = logging.getLogger(__name__)


class NLMRxNavApiFacade:
    """Facade for NLM RxNav API to map drugs to RxNorm Ingredient IDs and its ATC classifications."""

    TERM_TO_RXNORMID_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    RXNORMID_TO_CLASS_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
    RELATED_CONCEPTS_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
    DRUG_NAMES_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    HISTORY_STATUS_URL: Final[str] = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/historystatus.json"

    def get_rxnormid(self, term: str) -> str | None:
        """Maps a string to the most likely RxNorm ID."""
        logger.info(f"[get_rxnormid|in] ({term})")

        params = {"term": term, "maxEntries": 1}
        result = None
        try:
            response = requests.get(self.TERM_TO_RXNORMID_URL, params=params).json()  # noqa: S113
            candidates = response.get("approximateGroup", {}).get("candidate", [])
            result = candidates[0]["rxcui"] if candidates else None
        except Exception:
            logger.warning(f"[get_rxnormid] get failed for term: {term}")
        logger.info(f"[get_rxnormid|out] => {result}")
        return result

    def get_concepts(self, rxnormid: str, tty_filter: list[str] | None = None) -> list[Any]:
        """Get related concepts for a given RxNorm ID.

        Args:
            rxnormid: The RxNorm ID to retrieve concepts for.
            tty_filter: Optional list of TTY (term type) codes to filter concepts by.

        Returns:
            A list of concept properties (rxcui, name and tty).
        """
        logger.info(f"[get_concepts|in] ({rxnormid})")
        result: list[Any] = []
        try:
            entries = {}
            response = requests.get(self.RELATED_CONCEPTS_URL.format(rxcui=rxnormid)).json()  # noqa: S113
            related_concepts = response.get("allRelatedGroup", {}).get("conceptGroup", [])
            for entry in related_concepts:
                if (tty_filter and entry.get("tty") in tty_filter) or not tty_filter:
                    for property in entry.get("conceptProperties", []):
                        entries[property.get("rxcui")] = {
                            "rxcui": property.get("rxcui"),
                            "name": property.get("name"),
                            "tty": property.get("tty"),
                        }
            result = list(entries.values())
        except Exception:
            logger.warning(f"[get_concepts] get failed for rxnormid: {rxnormid}")
        logger.info(f"[get_concepts|out] => {result}")
        return result

    def get_history(self, rxnormid: str) -> dict[str, Any]:
        """Get historical status and related concepts for a given RxNorm ID.

        Args:
            rxnormid: The RxNorm ID to retrieve history for.

        Returns:
            A dictionary containing status, concepts, and ingredients information.
        """
        logger.info(f"[get_history|in] ({rxnormid})")
        concepts: dict[str, Any] = {}
        ingredients: dict[str, Any] = {}
        result: dict[str, Any] = {"status": None, "concepts": concepts, "ingredients": ingredients}
        try:
            response = requests.get(self.HISTORY_STATUS_URL.format(rxcui=rxnormid)).json()  # noqa: S113
            status_history = response.get("rxcuiStatusHistory")
            metadata = status_history.get("metaData")
            status = metadata.get("status", "").lower()
            result["status"] = status

            for feature in status_history.get("definitionalFeatures", {}).get("ingredientAndStrength", []):
                ingredients[feature.get("activeIngredientRxcui")] = {
                    "name": feature.get("activeIngredientName"),
                    "rxcui": feature.get("activeIngredientRxcui"),
                }
            derivedConcepts = status_history.get("derivedConcepts", None)
            if derivedConcepts is not None:
                for ingredientConcept in derivedConcepts.get("ingredientConcept", []):
                    ingredients[ingredientConcept.get("ingredientRxcui")] = {
                        "name": ingredientConcept.get("ingredientName"),
                        "rxcui": ingredientConcept.get("ingredientRxcui"),
                    }
                for quantifiedConcept in derivedConcepts.get("quantifiedConcept", []):
                    concepts[quantifiedConcept.get("quantifiedRxcui")] = {
                        "name": quantifiedConcept.get("quantifiedName"),
                        "tty": quantifiedConcept.get("quantifiedTTY"),
                        "active": quantifiedConcept.get("quantifiedActive"),
                        "rxcui": quantifiedConcept.get("quantifiedRxcui"),
                    }

                for remappedConcept in derivedConcepts.get("remappedConcept", []):
                    concepts[remappedConcept.get("remappedRxCui")] = {
                        "name": remappedConcept.get("remappedName"),
                        "tty": remappedConcept.get("remappedTTY"),
                        "active": remappedConcept.get("remappedActive"),
                        "rxcui": remappedConcept.get("remappedRxCui"),
                    }

                if "scdConcept" in derivedConcepts:
                    scdConcept = derivedConcepts.get("scdConcept")
                    # semantic clinical drugs
                    # you must perform a new API call to /rxcui/{SCD_RxCUI}/allrelated.json or /rxcui/{SCD_RxCUI}/property?propName=INGREDIENT
                    concepts[scdConcept.get("scdConceptRxcui")] = {
                        "name": scdConcept.get("scdConceptName"),
                        "tty": "SCD",
                        "rxcui": scdConcept.get("scdConceptRxcui"),
                    }
                if "qdFreeConcept" in derivedConcepts:
                    qdFreeConcept = derivedConcepts.get("qdFreeConcept")
                    # A QD (Quantified Clinical Drug) is simply an SCD that has been "wrapped" in a specific volume or count.
                    # Take the remappedRxCui of the QD and call the /rxcui/{rxcui}/allrelated.json endpoint , This will return the IN (Ingredient) directly, ignoring the "100 mL" wrapper.
                    concepts[qdFreeConcept.get("qdFreeRxcui")] = {
                        "name": qdFreeConcept.get("qdFreeName"),
                        "tty": "QD",
                        "rxcui": qdFreeConcept.get("qdFreeRxcui"),
                    }
        except Exception as x:
            logger.warning(f"[get_history] get failed for rxnormid: {rxnormid} - {x}")
        logger.info(f"[get_history|out] => {result}")
        return result

    def get_atc_classes(self, rxnormid: str) -> list[Any]:
        """Get ATC classifications for a given RxNorm ID.

        Args:
            rxnormid: The RxNorm ID to retrieve ATC classifications for.

        Returns:
            A list of ATC class information.
        """
        logger.info(f"[get_atc_classes|in] ({rxnormid})")
        params = {"rxcui": rxnormid, "relaSource": "ATC"}
        result: list[Any] = []
        try:
            response = requests.get(self.RXNORMID_TO_CLASS_URL, params=params).json()  # noqa: S113
            for clazz in response.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
                rxclass = clazz.get("rxclassMinConceptItem", None)
                if rxclass:
                    result.append(rxclass)
        except Exception:
            logger.warning(f"[get_atc_classes] get failed for RxCUI: {rxnormid}")
        logger.info(f"[get_atc_classes|out] => {result}")
        return result

    def find_concept_ingredients(self, concepts: dict[str, Any]) -> list[Any]:
        """Find ingredient concepts from a given set of concepts.

        Args:
            concepts: A dictionary of concepts with rxcui as key.

        Returns:
            A list of ingredient concepts found.
        """
        logger.info(f"[find_concept_ingredients|in] ({concepts})")
        result: list[Any] = []
        ingredients: dict[str, Any] = {}
        ingredient_filter = ["PIN", "IN"]
        for rxnormid, concept in concepts.items():
            if concept.get("tty") in ["SCD", "QD"]:
                related_concepts = self.get_concepts(rxnormid, tty_filter=ingredient_filter)
                for related_concept in related_concepts:
                    rxcui = related_concept["rxcui"]
                    ingredients[rxcui] = {
                        "rxcui": rxcui,
                        "name": related_concept["name"],
                        "tty": related_concept["tty"],
                    }
            # prefference to the ingredients found directly
            if concept.get("tty") in ingredient_filter:
                ingredients[rxnormid] = {
                    "rxcui": rxnormid,
                    "name": concept["name"],
                    "tty": concept["tty"],
                }
        result = list(ingredients.values())
        logger.info(f"[find_concept_ingredients|out] => {result}")
        return result

    def get_ingredients_from_history(self, rxnormid: str) -> list[Any]:
        """Get all ingredients for a given RxNorm ID from its history.

        Args:
            rxnormid: The RxNorm ID to retrieve ingredients for.

        Returns:
            A list of ingredients found in the history, combining concept-derived and direct ingredients.
        """
        logger.info(f"[get_ingredients_from_history|in] ({rxnormid})")
        result: list[Any] = []
        history_status: dict[str, Any] = self.get_history(rxnormid)
        concept_ingredients: list[Any] = self.find_concept_ingredients(history_status.get("concepts", {}))
        history_ingredients = history_status.get("ingredients", {}).values()
        result = concept_ingredients + [
            ingredient
            for ingredient in history_ingredients
            if ingredient["rxcui"] not in [ing["rxcui"] for ing in concept_ingredients]
        ]
        logger.info(f"[get_ingredients_from_history|out] => {result}")
        return result
