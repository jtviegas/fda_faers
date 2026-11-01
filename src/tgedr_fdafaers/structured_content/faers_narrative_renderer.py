"""Narrative rendering for structured FAERS case data."""

from collections.abc import Sequence
import logging
from typing import Any, ClassVar
from tgedr_dataops_abs.processor import Processor, ProcessorException


logger = logging.getLogger(__name__)


class FaersNarrativeRenderer(Processor):
    """Render structured FAERS case dictionaries as conservative prose summaries."""

    CONTEXT_KEY_CASES = "cases"
    _ROLE_LABELS: ClassVar[dict[str, str]] = {
        "PS": "primary suspect",
        "SS": "secondary suspect",
        "C": "concomitant",
        "I": "interacting",
    }
    _OUTCOME_LABELS: ClassVar[dict[str, str]] = {
        "CA": "congenital anomaly",
        "DE": "death",
        "DS": "disability",
        "HO": "hospitalization",
        "LT": "life-threatening outcome",
        "OT": "other outcome",
        "RI": "required intervention",
    }

    def process(self, context: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Render narrative summaries for each FAERS case in the context.

        Args:
            context: Processor context containing the cases under the
                `CONTEXT_KEY_CASES` key.

        Returns:
            A list of the input case dictionaries, each augmented with a
            generated `narrative` field.

        Raises:
            ProcessorException: If the context is missing or does not contain
                a sequence of case dictionaries.
        """
        logger.info("[process|in]")

        if not context or self.CONTEXT_KEY_CASES not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_CASES} must be provided in context")
        cases = context[self.CONTEXT_KEY_CASES]
        if not isinstance(cases, Sequence) or isinstance(cases, str | bytes):
            raise ProcessorException("cases must be a sequence of case dictionaries")
        result = [
            {"primaryid": case.get("demo", {}).get("primaryid"), "narrative": self._render_case(case)} for case in cases
        ]
        logger.info("[process|out]")
        return result

    def _render_case(self, case: dict[str, Any]) -> str:
        sentences: list[str] = []
        demo = case.get("demo", {})
        age = demo.get("age")
        age_code = demo.get("age_cod")
        sex = demo.get("sex")
        demographics = []
        if age is not None:
            demographics.append(f"{age} {age_code or 'age units'}-old")
        if sex and sex != "U":
            demographics.append({"F": "female", "M": "male"}.get(sex, str(sex)))
        if demographics:
            sentences.append(f"The report describes a {' '.join(demographics)} patient.")

        drug_phrases = []
        for entry in case.get("drugs", []):
            drug = entry.get("drug", {})
            name = drug.get("drugname") or drug.get("prod_ai")
            if not name:
                continue
            role = self._ROLE_LABELS.get(drug.get("role_cod"), drug.get("role_cod"))
            phrase = str(name)
            if role:
                phrase += f" ({role})"
            indications = [row.get("indi_pt") for row in entry.get("indications", []) if row.get("indi_pt")]
            if indications:
                phrase += f" for {', '.join(map(str, indications))}"
            drug_phrases.append(phrase)
        if drug_phrases:
            sentences.append(f"Reported drugs include {', '.join(drug_phrases)}.")

        reactions = self._unique_values(case.get("reactions", []), "pt")
        if reactions:
            sentences.append(f"Reported reactions include {', '.join(map(str, reactions))}.")
        outcomes = [
            self._OUTCOME_LABELS.get(value, value)
            for value in self._unique_values(case.get("outcomes", []), "outc_cod")
        ]
        if outcomes:
            sentences.append(f"Recorded outcomes include {', '.join(map(str, outcomes))}.")
        if not sentences:
            sentences.append("The report contains no narrative-ready demographic, drug, reaction, or outcome details.")
        return " ".join(sentences)

    def _unique_values(self, rows: Sequence[dict[str, Any]], column: str) -> list[Any]:
        values = []
        seen = set()
        for row in rows:
            value = row.get(column)
            if value is not None and value not in seen:
                values.append(value)
                seen.add(value)
        return values
