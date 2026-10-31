"""Processors for selecting, batching, assembling, and rendering FAERS case data."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import logging
from typing import Any, ClassVar
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import Array


from datasets import Dataset
from tgedr_dataops_abs.processor import Processor, ProcessorException

from tgedr_fdafaers.faers_processor import FaersProcessor
from tgedr_fdafaers.faers_tables import FaersTables


logger = logging.getLogger(__name__)


class FaersCaseBatcher(FaersProcessor):
    """Sort each table by PRIMARYID and yield batches without splitting a report."""

    CONTEXT_KEY_TABLES = "tables"
    CONTEXT_KEY_BATCH_SIZE = "batch_size"
    DEFAULT_BATCH_SIZE = 1000

    def process(self, context: dict[str, Any] | None = None) -> Iterator[FaersTables]:
        """Yield sorted table bundles, each containing complete PRIMARYID groups."""
        logger.info("[process|in]")
        if not context or self.CONTEXT_KEY_TABLES not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be provided in context")
        tables = context[self.CONTEXT_KEY_TABLES]
        if not isinstance(tables, FaersTables):
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be a FaersTables instance")

        batch_size = context.get(self.CONTEXT_KEY_BATCH_SIZE, self.DEFAULT_BATCH_SIZE)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ProcessorException("batch_size must be a positive integer")

        self._require_columns(tables.demo, "demo", {"primaryid"})
        sorted_tables = {}
        for name, dataset in tables.to_dict().items():
            self._require_columns(dataset, name, {"primaryid"})
            non_null_ids = dataset.filter(lambda row: row["primaryid"] is not None)
            sorted_tables[name] = non_null_ids.sort("primaryid")
        report_ids = sorted({row["primaryid"] for row in sorted_tables["demo"]})

        try:
            for offset in range(0, len(report_ids), batch_size):
                ids = report_ids[offset : offset + batch_size]
                batch_id_set = set(ids)
                first_id, last_id = ids[0], ids[-1]
                batch_tables: dict[str, Dataset] = {}
                for name, dataset in sorted_tables.items():
                    dataset_ids = dataset["primaryid"]
                    start = bisect_left(dataset_ids, first_id)
                    stop = bisect_right(dataset_ids, last_id)
                    candidate_rows = dataset.select(range(start, stop))
                    batch_tables[name] = candidate_rows.filter(lambda row, ids=batch_id_set: row["primaryid"] in ids)
                yield FaersTables.from_dict(batch_tables)
        finally:
            logger.info("[process|out]")


class FaersCaseAssembler(FaersProcessor):
    """Assemble each PRIMARYID and its linked drug, indication, and therapy rows."""

    CONTEXT_KEY_TABLES = "tables"

    def process(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return one nested structured representation per report PRIMARYID."""
        logger.info("[process|in]")
        if not context or self.CONTEXT_KEY_TABLES not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be provided in context")
        tables = context[self.CONTEXT_KEY_TABLES]
        if not isinstance(tables, FaersTables):
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be a FaersTables instance")
        self._require_columns(tables.demo, "demo", {"primaryid"})
        self._require_columns(tables.drug, "drug", {"primaryid", "drug_seq"})
        self._require_columns(tables.indi, "indi", {"primaryid", "indi_drug_seq"})
        self._require_columns(tables.ther, "ther", {"primaryid", "dsg_drug_seq"})

        tables_by_primaryid = {
            name: self._group_by_primaryid(dataset) for name, dataset in tables.to_dict().items() if name != "demo"
        }
        result: list[dict[str, Any]] = []
        seen_primaryids: set[int] = set()
        for demo in tables.demo:
            primaryid = demo["primaryid"]
            if primaryid is None:
                continue
            if primaryid in seen_primaryids:
                raise ProcessorException(f"DEMO contains multiple rows for PRIMARYID {primaryid}")
            seen_primaryids.add(primaryid)

            drugs = tables_by_primaryid["drug"].get(primaryid, [])
            indications = tables_by_primaryid["indi"].get(primaryid, [])
            therapies = tables_by_primaryid["ther"].get(primaryid, [])
            drug_entries = []
            for drug in drugs:
                drug_seq = drug.get("drug_seq")
                drug_entries.append(
                    {
                        "drug": drug,
                        "indications": [row for row in indications if row.get("indi_drug_seq") == drug_seq],
                        "therapies": [row for row in therapies if row.get("dsg_drug_seq") == drug_seq],
                    }
                )

            result.append(
                {
                    "primaryid": primaryid,
                    "caseid": demo.get("caseid"),
                    "caseversion": demo.get("caseversion"),
                    "demo": demo,
                    "drugs": drug_entries,
                    "reactions": tables_by_primaryid["reac"].get(primaryid, []),
                    "outcomes": tables_by_primaryid["outc"].get(primaryid, []),
                    "report_sources": tables_by_primaryid["rpsr"].get(primaryid, []),
                }
            )
        logger.info("[process|out]")
        return result

    def _group_by_primaryid(self, dataset: Dataset) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset:
            primaryid = row.get("primaryid")
            if primaryid is not None:
                grouped[primaryid].append(row)
        return dict(grouped)


class FaersNarrativeRenderer(FaersProcessor):
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

    def process(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return dictionaries containing each PRIMARYID and its rendered narrative."""
        logger.info("[process|in]")
        if not context or self.CONTEXT_KEY_CASES not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_CASES} must be provided in context")
        cases = context[self.CONTEXT_KEY_CASES]
        if not isinstance(cases, Sequence) or isinstance(cases, str | bytes):
            raise ProcessorException("cases must be a sequence of case dictionaries")
        result = [{**case, "narrative": self._render_case(case)} for case in cases]
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
            self._OUTCOME_LABELS.get(value, value) for value in self._unique_values(case.get("outcomes", []), "outc_cod")
        ]
        if outcomes:
            sentences.append(f"Recorded outcomes include {', '.join(map(str, outcomes))}.")
        if not sentences:
            sentences.append("The report contains no narrative-ready demographic, drug, reaction, or outcome details.")
        return " ".join(sentences)



