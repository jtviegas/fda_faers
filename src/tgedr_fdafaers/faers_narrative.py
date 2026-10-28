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


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaersTables:
    """The seven standardized FAERS tables used to assemble case representations."""

    demo: Dataset
    drug: Dataset
    reac: Dataset
    outc: Dataset
    rpsr: Dataset
    ther: Dataset
    indi: Dataset

    def to_dict(self) -> dict[str, Dataset]:
        """Return the seven datasets keyed by their FAERS table names."""
        return {
            "demo": self.demo,
            "drug": self.drug,
            "reac": self.reac,
            "outc": self.outc,
            "rpsr": self.rpsr,
            "ther": self.ther,
            "indi": self.indi,
        }

    @classmethod
    def from_dict(cls, tables: dict[str, Dataset]) -> FaersTables:
        """Create a table bundle from a mapping, requiring all seven table names.

        Args:
            tables: Mapping from lowercase FAERS table names to datasets.

        Returns:
            A bundle containing each required dataset; extra mapping keys are ignored.

        Raises:
            ProcessorException: If any required table name is absent.
        """
        names = ("demo", "drug", "reac", "outc", "rpsr", "ther", "indi")
        missing = [name for name in names if name not in tables]
        if missing:
            raise ProcessorException(f"Missing FAERS tables: {', '.join(missing)}")
        return cls(**{name: tables[name] for name in names})


class LatestCaseVersions(Processor):
    """Keep only the latest unambiguous report version for each CASEID."""

    CONTEXT_KEY_TABLES = "tables"

    def process(self, context: dict[str, Any] | None = None) -> FaersTables:
        """Filter every table to the selected latest PRIMARYIDs."""
        logger.info("[process|in]")
        if not context or self.CONTEXT_KEY_TABLES not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be provided in context")
        tables = context[self.CONTEXT_KEY_TABLES]
        if not isinstance(tables, FaersTables):
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be a FaersTables instance")

        demo_rows = list(tables.demo)
        _require_columns(tables.demo, "demo", {"primaryid", "caseid", "caseversion"})

        latest_by_case: dict[Any, dict[str, Any]] = {}
        ambiguous_latest_cases: set[Any] = set()
        selected_primaryids: set[int] = set()
        for row in demo_rows:
            primaryid = row["primaryid"]
            caseid = row["caseid"]
            version = row["caseversion"]
            if primaryid is None:
                continue
            if caseid is None:
                selected_primaryids.add(primaryid)
                continue

            previous = latest_by_case.get(caseid)
            if previous is None or _version_key(version) > _version_key(previous["caseversion"]):
                latest_by_case[caseid] = row
            elif _version_key(version) == _version_key(previous["caseversion"]):
                ambiguous_latest_cases.add(caseid)

        if ambiguous_latest_cases:
            caseid = next(iter(ambiguous_latest_cases))
            primaryids = [row["primaryid"] for row in demo_rows if row["caseid"] == caseid]
            raise ProcessorException(
                f"Ambiguous latest report version for CASEID {caseid}: "
                f"PRIMARYIDs {primaryids} have the same CASEVERSION"
            )

        selected_primaryids.update(row["primaryid"] for row in latest_by_case.values())
        # filtered = {
        #     name: dataset.filter(lambda row: row["primaryid"] in selected_primaryids)
        #     for name, dataset in tables.to_dict().items()
        # }

        selected = pa.array(sorted(selected_primaryids))
        filtered = {
            name: dataset.select(
                np.flatnonzero(pc.is_in(dataset.data.column("primaryid"), value_set=selected).to_numpy()).tolist()
            )
            for name, dataset in tables.to_dict().items()
        }

        logger.info("[process|out]")
        return FaersTables.from_dict(filtered)


class FaersCaseBatcher(Processor):
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

        _require_columns(tables.demo, "demo", {"primaryid"})
        sorted_tables = {}
        for name, dataset in tables.to_dict().items():
            _require_columns(dataset, name, {"primaryid"})
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


class FaersCaseAssembler(Processor):
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
        _require_columns(tables.demo, "demo", {"primaryid"})
        _require_columns(tables.drug, "drug", {"primaryid", "drug_seq"})
        _require_columns(tables.indi, "indi", {"primaryid", "indi_drug_seq"})
        _require_columns(tables.ther, "ther", {"primaryid", "dsg_drug_seq"})

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

        reactions = _unique_values(case.get("reactions", []), "pt")
        if reactions:
            sentences.append(f"Reported reactions include {', '.join(map(str, reactions))}.")
        outcomes = [
            self._OUTCOME_LABELS.get(value, value) for value in _unique_values(case.get("outcomes", []), "outc_cod")
        ]
        if outcomes:
            sentences.append(f"Recorded outcomes include {', '.join(map(str, outcomes))}.")
        if not sentences:
            sentences.append("The report contains no narrative-ready demographic, drug, reaction, or outcome details.")
        return " ".join(sentences)


def _require_columns(dataset: Dataset, table: str, columns: set[str]) -> None:
    missing = columns.difference(dataset.column_names)
    if missing:
        raise ProcessorException(f"{table} is missing required columns: {', '.join(sorted(missing))}")


def _version_key(version: Any) -> tuple[int, int]:
    return (0, -1) if version is None else (1, int(version))


def _unique_values(rows: Sequence[dict[str, Any]], column: str) -> list[Any]:
    values = []
    seen = set()
    for row in rows:
        value = row.get(column)
        if value is not None and value not in seen:
            values.append(value)
            seen.add(value)
    return values
