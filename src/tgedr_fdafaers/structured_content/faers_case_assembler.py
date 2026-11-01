"""Processors for selecting, batching, assembling, and rendering FAERS case data."""

from collections import defaultdict
import logging
from typing import Any
from datasets import Dataset
from tgedr_dataops_abs.processor import Processor, ProcessorException
from tgedr_fdafaers.faers_tables.faers_tables import FaersTables


logger = logging.getLogger(__name__)


class FaersCaseAssembler(Processor):
    """Assemble each PRIMARYID and its linked drug, indication, and therapy rows."""

    CONTEXT_KEY_TABLES = "tables"

    def process(self, context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return one nested structured representation per report PRIMARYID."""
        logger.info("[process|in]")

        if not context or self.CONTEXT_KEY_TABLES not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_TABLES} must be provided in context")
        tables: FaersTables = context[self.CONTEXT_KEY_TABLES]
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
                logger.warning(f"Duplicate PRIMARYID {primaryid} found in DEMO")
                # raise ProcessorException(f"DEMO contains multiple rows for PRIMARYID {primaryid}")  # noqa: ERA001
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

    def _require_columns(self, dataset: Dataset, table: str, columns: set[str]) -> None:
        missing = columns.difference(dataset.column_names)
        if missing:
            raise ProcessorException(f"{table} is missing required columns: {', '.join(sorted(missing))}")

    def _group_by_primaryid(self, dataset: Dataset) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset:
            primaryid = row.get("primaryid")
            if primaryid is not None:
                grouped[primaryid].append(row)
        return dict(grouped)
