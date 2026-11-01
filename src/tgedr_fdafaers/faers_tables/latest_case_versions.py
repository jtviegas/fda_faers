"""Select the latest unambiguous report version for each FAERS case."""

import logging
from typing import Any
from tgedr_dataops_abs.processor import Processor, ProcessorException
from tgedr_dataops.store.hf_dataset_file_arrow_based import Dataset

logger = logging.getLogger(__name__)


class LatestCaseVersions(Processor):
    """Keep only the latest unambiguous report version for each CASEID."""

    CONTEXT_KEY_DEMO_DATASET = "demo_dataset"

    def _version_key(self, version: Any) -> tuple[int, int]:
        return (0, -1) if version is None else (1, int(version))

    def process(self, context: dict[str, Any] | None = None) -> set[int]:
        """Filter every table to the selected latest PRIMARYIDs."""
        logger.info("[process|in]")
        if not context or self.CONTEXT_KEY_DEMO_DATASET not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_DEMO_DATASET} must be provided in context")
        ds: Dataset = (context[self.CONTEXT_KEY_DEMO_DATASET]).select_columns(["primaryid", "caseid", "caseversion"])

        latest_by_caseid: dict[Any, dict[str, Any]] = {}
        selected_primaryids: set[int] = set()

        for batch in ds.iter(batch_size=10_000):
            for row in zip(batch["primaryid"], batch["caseid"], batch["caseversion"], strict=True):
                primaryid, caseid, caseversion = row
                if primaryid is None:
                    continue  # drop this one
                if caseid is None:
                    selected_primaryids.add(primaryid)
                    continue

                previous = latest_by_caseid.get(
                    caseid
                )  # we assume we are fed a sorted stream by caseid and caseversion
                if previous is None or self._version_key(caseversion) > self._version_key(previous["caseversion"]):
                    latest_by_caseid[caseid] = {"primaryid": primaryid, "caseid": caseid, "caseversion": caseversion}
                elif self._version_key(caseversion) == self._version_key(previous["caseversion"]):
                    logger.warning(
                        f"Ambiguous latest case version for CASEID {caseid}: PRIMARYID {primaryid} has the same CASEVERSION as previous"
                    )

        selected_primaryids.update(row["primaryid"] for row in latest_by_caseid.values())
        logger.info(f"[process|out] PRIMARYIDs number: {len(selected_primaryids)}")
        return selected_primaryids
