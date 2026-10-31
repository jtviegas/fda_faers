

"""Select the latest unambiguous report version for each FAERS case."""

import logging
from typing import Any
import pyarrow as pa
from tgedr_dataops_abs.processor import ProcessorException
from tgedr_fdafaers.faers_processor import FaersProcessor
from tgedr_fdafaers.faers_tables import FaersTables
from tgedr_dataops.store.hf_dataset_file_arrow_based import HuggingFaceDatasetFileArrowBasedStore, IterableDataset

logger = logging.getLogger(__name__)


class LatestCaseVersions(FaersProcessor):
    """Keep only the latest unambiguous report version for each CASEID."""

    CONTEXT_KEY_DEMO_TABLE_URL = "demo_table_url"

    def process(self, context: dict[str, Any] | None = None) -> FaersTables:
        """Filter every table to the selected latest PRIMARYIDs."""
        logger.info("[process|in]")
        if not context or self.CONTEXT_KEY_DEMO_TABLE_URL not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_DEMO_TABLE_URL} must be provided in context")
        table_url: str = context[self.CONTEXT_KEY_DEMO_TABLE_URL]


        store = HuggingFaceDatasetFileArrowBasedStore()
        ds: IterableDataset = store.get(table_url, columns=["primaryid", "caseid", "caseversion"], streaming=True).train # type: ignore  # noqa: PGH003

        latest_by_case: dict[Any, dict[str, Any]] = {}
        ambiguous_latest_cases: set[Any] = set()
        selected_primaryids: set[int] = set()
        for row in ds:
            primaryid = row["primaryid"]
            caseid = row["caseid"]
            version = row["caseversion"]
            if primaryid is None:
                continue
            if caseid is None:
                selected_primaryids.add(primaryid)
                continue

            previous = latest_by_case.get(caseid)
            if previous is None or self._version_key(version) > self._version_key(previous["caseversion"]):
                latest_by_case[caseid] = row
            elif self._version_key(version) == self._version_key(previous["caseversion"]):
                ambiguous_latest_cases.add(caseid)

        if ambiguous_latest_cases:
            caseid = next(iter(ambiguous_latest_cases))
            primaryids = [row["primaryid"] for row in ds if row["caseid"] == caseid]
            raise ProcessorException(
                f"Ambiguous latest report version for CASEID {caseid}: "
                f"PRIMARYIDs {primaryids} have the same CASEVERSION"
            )

        selected_primaryids.update(row["primaryid"] for row in latest_by_case.values())
        # filtered = {
        #     name: dataset.filter(lambda row: row["primaryid"] in selected_primaryids)
        #     for name, dataset in tables.to_dict().items()
        # }

        result = pa.array(sorted(selected_primaryids))
        # filtered = {
        #     name: dataset.select(
        #         np.flatnonzero(pc.is_in(dataset.data.column("primaryid"), value_set=selected).to_numpy()).tolist()
        #     )
        #     for name, dataset in tables.to_dict().items()
        # }

        logger.info("[process|out]")
        return result




def main() -> None:

    o = LatestCaseVersions()
    o.process(context={"demo_table_url": "jtviegas/faers_bronze_demo"})


if __name__ == "__main__":
    main()

# uv run -m cProfile -s cumulative src/tgedr_fdafaers/latest_case_versions.py
