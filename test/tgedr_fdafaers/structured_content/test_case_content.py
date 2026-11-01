"""Tests for FAERS case selection, batching, assembly, and narrative rendering."""

# ruff: noqa: SLF001

from dataclasses import replace

import pytest
from datasets import Dataset, Features, Value

from tgedr_fdafaers.faers_tables.faers_tables import FaersTables
from tgedr_fdafaers.faers_tables.latest_case_versions import LatestCaseVersions
from tgedr_fdafaers.structured_content.faers_case_assembler import FaersCaseAssembler
from tgedr_fdafaers.structured_content.faers_narrative_renderer import FaersNarrativeRenderer
from tgedr_dataops_abs.processor import ProcessorException


def _tables() -> FaersTables:
    return FaersTables(
        demo=Dataset.from_list(
            [
                {"primaryid": 21, "caseid": 2, "caseversion": 1, "age": 44, "age_cod": "YR", "sex": "F"},
                {"primaryid": 10, "caseid": 1, "caseversion": 1, "age": 60, "age_cod": "YR", "sex": "M"},
                {"primaryid": 22, "caseid": 2, "caseversion": 2, "age": 44, "age_cod": "YR", "sex": "F"},
            ]
        ),
        drug=Dataset.from_list(
            [
                {"primaryid": 22, "drug_seq": 1, "drugname": "Drug A", "role_cod": "PS"},
                {"primaryid": 22, "drug_seq": 2, "drugname": "Drug B", "role_cod": "C"},
                {"primaryid": 10, "drug_seq": 1, "drugname": "Drug C", "role_cod": "SS"},
                {"primaryid": 21, "drug_seq": 1, "drugname": "Old Drug", "role_cod": "PS"},
            ]
        ),
        reac=Dataset.from_list(
            [
                {"primaryid": 22, "pt": "HEADACHE"},
                {"primaryid": 22, "pt": "NAUSEA"},
                {"primaryid": 10, "pt": "DIZZINESS"},
                {"primaryid": 21, "pt": "OLD REACTION"},
            ]
        ),
        outc=Dataset.from_list([{"primaryid": 22, "outc_cod": "HO"}]),
        rpsr=Dataset.from_list([{"primaryid": 22, "rpsr_cod": "MD"}]),
        ther=Dataset.from_list(
            [
                {"primaryid": 22, "dsg_drug_seq": 1, "start_dt": 20240101},
                {"primaryid": 22, "dsg_drug_seq": 2, "start_dt": 20240201},
            ]
        ),
        indi=Dataset.from_list(
            [
                {"primaryid": 22, "indi_drug_seq": 1, "indi_pt": "HYPERTENSION"},
                {"primaryid": 22, "indi_drug_seq": 2, "indi_pt": "PAIN"},
            ]
        ),
    )


def test_processors_reject_non_faers_tables_context() -> None:
    """Reject a context value that is not a FaersTables bundle."""
    with pytest.raises(ProcessorException, match="demo_dataset must be provided"):
        LatestCaseVersions().process({"tables": {}})
    with pytest.raises(ProcessorException, match="FaersTables instance"):
        FaersCaseAssembler().process({"tables": {}})


def test_latest_case_versions_selects_highest_primaryid_per_case() -> None:
    """Return the PRIMARYID with the highest case version for each CASEID."""
    result = LatestCaseVersions().process({"demo_dataset": _tables().demo})

    assert result == {10, 22}


def test_latest_case_versions_skips_null_primaryids_and_keeps_caseidless_reports() -> None:
    """Ignore DEMO rows without PRIMARYID and include reports lacking CASEID."""
    tables = _tables()
    demo = Dataset.from_list(
        [
            {"primaryid": None, "caseid": 3, "caseversion": 2},
            {"primaryid": 30, "caseid": None, "caseversion": 1},
            {"primaryid": 31, "caseid": 4, "caseversion": None},
            {"primaryid": 32, "caseid": 4, "caseversion": 2},
        ],
        features=Features(
            {
                "primaryid": Value("int64"),
                "caseid": Value("int64"),
                "caseversion": Value("int64"),
            }
        ),
    )

    result = LatestCaseVersions().process({"demo_dataset": demo})

    assert result == {30, 32}


def test_latest_case_versions_warns_on_tied_versions(caplog: pytest.LogCaptureFixture) -> None:
    """Warn about duplicate reports tied at the latest CASEVERSION."""
    demo = Dataset.from_list(
        [
            {"primaryid": 1, "caseid": 9, "caseversion": 3},
            {"primaryid": 2, "caseid": 9, "caseversion": 3},
        ]
    )

    result = LatestCaseVersions().process({"demo_dataset": demo})

    assert result == {1}
    assert "Ambiguous latest case version for CASEID 9" in caplog.text


def test_assembler_rejects_duplicate_demo_primaryid() -> None:
    """Reject a DEMO table that has duplicate rows for one report ID."""
    tables = _tables()
    duplicate_demo = Dataset.from_list(
        [
            {"primaryid": 7, "caseid": 70},
            {"primaryid": 7, "caseid": 70},
        ]
    )

    result = FaersCaseAssembler().process({"tables": replace(tables, demo=duplicate_demo)})

    assert [case["primaryid"] for case in result] == [7, 7]


def test_assembler_skips_null_primaryids_and_groups_non_null_rows() -> None:
    """Ignore rows with null IDs when grouping and assembling reports."""
    tables = _tables()
    altered = replace(
        tables,
        demo=Dataset.from_list([{"primaryid": None}, {"primaryid": 22, "caseid": 2, "caseversion": 2}]),
        reac=Dataset.from_list([{"primaryid": None, "pt": "ORPHAN"}, {"primaryid": 22, "pt": "NAUSEA"}]),
    )

    cases = FaersCaseAssembler().process({"tables": altered})

    assert len(cases) == 1
    assert cases[0]["primaryid"] == 22
    assert cases[0]["reactions"] == [{"primaryid": 22, "pt": "NAUSEA"}]


def test_assembler_rejects_required_column_gaps() -> None:
    """Validate primary and sequence columns used to relate table records."""
    tables = _tables()
    with pytest.raises(ProcessorException, match="demo is missing required columns: primaryid"):
        FaersCaseAssembler().process({"tables": replace(tables, demo=Dataset.from_list([{"caseid": 4}]))})
    with pytest.raises(ProcessorException, match="drug is missing required columns: drug_seq"):
        FaersCaseAssembler().process({"tables": replace(tables, drug=Dataset.from_list([{"primaryid": 1}]))})
    with pytest.raises(ProcessorException, match="indi is missing required columns: indi_drug_seq"):
        FaersCaseAssembler().process({"tables": replace(tables, indi=Dataset.from_list([{"primaryid": 1}]))})
    with pytest.raises(ProcessorException, match="ther is missing required columns: dsg_drug_seq"):
        FaersCaseAssembler().process({"tables": replace(tables, ther=Dataset.from_list([{"primaryid": 1}]))})


def test_assembler_links_indications_and_therapy_by_drug_sequence() -> None:
    """Attach INDI and THER rows to their matching drug sequence."""
    cases = FaersCaseAssembler().process({"tables": _tables()})
    case = next(row for row in cases if row["primaryid"] == 22)

    assert [entry["drug"]["drugname"] for entry in case["drugs"]] == ["Drug A", "Drug B"]
    assert case["drugs"][0]["indications"][0]["indi_pt"] == "HYPERTENSION"
    assert case["drugs"][1]["therapies"][0]["start_dt"] == 20240201
    assert [row["pt"] for row in case["reactions"]] == ["HEADACHE", "NAUSEA"]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ({"demo": {}, "drugs": [], "reactions": [], "outcomes": []}, "no narrative-ready"),
        (
            {
                "demo": {"age": 30, "sex": "U"},
                "drugs": [
                    {"drug": {"prod_ai": "Ingredient", "role_cod": "PS"}, "indications": [{"indi_pt": "PAIN"}]},
                    {"drug": {"drugname": "Unknown", "role_cod": "ZZ"}, "indications": []},
                    {"drug": {}, "indications": []},
                ],
                "reactions": [{"pt": "RASH"}, {"pt": "RASH"}, {"pt": None}],
                "outcomes": [
                    {"outc_cod": "DE"}, {"outc_cod": "DE"}, {"outc_cod": "XX"}, {"outc_cod": None}
                ],
            },
            "30 age units-old patient. Reported drugs include Ingredient (primary suspect) for PAIN, "
            "Unknown (ZZ). Reported reactions include RASH. Recorded outcomes include death, XX.",
        ),
        (
            {
                "demo": {"age": 0, "age_cod": "YR", "sex": "X"},
                "drugs": [{"drug": {"drugname": "Drug", "role_cod": "SS"}, "indications": []}],
                "reactions": [],
                "outcomes": [],
            },
            "The report describes a 0 YR-old X patient. Reported drugs include Drug (secondary suspect).",
        ),
    ],
)
def test_renderer_handles_optional_fields_duplicates_and_unknown_codes(
    case: dict, expected: str
) -> None:
    """Render optional demographics, drug fallbacks, duplicate terms, and unknown codes."""
    narrative = FaersNarrativeRenderer()._render_case(case)

    assert expected in narrative


def test_renderer_rejects_invalid_case_collection() -> None:
    """Reject strings and mappings instead of a sequence of case dictionaries."""
    for cases in ("not cases", {"primaryid": 1}):
        with pytest.raises(ProcessorException, match="sequence of case dictionaries"):
            FaersNarrativeRenderer().process({"cases": cases})


def test_renderer_uses_reported_language_and_keeps_case_fields() -> None:
    """Preserve case fields and avoid asserting causal language."""
    case = FaersCaseAssembler().process({"tables": _tables()})[-1]

    result = FaersNarrativeRenderer().process({"cases": [case]})[0]

    assert result["primaryid"] == 22
    assert "44 YR-old female patient" in result["narrative"]
    assert "Drug A (primary suspect) for HYPERTENSION" in result["narrative"]
    assert "Reported reactions include HEADACHE, NAUSEA." in result["narrative"]
    assert "Recorded outcomes include hospitalization." in result["narrative"]


def test_processors_validate_required_context() -> None:
    """Reject missing inputs for each processor."""
    with pytest.raises(ProcessorException):
        LatestCaseVersions().process(None)
    with pytest.raises(ValueError, match="Column name .* not in the dataset"):
        LatestCaseVersions().process({"demo_dataset": Dataset.from_list([{"primaryid": 1}])})
    with pytest.raises(ProcessorException):
        FaersCaseAssembler().process(None)
    with pytest.raises(ProcessorException):
        FaersNarrativeRenderer().process(None)
    with pytest.raises(ProcessorException):
        FaersNarrativeRenderer().process({"cases": None})
