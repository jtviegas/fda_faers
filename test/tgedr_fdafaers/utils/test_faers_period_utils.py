"""Unit tests for FAERS period primitives and URL resolution helpers."""

from unittest.mock import patch
from datetime import datetime, timezone

from tgedr_fdafaers.utils.faers_period import FaersPeriod, UtilsFaersPeriod


# --------------------------------------------------------------------------- #
# FaersPeriod comparison operators
# --------------------------------------------------------------------------- #


def test_faers_period_str_and_from_str_roundtrip() -> None:
    """String conversion should preserve year and quarter values."""
    period = FaersPeriod(2024, 1)

    parsed = FaersPeriod.from_str(str(period))

    assert parsed == period


def test_faers_period_hash() -> None:
    """Hashing should allow FaersPeriod to be used in sets and dict keys."""
    p1 = FaersPeriod(2024, 1)
    p2 = FaersPeriod(2024, 1)
    p3 = FaersPeriod(2024, 2)

    assert hash(p1) == hash(p2)
    assert {p1, p2, p3} == {p1, p3}


def test_faers_period_lt_same_year() -> None:
    """< should compare quarters within the same year."""
    assert FaersPeriod(2024, 1) < FaersPeriod(2024, 2)
    assert not FaersPeriod(2024, 2) < FaersPeriod(2024, 1)
    assert not FaersPeriod(2024, 2) < FaersPeriod(2024, 2)


def test_faers_period_lt_different_year() -> None:
    """< should compare across years."""
    assert FaersPeriod(2023, 4) < FaersPeriod(2024, 1)
    assert not FaersPeriod(2024, 1) < FaersPeriod(2023, 4)


def test_faers_period_le() -> None:
    """<= should return True for equal or lesser periods."""
    assert FaersPeriod(2024, 1) <= FaersPeriod(2024, 1)
    assert FaersPeriod(2024, 1) <= FaersPeriod(2024, 2)
    assert FaersPeriod(2023, 4) <= FaersPeriod(2024, 1)
    assert not FaersPeriod(2024, 2) <= FaersPeriod(2024, 1)


def test_faers_period_gt_same_year() -> None:
    """> should compare quarters within the same year."""
    assert FaersPeriod(2024, 2) > FaersPeriod(2024, 1)
    assert not FaersPeriod(2024, 1) > FaersPeriod(2024, 2)
    assert not FaersPeriod(2024, 2) > FaersPeriod(2024, 2)


def test_faers_period_gt_different_year() -> None:
    """> should compare across years."""
    assert FaersPeriod(2024, 1) > FaersPeriod(2023, 4)
    assert not FaersPeriod(2023, 4) > FaersPeriod(2024, 1)


def test_faers_period_ge() -> None:
    """>= should return True for equal or greater periods."""
    assert FaersPeriod(2024, 1) >= FaersPeriod(2024, 1)
    assert FaersPeriod(2024, 2) >= FaersPeriod(2024, 1)
    assert FaersPeriod(2024, 1) >= FaersPeriod(2023, 4)
    assert not FaersPeriod(2024, 1) >= FaersPeriod(2024, 2)


def test_faers_period_eq() -> None:
    """== should be True only when year and quarter match."""
    assert FaersPeriod(2024, 1) == FaersPeriod(2024, 1)
    assert not FaersPeriod(2024, 1) == FaersPeriod(2024, 2)
    assert not FaersPeriod(2024, 1) == FaersPeriod(2023, 1)


def test_faers_period_ne() -> None:
    """!= should be True when year or quarter differs."""
    assert FaersPeriod(2024, 1) != FaersPeriod(2024, 2)
    assert FaersPeriod(2024, 1) != FaersPeriod(2023, 1)
    assert not FaersPeriod(2024, 1) != FaersPeriod(2024, 1)


def test_faers_period_str() -> None:
    """__str__ should produce short form like '24q1'."""
    assert str(FaersPeriod(2024, 1)) == "24q1"
    assert str(FaersPeriod(2010, 3)) == "10q3"


# --------------------------------------------------------------------------- #
# UtilsFaersPeriod
# --------------------------------------------------------------------------- #


def test_next_and_previous_period_handle_year_boundaries() -> None:
    """Period navigation should roll quarter boundaries correctly."""
    assert UtilsFaersPeriod.next_period(FaersPeriod(2024, 4)) == FaersPeriod(2025, 1)
    assert UtilsFaersPeriod.previous_period(FaersPeriod(2024, 1)) == FaersPeriod(2023, 4)


def test_next_period_within_year() -> None:
    """next_period should increment quarter within a year."""
    assert UtilsFaersPeriod.next_period(FaersPeriod(2024, 1)) == FaersPeriod(2024, 2)
    assert UtilsFaersPeriod.next_period(FaersPeriod(2024, 2)) == FaersPeriod(2024, 3)
    assert UtilsFaersPeriod.next_period(FaersPeriod(2024, 3)) == FaersPeriod(2024, 4)


def test_previous_period_within_year() -> None:
    """previous_period should decrement quarter within a year."""
    assert UtilsFaersPeriod.previous_period(FaersPeriod(2024, 4)) == FaersPeriod(2024, 3)
    assert UtilsFaersPeriod.previous_period(FaersPeriod(2024, 3)) == FaersPeriod(2024, 2)
    assert UtilsFaersPeriod.previous_period(FaersPeriod(2024, 2)) == FaersPeriod(2024, 1)


def test_is_faers_period_respects_cutover_boundary() -> None:
    """2012q3 is AERS; periods after it are FAERS."""
    assert not UtilsFaersPeriod.is_faers_period(FaersPeriod(2012, 3))
    assert UtilsFaersPeriod.is_faers_period(FaersPeriod(2012, 4))


def test_get_url_uses_expected_prefix_and_quarter_case() -> None:
    """URL format should follow AERS vs FAERS conventions."""
    aers_url = UtilsFaersPeriod.get_url(FaersPeriod(2010, 2))
    faers_url = UtilsFaersPeriod.get_url(FaersPeriod(2024, 2))

    assert aers_url.endswith("aers_ascii_2010q2.zip")
    assert faers_url.endswith("faers_ascii_2024Q2.zip")


def test_resolve_period_parses_filenames_and_handles_invalid() -> None:
    """resolve_period should parse both q and Q, and return None when absent."""
    assert UtilsFaersPeriod.resolve_period("aers_ascii_2010q2.zip") == FaersPeriod(2010, 2)
    assert UtilsFaersPeriod.resolve_period("faers_ascii_2024Q3.zip") == FaersPeriod(2024, 3)
    assert UtilsFaersPeriod.resolve_period("no_period_here.zip") is None


def test_get_current_period_returns_expected_for_known_date() -> None:
    """get_current_period should return the correct period for a mocked date."""
    # Mock datetime.now to return a known date: 2024-07-15 (Q3 -> current = Q2 since formula subtracts 1)
    mock_dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
    with patch("tgedr_fdafaers.utils.faers_period.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        result = UtilsFaersPeriod.get_current_period()

    # month 7 -> (7-1)/3 + 1 = 3, then -1 = 2
    assert result == FaersPeriod(2024, 2)


def test_get_faers_periods_returns_periods_after_aers_end() -> None:
    """get_faers_periods should return all periods from AERS_END+1 up to (not including) current."""
    # Mock current period to a known value
    with patch.object(UtilsFaersPeriod, "get_current_period", return_value=FaersPeriod(2013, 3)):
        result = UtilsFaersPeriod.get_faers_periods()

    # AERS_END_PERIOD = 2012q3, so FAERS starts at 2012q4
    expected = [FaersPeriod(2012, 4), FaersPeriod(2013, 1), FaersPeriod(2013, 2)]
    assert result == expected


def test_get_aers_periods_returns_fixed_range() -> None:
    """get_aers_periods should return all periods from 2004q1 to 2012q3 inclusive."""
    result = UtilsFaersPeriod.get_aers_periods()

    assert result[0] == FaersPeriod(2004, 1)
    assert result[-1] == FaersPeriod(2012, 3)
    # 2004q1 to 2012q3 = 8 full years (32 quarters) + 3 quarters = 35
    assert len(result) == 35


def test_get_all_faers_periods_includes_both_eras() -> None:
    """get_all_faers_periods should include both FAERS and AERS periods."""
    with patch.object(UtilsFaersPeriod, "get_current_period", return_value=FaersPeriod(2013, 2)):
        result = UtilsFaersPeriod.get_all_faers_periods()

    # FAERS: 2012q4, 2013q1 (2 periods) + AERS: 2004q1..2012q3 (35 periods) = 37
    assert len(result) == 37
    # FAERS periods come first in the list
    assert result[0] == FaersPeriod(2012, 4)
    # AERS periods come after
    assert result[-1] == FaersPeriod(2012, 3)
