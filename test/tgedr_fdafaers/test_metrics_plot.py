"""Unit tests for the rows-per-period metrics plotting helper."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tgedr_fdafaers.utils.metrics_plot import load_rows_by_period, plot_rows_by_period


def _point(table: str, period: str, value: float, ts: int) -> dict:
    return {
        "attributes": {"table": table, "period": period},
        "time_unix_nano": ts,
        "value": value,
    }


def _document(metric_name: str, points: list[dict]) -> str:
    import json

    doc = {
        "resource_metrics": [
            {
                "scope_metrics": [
                    {
                        "metrics": [
                            {
                                "name": metric_name,
                                "data": {"data_points": points},
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return json.dumps(doc)


@pytest.fixture()
def metrics_file(tmp_path) -> Path:
    """Write a metrics file with two flushes, two tables, and out-of-order periods."""
    path = tmp_path / "metrics"
    # First flush: '13q1' appears before '12q4' (out of order); an unrelated metric is present.
    first = _document(
        "rows",
        [
            _point("reac", "13q1", 3, 1_000_000_000),
            _point("reac", "12q4", 2, 1_000_000_000),
            _point("drug", "13q1", 7, 1_000_000_000),
        ],
    )
    # An unrelated metric that must be ignored.
    other = _document("new_rows", [_point("reac", "12q4", 99, 1_000_000_000)])
    # Second flush: 'reac'/'12q4' is overwritten with a newer value (latest wins).
    second = _document("rows", [_point("reac", "12q4", 5, 2_000_000_000)])
    path.write_text(first + "\n   \n" + other + "\n" + second + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# load_rows_by_period
# --------------------------------------------------------------------------- #


def test_load_rows_by_period_returns_latest_value_per_table_period(metrics_file) -> None:
    """load_rows_by_period should keep the latest value per (table, period) and filter by metric name."""
    result = load_rows_by_period(metrics_file, metric_name="rows")

    assert result == {
        "reac": {"13q1": 3.0, "12q4": 5.0},
        "drug": {"13q1": 7.0},
    }


def test_load_rows_by_period_raises_when_metric_not_found(metrics_file) -> None:
    """load_rows_by_period should raise ValueError when the metric is absent."""
    with pytest.raises(ValueError, match="no matching metric"):
        load_rows_by_period(metrics_file, metric_name="does_not_exist")


def test_load_rows_by_period_raises_on_empty_file(tmp_path) -> None:
    """load_rows_by_period should raise ValueError when the file has no matching points."""
    path = tmp_path / "empty"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no matching metric"):
        load_rows_by_period(path, metric_name="rows")


# --------------------------------------------------------------------------- #
# plot_rows_by_period
# --------------------------------------------------------------------------- #


def test_plot_rows_by_period_writes_file_and_returns_path(metrics_file, tmp_path) -> None:
    """plot_rows_by_period should save the figure and return its path string."""
    save_path = tmp_path / "out" / "rows_by_period.png"

    result = plot_rows_by_period(metrics_file, metric_name="rows", save_path=save_path)

    assert result == str(save_path)
    assert Path(save_path).exists()


def test_plot_rows_by_period_returns_none_when_no_save_path(metrics_file) -> None:
    """plot_rows_by_period should return None and call plt.show when save_path is None."""
    with patch("tgedr_fdafaers.utils.metrics_plot.plt.show") as mock_show:
        result = plot_rows_by_period(metrics_file, metric_name="rows", save_path=None)

    mock_show.assert_called_once()
    assert result is None
