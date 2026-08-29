"""Plot the rows-per-(table, period) gauge metric exported to an OpenTelemetry metrics file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend, safe for headless/CI usage
import matplotlib.pyplot as plt

from tgedr_fdafaers.utils.faers_period import FaersPeriod

if TYPE_CHECKING:
    from collections.abc import Iterator


def _iter_json_documents(text: str) -> Iterator[dict]:
    """Yield successive JSON documents from concatenated exporter output."""
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        document, offset = decoder.raw_decode(text, index)
        yield document
        index = offset


def load_rows_by_period(
    path: str | Path,
    metric_name: str = "rows",
) -> dict[str, dict[str, float]]:
    """Extract, per table, the latest row count for each period from a metrics file.

    Args:
        path: path to the metrics export file (one or more JSON documents).
        metric_name: metric to read; defaults to ``"rows"``.

    Returns:
        A dict mapping each table to a dict of ``{period: rows}``, where the
        value is the latest one recorded for that ``(table, period)`` pair
        (gauges are set on every run, so a pair may appear in several flushes).

    Raises:
        ValueError: when no matching metric is found.
    """
    text = Path(path).read_text(encoding="utf-8")

    # (table, period) -> (time_unix_nano, value)
    latest: dict[tuple[str, str], tuple[int, float]] = {}

    for document in _iter_json_documents(text):
        for resource_metric in document.get("resource_metrics", []):
            for scope_metric in resource_metric.get("scope_metrics", []):
                for metric in scope_metric.get("metrics", []):
                    if metric["name"] != metric_name:
                        continue
                    for point in metric.get("data", {}).get("data_points", []):
                        attributes = point.get("attributes", {}) or {}
                        table = str(attributes.get("table", "?"))
                        period = str(attributes.get("period", "?"))
                        value = float(point["value"])
                        ts = int(point["time_unix_nano"])
                        prev = latest.get((table, period))
                        if prev is None or ts >= prev[0]:
                            latest[(table, period)] = (ts, value)

    if not latest:
        msg = f"no matching metric '{metric_name}' found in {path}"
        raise ValueError(msg)

    result: dict[str, dict[str, float]] = {}
    for (table, period), (_ts, value) in latest.items():
        result.setdefault(table, {})[period] = value

    return result


def _sort_periods(periods: list[str]) -> list[str]:
    """Sort FAERS period strings chronologically (e.g. ``['13q2', '12q4']`` -> ``['12q4', '13q2']``)."""

    def _key(period: str) -> FaersPeriod:
        return FaersPeriod.from_str(period)

    return sorted(periods, key=_key)


def plot_rows_by_period(
    path: str | Path,
    metric_name: str = "rows",
    save_path: str | Path | None = "./plots/rows_by_period.png",
) -> str | None:
    """Read a metrics export file and plot rows per period, one line per table.

    Args:
        path: path to the metrics export file.
        metric_name: metric to plot; defaults to ``"rows"``.
        save_path: when set, the figure is written here and the path returned;
            otherwise the figure is shown interactively and ``None`` is returned.

    Returns:
        The saved file path (as a string) when ``save_path`` is provided, else
        ``None``.
    """
    series = load_rows_by_period(path, metric_name=metric_name)

    all_periods = _sort_periods({period for counts in series.values() for period in counts})

    fig, ax = plt.subplots(figsize=(9, 5))
    for table in sorted(series):
        counts = series[table]
        xs = all_periods
        ys = [counts.get(period) for period in all_periods]
        ax.plot(xs, ys, marker="o", linewidth=2, label=table)

    ax.set_title(f"{metric_name} per period")
    ax.set_xlabel("period")
    ax.set_ylabel(metric_name)
    ax.legend(title="table")
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return str(save_path)

    plt.show()
    plt.close(fig)
    return None
