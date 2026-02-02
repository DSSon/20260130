from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BacktestReport:
    metrics: dict[str, float]
    impact_scores: np.ndarray
    strategy_returns: np.ndarray
    dates: list


def save_backtest_report(report: BacktestReport, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "summary.txt"
    plot_path = report_dir / "strategy_returns.png"

    mean_score = _safe_mean(report.impact_scores)
    total_return = _safe_sum(report.strategy_returns)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Backtest Summary\n")
        handle.write("================\n")
        for key, value in report.metrics.items():
            handle.write(f"{key}: {value:.4f}\n")
        handle.write(f"Average impact score: {mean_score:.4f}\n")
        handle.write(f"Strategy total return: {total_return:.4f}\n")

    _plot_returns(report.strategy_returns, plot_path)


def _plot_returns(strategy_returns: np.ndarray, plot_path: Path) -> None:
    cumulative = np.cumsum(strategy_returns)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumulative, label="Strategy")
    ax.set_title("Cumulative Strategy Returns")
    ax.set_xlabel("Event")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _safe_sum(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nansum(values))
