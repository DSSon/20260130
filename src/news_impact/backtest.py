from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from news_impact.data_sources.base import NewsItem, PriceBar
from news_impact.model import ImpactModel, build_labels, train_model
from news_impact.reporting import BacktestReport, save_backtest_report


@dataclass
class BacktestResult:
    report: BacktestReport
    model: ImpactModel


def run_backtest(
    news_items: Sequence[NewsItem],
    price_bars: Sequence[PriceBar],
    report_dir: Path,
) -> BacktestResult:
    prices = _prices_to_frame(price_bars)
    news = _news_to_frame(news_items)

    merged = _align_news_prices(news, prices)
    merged = merged.dropna(subset=["next_return"]).reset_index(drop=True)

    aligned_news = _frame_to_news(merged, news_items)
    labels = build_labels(merged["next_return"])
    model, metrics = train_model(aligned_news, labels)

    impact_scores = model.predict_proba(aligned_news)
    strategy_returns = _compute_strategy_returns(merged, impact_scores)

    report = BacktestReport(
        metrics=metrics,
        impact_scores=impact_scores,
        strategy_returns=strategy_returns,
        dates=merged["date"].tolist(),
    )
    save_backtest_report(report, report_dir)
    return BacktestResult(report=report, model=model)


def _prices_to_frame(price_bars: Sequence[PriceBar]) -> pd.DataFrame:
    data = {
        "date": [bar.date.date() for bar in price_bars],
        "close": [bar.close for bar in price_bars],
    }
    frame = pd.DataFrame(data).sort_values("date")
    frame["next_close"] = frame["close"].shift(-1)
    frame["next_return"] = frame["next_close"] / frame["close"] - 1.0
    return frame


def _news_to_frame(news_items: Sequence[NewsItem]) -> pd.DataFrame:
    data = {
        "date": [item.published_at.date() for item in news_items],
        "headline": [item.headline for item in news_items],
        "body": [item.body for item in news_items],
        "source": [item.source for item in news_items],
    }
    return pd.DataFrame(data)


def _frame_to_news(merged: pd.DataFrame, original: Sequence[NewsItem]) -> list[NewsItem]:
    aligned_news: list[NewsItem] = []
    for row in merged.itertuples(index=False):
        aligned_news.append(
            NewsItem(
                published_at=pd.Timestamp(row.date).to_pydatetime(),
                headline=str(row.headline),
                body=str(row.body),
                source=str(row.source),
            )
        )
    return aligned_news


def _align_news_prices(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return news.merge(prices, on="date", how="left")


def _compute_strategy_returns(merged: pd.DataFrame, impact_scores: np.ndarray) -> np.ndarray:
    threshold = np.percentile(impact_scores, 75)
    signals = impact_scores >= threshold
    returns = merged["next_return"].fillna(0.0).to_numpy()
    return returns * signals
