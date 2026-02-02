from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

from .types import NewsItem, PriceBar


def _window_delta(window: str) -> timedelta:
    if window == "2h":
        return timedelta(hours=2)
    return timedelta(days=1)


def _next_bar(bars: Iterable[PriceBar], timestamp: datetime) -> Optional[PriceBar]:
    for bar in bars:
        if bar.timestamp >= timestamp:
            return bar
    return None


def event_window_return(
    items: List[NewsItem],
    price_map: Dict[str, List[PriceBar]],
    window: str = "1d",
) -> List[Optional[float]]:
    delta = _window_delta(window)
    returns: List[Optional[float]] = []
    for item in items:
        bars = price_map.get(item.symbol, [])
        start_bar = _next_bar(bars, item.timestamp)
        end_bar = _next_bar(bars, item.timestamp + delta)
        if start_bar is None or end_bar is None:
            returns.append(None)
        else:
            returns.append(end_bar.close / start_bar.close - 1)
    return returns


def volume_spike_zscore(
    items: List[NewsItem],
    price_map: Dict[str, List[PriceBar]],
    window: int = 5,
    min_periods: int = 3,
) -> List[Optional[float]]:
    scores: List[Optional[float]] = []
    for item in items:
        bars = price_map.get(item.symbol, [])
        history = [bar for bar in bars if bar.timestamp <= item.timestamp][-window:]
        volumes = [bar.volume for bar in history]
        if len(volumes) < min_periods:
            scores.append(None)
            continue
        avg = mean(volumes)
        std = pstdev(volumes) if len(volumes) > 1 else 0.0
        if std == 0.0:
            scores.append(0.0)
            continue
        scores.append((volumes[-1] - avg) / std)
    return scores
