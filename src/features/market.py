from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, Iterable, List

from .types import NewsItem, PriceBar


@dataclass(frozen=True)
class MarketState:
    volatility: float
    trend: float
    avg_volume: float


def _bars_before(bars: Iterable[PriceBar], timestamp: datetime) -> List[PriceBar]:
    return [bar for bar in bars if bar.timestamp <= timestamp]


def market_state_features(
    items: List[NewsItem], price_map: Dict[str, List[PriceBar]], window: int = 5
) -> List[Dict[str, float]]:
    features: List[Dict[str, float]] = []
    for item in items:
        bars = price_map.get(item.symbol, [])
        history = _bars_before(bars, item.timestamp)[-window:]
        closes = [bar.close for bar in history]
        volumes = [bar.volume for bar in history]
        if len(closes) >= 2:
            returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
            volatility = pstdev(returns) if len(returns) > 1 else 0.0
            trend = closes[-1] / closes[0] - 1
        else:
            volatility = 0.0
            trend = 0.0
        avg_volume = mean(volumes) if volumes else 0.0
        features.append(
            {
                "recent_volatility": float(volatility),
                "recent_trend": float(trend),
                "recent_avg_volume": float(avg_volume),
            }
        )
    return features
