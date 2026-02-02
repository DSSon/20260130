from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class NewsItem:
    news_id: str
    symbol: str
    text: str
    timestamp: datetime
    source: str
    language: str


@dataclass(frozen=True)
class PriceBar:
    timestamp: datetime
    close: float
    volume: float


@dataclass(frozen=True)
class FeatureConfig:
    tfidf_max_features: int = 500
    label_window: str = "1d"  # "1d" or "2h"
    volume_zscore_window: int = 5
    volume_zscore_min_periods: int = 3
    version: str = "v1"


@dataclass(frozen=True)
class PipelineOutput:
    features: List[dict]
    labels: List[dict]
    metadata: dict
