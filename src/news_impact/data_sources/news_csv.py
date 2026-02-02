from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from news_impact.data_sources.base import NewsCollector, NewsItem


class CsvNewsCollector(NewsCollector):
    def __init__(self, path: Path) -> None:
        self._path = path

    def fetch(self) -> Iterable[NewsItem]:
        data = pd.read_csv(self._path)
        required = {"published_at", "headline", "body", "source"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in news CSV: {sorted(missing)}")

        for row in data.itertuples(index=False):
            published_at = _parse_datetime(row.published_at)
            yield NewsItem(
                published_at=published_at,
                headline=str(row.headline),
                body=str(row.body),
                source=str(row.source),
            )


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)
