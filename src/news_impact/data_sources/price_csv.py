from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from news_impact.data_sources.base import PriceBar, PriceCollector


class CsvPriceCollector(PriceCollector):
    def __init__(self, path: Path) -> None:
        self._path = path

    def fetch(self) -> Iterable[PriceBar]:
        data = pd.read_csv(self._path)
        required = {"date", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns in prices CSV: {sorted(missing)}")

        for row in data.itertuples(index=False):
            yield PriceBar(date=_parse_date(row.date), close=float(row.close))


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)
