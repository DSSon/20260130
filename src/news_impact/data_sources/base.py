from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class NewsItem:
    published_at: datetime
    headline: str
    body: str
    source: str


@dataclass(frozen=True)
class PriceBar:
    date: datetime
    close: float


class NewsCollector(ABC):
    @abstractmethod
    def fetch(self) -> Iterable[NewsItem]:
        raise NotImplementedError


class PriceCollector(ABC):
    @abstractmethod
    def fetch(self) -> Iterable[PriceBar]:
        raise NotImplementedError
