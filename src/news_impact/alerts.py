from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from news_impact.data_sources.base import NewsItem


@dataclass(frozen=True)
class AlertEvent:
    news: NewsItem
    score: float
    threshold: float
    triggered_at: datetime


class AlertSink:
    def notify(self, event: AlertEvent) -> None:
        raise NotImplementedError


class PrintAlertSink(AlertSink):
    def notify(self, event: AlertEvent) -> None:
        print(
            f"[ALERT] {event.news.headline} score={event.score:.3f} "
            f"threshold={event.threshold:.2f} at {event.triggered_at.isoformat()}"
        )
