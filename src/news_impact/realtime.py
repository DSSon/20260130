from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from news_impact.alerts import AlertEvent, AlertSink
from news_impact.data_sources.base import NewsCollector, NewsItem
from news_impact.model import ImpactModel


def monitor_news(
    collector: NewsCollector,
    model: ImpactModel,
    alert_threshold: float,
    sink: AlertSink,
) -> list[AlertEvent]:
    alerts: list[AlertEvent] = []
    news_items = list(collector.fetch())
    if not news_items:
        return alerts

    scores = model.predict_proba(news_items)
    for item, score in zip(news_items, scores):
        if score >= alert_threshold:
            event = AlertEvent(
                news=item,
                score=float(score),
                threshold=alert_threshold,
                triggered_at=datetime.utcnow(),
            )
            sink.notify(event)
            alerts.append(event)
    return alerts


def score_news(model: ImpactModel, news_items: Sequence[NewsItem]) -> list[float]:
    return [float(score) for score in model.predict_proba(news_items)]
