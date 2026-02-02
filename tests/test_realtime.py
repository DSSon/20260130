from datetime import datetime

from news_impact.alerts import AlertSink
from news_impact.data_sources.base import NewsCollector, NewsItem
from news_impact.model import ImpactModel, train_model
from news_impact.realtime import monitor_news


class InMemoryNewsCollector(NewsCollector):
    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items

    def fetch(self):
        return self._items


class CaptureSink(AlertSink):
    def __init__(self) -> None:
        self.events = []

    def notify(self, event):
        self.events.append(event)


def test_monitor_triggers_alert() -> None:
    news_items = [
        NewsItem(
            published_at=datetime(2024, 1, 2, 9, 0),
            headline="Positive earnings",
            body="Strong revenue growth and margin expansion.",
            source="Test",
        ),
        NewsItem(
            published_at=datetime(2024, 1, 3, 9, 0),
            headline="Regulatory concerns",
            body="Potential fines weigh on outlook.",
            source="Test",
        ),
    ]
    labels = [1, 0]
    model, _ = train_model(news_items, labels)

    collector = InMemoryNewsCollector(news_items)
    sink = CaptureSink()

    alerts = monitor_news(collector, model, alert_threshold=0.5, sink=sink)

    assert alerts
    assert len(alerts) == len(sink.events)
