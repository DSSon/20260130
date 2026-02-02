from datetime import datetime
from pathlib import Path

from news_impact.backtest import run_backtest
from news_impact.data_sources.base import NewsItem, PriceBar


def test_backtest_generates_report(tmp_path: Path) -> None:
    news = [
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
        NewsItem(
            published_at=datetime(2024, 1, 4, 9, 0),
            headline="New partnership",
            body="Strategic alliance opens new market.",
            source="Test",
        ),
    ]
    prices = [
        PriceBar(date=datetime(2024, 1, 2), close=100),
        PriceBar(date=datetime(2024, 1, 3), close=102),
        PriceBar(date=datetime(2024, 1, 4), close=101),
        PriceBar(date=datetime(2024, 1, 5), close=103),
    ]

    result = run_backtest(news, prices, tmp_path)

    summary = tmp_path / "summary.txt"
    chart = tmp_path / "strategy_returns.png"
    assert summary.exists()
    assert chart.exists()
    assert result.report.metrics
