import sys
from datetime import datetime, timezone

sys.path.insert(0, "./src")

from market.schema import PriceRecord, apply_missing_rules
from news.schema import NewsRecord, dedupe_news, extract_tickers, map_tickers


def test_news_schema_and_deduplication():
    base = {
        "id": "1",
        "published_at": "2024-01-02T03:04:05+09:00",
        "source": "Example",
        "title": "Apple launches new product",
        "summary": "Apple (AAPL) announced a new device.",
        "tickers": ["AAPL"],
        "url": "https://example.com/news/1",
        "language": "en",
    }
    record = NewsRecord.from_dict(base)
    assert record.published_at.tzinfo == timezone.utc

    dupe = NewsRecord.from_dict({**base, "id": "2", "title": "Apple launches new product!"})
    deduped = dedupe_news([record, dupe])
    assert len(deduped) == 1


def test_ticker_mapping_rules():
    record = NewsRecord.from_dict(
        {
            "id": "3",
            "published_at": datetime(2024, 1, 5, tzinfo=timezone.utc),
            "source": "Example",
            "title": "Samsung partners with Apple",
            "body": "Samsung Electronics and Apple plan to collaborate.",
            "url": "https://example.com/news/2",
            "language": "en",
        }
    )
    symbol_map = {"AAPL": ["Apple"], "005930.KS": ["Samsung", "Samsung Electronics"]}
    mapped = map_tickers(record, symbol_map)
    assert mapped.tickers == ("005930.KS", "AAPL")
    assert extract_tickers("TSLA shares rise") == ["TSLA"]


def test_price_schema_and_missing_rules():
    record = PriceRecord.from_dict(
        {
            "ticker": "aapl",
            "timestamp": "2024-01-02T00:00:00-05:00",
            "open": 100,
            "high": 110,
            "low": 99,
            "close": 105,
            "volume": 10,
            "adjusted_close": 104.5,
        }
    )
    assert record.timestamp.tzinfo == timezone.utc

    missing = PriceRecord.from_dict(
        {
            "ticker": "AAPL",
            "timestamp": "2024-01-03T00:00:00Z",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        }
    )
    filled = apply_missing_rules([record, missing])
    assert filled[1].close == record.close
