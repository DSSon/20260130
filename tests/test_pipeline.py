from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features.pipeline import build_features_and_labels
from features.types import FeatureConfig, NewsItem, PriceBar


def test_end_to_end_pipeline():
    base_time = datetime(2024, 1, 2, 9, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for i in range(6):
        bars.append(
            PriceBar(
                timestamp=base_time + timedelta(hours=i),
                close=price,
                volume=1000 + i * 100,
            )
        )
        price *= 1.01

    items = [
        NewsItem(
            news_id="n1",
            symbol="AAA",
            text="Company AAA reports record profit and growth",
            timestamp=base_time + timedelta(hours=1),
            source="wire",
            language="ko",
        ),
        NewsItem(
            news_id="n2",
            symbol="AAA",
            text="AAA faces downgrade after weak results",
            timestamp=base_time + timedelta(hours=3),
            source="blog",
            language="en",
        ),
    ]

    config = FeatureConfig(
        label_window="2h", volume_zscore_window=4, volume_zscore_min_periods=2
    )
    output = build_features_and_labels(items, {"AAA": bars}, config)

    assert len(output.features) == len(items)
    assert len(output.labels) == len(items)

    first_label = output.labels[0]
    assert first_label["event_window_return"] is not None
    assert first_label["volume_spike_zscore"] is not None

    assert "tfidf_company" in output.features[0]
    assert "sentiment" in output.features[0]
    assert "recent_trend" in output.features[0]
