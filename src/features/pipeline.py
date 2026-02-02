from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

from .cache import cache_key
from .labels import event_window_return, volume_spike_zscore
from .market import market_state_features
from .meta import build_meta_vocab, meta_features
from .text import EmbeddingProvider, embedding_features, tfidf_features
from .types import FeatureConfig, NewsItem, PipelineOutput, PriceBar


def _merge_feature_dicts(rows: List[List[dict]]) -> List[dict]:
    merged: List[dict] = []
    for row_parts in zip(*rows):
        row: dict = {}
        for part in row_parts:
            row.update(part)
        merged.append(row)
    return merged


def build_features_and_labels(
    items: List[NewsItem],
    price_map: Dict[str, List[PriceBar]],
    config: FeatureConfig,
    embedder: Optional[EmbeddingProvider] = None,
) -> PipelineOutput:
    vocab = build_meta_vocab(items)
    texts = [item.text for item in items]
    tfidf_rows = tfidf_features(texts, max_features=config.tfidf_max_features)
    embedding_rows = embedding_features(texts, embedder=embedder)
    meta_rows = meta_features(items, vocab)
    market_rows = market_state_features(items, price_map)

    features = _merge_feature_dicts([tfidf_rows, embedding_rows, meta_rows, market_rows])

    labels = [
        {
            "event_window_return": ret,
            "volume_spike_zscore": spike,
        }
        for ret, spike in zip(
            event_window_return(items, price_map, window=config.label_window),
            volume_spike_zscore(
                items,
                price_map,
                window=config.volume_zscore_window,
                min_periods=config.volume_zscore_min_periods,
            ),
        )
    ]

    metadata = {
        "config": asdict(config),
        "cache_key": cache_key({"items": items, "config": config}),
        "feature_count": len(features),
        "label_count": len(labels),
    }

    return PipelineOutput(features=features, labels=labels, metadata=metadata)
