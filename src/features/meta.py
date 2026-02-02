from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

from .text import tokenize
from .types import NewsItem

POSITIVE_LEXICON = {"surge", "beat", "growth", "upgrade", "profit", "record", "strong"}
NEGATIVE_LEXICON = {"drop", "miss", "loss", "downgrade", "weak", "fraud", "decline"}


@dataclass(frozen=True)
class MetaVocab:
    sources: Dict[str, int]
    languages: Dict[str, int]


def build_meta_vocab(items: Iterable[NewsItem]) -> MetaVocab:
    source_counts = Counter(item.source for item in items)
    lang_counts = Counter(item.language for item in items)
    sources = {source: idx for idx, (source, _) in enumerate(source_counts.items())}
    languages = {lang: idx for idx, (lang, _) in enumerate(lang_counts.items())}
    return MetaVocab(sources=sources, languages=languages)


def _timezone_offset_hours(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        return 0
    offset = timestamp.utcoffset()
    if offset is None:
        return 0
    return int(offset.total_seconds() // 3600)


def _sentiment_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in POSITIVE_LEXICON)
    negative = sum(1 for token in tokens if token in NEGATIVE_LEXICON)
    return (positive - negative) / len(tokens)


def meta_features(items: List[NewsItem], vocab: MetaVocab) -> List[Dict[str, float]]:
    features: List[Dict[str, float]] = []
    for item in items:
        row: Dict[str, float] = {}
        if item.source in vocab.sources:
            row[f"source_{item.source}"] = 1.0
        if item.language in vocab.languages:
            row[f"lang_{item.language}"] = 1.0

        row["text_length"] = float(len(item.text))
        row["token_length"] = float(len(tokenize(item.text)))
        row["sentiment"] = _sentiment_score(item.text)
        row["hour"] = float(item.timestamp.hour)
        row["tz_offset_hours"] = float(_timezone_offset_hours(item.timestamp))
        features.append(row)
    return features
