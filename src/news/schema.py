from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
from typing import Iterable, Mapping, Sequence


_TITLE_CLEAN_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
_URL_CLEAN_RE = re.compile(r"#.*$")


@dataclass(frozen=True)
class NewsRecord:
    id: str
    published_at: datetime
    source: str
    title: str
    body: str
    tickers: tuple[str, ...] = field(default_factory=tuple)
    url: str = ""
    language: str = ""

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "NewsRecord":
        published_at = normalize_datetime(data["published_at"])
        tickers = tuple(sorted(set(data.get("tickers", []) or [])))
        return NewsRecord(
            id=str(data["id"]),
            published_at=published_at,
            source=str(data.get("source", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body") or data.get("summary") or ""),
            tickers=tickers,
            url=str(data.get("url", "")),
            language=str(data.get("language", "")),
        )


def normalize_datetime(value: object) -> datetime:
    """Normalize datetimes to UTC with timezone awareness."""
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_title(title: str) -> str:
    cleaned = _TITLE_CLEAN_RE.sub(" ", title.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def normalize_url(url: str) -> str:
    cleaned = _URL_CLEAN_RE.sub("", url.strip())
    return cleaned.lower()


def similarity_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def is_duplicate(
    existing: NewsRecord,
    candidate: NewsRecord,
    *,
    similarity_threshold: float = 0.9,
) -> bool:
    """Detect duplicates via URL + normalized title similarity."""
    if normalize_url(existing.url) and normalize_url(existing.url) == normalize_url(candidate.url):
        return True
    left = normalize_title(existing.title)
    right = normalize_title(candidate.title)
    if left and right and similarity_ratio(left, right) >= similarity_threshold:
        return True
    return False


def dedupe_news(records: Sequence[NewsRecord]) -> list[NewsRecord]:
    """Remove duplicates while preserving order."""
    unique: list[NewsRecord] = []
    for record in records:
        if any(is_duplicate(existing, record) for existing in unique):
            continue
        unique.append(record)
    return unique


def extract_tickers(
    text: str,
    *,
    symbol_map: Mapping[str, Iterable[str]] | None = None,
    enable_ner: bool = False,
) -> list[str]:
    """Extract tickers using rule-based matching with optional NER hook."""
    symbols: set[str] = set()
    text_lower = text.lower()
    if symbol_map:
        for ticker, aliases in symbol_map.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    symbols.add(ticker.upper())
                    break
    ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")
    symbols.update({match.group(0) for match in ticker_pattern.finditer(text)})

    if enable_ner:
        raise NotImplementedError("NER extension hook not configured")

    return sorted(symbols)


def map_tickers(record: NewsRecord, symbol_map: Mapping[str, Iterable[str]] | None = None) -> NewsRecord:
    text = f"{record.title} {record.body}".strip()
    tickers = extract_tickers(text, symbol_map=symbol_map)
    return NewsRecord(
        id=record.id,
        published_at=record.published_at,
        source=record.source,
        title=record.title,
        body=record.body,
        tickers=tuple(tickers),
        url=record.url,
        language=record.language,
    )
