from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PriceRecord:
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "PriceRecord":
        timestamp = normalize_datetime(data["timestamp"])
        adjusted = data.get("adjusted_close")
        return PriceRecord(
            ticker=str(data["ticker"]).upper(),
            timestamp=timestamp,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            adjusted_close=float(adjusted) if adjusted is not None else None,
        )


def normalize_datetime(value: object) -> datetime:
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


def apply_missing_rules(
    records: Iterable[PriceRecord],
    *,
    last_close: float | None = None,
    allow_forward_fill: bool = True,
) -> list[PriceRecord]:
    """Handle missing sessions with optional forward fill based on last close."""
    normalized: list[PriceRecord] = []
    previous_close = last_close
    for record in records:
        if record.volume == 0 and allow_forward_fill and previous_close is not None:
            filled = PriceRecord(
                ticker=record.ticker,
                timestamp=record.timestamp,
                open=previous_close,
                high=previous_close,
                low=previous_close,
                close=previous_close,
                volume=0.0,
                adjusted_close=record.adjusted_close,
            )
            normalized.append(filled)
        else:
            normalized.append(record)
        previous_close = normalized[-1].close
    return normalized
