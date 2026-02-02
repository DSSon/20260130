from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _default_serializer(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def cache_key(payload: Any) -> str:
    serialized = json.dumps(payload, default=_default_serializer, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
