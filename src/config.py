"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ParsedConfig:
    raw: Dict[str, Any]


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _tokenize(text: str) -> List[Tuple[int, str]]:
    tokens: List[Tuple[int, str]] = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip("\n")
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        tokens.append((indent, stripped.strip()))
    return tokens


def _parse_block(tokens: List[Tuple[int, str]], start: int, parent_indent: int) -> Tuple[Any, int]:
    if start >= len(tokens):
        return {}, start

    indent, content = tokens[start]
    if indent <= parent_indent:
        return {}, start

    if content.startswith("- "):
        items: List[Any] = []
        idx = start
        while idx < len(tokens) and tokens[idx][0] > parent_indent:
            indent, content = tokens[idx]
            if not content.startswith("- "):
                break
            item_content = content[2:].strip()
            if item_content:
                items.append(_parse_scalar(item_content))
                idx += 1
                continue
            item, idx = _parse_block(tokens, idx + 1, indent)
            items.append(item)
        return items, idx

    mapping: Dict[str, Any] = {}
    idx = start
    while idx < len(tokens) and tokens[idx][0] > parent_indent:
        indent, content = tokens[idx]
        if ":" not in content:
            idx += 1
            continue
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            mapping[key] = _parse_scalar(value)
            idx += 1
        else:
            nested, idx = _parse_block(tokens, idx + 1, indent)
            mapping[key] = nested
    return mapping, idx


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except ImportError:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        tokens = _tokenize(text)
        parsed, _ = _parse_block(tokens, 0, -1)
        if not isinstance(parsed, dict):
            raise ValueError("Configuration root must be a mapping.")
        return parsed
