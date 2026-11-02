from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import deque
from collections.abc import Set as AbstractSet
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, Callable, Dict

from fastapi import Request, Response

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def orjson_default(obj: Any) -> Any:
    """Fallback serializer for orjson handling Decimal, Path, sets, Pydantic, etc."""
    if isinstance(obj, Decimal):
        if obj.is_nan() or obj.is_infinite():
            return None
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, (AbstractSet, deque)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, Exception):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def orjson_dumps(
    content: Any,
    *,
    default: Callable[[Any], Any] = orjson_default,
    sort_keys: bool = False,
) -> bytes:
    """Serialize Python object to JSON bytes using orjson with fallback support."""
    if orjson is not None:
        option = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS
        return orjson.dumps(content, default=default, option=option)
    return json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        default=default,
    ).encode("utf-8")


class ORJSONResponse(Response):
    """High-performance JSON Response using orjson and custom type handling."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        if content is None:
            return b""
        if isinstance(content, (bytes, memoryview)):
            return bytes(content)
        return orjson_dumps(content)


def weak_etag(revision: Any) -> str:
    """Build a stable weak validator from small source-version metadata."""
    serialized = orjson_dumps(revision, sort_keys=True)
    digest = hashlib.sha256(serialized).hexdigest()
    return f'W/"{digest}"'


def _entity_tag_opaque_value(tag: str) -> str | None:
    value = tag.strip()
    if value.startswith("W/"):
        value = value[2:].lstrip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return None
    return value[1:-1]


def _split_entity_tag_list(value: str) -> list[str]:
    tags: list[str] = []
    start = 0
    in_quotes = False
    for index, character in enumerate(value):
        if character == '"':
            in_quotes = not in_quotes
        elif character == "," and not in_quotes:
            tags.append(value[start:index].strip())
            start = index + 1
    tags.append(value[start:].strip())
    return [tag for tag in tags if tag]


def if_none_match_matches(header_value: str | None, current_etag: str) -> bool:
    """Apply If-None-Match's weak comparison semantics for GET responses."""
    if not header_value:
        return False
    current = _entity_tag_opaque_value(current_etag)
    if current is None:
        return False
    for candidate in _split_entity_tag_list(header_value):
        if candidate == "*":
            return True
        if _entity_tag_opaque_value(candidate) == current:
            return True
    return False


def conditional_json_response(
    request: Request,
    payload: Dict[str, Any],
    revision: Any | None,
    status_code: int = 200,
) -> Response:
    """Return JSON with a validator, or an empty 304 before JSON encoding."""
    headers = {"Cache-Control": "private, no-cache"}
    if revision is not None:
        etag = weak_etag(revision)
        headers["ETag"] = etag
        if if_none_match_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
    return ORJSONResponse(content=payload, status_code=status_code, headers=headers)
