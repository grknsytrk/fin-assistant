from __future__ import annotations

import collections
import dataclasses
import datetime
import hashlib
import json
import math
import pathlib
import re
import uuid
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict

from fastapi import Request, Response

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def orjson_default(obj: Any) -> Any:
    """Fallback serializer for orjson handling Decimal, Path, sets, Pydantic, etc."""
    if isinstance(obj, Decimal):
        if obj.is_nan() or obj.is_infinite():
            return None
        try:
            if obj == obj.to_integral_value():
                val = int(obj)
                if -9223372036854775808 <= val <= 9223372036854775807:
                    return val
            as_float = float(obj)
            if math.isnan(as_float) or math.isinf(as_float):
                return None
            return as_float
        except Exception:
            return str(obj)
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, (collections.abc.Set, collections.deque)):
        return list(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Enum):
        return obj.value
    if (pd is not None and obj is pd.NaT) or type(obj).__name__ == "NaTType":
        return None
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, re.Pattern):
        return obj.pattern
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if obj is pd.NaT:
            return None
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if pd.isna(obj) is True:
            return None
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict(orient="records")
        except TypeError:
            return obj.to_dict()
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
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
            return b"null"
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
