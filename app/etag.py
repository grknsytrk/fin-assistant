from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def weak_etag(revision: Any) -> str:
    """Build a stable weak validator from small source-version metadata."""

    serialized = json.dumps(revision, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
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
) -> Response:
    """Return JSON with a validator, or an empty 304 before JSON encoding."""

    headers = {"Cache-Control": "private, no-cache"}
    if revision is not None:
        etag = weak_etag(revision)
        headers["ETag"] = etag
        if if_none_match_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
    return JSONResponse(content=jsonable_encoder(payload), headers=headers)
