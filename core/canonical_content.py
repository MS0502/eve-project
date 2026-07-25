"""Pure deterministic A11 canonical material + structural reference helpers.

This module deliberately has no I/O and does not relax EventEnvelope limits.
It exists so event/snapshot persistence and replay can hash large state with one
versioned canonical representation while leaving logical envelopes unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

CONTENT_REFERENCE_SCHEMA_VERSION = "eve.content-addressed-json-reference.v1"
CONTENT_SERIALIZATION_SCHEMA_VERSION = "eve.canonical-json-content.v1"
MAX_CONTENT_NESTING_DEPTH = 32


class CanonicalContentError(ValueError):
    pass


def _validate(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_CONTENT_NESTING_DEPTH:
        raise CanonicalContentError("content JSON exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalContentError("content JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalContentError("content JSON object keys must be strings")
            _validate(item, depth=depth + 1)
        return
    raise CanonicalContentError(f"unsupported content JSON value type: {type(value).__name__}")


def canonical_content_json(value: Mapping[str, Any], *, field: str) -> str:
    if not isinstance(value, Mapping):
        raise CanonicalContentError(f"{field} must be a mapping")
    plain = dict(value)
    _validate(plain)
    return json.dumps(
        plain,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def collection_counts(value: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in sorted(value):
        item = value[key]
        if isinstance(item, (list, dict)):
            counts[key] = len(item)
        if isinstance(item, dict):
            for nested_key in sorted(item):
                nested = item[nested_key]
                if isinstance(nested, (list, dict)):
                    counts[f"{key}.{nested_key}"] = len(nested)
    return counts


def content_manifest(
    value: Mapping[str, Any],
    material_json: str,
    *,
    content_schema_version: str,
) -> dict[str, Any]:
    if not isinstance(content_schema_version, str) or not content_schema_version.strip():
        raise CanonicalContentError("content_schema_version must be non-empty")
    return {
        "canonical_bytes": len(material_json.encode("utf-8")),
        "collection_counts": collection_counts(value),
        "content_schema_version": content_schema_version,
        "hash_algorithm": "sha256",
        "reference_schema_version": CONTENT_REFERENCE_SCHEMA_VERSION,
        "serialization_schema": CONTENT_SERIALIZATION_SCHEMA_VERSION,
        "top_level_key_count": len(value),
        "top_level_keys": sorted(value),
    }


def build_content_reference(
    value: Mapping[str, Any],
    *,
    content_schema_version: str,
) -> tuple[dict[str, Any], str]:
    material_json = canonical_content_json(value, field="content_material")
    reference = {
        "content_digest": hashlib.sha256(material_json.encode("utf-8")).hexdigest(),
        "manifest": content_manifest(
            value,
            material_json,
            content_schema_version=content_schema_version,
        ),
        "reference_schema_version": CONTENT_REFERENCE_SCHEMA_VERSION,
    }
    return reference, material_json


def verify_content_reference(
    reference: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    expected_schema_version: str,
) -> None:
    expected, _material_json = build_content_reference(
        value,
        content_schema_version=expected_schema_version,
    )
    if dict(reference) != expected:
        raise CanonicalContentError("content reference does not match canonical material")
