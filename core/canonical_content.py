"""Pure deterministic A11 canonical material + structural reference helpers.

This module deliberately has no I/O and does not relax EventEnvelope limits.
It exists so snapshot persistence, habitat compact events, and replay all hash
large state with one versioned canonical representation.
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


APPEND_STATE_REPRESENTATION_SCHEMA_VERSION = "eve.a11-append-state-reference.v1"
APPEND_STATE_CONTENT_SCHEMA_VERSION = "eve.event-append-state-material.v1"


def _append_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if set(before) != set(after):
        raise CanonicalContentError("append-state mappings must have identical key domains")
    appended: dict[str, list[Any]] = {}
    for key in sorted(before):
        before_value = before[key]
        after_value = after[key]
        if not isinstance(before_value, list) or not isinstance(after_value, list):
            raise CanonicalContentError("append-state mappings must contain JSON lists")
        if after_value[: len(before_value)] != before_value:
            raise CanonicalContentError("append-state after value must preserve the before prefix")
        appended[key] = after_value[len(before_value) :]
    return {"append": appended}


def compact_append_state_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Replace large before/after list-state material with A11 refs + append delta."""
    if not isinstance(payload, Mapping):
        raise CanonicalContentError("payload must be a mapping")
    before = payload.get("before")
    after = payload.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise CanonicalContentError("payload is not an append-state candidate")
    before_plain = dict(before)
    after_plain = dict(after)
    delta = _append_delta(before_plain, after_plain)
    before_ref, _ = build_content_reference(
        before_plain,
        content_schema_version=APPEND_STATE_CONTENT_SCHEMA_VERSION,
    )
    after_ref, _ = build_content_reference(
        after_plain,
        content_schema_version=APPEND_STATE_CONTENT_SCHEMA_VERSION,
    )
    compact = {key: value for key, value in payload.items() if key not in {"before", "after"}}
    compact.update(
        {
            "after_ref": after_ref,
            "before_ref": before_ref,
            "state_delta": delta,
            "state_representation": APPEND_STATE_REPRESENTATION_SCHEMA_VERSION,
        }
    )
    return compact


def apply_append_state_delta(
    before: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically reconstruct an append-only after-state for revalidation."""
    if not isinstance(before, Mapping) or not isinstance(delta, Mapping) or set(delta) != {"append"}:
        raise CanonicalContentError("append-state delta is malformed")
    appended = delta["append"]
    if not isinstance(appended, Mapping) or set(appended) != set(before):
        raise CanonicalContentError("append-state delta key domain mismatch")
    result: dict[str, Any] = {}
    for key in sorted(before):
        base = before[key]
        extra = appended[key]
        if not isinstance(base, list) or not isinstance(extra, list):
            raise CanonicalContentError("append-state delta values must be JSON lists")
        result[key] = list(base) + list(extra)
    return result
