#!/usr/bin/env python3
"""Validate compact, exact-coverage M2-B decision groups.

The compact file shares review text across exact edge/finding ID sets. The
origin report digest remains immutable provenance; current validity is decided
by exact, non-stale coverage of evidence IDs, whose hashes already bind source,
call-path, sink, and parse evidence. Append-only per-PR fragments may extend the
compact groups and advance only the candidate-surface digest. This module is
audit-only and cannot set human acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.audit import m2_b_read_capability_manifest as manifest
except ModuleNotFoundError:  # direct execution from scripts/audit
    import m2_b_read_capability_manifest as manifest

SCHEMA_VERSION = "eve.m2-b-read-capability-decision-groups.v2"
ADDITIONS_SCHEMA_VERSION = "eve.m2-b-read-capability-decision-additions.v1"
EDGE_SHARED_FIELDS = manifest.DECISION_FIELDS - {"edge_id"}
FINDING_SHARED_FIELDS = manifest.REVIEW_FIELDS
SURFACE_FIELDS = (
    "schema_version",
    "candidate_edges",
    "unresolved_boundary_calls",
    "parse_errors",
)
_GROUP_FIELDS = (
    "edge_decision_groups",
    "unresolved_call_decision_groups",
    "parse_error_decision_groups",
)


def candidate_surface_digest(report: Mapping[str, Any]) -> str:
    """Digest only capability evidence, excluding audit inventory counters."""
    return manifest.digest({field: report.get(field) for field in SURFACE_FIELDS})


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reviewed(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value != manifest.REVIEW_REQUIRED


def _expand_groups(
    groups: Any,
    *,
    ids_field: str,
    output_id_field: str,
    required_fields: set[str],
    label: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(groups, list):
        errors.append(f"{label} must be a list")
        return []
    expanded: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"{label}[{index}] must be an object")
            continue
        identifiers = group.get(ids_field)
        if not isinstance(identifiers, list) or not identifiers or not all(
            isinstance(value, str) and value for value in identifiers
        ):
            errors.append(f"{label}[{index}].{ids_field} must be a non-empty string list")
            continue
        missing = sorted(field for field in required_fields if not _reviewed(group.get(field)))
        if missing:
            errors.append(f"{label}[{index}] fields require review: {','.join(missing)}")
            continue
        shared = {field: group[field] for field in required_fields}
        expanded.extend({output_id_field: identifier, **shared} for identifier in identifiers)
    return expanded


def apply_additions(
    compact: Mapping[str, Any],
    additions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Apply ordered append-only review fragments without rewriting the base record."""
    errors: list[str] = []
    if not isinstance(compact, Mapping):
        return {}, ["compact decision root must be an object"]
    merged = json.loads(json.dumps(compact, ensure_ascii=False))
    for field in _GROUP_FIELDS:
        if not isinstance(merged.get(field), list):
            errors.append(f"compact {field} must be a list")
            merged[field] = []
    current_surface = merged.get("candidate_surface_digest")
    seen_prs: set[int] = set()
    for index, fragment in enumerate(additions):
        label = f"decision_additions[{index}]"
        if not isinstance(fragment, Mapping):
            errors.append(f"{label} must be an object")
            continue
        if fragment.get("schema_version") != ADDITIONS_SCHEMA_VERSION:
            errors.append(f"{label} schema_version mismatch")
        introduced_by_pr = fragment.get("introduced_by_pr")
        if (
            isinstance(introduced_by_pr, bool)
            or not isinstance(introduced_by_pr, int)
            or introduced_by_pr <= 0
        ):
            errors.append(f"{label} introduced_by_pr must be a positive integer")
        elif introduced_by_pr in seen_prs:
            errors.append(f"{label} duplicates introduced_by_pr {introduced_by_pr}")
        else:
            seen_prs.add(introduced_by_pr)
        base_surface = fragment.get("base_candidate_surface_digest")
        next_surface = fragment.get("candidate_surface_digest")
        if not _sha256(base_surface) or base_surface != current_surface:
            errors.append(f"{label} base candidate-surface digest mismatch")
        if not _sha256(next_surface):
            errors.append(f"{label} candidate_surface_digest must be a lowercase SHA-256")
        for field in _GROUP_FIELDS:
            groups = fragment.get(field, [])
            if not isinstance(groups, list):
                errors.append(f"{label}.{field} must be a list")
                continue
            merged[field].extend(json.loads(json.dumps(groups, ensure_ascii=False)))
        if _sha256(next_surface):
            current_surface = next_surface
    merged["candidate_surface_digest"] = current_surface
    return merged, errors


def expand_decisions(
    report: Mapping[str, Any], compact: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if compact.get("schema_version") != SCHEMA_VERSION:
        errors.append("compact decision schema_version mismatch")
    if not _sha256(compact.get("candidate_report_digest")):
        errors.append("compact candidate_report_digest must be a lowercase SHA-256")
    expected_surface_digest = candidate_surface_digest(report)
    if not _sha256(compact.get("candidate_surface_digest")):
        errors.append("compact candidate_surface_digest must be a lowercase SHA-256")
    elif compact.get("candidate_surface_digest") != expected_surface_digest:
        errors.append("compact candidate_surface_digest mismatch")
    edge_decisions = _expand_groups(
        compact.get("edge_decision_groups", []),
        ids_field="edge_ids",
        output_id_field="edge_id",
        required_fields=EDGE_SHARED_FIELDS,
        label="edge_decision_groups",
        errors=errors,
    )
    unresolved = _expand_groups(
        compact.get("unresolved_call_decision_groups", []),
        ids_field="finding_ids",
        output_id_field="finding_id",
        required_fields=FINDING_SHARED_FIELDS,
        label="unresolved_call_decision_groups",
        errors=errors,
    )
    parse_errors = _expand_groups(
        compact.get("parse_error_decision_groups", []),
        ids_field="finding_ids",
        output_id_field="finding_id",
        required_fields=FINDING_SHARED_FIELDS,
        label="parse_error_decision_groups",
        errors=errors,
    )
    expanded = {
        "schema_version": manifest.DECISION_SCHEMA_VERSION,
        "candidate_report_digest": report.get("report_digest"),
        "edge_decisions": edge_decisions,
        "unresolved_call_decisions": unresolved,
        "parse_error_decisions": parse_errors,
    }
    return expanded, errors


def validate_compact_decisions(
    report: Mapping[str, Any],
    compact: Mapping[str, Any],
    additions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    merged, addition_errors = apply_additions(compact, additions)
    expanded, compact_errors = expand_decisions(report, merged)
    delegated = manifest.validate_decisions(report, expanded)
    errors = [*addition_errors, *compact_errors, *delegated["errors"]]
    result = {
        **delegated,
        "schema_version": SCHEMA_VERSION,
        "origin_candidate_report_digest": compact.get("candidate_report_digest"),
        "candidate_surface_digest": candidate_surface_digest(report),
        "valid": not errors,
        "errors": errors,
        "eligible_for_human_review": not errors,
        "human_accepted": False,
        "authority": manifest.AUTHORITY,
        "addition_fragment_count": len(additions),
        "compact_group_counts": {
            "edge": len(merged.get("edge_decision_groups", []))
            if isinstance(merged.get("edge_decision_groups", []), list)
            else 0,
            "unresolved_call": len(merged.get("unresolved_call_decision_groups", []))
            if isinstance(merged.get("unresolved_call_decision_groups", []), list)
            else 0,
            "parse_error": len(merged.get("parse_error_decision_groups", []))
            if isinstance(merged.get("parse_error_decision_groups", []), list)
            else 0,
        },
    }
    result["validation_digest"] = manifest.digest(result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--additions", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = json.loads(args.candidate.read_text(encoding="utf-8"))
    compact = json.loads(args.decisions.read_text(encoding="utf-8"))
    additions = [json.loads(path.read_text(encoding="utf-8")) for path in args.additions]
    if not isinstance(report, dict) or not isinstance(compact, dict) or not all(
        isinstance(value, dict) for value in additions
    ):
        raise ValueError("candidate, decision, and addition roots must be objects")
    result = validate_compact_decisions(report, compact, additions)
    text = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return int(args.strict and not result["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
