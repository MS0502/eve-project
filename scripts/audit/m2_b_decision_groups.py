#!/usr/bin/env python3
"""Validate compact, exact-coverage M2-B decision groups.

The compact file shares review text across exact edge/finding ID sets. This
module expands those groups and delegates fail-closed validation to the M2-B
candidate extractor. It is audit-only and cannot set human acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from scripts.audit import m2_b_read_capability_manifest as manifest
except ModuleNotFoundError:  # direct execution from scripts/audit
    import m2_b_read_capability_manifest as manifest

SCHEMA_VERSION = "eve.m2-b-read-capability-decision-groups.v1"
EDGE_SHARED_FIELDS = manifest.DECISION_FIELDS - {"edge_id"}
FINDING_SHARED_FIELDS = manifest.REVIEW_FIELDS
SURFACE_FIELDS = (
    "schema_version",
    "candidate_edges",
    "unresolved_boundary_calls",
    "parse_errors",
)


def candidate_surface_digest(report: Mapping[str, Any]) -> str:
    """Bind decisions to capability evidence, not audit-tool inventory counts."""
    material = {field: report.get(field) for field in SURFACE_FIELDS}
    return manifest.digest(material)


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


def expand_decisions(report: Mapping[str, Any], compact: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if compact.get("schema_version") != SCHEMA_VERSION:
        errors.append("compact decision schema_version mismatch")
    expected_surface_digest = candidate_surface_digest(report)
    if compact.get("candidate_surface_digest") != expected_surface_digest:
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


def validate_compact_decisions(report: Mapping[str, Any], compact: Mapping[str, Any]) -> dict[str, Any]:
    expanded, compact_errors = expand_decisions(report, compact)
    delegated = manifest.validate_decisions(report, expanded)
    errors = [*compact_errors, *delegated["errors"]]
    result = {
        **delegated,
        "schema_version": SCHEMA_VERSION,
        "candidate_surface_digest": candidate_surface_digest(report),
        "valid": not errors,
        "errors": errors,
        "eligible_for_human_review": not errors,
        "human_accepted": False,
        "authority": manifest.AUTHORITY,
        "compact_group_counts": {
            "edge": len(compact.get("edge_decision_groups", [])) if isinstance(compact.get("edge_decision_groups", []), list) else 0,
            "unresolved_call": len(compact.get("unresolved_call_decision_groups", [])) if isinstance(compact.get("unresolved_call_decision_groups", []), list) else 0,
            "parse_error": len(compact.get("parse_error_decision_groups", [])) if isinstance(compact.get("parse_error_decision_groups", []), list) else 0,
        },
    }
    result["validation_digest"] = manifest.digest(result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = json.loads(args.candidate.read_text(encoding="utf-8"))
    compact = json.loads(args.decisions.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(compact, dict):
        raise ValueError("candidate and decision roots must be objects")
    result = validate_compact_decisions(report, compact)
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
