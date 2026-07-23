#!/usr/bin/env python3
"""Static M3-B source-ownership preflight for all 63 affect axes.

The checker parses source text only. It imports no EVE production module, reads no
live state, installs no observer, accesses no persistence, emits no event, and
cannot start an observation window or grant M3-C/M3-E/cutover authority.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "eve.m3-b.observation-source-ownership-preflight.v1"
BASELINE_SHA = "0d755c35c994fa5b1ed3f2768c7905cda83c9a95"
LEGACY_PATH = Path("hormone_system.py")
LEGACY_ADAPTER_PATH = Path("adapters/hormone_adapter.py")
REGISTRY_PATH = Path("adapters/affect_hormone_neural_rhythm_registry.py")
PERSISTENCE_PATH = Path("adapters/persistence_adapter.py")
PROPOSAL_MAP_PATH = Path("adapters/affect_event_to_axis_proposal_map.py")
PROPOSAL_POLICY_PATHS = {
    PROPOSAL_MAP_PATH.as_posix(),
    "adapters/affect_event_proposal_validator.py",
    "adapters/affect_hormone_interaction_matrix.py",
}
NONPRODUCTION_PREFIXES = ("scripts/audit/", "tests/", "eve_v3_autonomous_handoff/")
REQUIRED_OBSERVATION_FIELDS = (
    "source_snapshot_id",
    "source_schema_version",
    "source_integrity_digest",
    "value",
    "baseline",
    "floor",
    "ceiling",
    "confidence",
)
EXPECTED_BLOCKERS = (
    "LEGACY_IMMUTABLE_SOURCE_ENVELOPE_ABSENT",
    "REGISTRY_OBSERVED_VALUE_OWNER_ABSENT",
)


class SourceOwnershipAuditError(ValueError):
    """Raised when an authoritative source contract cannot be parsed exactly."""


def _parse(root: Path, relative: Path) -> ast.AST:
    return ast.parse(
        (root / relative).read_text(encoding="utf-8"),
        filename=relative.as_posix(),
    )


def _string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(rows) != 1:
        raise SourceOwnershipAuditError(f"expected exactly one function {name!r}")
    return rows[0]


def _tracked_python_files(root: Path) -> Iterator[Path]:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        yield from sorted(root.rglob("*.py"))
        return
    for value in raw.split(b"\0"):
        if value:
            yield root / Path(os.fsdecode(value))


def _nonproduction(relative: str) -> bool:
    return relative.startswith(NONPRODUCTION_PREFIXES)


def extract_legacy_axes(root: Path = ROOT) -> tuple[dict[str, Any], ...]:
    function = _function(_parse(root, LEGACY_PATH), "_init_all_hormones")
    rows: list[dict[str, Any]] = []
    for node in ast.walk(function):
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "config" for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "config"
        ):
            value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            continue
        for item in value.elts:
            if not isinstance(item, (ast.List, ast.Tuple)) or len(item.elts) != 6:
                continue
            axis = _string(item.elts[0])
            if axis is None:
                continue
            try:
                baseline, reactivity, decay_rate, tier, phase = (
                    ast.literal_eval(part) for part in item.elts[1:]
                )
            except (ValueError, TypeError) as exc:
                raise SourceOwnershipAuditError(f"{axis}: non-literal legacy config") from exc
            rows.append(
                {
                    "axis": axis,
                    "baseline": float(baseline),
                    "reactivity": float(reactivity),
                    "decay_rate": float(decay_rate),
                    "tier": str(tier),
                    "phase": int(phase),
                    "path": LEGACY_PATH.as_posix(),
                    "line": int(getattr(item, "lineno", 1)),
                }
            )
    rows.sort(key=lambda row: row["axis"])
    if len(rows) != 26 or len({row["axis"] for row in rows}) != 26:
        raise SourceOwnershipAuditError("legacy axis catalog must contain 26 unique axes")
    return tuple(rows)


def extract_registry_axes(root: Path = ROOT) -> tuple[dict[str, Any], ...]:
    tree = _parse(root, REGISTRY_PATH)
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        target: str | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = _name(node.targets[0]), node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = _name(node.target), node.value
        if target != "AXIS_GROUPS" or not isinstance(value, ast.Dict):
            continue
        for group_node, axes_node in zip(value.keys, value.values):
            group = _string(group_node)
            if group is None or not isinstance(axes_node, (ast.Tuple, ast.List)):
                continue
            for axis_node in axes_node.elts:
                axis = _string(axis_node)
                if axis is not None:
                    rows.append(
                        {
                            "axis": axis,
                            "group": group,
                            "path": REGISTRY_PATH.as_posix(),
                            "line": int(getattr(axis_node, "lineno", 1)),
                            "source_kind": "schema_definition_only",
                        }
                    )
    rows.sort(key=lambda row: row["axis"])
    if len(rows) != 37 or len({row["axis"] for row in rows}) != 37:
        raise SourceOwnershipAuditError("registry axis catalog must contain 37 unique axes")
    return tuple(rows)


def extract_proposal_rule_axes(
    root: Path,
    registry_axes: Iterable[str],
) -> dict[str, Any]:
    """Extract bounded `_EVENT_ROWS` proposal keys, never current state values."""
    registry = frozenset(registry_axes)
    tree = _parse(root, PROPOSAL_MAP_PATH)
    occurrences: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _name(node.func) != "_entry":
            continue
        delta_node: ast.AST | None = node.args[3] if len(node.args) >= 4 else None
        for keyword in node.keywords:
            if keyword.arg == "allowed_axis_deltas":
                delta_node = keyword.value
        if not isinstance(delta_node, ast.Dict):
            continue
        event = _string(node.args[0]) if node.args else None
        for key_node in delta_node.keys:
            axis = _string(key_node)
            if axis is None:
                continue
            if axis not in registry:
                raise SourceOwnershipAuditError(f"proposal map contains unknown axis {axis!r}")
            occurrences.append(
                {
                    "axis": axis,
                    "event_category": event,
                    "path": PROPOSAL_MAP_PATH.as_posix(),
                    "line": int(getattr(key_node, "lineno", 1)),
                    "classification": "bounded_proposal_delta_metadata",
                }
            )
    occurrences.sort(key=lambda row: (row["line"], row["axis"]))
    unique_axes = sorted({row["axis"] for row in occurrences})
    return {
        "proposal_rule_occurrences": occurrences,
        "proposal_rule_occurrence_count": len(occurrences),
        "proposal_rule_unique_axes": unique_axes,
        "proposal_rule_unique_axis_count": len(unique_axes),
        "proposal_rules_are_not_current_axis_values": True,
    }


def inspect_legacy_read_surface(root: Path = ROOT) -> dict[str, Any]:
    tree = _parse(root, LEGACY_ADAPTER_PATH)
    init = _function(tree, "__init__")
    as_dict = _function(tree, "as_dict")
    owns_reference = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            _name(target) == "self.hs"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in ast.walk(init)
    )
    iterates_all = any(
        isinstance(node, ast.Call) and _name(node.func) == "self.hs.hormones.items"
        for node in ast.walk(as_dict)
    )
    reads_level = any(
        isinstance(node, ast.Attribute) and _name(node) == "h.level"
        for node in ast.walk(as_dict)
    )
    literals = {_string(node) for node in ast.walk(as_dict)}
    missing = tuple(field for field in REQUIRED_OBSERVATION_FIELDS if field not in literals)
    return {
        "path": LEGACY_ADAPTER_PATH.as_posix(),
        "owns_hormone_system_reference": owns_reference,
        "iterates_all_hormones": iterates_all,
        "reads_current_level": reads_level,
        "axis_count_readable": 26 if owns_reference and iterates_all and reads_level else 0,
        "required_observation_fields": list(REQUIRED_OBSERVATION_FIELDS),
        "missing_immutable_envelope_fields": list(missing),
        "immutable_source_envelope_complete": not missing,
        "derived_compatibility_keys_are_not_axes": ["stress", "energy", "curiosity"],
        "authority": "legacy_runtime_authoritative_read_surface_only",
    }


def inspect_persistence_container(root: Path, legacy_axes: Iterable[str]) -> dict[str, Any]:
    names = frozenset(legacy_axes)
    tree = _parse(root, PERSISTENCE_PATH)
    whole_container = False
    axis_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            whole_container = whole_container or any(_name(target) == "self.hs" for target in targets)
        if isinstance(node, ast.Subscript) and _string(node.slice) in names:
            axis_keys.add(str(_string(node.slice)))
        if isinstance(node, ast.Dict):
            axis_keys.update(key for key in (_string(item) for item in node.keys) if key in names)
    return {
        "path": PERSISTENCE_PATH.as_posix(),
        "whole_hormone_system_reference": whole_container,
        "axis_specific_snapshot_keys": sorted(axis_keys),
        "axis_specific_snapshot_contract_found": bool(axis_keys),
        "persistence_authority_changed": False,
    }


def scan_registry_usage(root: Path, registry_axes: Iterable[str]) -> dict[str, Any]:
    """Find explicit keyed writes outside definition/proposal/audit paths.

    Dict literals are not treated as current-value ownership: they are commonly
    schemas, detached result packets, interaction matrices, or proposal metadata.
    A candidate owner must at least perform an exact-axis subscript Store outside
    the reviewed policy-only and non-production paths.
    """
    axes = frozenset(registry_axes)
    calls: list[dict[str, Any]] = []
    observed_stores: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in _tracked_python_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == REGISTRY_PATH.as_posix():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_errors.append({"path": relative, "error": type(exc).__name__})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func).split(".")[-1] == "affect_hormone_axis_registry":
                calls.append(
                    {
                        "path": relative,
                        "line": int(getattr(node, "lineno", 1)),
                        "classification": (
                            "proposal_or_schema_consumer"
                            if relative in PROPOSAL_POLICY_PATHS
                            else "other_consumer"
                        ),
                    }
                )
            if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Store):
                continue
            axis = _string(node.slice)
            if (
                axis not in axes
                or _nonproduction(relative)
                or relative in PROPOSAL_POLICY_PATHS
            ):
                continue
            observed_stores.append(
                {
                    "axis": axis,
                    "path": relative,
                    "line": int(getattr(node, "lineno", 1)),
                    "target": _name(node.value),
                    "classification": "explicit_current_value_owner_candidate",
                }
            )
    observed_stores.sort(key=lambda row: (row["path"], row["line"], row["axis"]))
    return {
        "registry_factory_call_sites": sorted(calls, key=lambda row: (row["path"], row["line"])),
        "proposal_policy_paths": sorted(PROPOSAL_POLICY_PATHS),
        "observed_value_store_candidates": observed_stores,
        "observed_value_store_candidate_count": len(observed_stores),
        "observed_value_owner_found": bool(observed_stores),
        "tracked_parse_errors": sorted(parse_errors, key=lambda row: row["path"]),
        "tracked_parse_errors_are_not_source_ownership_evidence": True,
    }


def _digest(value: Mapping[str, Any]) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    legacy_axes = extract_legacy_axes(root)
    registry_axes = extract_registry_axes(root)
    registry_names = tuple(row["axis"] for row in registry_axes)
    proposal_rules = extract_proposal_rule_axes(root, registry_names)
    legacy_surface = inspect_legacy_read_surface(root)
    persistence = inspect_persistence_container(root, (row["axis"] for row in legacy_axes))
    usage = scan_registry_usage(root, registry_names)

    blockers: list[str] = []
    if not legacy_surface["immutable_source_envelope_complete"]:
        blockers.append("LEGACY_IMMUTABLE_SOURCE_ENVELOPE_ABSENT")
    if not usage["observed_value_owner_found"]:
        blockers.append("REGISTRY_OBSERVED_VALUE_OWNER_ABSENT")

    errors: list[str] = []
    if tuple(blockers) != EXPECTED_BLOCKERS:
        errors.append(f"unexpected blocker set: {blockers}")
    if persistence["axis_specific_snapshot_contract_found"]:
        errors.append("legacy persistence unexpectedly claims axis-specific snapshot ownership")
    if legacy_surface["axis_count_readable"] != 26:
        errors.append("legacy read surface does not expose all 26 axes")
    if proposal_rules["proposal_rule_unique_axis_count"] != 27:
        errors.append("proposal map must cover exactly 27 unique registry axes")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "m3_b_technical_merge_sha": BASELINE_SHA,
        "authority": "shadow_only_preflight",
        "legacy_runtime_authoritative": True,
        "legacy_persistence_authoritative": True,
        "live_source_acquisition_installed": False,
        "observation_window_started": False,
        "observation_window_satisfied": False,
        "m3_b_complete": False,
        "m3_c_open": False,
        "m3_e_authority_open": False,
        "cutover_authorized": False,
        "event_append_performed": False,
        "persistence_accessed": False,
        "axis_counts": {
            "legacy_mutable_hormone": len(legacy_axes),
            "read_only_affect_registry": len(registry_axes),
            "total": len(legacy_axes) + len(registry_axes),
        },
        "legacy_axes": list(legacy_axes),
        "registry_axes": list(registry_axes),
        "legacy_read_surface": legacy_surface,
        "legacy_persistence_container": persistence,
        "registry_proposal_metadata": proposal_rules,
        "registry_usage": usage,
        "source_family_readiness": {
            "legacy_mutable_hormone": "READABLE_UNVERSIONED_LEGACY_CONTAINER",
            "read_only_affect_registry": "PROPOSAL_METADATA_EXISTS_NO_OBSERVED_VALUE_OWNER",
        },
        "strict_63_axis_observation_ready": False,
        "observation_ready_axis_count": 0,
        "blockers": blockers,
        "next_required_artifacts": [
            "versioned immutable after-the-fact legacy 26-axis source envelope with identity/schema/integrity digest and no mutation",
            "separately reviewed current-value producer/ownership contract for all 37 registry axes; defaults and proposal deltas are not observations",
            "real read-only observation-window packet only after both source families are admissible",
        ],
        "errors": errors,
    }
    report["report_digest"] = _digest(report)
    return report


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _write(value: Mapping[str, Any], output: Path | None, pretty: bool) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    ) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    report = audit_repository()
    output: Mapping[str, Any] = report
    if args.summary_only:
        output = {
            "axis_counts": report["axis_counts"],
            "blockers": report["blockers"],
            "errors": report["errors"],
            "m3_b_complete": report["m3_b_complete"],
            "m3_c_open": report["m3_c_open"],
            "observation_window_started": report["observation_window_started"],
            "proposal_rule_occurrence_count": report["registry_proposal_metadata"]["proposal_rule_occurrence_count"],
            "proposal_rule_unique_axis_count": report["registry_proposal_metadata"]["proposal_rule_unique_axis_count"],
            "report_digest": report["report_digest"],
            "source_family_readiness": report["source_family_readiness"],
            "strict_63_axis_observation_ready": report["strict_63_axis_observation_ready"],
        }
    _write(output, args.output, args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
