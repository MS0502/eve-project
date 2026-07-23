#!/usr/bin/env python3
"""Audit whether M3-B has real read-only source ownership for all 63 axes.

This is static preflight evidence. It imports no production module, constructs no
runtime, reads no live state, installs no observer, accesses no persistence, emits
no event, and grants no observation-window, M3-C, cutover, or M3-E authority.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "eve.m3-b.observation-source-ownership-preflight.v1"
BASELINE_SHA = "0d755c35c994fa5b1ed3f2768c7905cda83c9a95"
M3_B_TECHNICAL_MERGE_SHA = BASELINE_SHA
LEGACY_PATH = Path("hormone_system.py")
LEGACY_ADAPTER_PATH = Path("adapters/hormone_adapter.py")
REGISTRY_PATH = Path("adapters/affect_hormone_neural_rhythm_registry.py")
PERSISTENCE_PATH = Path("adapters/persistence_adapter.py")
AUDIT_PREFIXES = ("scripts/audit/", "tests/", "eve_v3_autonomous_handoff/")
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
    """Raised when authoritative source files cannot be parsed deterministically."""


def _parse(root: Path, relative: Path) -> ast.AST:
    path = root / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=relative.as_posix())


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _target_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _call_name(node: ast.Call) -> str:
    return _target_name(node.func)


def _node_end(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))


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


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise SourceOwnershipAuditError(f"expected exactly one function {name!r}")
    return matches[0]


def extract_legacy_axes(root: Path = ROOT) -> tuple[dict[str, Any], ...]:
    tree = _parse(root, LEGACY_PATH)
    function = _function(tree, "_init_all_hormones")
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
            axis = _literal_string(item.elts[0])
            if axis is None:
                continue
            scalars: list[Any] = []
            for scalar in item.elts[1:]:
                try:
                    scalars.append(ast.literal_eval(scalar))
                except (ValueError, TypeError) as exc:
                    raise SourceOwnershipAuditError(f"{axis}: non-literal legacy config") from exc
            baseline, reactivity, decay_rate, tier, phase = scalars
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
    groups: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        target: str | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = _target_name(node.targets[0]), node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = _target_name(node.target), node.value
        if target != "AXIS_GROUPS" or not isinstance(value, ast.Dict):
            continue
        for group_node, axes_node in zip(value.keys, value.values):
            group = _literal_string(group_node)
            if group is None or not isinstance(axes_node, (ast.Tuple, ast.List)):
                continue
            axes = tuple(
                axis
                for axis in (_literal_string(item) for item in axes_node.elts)
                if axis is not None
            )
            groups.append((group, axes))
    rows: list[dict[str, Any]] = []
    for group, axes in groups:
        for axis in axes:
            rows.append(
                {
                    "axis": axis,
                    "group": group,
                    "path": REGISTRY_PATH.as_posix(),
                    "source_kind": "schema_definition_only",
                }
            )
    rows.sort(key=lambda row: row["axis"])
    if len(rows) != 37 or len({row["axis"] for row in rows}) != 37:
        raise SourceOwnershipAuditError("registry axis catalog must contain 37 unique axes")
    return tuple(rows)


def inspect_legacy_read_surface(root: Path = ROOT) -> dict[str, Any]:
    tree = _parse(root, LEGACY_ADAPTER_PATH)
    init = _function(tree, "__init__")
    as_dict = _function(tree, "as_dict")
    owns_hormone_system = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            _target_name(target) == "self.hs"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in ast.walk(init)
    )
    reads_levels = any(
        isinstance(node, ast.Attribute) and _target_name(node) == "h.level"
        for node in ast.walk(as_dict)
    )
    iterates_hormones = any(
        isinstance(node, ast.Call) and _call_name(node) == "self.hs.hormones.items"
        for node in ast.walk(as_dict)
    )
    literal_fields = {
        value
        for value in (_literal_string(node) for node in ast.walk(as_dict))
        if value is not None
    }
    missing_fields = tuple(field for field in REQUIRED_OBSERVATION_FIELDS if field not in literal_fields)
    return {
        "adapter_path": LEGACY_ADAPTER_PATH.as_posix(),
        "owns_hormone_system_reference": owns_hormone_system,
        "iterates_all_hormones": iterates_hormones,
        "reads_current_level": reads_levels,
        "axis_count_readable": 26 if owns_hormone_system and iterates_hormones and reads_levels else 0,
        "required_observation_fields": list(REQUIRED_OBSERVATION_FIELDS),
        "missing_immutable_envelope_fields": list(missing_fields),
        "immutable_source_envelope_complete": not missing_fields,
        "derived_compatibility_keys_are_not_axes": ["stress", "energy", "curiosity"],
        "authority": "legacy_runtime_authoritative_read_surface_only",
    }


def inspect_persistence_container(root: Path = ROOT) -> dict[str, Any]:
    tree = _parse(root, PERSISTENCE_PATH)
    whole_container_reference = False
    axis_specific_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_target_name(target) == "self.hs" for target in targets):
                whole_container_reference = True
        if isinstance(node, ast.Subscript):
            key = _literal_string(node.slice)
            if key is not None:
                axis_specific_keys.add(key)
    return {
        "path": PERSISTENCE_PATH.as_posix(),
        "whole_hormone_system_reference": whole_container_reference,
        "axis_specific_snapshot_keys": sorted(axis_specific_keys),
        "axis_specific_snapshot_contract_found": bool(axis_specific_keys),
        "persistence_authority_changed": False,
    }


def _is_nonproduction(relative: str) -> bool:
    return relative.startswith(AUDIT_PREFIXES)


def scan_registry_usage(
    root: Path,
    registry_axes: Iterable[str],
) -> dict[str, Any]:
    axes = frozenset(registry_axes)
    call_sites: list[dict[str, Any]] = []
    literal_sites: list[dict[str, Any]] = []
    value_store_candidates: list[dict[str, Any]] = []
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
        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node).split(".")[-1] == "affect_hormone_axis_registry":
                call_sites.append(
                    {
                        "path": relative,
                        "line": int(getattr(node, "lineno", 1)),
                        "production": not _is_nonproduction(relative),
                    }
                )
            axis = _literal_string(node)
            if axis not in axes:
                continue
            owner = parent.get(node)
            context = type(owner).__name__ if owner is not None else "Unknown"
            literal_sites.append(
                {
                    "axis": axis,
                    "path": relative,
                    "line": int(getattr(node, "lineno", 1)),
                    "context": context,
                    "production": not _is_nonproduction(relative),
                }
            )
            is_store = False
            if isinstance(owner, ast.Subscript) and owner.slice is node and isinstance(owner.ctx, ast.Store):
                is_store = True
            elif isinstance(owner, ast.Dict):
                for key, value in zip(owner.keys, owner.values):
                    if key is node and not isinstance(value, ast.Constant):
                        is_store = True
                        break
            if is_store and not _is_nonproduction(relative):
                value_store_candidates.append(
                    {
                        "axis": axis,
                        "path": relative,
                        "line": int(getattr(node, "lineno", 1)),
                        "context": context,
                    }
                )
    production_calls = [site for site in call_sites if site["production"]]
    production_literal_sites = [site for site in literal_sites if site["production"]]
    return {
        "registry_factory_call_sites": sorted(call_sites, key=lambda item: (item["path"], item["line"])),
        "production_registry_factory_call_count": len(production_calls),
        "production_axis_literal_site_count": len(production_literal_sites),
        "production_value_store_candidates": sorted(
            value_store_candidates,
            key=lambda item: (item["path"], item["line"], item["axis"]),
        ),
        "production_value_store_candidate_count": len(value_store_candidates),
        "parse_errors": sorted(parse_errors, key=lambda item: item["path"]),
        "observed_value_owner_found": bool(value_store_candidates),
    }


def _canonical_digest(value: Mapping[str, Any]) -> str:
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
    legacy_surface = inspect_legacy_read_surface(root)
    persistence = inspect_persistence_container(root)
    registry_usage = scan_registry_usage(root, (row["axis"] for row in registry_axes))

    blockers: list[str] = []
    if not legacy_surface["immutable_source_envelope_complete"]:
        blockers.append("LEGACY_IMMUTABLE_SOURCE_ENVELOPE_ABSENT")
    if not registry_usage["observed_value_owner_found"]:
        blockers.append("REGISTRY_OBSERVED_VALUE_OWNER_ABSENT")

    errors: list[str] = []
    if tuple(blockers) != EXPECTED_BLOCKERS:
        errors.append(f"unexpected blocker set: {blockers}")
    if registry_usage["parse_errors"]:
        errors.append(f"tracked Python parse errors: {registry_usage['parse_errors']}")
    if persistence["axis_specific_snapshot_contract_found"]:
        errors.append("legacy persistence unexpectedly claims axis-specific snapshot ownership")
    if legacy_surface["axis_count_readable"] != 26:
        errors.append("legacy read surface does not expose all 26 axes")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "m3_b_technical_merge_sha": M3_B_TECHNICAL_MERGE_SHA,
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
        "registry_usage": registry_usage,
        "source_family_readiness": {
            "legacy_mutable_hormone": "READABLE_UNVERSIONED_LEGACY_CONTAINER",
            "read_only_affect_registry": "DEFINITION_ONLY_NO_OBSERVED_VALUE_OWNER",
        },
        "strict_63_axis_observation_ready": False,
        "observation_ready_axis_count": 0,
        "blockers": blockers,
        "next_required_artifacts": [
            "versioned immutable after-the-fact legacy 26-axis source envelope with source identity/schema/integrity digest and no mutation",
            "separately reviewed producer/ownership contract for actual observed values of all 37 registry axes; registry defaults cannot be used as observations",
            "real read-only observation-window packet proving repeated deterministic projection and zero live effects only after both source families are ready",
        ],
        "errors": errors,
    }
    report["report_digest"] = _canonical_digest(report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _write(value: Mapping[str, Any], output: Path | None, *, pretty: bool) -> None:
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
    args = _parse_args(argv)
    report = audit_repository()
    output: Mapping[str, Any]
    if args.summary_only:
        output = {
            "axis_counts": report["axis_counts"],
            "blockers": report["blockers"],
            "errors": report["errors"],
            "m3_b_complete": report["m3_b_complete"],
            "m3_c_open": report["m3_c_open"],
            "observation_window_started": report["observation_window_started"],
            "report_digest": report["report_digest"],
            "source_family_readiness": report["source_family_readiness"],
            "strict_63_axis_observation_ready": report["strict_63_axis_observation_ready"],
        }
    else:
        output = report
    _write(output, args.output, pretty=args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
