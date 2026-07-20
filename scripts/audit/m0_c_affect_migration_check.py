#!/usr/bin/env python3
"""Audit coverage for the EVE M0-C affect migration plan.

The command performs static analysis only. It extracts the legacy mutable
hormone axes, the read-only affect-axis registry, derived compatibility keys,
and persistence-container evidence. It then parses the plan mapping table and
emits canonical JSON to stdout.

No production module is imported, no runtime is executed, no state is read or
written, and no migration or projection code is implemented.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "1.0.0-m0-c-affect-migration"
BASELINE_SHA = "28ec113a8ee371fdc6ac13341c0d70e00db26ce4"
PLAN_PATH = Path("docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md")
LEGACY_PATH = Path("hormone_system.py")
REGISTRY_PATH = Path("adapters/affect_hormone_neural_rhythm_registry.py")
ADAPTER_PATH = Path("adapters/hormone_adapter.py")
PERSISTENCE_ADAPTER_PATH = Path("adapters/persistence_adapter.py")

ALLOWED_DRIVES = {
    "energy", "safety", "affiliation", "curiosity",
    "agency", "coherence", "competence", "expression",
}
ALLOWED_STATUSES = {"MAPPED", "PROPOSED-DROP", "UNRESOLVED"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
REQUIRED_SECTIONS = (
    "## Mechanical axis enumeration",
    "## Target-state mapping table",
    "## Phased migration",
    "## Backward-compatibility projection",
    "## Event and snapshot conversion boundary",
    "## Rollback",
    "## Identity and memory continuity",
    "## Acceptance criteria",
    "## Explicit non-goals",
    "## M0 cross-references and conceptual module dispositions",
)
TABLE_BEGIN = "<!-- BEGIN AFFECT AXIS MAPPING TABLE -->"
TABLE_END = "<!-- END AFFECT AXIS MAPPING TABLE -->"


def _node_end(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _target_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


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


def _parse(root: Path, relative: Path) -> ast.AST:
    path = root / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=relative.as_posix())


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_legacy_axes(root: Path) -> list[dict[str, Any]]:
    tree = _parse(root, LEGACY_PATH)
    axes: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "_init_all_hormones":
            continue
        for child in ast.walk(node):
            value: ast.AST | None = None
            if isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "config"
                for target in child.targets
            ):
                value = child.value
            elif (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and child.target.id == "config"
            ):
                value = child.value
            if not isinstance(value, (ast.List, ast.Tuple)):
                continue
            for item in value.elts:
                if not isinstance(item, (ast.Tuple, ast.List)) or not item.elts:
                    continue
                axis = _literal_string(item.elts[0])
                if axis is None:
                    continue
                axes.append({
                    "axis": axis,
                    "source_family": "legacy_mutable_hormone",
                    "path": LEGACY_PATH.as_posix(),
                    "line_start": int(getattr(item.elts[0], "lineno", 1)),
                    "line_end": _node_end(item.elts[0]),
                    "callable": "HormoneSystem._init_all_hormones",
                    "mechanical_evidence": "first string element of config tuple",
                })
    axes.sort(key=lambda item: item["axis"])
    if not axes:
        raise AssertionError("no legacy axes found")
    return axes


def _extract_registry_axes(root: Path) -> list[dict[str, Any]]:
    tree = _parse(root, REGISTRY_PATH)
    axes: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        target = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1:
                target = _target_name(node.targets[0])
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = _target_name(node.target)
            value = node.value
        if target != "AXIS_GROUPS" or not isinstance(value, ast.Dict):
            continue
        for group_node, axes_node in zip(value.keys, value.values):
            group = _literal_string(group_node)
            if group is None or not isinstance(axes_node, (ast.Tuple, ast.List)):
                continue
            for axis_node in axes_node.elts:
                axis = _literal_string(axis_node)
                if axis is None:
                    continue
                axes.append({
                    "axis": axis,
                    "source_family": "read_only_affect_registry",
                    "group": group,
                    "path": REGISTRY_PATH.as_posix(),
                    "line_start": int(getattr(axis_node, "lineno", 1)),
                    "line_end": _node_end(axis_node),
                    "callable": "<module>.AXIS_GROUPS",
                    "mechanical_evidence": f"string member of AXIS_GROUPS[{group!r}]",
                })
    axes.sort(key=lambda item: item["axis"])
    if not axes:
        raise AssertionError("no registry axes found")
    return axes


def _dict_keys_in_function(tree: ast.AST, function_name: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                for key in child.keys:
                    value = _literal_string(key)
                    if value is not None:
                        findings.append({
                            "key": value,
                            "line": int(getattr(key, "lineno", 1)),
                            "detector": "ast.Dict string key",
                        })
            elif isinstance(child, ast.Subscript) and isinstance(child.ctx, ast.Store):
                value = _literal_string(child.slice)
                if value is not None:
                    findings.append({
                        "key": value,
                        "line": int(getattr(child.slice, "lineno", 1)),
                        "detector": "ast.Subscript stored string key",
                    })
    return findings


def _extract_compatibility_keys(root: Path) -> list[dict[str, Any]]:
    adapter_tree = _parse(root, ADAPTER_PATH)
    legacy_tree = _parse(root, LEGACY_PATH)
    findings: list[dict[str, Any]] = []
    for item in _dict_keys_in_function(adapter_tree, "as_dict"):
        findings.append({
            "key": item["key"],
            "kind": "adapter_derived_compatibility_key",
            "path": ADAPTER_PATH.as_posix(),
            "line_start": item["line"],
            "line_end": item["line"],
            "callable": "HormoneAdapter.as_dict",
            "mechanical_evidence": item["detector"],
        })
    for item in _dict_keys_in_function(legacy_tree, "compute_mood"):
        findings.append({
            "key": item["key"],
            "kind": "derived_mood_projection_key",
            "path": LEGACY_PATH.as_posix(),
            "line_start": item["line"],
            "line_end": item["line"],
            "callable": "HormoneSystem.compute_mood",
            "mechanical_evidence": item["detector"],
        })
    dedup = {
        (item["key"], item["kind"], item["path"], item["line_start"]): item
        for item in findings
    }
    return sorted(dedup.values(), key=lambda item: (item["kind"], item["key"], item["line_start"]))


def _extract_persistence_container_evidence(root: Path) -> list[dict[str, Any]]:
    tree = _parse(root, PERSISTENCE_ADAPTER_PATH)
    findings: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _target_name(target) == "self.hs":
                    findings.append({
                        "path": PERSISTENCE_ADAPTER_PATH.as_posix(),
                        "line_start": int(getattr(node, "lineno", 1)),
                        "line_end": _node_end(node),
                        "callable": "_MockEve.__init__",
                        "mechanical_evidence": "legacy persistence receives the whole HormoneSystem as self.hs",
                        "axis_specific_snapshot_keys_found": False,
                    })
    return sorted(findings, key=lambda item: (item["path"], item["line_start"]))


def _split_cell(value: str) -> list[str]:
    value = value.strip()
    if value in {"", "—", "-"}:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def parse_mapping_table(plan_text: str) -> list[dict[str, Any]]:
    if TABLE_BEGIN not in plan_text or TABLE_END not in plan_text:
        raise AssertionError("mapping table markers missing")
    block = plan_text.split(TABLE_BEGIN, 1)[1].split(TABLE_END, 1)[0]
    rows: list[dict[str, Any]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] in {"Axis", "---"} or set(cells[0]) == {"-"}:
            continue
        if len(cells) != 12:
            raise AssertionError(f"mapping row has {len(cells)} cells: {line}")
        (
            axis, source_family, status, target_drives, appraisal_dimensions,
            derived_emotion, projection_sketch, rationale, confidence,
            evidence, preservation, open_question,
        ) = cells
        rows.append({
            "axis": axis.strip("`"),
            "source_family": source_family.strip("`"),
            "status": status.strip("`"),
            "target_drives": _split_cell(target_drives.strip("`")),
            "appraisal_dimensions": _split_cell(appraisal_dimensions.strip("`")),
            "derived_emotion": _split_cell(derived_emotion.strip("`")),
            "projection_sketch": projection_sketch,
            "rationale": rationale,
            "confidence": confidence.strip("`"),
            "evidence": evidence.strip("`"),
            "preservation": preservation,
            "open_question": open_question,
        })
    return rows


def _validate_plan(
    plan_text: str,
    axes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in plan_text:
            errors.append(f"missing section: {section}")
    for required in ("1,777", "54", "43"):
        if required not in plan_text:
            errors.append(f"missing M0-C evidence count: {required}")

    expected = {entry["axis"]: entry for entry in axes}
    counts = Counter(row["axis"] for row in rows)
    duplicates = sorted(axis for axis, count in counts.items() if count != 1)
    missing = sorted(set(expected) - set(counts))
    extra = sorted(set(counts) - set(expected))
    if duplicates:
        errors.append(f"duplicate mapping rows: {duplicates}")
    if missing:
        errors.append(f"missing mapping rows: {missing}")
    if extra:
        errors.append(f"extra mapping rows: {extra}")

    for row in rows:
        axis = row["axis"]
        source = expected.get(axis)
        if source is None:
            continue
        if row["source_family"] != source["source_family"]:
            errors.append(f"{axis}: source family mismatch")
        if row["status"] not in ALLOWED_STATUSES:
            errors.append(f"{axis}: invalid status {row['status']}")
        if row["confidence"] not in ALLOWED_CONFIDENCE:
            errors.append(f"{axis}: invalid confidence {row['confidence']}")
        invalid_drives = sorted(set(row["target_drives"]) - ALLOWED_DRIVES)
        if invalid_drives:
            errors.append(f"{axis}: invalid target drives {invalid_drives}")
        if row["status"] == "MAPPED" and not (
            row["target_drives"] or row["appraisal_dimensions"] or row["derived_emotion"]
        ):
            errors.append(f"{axis}: mapped row has no target")
        if row["status"] == "PROPOSED-DROP":
            if row["target_drives"] or row["appraisal_dimensions"] or row["derived_emotion"]:
                errors.append(f"{axis}: drop row still has target semantics")
            if not row["preservation"] or row["preservation"] == "—":
                errors.append(f"{axis}: drop row lacks preservation note")
        if row["status"] == "UNRESOLVED" and (
            not row["open_question"] or row["open_question"] == "—"
        ):
            errors.append(f"{axis}: unresolved row lacks open question")
        evidence_token = f"{source['path']}:{source['line_start']}"
        if evidence_token not in row["evidence"]:
            errors.append(f"{axis}: evidence does not cite {evidence_token}")
        if not row["rationale"] or row["rationale"] == "—":
            errors.append(f"{axis}: rationale missing")
    return errors


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    legacy_axes = _extract_legacy_axes(root)
    registry_axes = _extract_registry_axes(root)
    axes = sorted(legacy_axes + registry_axes, key=lambda item: item["axis"])
    names = [entry["axis"] for entry in axes]
    if len(names) != len(set(names)):
        duplicates = sorted(axis for axis, count in Counter(names).items() if count > 1)
        raise AssertionError(f"axis names overlap across source families: {duplicates}")

    plan_path = root / PLAN_PATH
    plan_text = plan_path.read_text(encoding="utf-8")
    rows = parse_mapping_table(plan_text)
    errors = _validate_plan(plan_text, axes, rows)
    row_by_axis = {row["axis"]: row for row in rows}
    axis_records = []
    for entry in axes:
        record = dict(entry)
        record["mapping"] = row_by_axis.get(entry["axis"])
        axis_records.append(record)

    status_counts = Counter(row["status"] for row in rows)
    source_counts = Counter(entry["source_family"] for entry in axes)
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "root": ".",
        "summary": {
            "legacy_mutable_hormone_axes": source_counts.get("legacy_mutable_hormone", 0),
            "read_only_affect_registry_axes": source_counts.get("read_only_affect_registry", 0),
            "authoritative_found_axes": len(axes),
            "mapping_rows": len(rows),
            "mapped": status_counts.get("MAPPED", 0),
            "proposed_drop": status_counts.get("PROPOSED-DROP", 0),
            "unresolved": status_counts.get("UNRESOLVED", 0),
            "compatibility_keys": len(_extract_compatibility_keys(root)),
            "persistence_container_evidence": len(_extract_persistence_container_evidence(root)),
            "validation_errors": len(errors),
        },
        "axes": axis_records,
        "compatibility_keys": _extract_compatibility_keys(root),
        "persistence_container_evidence": _extract_persistence_container_evidence(root),
        "validation_errors": errors,
        "scope": {
            "static_analysis_only": True,
            "runtime_import_performed": False,
            "runtime_execution_performed": False,
            "migration_execution_performed": False,
            "projection_implementation_performed": False,
            "production_state_read_performed": False,
            "production_state_write_performed": False,
            "production_source_mutation_performed": False,
            "generated_json_committed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = audit_repository(args.root)
    if report["validation_errors"]:
        for error in report["validation_errors"]:
            print(error, file=sys.stderr)
        return 2
    if args.fail_on_unresolved and report["summary"]["unresolved"]:
        print("unresolved mappings remain", file=sys.stderr)
        return 3
    payload: Any = report["summary"] if args.summary_only else report
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
