#!/usr/bin/env python3
"""Enforce EVE v4.1 current-tree forward-regression governance.

The historical M0 scanners remain snapshot-pinned. This scanner reuses their AST
visitor detector families against both the frozen v4.1 forward baseline and the
current tracked tree, then rejects only additions not registered in the reviewed
forward-additions manifest.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SCHEMA_VERSION = "1.0.0-forward-regression"
MANIFEST_SCHEMA_VERSION = "1.0.0-forward-additions"
CONSTITUTION_BASELINE_SHA = "8cd1a0ad0ed8aaa2810da0730c17b6168bd2fb7b"
DEFAULT_MANIFEST = Path("docs/audit/FORWARD_ADDITIONS_MANIFEST.json")

EXCLUDED_PARTS = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "__pycache__", "build", "dist", "node_modules", "venv",
}
NON_RUNTIME_ROOTS = {".codex", ".github", "docs", "scripts", "tests"}
ADAPTIVE_CLASSIFICATIONS = {
    "ADAPTIVE_OR_LEARNING_METHOD_CANDIDATE",
    "ADAPTIVE_STATE_TRANSITION_CANDIDATE",
    "NUMERIC_OR_LEARNED_STATE_CANDIDATE",
    "VECTOR_OR_NUMERIC_COMPONENT_CANDIDATE",
    "VECTOR_OR_NUMERIC_METHOD_CANDIDATE",
    "VECTOR_OR_VOCAB_ARTIFACT_IO_CANDIDATE",
}
RAW_NAME_TOKENS = {
    "body", "chat", "content", "document", "external", "input", "message",
    "ocr", "payload", "prompt", "query", "raw", "request", "source", "stt",
    "text", "transcript", "utterance",
}
RAW_CALL_TOKENS = {
    "fetch", "input", "listen", "ocr", "read", "receive", "request", "source",
    "stt", "transcribe",
}
EXPRESSION_SINK_TOKENS = {
    "compose", "emit", "express", "expression", "generate", "generator", "output",
    "publish", "render", "reply", "respond", "response", "send", "speak", "speech",
    "stream",
}
REQUIRED_REGISTRATION_FIELDS = {
    "fingerprint", "count", "category", "path", "symbol", "rationale", "owner",
    "disposition", "introduced_by_pr",
}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detector_modules(root: Path) -> tuple[Any, Any, Any]:
    audit = root / "scripts/audit"
    return (
        _load_module(audit / "m0_a_runtime_inventory.py", "_eve_forward_m0_a"),
        _load_module(audit / "m0_b_controlflow_concurrency_inventory.py", "_eve_forward_m0_b"),
        _load_module(audit / "m0_d_component_inventory.py", "_eve_forward_m0_d"),
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _current_sources(root: Path) -> dict[str, str]:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"],
            stderr=subprocess.DEVNULL,
        )
        relative_paths = [
            Path(os.fsdecode(item))
            for item in raw.split(b"\0")
            if item and Path(os.fsdecode(item)).suffix == ".py"
        ]
    except (OSError, subprocess.CalledProcessError):
        relative_paths = [path.relative_to(root) for path in root.rglob("*.py")]
    sources: dict[str, str] = {}
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        path = root / relative
        if path.is_file():
            sources[relative.as_posix()] = _read_text(path)
    return sources


def _snapshot_sources(root: Path, sha: str) -> dict[str, str]:
    try:
        archive = subprocess.check_output(
            ["git", "-C", str(root), "archive", "--format=tar", sha],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read frozen forward baseline {sha}") from exc
    sources: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        for member in handle.getmembers():
            relative = Path(member.name)
            if (
                not member.isfile()
                or relative.suffix != ".py"
                or any(part in EXCLUDED_PARTS for part in relative.parts)
            ):
                continue
            extracted = handle.extractfile(member)
            if extracted is not None:
                sources[relative.as_posix()] = extracted.read().decode(
                    "utf-8", errors="replace"
                )
    return dict(sorted(sources.items()))


def _symbol_tokens(value: str) -> set[str]:
    normalized = "".join(
        character if character.isalnum() else "_" for character in value.lower()
    )
    return {part for part in normalized.split("_") if part}


def _dotted_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return ""


def _iter_function_nodes(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        yield child
        yield from _iter_function_nodes(child)


def _names_in(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            dotted = _dotted_name(child)
            if dotted:
                names.add(dotted)
                names.add(child.attr)
    return names


def _raw_named(name: str) -> bool:
    return bool(_symbol_tokens(name) & RAW_NAME_TOKENS)


def _expression_is_raw(node: ast.AST | None, tainted: set[str]) -> bool:
    if node is None:
        return False
    names = _names_in(node)
    if names & tainted or any(_raw_named(name) for name in names):
        return True
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if _symbol_tokens(_dotted_name(child.func)) & RAW_CALL_TOKENS:
                return True
    return False


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        dotted = _dotted_name(node)
        return {dotted, node.attr} if dotted else {node.attr}
    if isinstance(node, ast.Subscript):
        dotted = _dotted_name(node.value)
        return {dotted} if dotted else set()
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_target_names(item))
        return names
    return set()


def _function_arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    values = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    if node.args.vararg is not None:
        values.append(node.args.vararg)
    if node.args.kwarg is not None:
        values.append(node.args.kwarg)
    return [argument.arg for argument in values]


def _raw_capability_findings(path: str, tree: ast.AST) -> list[dict[str, Any]]:
    if Path(path).parts and Path(path).parts[0] in NON_RUNTIME_ROOTS:
        return []
    findings: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            for child in node.body:
                self.visit(child)
            self.scope.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.scope.append(node.name)
            symbol = ".".join(self.scope)
            tainted = {name for name in _function_arguments(node) if _raw_named(name)}
            assignments: list[tuple[set[str], ast.AST | None]] = []
            nodes = list(_iter_function_nodes(node))
            for child in nodes:
                if isinstance(child, ast.Assign):
                    targets: set[str] = set()
                    for target in child.targets:
                        targets.update(_target_names(target))
                    assignments.append((targets, child.value))
                elif isinstance(child, ast.AnnAssign):
                    assignments.append((_target_names(child.target), child.value))
                elif isinstance(child, ast.NamedExpr):
                    assignments.append((_target_names(child.target), child.value))
            changed = True
            while changed:
                changed = False
                for targets, value in assignments:
                    if targets - tainted and _expression_is_raw(value, tainted):
                        tainted.update(targets)
                        changed = True
            for child in nodes:
                if isinstance(child, ast.Call):
                    target = _dotted_name(child.func)
                    if not (_symbol_tokens(target) & EXPRESSION_SINK_TOKENS):
                        continue
                    arguments = list(child.args) + [
                        keyword.value for keyword in child.keywords
                    ]
                    if not any(
                        _expression_is_raw(argument, tainted) for argument in arguments
                    ):
                        continue
                    findings.append(
                        {
                            "category": "raw_capability",
                            "path": path,
                            "line_start": int(getattr(child, "lineno", 1)),
                            "line_end": int(
                                getattr(child, "end_lineno", getattr(child, "lineno", 1))
                            ),
                            "symbol": symbol,
                            "detector_family": "v4_1_raw_capability",
                            "classification": "RAW_EXTERNAL_TEXT_TO_EXPRESSION_CAPABILITY_CANDIDATE",
                            "evidence": f"raw_to_expression_call={target}",
                            "details": {
                                "sink": target,
                                "tainted_names": sorted(tainted),
                            },
                        }
                    )
                elif isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom)):
                    if not (_symbol_tokens(symbol) & EXPRESSION_SINK_TOKENS):
                        continue
                    value = getattr(child, "value", None)
                    if _expression_is_raw(value, tainted):
                        findings.append(
                            {
                                "category": "raw_capability",
                                "path": path,
                                "line_start": int(getattr(child, "lineno", 1)),
                                "line_end": int(
                                    getattr(
                                        child,
                                        "end_lineno",
                                        getattr(child, "lineno", 1),
                                    )
                                ),
                                "symbol": symbol,
                                "detector_family": "v4_1_raw_capability",
                                "classification": "RAW_EXTERNAL_TEXT_RETURNED_BY_EXPRESSION_CALLABLE_CANDIDATE",
                                "evidence": f"raw_expression_return={type(child).__name__}",
                                "details": {"tainted_names": sorted(tainted)},
                            }
                        )
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    self.visit(child)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

    Visitor().visit(tree)
    return findings


def _normalise_m0_a(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": str(entry["category"]),
        "path": str(entry["path"]),
        "line_start": int(entry["line_start"]),
        "line_end": int(entry["line_end"]),
        "symbol": str(entry["callable"]),
        "detector_family": "m0_a_current_tree",
        "classification": str(entry["manual_classification"]),
        "evidence": str(entry["mechanical_evidence"]),
        "details": entry.get("details", {}),
    }


def _normalise_m0_b(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": "silent_broad",
        "path": str(entry["path"]),
        "line_start": int(entry["line_start"]),
        "line_end": int(entry["line_end"]),
        "symbol": str(entry["callable"]),
        "detector_family": "m0_b_current_tree",
        "classification": str(entry["manual_classification"]),
        "evidence": str(entry["mechanical_evidence"]),
        "details": entry.get("details", {}),
    }


def _normalise_m0_d(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": "adaptive_numeric",
        "path": str(entry["path"]),
        "line_start": int(entry["line_start"]),
        "line_end": int(entry["line_end"]),
        "symbol": str(entry["symbol"]),
        "detector_family": "m0_d_current_tree",
        "classification": str(entry["classification"]),
        "evidence": str(entry["evidence"]),
        "details": entry.get("details", {}),
    }


def scan_sources(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    root = root.resolve()
    m0_a, m0_b, m0_d = _detector_modules(root)
    roots = {
        path.stem if len(path.parts) == 1 else path.parts[0]
        for path in (Path(value) for value in sources)
    }
    findings: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path, source in sorted(sources.items()):
        try:
            tree = ast.parse(source, filename=path, type_comments=True)
        except (SyntaxError, ValueError) as exc:
            parse_errors.append(
                {
                    "path": path,
                    "line": int(getattr(exc, "lineno", 1) or 1),
                    "error": str(exc),
                }
            )
            continue
        visitor_a = m0_a.InventoryVisitor(path, roots)
        visitor_a.visit(tree)
        findings.extend(
            _normalise_m0_a(entry)
            for entry in visitor_a.findings
            if entry["category"] in {"mutation", "direct_write"}
        )
        visitor_b = m0_b.ControlFlowVisitor(path)
        visitor_b.visit(tree)
        findings.extend(
            _normalise_m0_b(entry)
            for entry in visitor_b.findings
            if entry["manual_classification"] == "SILENT_BROAD_EXCEPTION_PATH"
        )
        visitor_d = m0_d.ComponentVisitor(path)
        visitor_d.visit(tree)
        findings.extend(
            _normalise_m0_d(entry)
            for entry in visitor_d.entries
            if entry["classification"] in ADAPTIVE_CLASSIFICATIONS
        )
        findings.extend(_raw_capability_findings(path, tree))
    for finding in findings:
        finding["fingerprint"] = finding_fingerprint(finding)
    findings.sort(
        key=lambda entry: (
            entry["fingerprint"],
            entry["path"],
            entry["line_start"],
            entry["evidence"],
        )
    )
    return {
        "python_files_scanned": len(sources),
        "findings": findings,
        "parse_errors": sorted(
            parse_errors, key=lambda entry: (entry["path"], entry["line"])
        ),
    }


def finding_fingerprint(finding: Mapping[str, Any]) -> str:
    payload = {
        "category": finding["category"],
        "path": finding["path"],
        "symbol": finding["symbol"],
        "detector_family": finding["detector_family"],
        "classification": finding["classification"],
        "evidence": finding["evidence"],
        "details": finding.get("details", {}),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _counter(findings: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(finding["fingerprint"]) for finding in findings)


def _finding_index(
    findings: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        index[str(finding["fingerprint"])].append(dict(finding))
    for values in index.values():
        values.sort(
            key=lambda entry: (entry["path"], entry["line_start"], entry["line_end"])
        )
    return dict(index)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return payload


def _validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if manifest.get("baseline_sha") != CONSTITUTION_BASELINE_SHA:
        errors.append("manifest baseline_sha is not the frozen v4.1 merge SHA")
    baseline = manifest.get("baseline_fingerprints")
    if not isinstance(baseline, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and value > 0
        for key, value in baseline.items()
    ):
        errors.append(
            "baseline_fingerprints must be a fingerprint->positive-count object"
        )
    registrations = manifest.get("registered_additions")
    if not isinstance(registrations, list):
        errors.append("registered_additions must be a list")
        return errors
    seen: set[str] = set()
    for index, registration in enumerate(registrations):
        if not isinstance(registration, dict):
            errors.append(f"registration[{index}] must be an object")
            continue
        missing = REQUIRED_REGISTRATION_FIELDS - set(registration)
        if missing:
            errors.append(
                f"registration[{index}] missing fields: {','.join(sorted(missing))}"
            )
            continue
        fingerprint = registration["fingerprint"]
        if fingerprint in seen:
            errors.append(f"duplicate registration fingerprint: {fingerprint}")
        seen.add(fingerprint)
        if not isinstance(registration["count"], int) or registration["count"] <= 0:
            errors.append(f"registration[{index}] count must be positive")
        for field in (
            "category",
            "path",
            "symbol",
            "rationale",
            "owner",
            "disposition",
        ):
            if not isinstance(registration[field], str) or not registration[field].strip():
                errors.append(f"registration[{index}] {field} must be non-empty")
        if (
            not isinstance(registration["introduced_by_pr"], int)
            or registration["introduced_by_pr"] <= 0
        ):
            errors.append(
                f"registration[{index}] introduced_by_pr must be a positive integer"
            )
    return errors


def evaluate(
    baseline_scan: Mapping[str, Any],
    current_scan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    errors = _validate_manifest(manifest)
    computed_baseline = _counter(baseline_scan["findings"])
    frozen_baseline = Counter(
        {
            str(key): int(value)
            for key, value in manifest.get("baseline_fingerprints", {}).items()
        }
    )
    baseline_drift = {
        "missing": dict(sorted((frozen_baseline - computed_baseline).items())),
        "unexpected": dict(sorted((computed_baseline - frozen_baseline).items())),
    }
    if baseline_drift["missing"] or baseline_drift["unexpected"]:
        errors.append("frozen forward baseline fingerprint drift")
    current = _counter(current_scan["findings"])
    additions = current - frozen_baseline
    representatives = _finding_index(current_scan["findings"])
    registered = Counter()
    stale: list[dict[str, Any]] = []
    metadata_errors: list[str] = []
    for registration in manifest.get("registered_additions", []):
        if not isinstance(registration, dict) or "fingerprint" not in registration:
            continue
        fingerprint = str(registration["fingerprint"])
        count = int(registration.get("count", 0) or 0)
        registered[fingerprint] += count
        candidates = representatives.get(fingerprint, [])
        representative = candidates[0] if candidates else None
        if representative is None or additions.get(fingerprint, 0) < count:
            stale.append(
                {
                    "fingerprint": fingerprint,
                    "registered": count,
                    "actual_addition": additions.get(fingerprint, 0),
                }
            )
            continue
        for field in ("category", "path", "symbol"):
            if registration.get(field) != representative.get(field):
                metadata_errors.append(
                    f"registration metadata mismatch for {fingerprint}: {field}"
                )
    errors.extend(metadata_errors)
    if stale:
        errors.append("stale or over-counted forward registrations")
    unregistered_counter = additions - registered
    over_registered = registered - additions
    if over_registered:
        errors.append("registered count exceeds current addition count")
    unregistered: list[dict[str, Any]] = []
    for fingerprint, count in sorted(unregistered_counter.items()):
        representative = representatives[fingerprint][0]
        unregistered.append(
            {
                "fingerprint": fingerprint,
                "count": count,
                "category": representative["category"],
                "path": representative["path"],
                "symbol": representative["symbol"],
                "classification": representative["classification"],
                "evidence": representative["evidence"],
                "occurrences": representatives[fingerprint][:count],
            }
        )
    if unregistered:
        errors.append("unregistered current-tree additions")
    category_counts = Counter(
        finding["category"] for finding in current_scan["findings"]
    )
    addition_category_counts = Counter()
    for fingerprint, count in additions.items():
        if representatives.get(fingerprint):
            addition_category_counts[
                representatives[fingerprint][0]["category"]
            ] += count
    return {
        "pass": not errors,
        "errors": errors,
        "summary": {
            "baseline_python_files": baseline_scan["python_files_scanned"],
            "current_python_files": current_scan["python_files_scanned"],
            "baseline_findings": sum(computed_baseline.values()),
            "current_findings": sum(current.values()),
            "current_category_counts": dict(sorted(category_counts.items())),
            "addition_counts": dict(sorted(addition_category_counts.items())),
            "registered_addition_occurrences": sum(registered.values()),
            "unregistered_addition_occurrences": sum(unregistered_counter.values()),
            "stale_registration_count": len(stale),
            "baseline_parse_errors": len(baseline_scan["parse_errors"]),
            "current_parse_errors": len(current_scan["parse_errors"]),
        },
        "baseline_drift": baseline_drift,
        "unregistered_additions": unregistered,
        "stale_registrations": stale,
        "parse_errors": {
            "baseline": baseline_scan["parse_errors"],
            "current": current_scan["parse_errors"],
        },
    }


def suggested_manifest(
    baseline_scan: Mapping[str, Any],
    current_scan: Mapping[str, Any],
    *,
    introduced_by_pr: int,
) -> dict[str, Any]:
    baseline = _counter(baseline_scan["findings"])
    current = _counter(current_scan["findings"])
    additions = current - baseline
    index = _finding_index(current_scan["findings"])
    registrations: list[dict[str, Any]] = []
    for fingerprint, count in sorted(additions.items()):
        representative = index[fingerprint][0]
        registrations.append(
            {
                "fingerprint": fingerprint,
                "count": count,
                "category": representative["category"],
                "path": representative["path"],
                "symbol": representative["symbol"],
                "rationale": (
                    "Forward-regression scanner infrastructure introduced by "
                    "the same reviewed PR."
                ),
                "owner": "forward-regression infrastructure",
                "disposition": "AUDIT_TOOLING",
                "introduced_by_pr": introduced_by_pr,
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "baseline_sha": CONSTITUTION_BASELINE_SHA,
        "baseline_fingerprints": dict(sorted(baseline.items())),
        "registered_additions": registrations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--suggest-manifest-for-pr", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    manifest_path = args.manifest or (root / DEFAULT_MANIFEST)
    baseline_sources = _snapshot_sources(root, CONSTITUTION_BASELINE_SHA)
    current_sources = _current_sources(root)
    baseline_scan = scan_sources(root, baseline_sources)
    current_scan = scan_sources(root, current_sources)
    if args.suggest_manifest_for_pr is not None:
        payload: dict[str, Any] = suggested_manifest(
            baseline_scan,
            current_scan,
            introduced_by_pr=args.suggest_manifest_for_pr,
        )
        exit_code = 0
    else:
        manifest = _load_manifest(manifest_path)
        result = evaluate(baseline_scan, current_scan, manifest)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "baseline_sha": CONSTITUTION_BASELINE_SHA,
            "manifest": (
                manifest_path.relative_to(root).as_posix()
                if manifest_path.is_relative_to(root)
                else manifest_path.as_posix()
            ),
            **result,
        }
        exit_code = 0 if result["pass"] or args.report_only else 1
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
