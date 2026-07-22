#!/usr/bin/env python3
"""Extract and validate EVE M2-B raw-text capability candidates.

This module is audit-only: it parses tracked Python source, executes no runtime
module, grants no capability, and cannot mark itself human-accepted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = "eve.m2-b-read-capability-candidates.v1"
DECISION_SCHEMA_VERSION = "eve.m2-b-read-capability-decisions.v1"
AUTHORITY = "audit_only"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
MAX_CALL_DEPTH = 12
EXCLUDED = {".git", ".venv", "venv", "__pycache__", "build", "dist", "node_modules"}
NON_RUNTIME = {".codex", ".github", "docs", "scripts", "tests"}
RAW_NAMES = {
    "body", "chat", "content", "document", "external", "input", "message",
    "ocr", "payload", "prompt", "query", "raw", "request", "source", "stt",
    "text", "transcript", "utterance",
}
RAW_CALLS = {"fetch", "get_json", "input", "listen", "ocr", "read", "read_text", "receive", "request", "source", "stt", "transcribe"}
SINK_CALLS = {"compose", "emit", "express", "expression", "generate", "generator", "output", "publish", "render", "reply", "respond", "response", "send", "speak", "speech", "stream", "yield"}
MARKERS = {
    "provenance": {"confidence", "origin", "provenance", "source_id", "source_type", "verification", "version"},
    "quarantine": {"canonical", "claim", "meaning", "normalize", "parse", "quarantine", "sanitize", "semantic", "understand", "validate"},
    "quotation": {"attribute", "attribution", "cite", "quote", "quotation"},
    "denial": {"allow", "authorize", "block", "deny", "fail", "forbid", "permission", "raise", "reject"},
}
DECISIONS = {"APPROVED_QUARANTINED", "DENIED_NO_CAPABILITY", "LEGACY_REWRITE", "NOT_RAW_TEXT_FALSE_POSITIVE"}
DECISION_FIELDS = {"edge_id", "decision", "capability", "provenance", "quarantine", "quotation", "denial_semantics", "rationale", "owner"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    text = value if isinstance(value, str) else canonical(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokens(value: str) -> set[str]:
    normalized = "".join(c if c.isalnum() else "_" for c in value.lower())
    return {part for part in normalized.split("_") if part}


def dotted(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted(node.func)
    return ""


def function_nodes(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from function_nodes(child)


def arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    values = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    if node.args.vararg:
        values.append(node.args.vararg)
    if node.args.kwarg:
        values.append(node.args.kwarg)
    return [value.arg for value in values]


def tracked_sources(root: Path) -> dict[str, str]:
    try:
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.DEVNULL)
        paths = [Path(os.fsdecode(item)) for item in raw.split(b"\0") if item and Path(os.fsdecode(item)).suffix == ".py"]
    except (OSError, subprocess.CalledProcessError):
        paths = [path.relative_to(root) for path in root.rglob("*.py")]
    result = {}
    for relative in sorted(paths, key=lambda value: value.as_posix()):
        if any(part in EXCLUDED for part in relative.parts):
            continue
        path = root / relative
        if path.is_file():
            result[relative.as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return result


def module_name(path: str) -> str:
    value = Path(path).with_suffix("").as_posix().replace("/", ".")
    return value[:-9] if value.endswith(".__init__") else value


def import_aliases(tree: ast.Module) -> dict[str, str]:
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                result[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return result


def marker_evidence(nodes: list[ast.AST]) -> dict[str, list[str]]:
    names = set()
    for node in nodes:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.update({node.attr, dotted(node)})
        elif isinstance(node, ast.Call):
            names.add(dotted(node.func))
    return {kind: sorted(name for name in names if tokens(name) & expected) for kind, expected in MARKERS.items()}


def extract_functions(sources: Mapping[str, str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    functions: dict[str, dict[str, Any]] = {}
    errors = []
    for path, source in sorted(sources.items()):
        try:
            tree = ast.parse(source, filename=path, type_comments=True)
        except (SyntaxError, ValueError) as exc:
            errors.append({"path": path, "line": int(getattr(exc, "lineno", 1) or 1), "error": str(getattr(exc, "msg", None) or exc)})
            continue
        module = module_name(path)
        aliases = import_aliases(tree)

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.scope.append(node.name)
                for child in node.body:
                    self.visit(child)
                self.scope.pop()

            def add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                self.scope.append(node.name)
                qualname = ".".join(self.scope)
                key = f"{module}:{qualname}"
                nodes = list(function_nodes(node))
                sources_found = [
                    {"kind": "raw_parameter", "evidence": name, "line_start": node.lineno, "line_end": node.lineno}
                    for name in arguments(node) if tokens(name) & RAW_NAMES
                ]
                calls, sinks = [], []
                for child in nodes:
                    if isinstance(child, ast.Call):
                        target = dotted(child.func)
                        item = {"target": target, "line_start": child.lineno, "line_end": int(getattr(child, "end_lineno", child.lineno))}
                        calls.append(item)
                        if tokens(target) & RAW_CALLS:
                            sources_found.append({"kind": "raw_source_call", "evidence": target, "line_start": item["line_start"], "line_end": item["line_end"]})
                        if tokens(target) & SINK_CALLS:
                            sinks.append({"kind": "expression_call", "evidence": target, "line_start": item["line_start"], "line_end": item["line_end"]})
                    elif isinstance(child, (ast.Yield, ast.YieldFrom)) and tokens(qualname) & SINK_CALLS:
                        sinks.append({"kind": "stream_yield", "evidence": type(child).__name__, "line_start": child.lineno, "line_end": int(getattr(child, "end_lineno", child.lineno))})
                    elif isinstance(child, ast.Return) and child.value is not None and tokens(qualname) & SINK_CALLS:
                        sinks.append({"kind": "expression_return", "evidence": "Return", "line_start": child.lineno, "line_end": int(getattr(child, "end_lineno", child.lineno))})
                functions[key] = {
                    "module": module, "path": path, "qualname": qualname,
                    "source_evidence": sorted(sources_found, key=lambda item: (item["line_start"], item["kind"], item["evidence"])),
                    "sink_evidence": sorted(sinks, key=lambda item: (item["line_start"], item["kind"], item["evidence"])),
                    "calls": sorted(calls, key=lambda item: (item["line_start"], item["target"])),
                    "aliases": dict(sorted(aliases.items())), "markers": marker_evidence(nodes),
                }
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.visit(child)
                self.scope.pop()

            visit_FunctionDef = add_function
            visit_AsyncFunctionDef = add_function

        Visitor().visit(tree)
    return dict(sorted(functions.items())), sorted(errors, key=lambda item: (item["path"], item["line"]))


def resolve(info: Mapping[str, Any], target: str, functions: Mapping[str, Any], leaf_index: Mapping[str, list[str]]) -> str | None:
    module, qualname = str(info["module"]), str(info["qualname"])
    class_prefix = qualname.rsplit(".", 1)[0] if "." in qualname else ""
    candidates = []
    if target.startswith(("self.", "cls.")) and class_prefix:
        candidates.append(f"{module}:{class_prefix}.{target.split('.', 1)[1]}")
    if "." not in target:
        candidates.extend([f"{module}:{target}", f"{module}:{class_prefix}.{target}" if class_prefix else ""])
    head, _, tail = target.partition(".")
    alias = info.get("aliases", {}).get(head)
    if alias:
        expanded = f"{alias}.{tail}" if tail else alias
        module_part, _, symbol = expanded.rpartition(".")
        if module_part:
            candidates.append(f"{module_part}:{symbol}")
    module_part, _, symbol = target.rpartition(".")
    if module_part:
        candidates.append(f"{module_part}:{symbol}")
    for candidate in candidates:
        if candidate in functions:
            return candidate
    matches = leaf_index.get(target.rsplit(".", 1)[-1], [])
    return matches[0] if len(matches) == 1 else None


def build_graph(functions: Mapping[str, dict[str, Any]]) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    leaf_index: dict[str, list[str]] = defaultdict(list)
    for key, info in functions.items():
        leaf_index[info["qualname"].rsplit(".", 1)[-1]].append(key)
    graph, unresolved = {}, []
    for key, info in functions.items():
        edges = set()
        for call in info["calls"]:
            target = call["target"]
            destination = resolve(info, target, functions, leaf_index)
            if destination:
                edges.add(destination)
            elif tokens(target) & (RAW_CALLS | SINK_CALLS):
                material = {"caller": key, "path": info["path"], **call}
                unresolved.append({"finding_id": digest(material), **material, "review_status": REVIEW_REQUIRED})
        graph[key] = tuple(sorted(edges))
    return graph, sorted(unresolved, key=lambda item: item["finding_id"])


def shortest_path(graph: Mapping[str, tuple[str, ...]], source: str, sink: str) -> tuple[str, ...] | None:
    queue = deque([(source, (source,))])
    seen = {source}
    while queue:
        current, path = queue.popleft()
        if current == sink:
            return path
        if len(path) >= MAX_CALL_DEPTH:
            continue
        for next_key in graph.get(current, ()):
            if next_key not in seen:
                seen.add(next_key)
                queue.append((next_key, path + (next_key,)))
    return None


def merged_markers(path: tuple[str, ...], functions: Mapping[str, dict[str, Any]]) -> dict[str, list[str]]:
    result = {kind: set() for kind in MARKERS}
    for key in path:
        for kind, values in functions[key]["markers"].items():
            result[kind].update(values)
    return {kind: sorted(values) for kind, values in result.items()}


def extract_candidates(root: Path) -> dict[str, Any]:
    sources = tracked_sources(root)
    functions, parse_errors = extract_functions(sources)
    functions = {key: info for key, info in functions.items() if Path(info["path"]).parts[0] not in NON_RUNTIME}
    graph, unresolved = build_graph(functions)
    source_keys = sorted(key for key, info in functions.items() if info["source_evidence"])
    sink_keys = sorted(key for key, info in functions.items() if info["sink_evidence"])
    candidates = []
    for source_key in source_keys:
        for sink_key in sink_keys:
            path = shortest_path(graph, source_key, sink_key)
            if not path:
                continue
            source, sink = functions[source_key], functions[sink_key]
            material = {"source": source_key, "source_evidence": source["source_evidence"], "sink": sink_key, "sink_evidence": sink["sink_evidence"], "call_path": list(path)}
            candidates.append({
                "edge_id": digest(material),
                "source": {"path": source["path"], "symbol": source["qualname"], "evidence": source["source_evidence"]},
                "cognition_hops": [{"path": functions[key]["path"], "symbol": functions[key]["qualname"]} for key in path[1:-1]],
                "sink": {"path": sink["path"], "symbol": sink["qualname"], "evidence": sink["sink_evidence"]},
                "call_path": list(path), "semantic_markers": merged_markers(path, functions),
                "mechanical_confidence": "high" if len(path) == 1 else "medium", "review_status": REVIEW_REQUIRED,
            })
    candidates = list({item["edge_id"]: item for item in candidates}.values())
    candidates.sort(key=lambda item: item["edge_id"])
    report = {
        "schema_version": SCHEMA_VERSION, "authority": AUTHORITY, "human_accepted": False, "runtime_integrated": False,
        "python_files_scanned": len(sources), "function_count": len(functions),
        "candidate_edges": candidates, "unresolved_boundary_calls": unresolved, "parse_errors": parse_errors,
        "summary": {"candidate_edge_count": len(candidates), "unresolved_boundary_call_count": len(unresolved), "parse_error_count": len(parse_errors), "source_callable_count": len(source_keys), "sink_callable_count": len(sink_keys)},
    }
    report["report_digest"] = digest(report)
    return report


def reviewed(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value != REVIEW_REQUIRED


def validate_decisions(report: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        errors.append("decision schema_version mismatch")
    if payload.get("candidate_report_digest") != report.get("report_digest"):
        errors.append("candidate_report_digest mismatch")
    expected = {item["edge_id"] for item in report["candidate_edges"]}
    seen = set()
    decisions = payload.get("edge_decisions", [])
    if not isinstance(decisions, list):
        decisions, errors = [], errors + ["edge_decisions must be a list"]
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"edge_decisions[{index}] must be an object")
            continue
        missing = DECISION_FIELDS - set(decision)
        if missing:
            errors.append(f"edge_decisions[{index}] missing fields: {','.join(sorted(missing))}")
            continue
        edge_id = decision["edge_id"]
        if edge_id in seen:
            errors.append(f"duplicate edge decision: {edge_id}")
        seen.add(edge_id)
        if edge_id not in expected:
            errors.append(f"stale edge decision: {edge_id}")
        if decision["decision"] not in DECISIONS:
            errors.append(f"invalid edge decision: {edge_id}")
        for field in DECISION_FIELDS - {"edge_id", "decision"}:
            if not reviewed(decision.get(field)):
                errors.append(f"edge {edge_id} field {field} requires review")
    if expected - seen:
        errors.append(f"missing edge decisions: {len(expected - seen)}")
    expected_calls = {item["finding_id"] for item in report["unresolved_boundary_calls"]}
    seen_calls = set()
    call_decisions = payload.get("unresolved_call_decisions", [])
    if not isinstance(call_decisions, list):
        call_decisions, errors = [], errors + ["unresolved_call_decisions must be a list"]
    for index, decision in enumerate(call_decisions):
        if not isinstance(decision, dict) or not isinstance(decision.get("finding_id"), str):
            errors.append(f"unresolved_call_decisions[{index}] is malformed")
            continue
        finding_id = decision["finding_id"]
        if finding_id in seen_calls:
            errors.append(f"duplicate unresolved decision: {finding_id}")
        seen_calls.add(finding_id)
        if finding_id not in expected_calls:
            errors.append(f"stale unresolved decision: {finding_id}")
        for field in ("decision", "rationale", "owner", "denial_semantics"):
            if not reviewed(decision.get(field)):
                errors.append(f"unresolved {finding_id} field {field} requires review")
    if expected_calls - seen_calls:
        errors.append(f"missing unresolved call decisions: {len(expected_calls - seen_calls)}")
    if report["parse_errors"]:
        errors.append("candidate report contains parse errors")
    result = {"schema_version": DECISION_SCHEMA_VERSION, "candidate_report_digest": report["report_digest"], "valid": not errors, "errors": errors, "edge_decision_count": len(seen), "unresolved_call_decision_count": len(seen_calls), "eligible_for_human_review": not errors, "human_accepted": False, "authority": AUTHORITY}
    result["validation_digest"] = digest(result)
    return result


def build_output(root: Path, decisions: Path | None = None) -> dict[str, Any]:
    report = extract_candidates(root)
    if decisions is None:
        report["decision_validation"] = {"status": "not_requested", "eligible_for_human_review": False, "human_accepted": False, "authority": AUTHORITY}
    else:
        payload = json.loads(decisions.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("decision file root must be an object")
        report["decision_validation"] = validate_decisions(report, payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    report = build_output(args.root.resolve(), args.decisions)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return int(args.strict and not report["decision_validation"].get("eligible_for_human_review", False))


if __name__ == "__main__":
    raise SystemExit(main())
