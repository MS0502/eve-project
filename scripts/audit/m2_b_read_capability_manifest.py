#!/usr/bin/env python3
"""Extract and validate EVE M2-B raw-text read-capability candidates.

The extractor is audit-only. It parses tracked Python source without importing or
executing runtime modules, follows only concrete tainted argument flows, and can
never grant runtime capability or human acceptance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

SCHEMA_VERSION = "eve.m2-b-read-capability-candidates.v2"
DECISION_SCHEMA_VERSION = "eve.m2-b-read-capability-decisions.v2"
AUTHORITY = "audit_only"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
MAX_CALL_DEPTH = 16
EXCLUDED = {".git", ".venv", "venv", "__pycache__", "build", "dist", "node_modules"}
NON_RUNTIME = {".codex", ".github", "docs", "scripts", "tests"}
RAW_NAME_TOKENS = {
    "body", "chat", "content", "document", "external", "input", "message",
    "ocr", "payload", "prompt", "query", "raw", "request", "source", "stt",
    "text", "transcript", "utterance",
}
RAW_SOURCE_LEAVES = {
    "fetch", "get_json", "input", "listen", "ocr", "read", "read_bytes",
    "read_text", "readline", "receive", "request", "stt", "transcribe",
    "urlopen",
}
SINK_TOKENS = {
    "chat", "emit", "express", "expression", "generate", "output", "publish",
    "render", "reply", "respond", "response", "send", "speak", "speech", "stream",
}
SINK_EXCLUSIONS = {
    "analyze", "candidate", "candidates", "classify", "count", "decide", "digest",
    "graph", "length", "meaning", "metric", "observe", "plan", "policy", "record",
    "score", "scoring", "state", "status", "trace", "validate",
}
EXACT_EXTERNAL_SINKS = {"print", "send", "send_text", "speak", "write", "write_text"}
MARKERS = {
    "provenance": {"confidence", "origin", "provenance", "source_id", "source_type", "verification", "version"},
    "quarantine": {"canonical", "claim", "meaning", "normalize", "parse", "quarantine", "sanitize", "semantic", "understand", "validate"},
    "quotation": {"attribute", "attribution", "cite", "quote", "quotation"},
    "denial": {"allow", "authorize", "block", "deny", "fail", "forbid", "permission", "raise", "reject"},
}
DECISIONS = {
    "APPROVED_QUARANTINED", "DENIED_NO_CAPABILITY", "LEGACY_REWRITE",
    "NOT_RAW_TEXT_FALSE_POSITIVE",
}
DECISION_FIELDS = {
    "edge_id", "decision", "capability", "provenance", "quarantine", "quotation",
    "denial_semantics", "rationale", "owner",
}
REVIEW_FIELDS = {"decision", "rationale", "owner", "denial_semantics"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    text = value if isinstance(value, str) else canonical(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokens(value: str) -> set[str]:
    normalized = "".join(character if character.isalnum() else "_" for character in value.lower())
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


def arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    values = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    if node.args.vararg is not None:
        values.append(node.args.vararg)
    if node.args.kwarg is not None:
        values.append(node.args.kwarg)
    return [value.arg for value in values]


def tracked_sources(root: Path) -> dict[str, str]:
    try:
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.DEVNULL)
        paths = [
            Path(os.fsdecode(item)) for item in raw.split(b"\0")
            if item and Path(os.fsdecode(item)).suffix == ".py"
        ]
    except (OSError, subprocess.CalledProcessError):
        paths = [path.relative_to(root) for path in root.rglob("*.py")]
    result: dict[str, str] = {}
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
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                result[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return result


def iter_local_nodes(node: ast.AST) -> Iterator[ast.AST]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from iter_local_nodes(child)


def marker_evidence(node: ast.AST) -> dict[str, list[str]]:
    names: set[str] = set()
    for child in iter_local_nodes(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.update({child.attr, dotted(child)})
        elif isinstance(child, ast.Call):
            names.add(dotted(child.func))
    return {
        kind: sorted(name for name in names if tokens(name) & expected)
        for kind, expected in MARKERS.items()
    }


def raw_parameter(name: str) -> bool:
    return bool(tokens(name) & RAW_NAME_TOKENS) and name not in {"self", "cls"}


def raw_source_call(target: str) -> bool:
    leaf = target.rsplit(".", 1)[-1].lower()
    if leaf not in RAW_SOURCE_LEAVES:
        return False
    if leaf == "read":
        receiver = target.rsplit(".", 1)[0] if "." in target else ""
        return not receiver or bool(tokens(receiver) & {"file", "input", "request", "response", "socket", "stdin", "stream"})
    return True


def sink_name(value: str) -> bool:
    parts = tokens(value.rsplit(":", 1)[-1])
    return bool(parts & SINK_TOKENS) and not bool(parts & SINK_EXCLUSIONS)


def external_sink(target: str) -> bool:
    leaf = target.rsplit(".", 1)[-1].lower()
    return leaf in EXACT_EXTERNAL_SINKS or sink_name(target)


@dataclass(frozen=True)
class FunctionInfo:
    key: str
    module: str
    path: str
    qualname: str
    args: tuple[str, ...]
    node: ast.FunctionDef | ast.AsyncFunctionDef
    aliases: Mapping[str, str]
    markers: Mapping[str, list[str]]


@dataclass(frozen=True)
class SourceSeed:
    function_key: str
    kind: str
    evidence: str
    line_start: int
    line_end: int
    parameter_index: int | None = None

    @property
    def source_id(self) -> str:
        return digest({
            "function": self.function_key, "kind": self.kind, "evidence": self.evidence,
            "line_start": self.line_start, "line_end": self.line_end,
            "parameter_index": self.parameter_index,
        })


def extract_functions(sources: Mapping[str, str]) -> tuple[dict[str, FunctionInfo], list[dict[str, Any]], list[SourceSeed]]:
    functions: dict[str, FunctionInfo] = {}
    parse_errors: list[dict[str, Any]] = []
    seeds: list[SourceSeed] = []
    for path, source in sorted(sources.items()):
        try:
            tree = ast.parse(source, filename=path, type_comments=True)
        except (SyntaxError, ValueError) as exc:
            material = {
                "path": path, "line": int(getattr(exc, "lineno", 1) or 1),
                "error": str(getattr(exc, "msg", None) or exc),
            }
            parse_errors.append({"finding_id": digest(material), **material, "review_status": REVIEW_REQUIRED})
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
                args = tuple(arguments(node))
                info = FunctionInfo(key, module, path, qualname, args, node, dict(sorted(aliases.items())), marker_evidence(node))
                functions[key] = info
                for index, name in enumerate(args):
                    if raw_parameter(name):
                        seeds.append(SourceSeed(key, "raw_parameter", name, node.lineno, node.lineno, index))
                for child in iter_local_nodes(node):
                    if isinstance(child, ast.Call):
                        target = dotted(child.func)
                        if raw_source_call(target):
                            seeds.append(SourceSeed(
                                key, "raw_source_call", target, child.lineno,
                                int(getattr(child, "end_lineno", child.lineno)), None,
                            ))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.visit(child)
                self.scope.pop()

            visit_FunctionDef = add_function
            visit_AsyncFunctionDef = add_function

        Visitor().visit(tree)
    parse_errors.sort(key=lambda item: (item["path"], item["line"], item["finding_id"]))
    unique_seeds = {seed.source_id: seed for seed in seeds}
    return dict(sorted(functions.items())), parse_errors, [unique_seeds[key] for key in sorted(unique_seeds)]


def resolve(info: FunctionInfo, target: str, functions: Mapping[str, FunctionInfo], leaf_index: Mapping[str, list[str]]) -> str | None:
    class_prefix = info.qualname.rsplit(".", 1)[0] if "." in info.qualname else ""
    candidates: list[str] = []
    if target.startswith(("self.", "cls.")) and class_prefix:
        candidates.append(f"{info.module}:{class_prefix}.{target.split('.', 1)[1]}")
    if "." not in target:
        candidates.append(f"{info.module}:{target}")
        if class_prefix:
            candidates.append(f"{info.module}:{class_prefix}.{target}")
    head, _, tail = target.partition(".")
    alias = info.aliases.get(head)
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


def target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        value = dotted(node)
        return {value, node.attr} if value else {node.attr}
    if isinstance(node, ast.Subscript):
        value = dotted(node.value)
        return {value} if value else set()
    if isinstance(node, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for item in node.elts:
            result.update(target_names(item))
        return result
    return set()


def merge_envs(*values: Mapping[str, bool]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for value in values:
        for name, tainted in value.items():
            result[name] = result.get(name, False) or tainted
    return result


class FlowAnalyzer:
    def __init__(self, functions: Mapping[str, FunctionInfo]) -> None:
        self.functions = functions
        self.leaf_index: dict[str, list[str]] = defaultdict(list)
        for key, info in functions.items():
            self.leaf_index[info.qualname.rsplit(".", 1)[-1]].append(key)

    def call_argument_taint(self, call: ast.Call, env: Mapping[str, bool], seed: SourceSeed, info: FunctionInfo) -> tuple[list[bool], dict[str, bool]]:
        positional = [self.expr_tainted(value, env, seed, info) for value in call.args]
        keywords = {
            keyword.arg: self.expr_tainted(keyword.value, env, seed, info)
            for keyword in call.keywords if keyword.arg is not None
        }
        return positional, keywords

    def mapped_tainted_parameters(self, destination: FunctionInfo, target: str, positional: list[bool], keywords: Mapping[str, bool]) -> tuple[int, ...]:
        offset = 1 if destination.args and destination.args[0] in {"self", "cls"} and target.startswith(("self.", "cls.")) else 0
        indexes: set[int] = set()
        for index, tainted in enumerate(positional):
            parameter_index = index + offset
            if tainted and parameter_index < len(destination.args):
                indexes.add(parameter_index)
        by_name = {name: index for index, name in enumerate(destination.args)}
        for name, tainted in keywords.items():
            if tainted and name in by_name:
                indexes.add(by_name[name])
        return tuple(sorted(indexes))

    def expr_tainted(self, node: ast.AST | None, env: Mapping[str, bool], seed: SourceSeed, info: FunctionInfo) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return bool(env.get(node.id, False))
        if isinstance(node, ast.Attribute):
            return bool(env.get(dotted(node), False) or env.get(node.attr, False) or self.expr_tainted(node.value, env, seed, info))
        if isinstance(node, ast.Subscript):
            return self.expr_tainted(node.value, env, seed, info) or self.expr_tainted(node.slice, env, seed, info)
        if isinstance(node, ast.Call):
            target = dotted(node.func)
            if seed.kind == "raw_source_call" and info.key == seed.function_key and target == seed.evidence and node.lineno == seed.line_start:
                return True
            positional, keywords = self.call_argument_taint(node, env, seed, info)
            return any(positional) or any(keywords.values())
        if isinstance(node, ast.Lambda):
            return False
        return any(self.expr_tainted(child, env, seed, info) for child in ast.iter_child_nodes(node))

    def analyze(self, info: FunctionInfo, tainted_parameters: tuple[int, ...], seed: SourceSeed) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        env = {name: index in tainted_parameters for index, name in enumerate(info.args)}
        next_calls: list[dict[str, Any]] = []
        sinks: list[dict[str, Any]] = []
        returned_tainted = False

        def process(statements: Iterable[ast.stmt], current: dict[str, bool]) -> dict[str, bool]:
            nonlocal returned_tainted
            env_local = dict(current)
            for statement in statements:
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    if isinstance(statement, ast.Assign):
                        targets = statement.targets
                        value = statement.value
                    else:
                        targets = [statement.target]
                        value = statement.value
                    tainted = self.expr_tainted(value, env_local, seed, info)
                    for target in targets:
                        for name in target_names(target):
                            env_local[name] = tainted
                    inspect_calls(value, env_local)
                elif isinstance(statement, ast.AugAssign):
                    tainted = self.expr_tainted(statement.target, env_local, seed, info) or self.expr_tainted(statement.value, env_local, seed, info)
                    for name in target_names(statement.target):
                        env_local[name] = tainted
                    inspect_calls(statement.value, env_local)
                elif isinstance(statement, ast.Expr):
                    inspect_calls(statement.value, env_local)
                elif isinstance(statement, ast.Return):
                    tainted = self.expr_tainted(statement.value, env_local, seed, info)
                    returned_tainted = returned_tainted or tainted
                    inspect_calls(statement.value, env_local)
                    if tainted and sink_name(info.qualname):
                        direct_sink = (
                            isinstance(statement.value, ast.Call)
                            and external_sink(dotted(statement.value.func))
                            and resolve(info, dotted(statement.value.func), self.functions, self.leaf_index) is None
                        )
                        if not direct_sink:
                            sinks.append(sink_record(info, statement, "expression_return", "Return"))
                elif isinstance(statement, (ast.If, ast.Try, ast.Match)):
                    branches: list[list[ast.stmt]] = []
                    if isinstance(statement, ast.If):
                        inspect_calls(statement.test, env_local)
                        branches = [statement.body, statement.orelse]
                    elif isinstance(statement, ast.Try):
                        branches = [statement.body, statement.orelse, statement.finalbody, *(handler.body for handler in statement.handlers)]
                    else:
                        inspect_calls(statement.subject, env_local)
                        branches = [case.body for case in statement.cases]
                    branch_envs = [process(branch, dict(env_local)) for branch in branches if branch]
                    if branch_envs:
                        env_local = merge_envs(env_local, *branch_envs)
                elif isinstance(statement, (ast.For, ast.AsyncFor)):
                    inspect_calls(statement.iter, env_local)
                    item_tainted = self.expr_tainted(statement.iter, env_local, seed, info)
                    branch = dict(env_local)
                    for name in target_names(statement.target):
                        branch[name] = item_tainted
                    body_env = process(statement.body, branch)
                    else_env = process(statement.orelse, dict(env_local)) if statement.orelse else env_local
                    env_local = merge_envs(env_local, body_env, else_env)
                elif isinstance(statement, ast.While):
                    inspect_calls(statement.test, env_local)
                    body_env = process(statement.body, dict(env_local))
                    else_env = process(statement.orelse, dict(env_local)) if statement.orelse else env_local
                    env_local = merge_envs(env_local, body_env, else_env)
                elif isinstance(statement, (ast.With, ast.AsyncWith)):
                    branch = dict(env_local)
                    for item in statement.items:
                        inspect_calls(item.context_expr, branch)
                        value_tainted = self.expr_tainted(item.context_expr, branch, seed, info)
                        if item.optional_vars is not None:
                            for name in target_names(item.optional_vars):
                                branch[name] = value_tainted
                    env_local = merge_envs(env_local, process(statement.body, branch))
                else:
                    inspect_calls(statement, env_local)
                    for child in ast.iter_child_nodes(statement):
                        if isinstance(child, (ast.Yield, ast.YieldFrom)) and self.expr_tainted(getattr(child, "value", None), env_local, seed, info):
                            sinks.append(sink_record(info, child, "stream_yield", type(child).__name__))
            return env_local

        def inspect_calls(node: ast.AST | None, current: Mapping[str, bool]) -> None:
            if node is None:
                return
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and child is not node:
                    continue
                if isinstance(child, (ast.Yield, ast.YieldFrom)):
                    if self.expr_tainted(getattr(child, "value", None), current, seed, info):
                        sinks.append(sink_record(info, child, "stream_yield", type(child).__name__))
                if not isinstance(child, ast.Call):
                    continue
                target = dotted(child.func)
                positional, keywords = self.call_argument_taint(child, current, seed, info)
                if not (any(positional) or any(keywords.values())):
                    continue
                destination_key = resolve(info, target, self.functions, self.leaf_index)
                if destination_key is not None:
                    destination = self.functions[destination_key]
                    indexes = self.mapped_tainted_parameters(destination, target, positional, keywords)
                    if indexes:
                        next_calls.append({
                            "destination": destination_key, "tainted_parameters": indexes,
                            "target": target, "line_start": child.lineno,
                            "line_end": int(getattr(child, "end_lineno", child.lineno)),
                        })
                elif external_sink(target):
                    sinks.append(sink_record(info, child, "external_expression_call", target))

        process(info.node.body, env)
        unique_calls = {
            (item["destination"], item["tainted_parameters"], item["line_start"], item["target"]): item
            for item in next_calls
        }
        unique_sinks = {
            (item["path"], item["symbol"], item["evidence"]["line_start"], item["evidence"]["kind"], item["evidence"]["target"]): item
            for item in sinks
        }
        return (
            [unique_calls[key] for key in sorted(unique_calls)],
            [unique_sinks[key] for key in sorted(unique_sinks)],
            returned_tainted,
        )


def sink_record(info: FunctionInfo, node: ast.AST, kind: str, target: str) -> dict[str, Any]:
    return {
        "path": info.path,
        "symbol": info.qualname,
        "evidence": {
            "kind": kind, "target": target, "line_start": int(getattr(node, "lineno", 1)),
            "line_end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
        },
    }


def merged_markers(path: tuple[str, ...], functions: Mapping[str, FunctionInfo]) -> dict[str, list[str]]:
    result = {kind: set() for kind in MARKERS}
    for key in path:
        for kind, values in functions[key].markers.items():
            result[kind].update(values)
    return {kind: sorted(values) for kind, values in result.items()}


def extract_candidates(root: Path) -> dict[str, Any]:
    sources = tracked_sources(root)
    functions, parse_errors, seeds = extract_functions(sources)
    functions = {
        key: info for key, info in functions.items()
        if Path(info.path).parts and Path(info.path).parts[0] not in NON_RUNTIME
    }
    seeds = [seed for seed in seeds if seed.function_key in functions]
    analyzer = FlowAnalyzer(functions)
    candidates: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, dict[str, Any]] = {}
    analyzed_state_count = 0
    for seed in seeds:
        initial = (seed.parameter_index,) if seed.parameter_index is not None else ()
        queue = deque([(seed.function_key, tuple(initial), (seed.function_key,), ())])
        seen: set[tuple[str, tuple[int, ...]]] = set()
        while queue:
            function_key, tainted_parameters, path, call_evidence = queue.popleft()
            state = (function_key, tainted_parameters)
            if state in seen or len(path) > MAX_CALL_DEPTH:
                continue
            seen.add(state)
            analyzed_state_count += 1
            info = functions[function_key]
            next_calls, sinks, _returned_tainted = analyzer.analyze(info, tainted_parameters, seed)
            for sink in sinks:
                material = {
                    "source_id": seed.source_id,
                    "source_function": seed.function_key,
                    "source_kind": seed.kind,
                    "source_evidence": seed.evidence,
                    "call_path": list(path),
                    "call_evidence": list(call_evidence),
                    "sink": sink,
                }
                edge_id = digest(material)
                source_info = functions[seed.function_key]
                candidates[edge_id] = {
                    "edge_id": edge_id,
                    "source": {
                        "source_id": seed.source_id, "path": source_info.path,
                        "symbol": source_info.qualname, "kind": seed.kind,
                        "evidence": seed.evidence, "line_start": seed.line_start,
                        "line_end": seed.line_end,
                    },
                    "cognition_hops": [
                        {"path": functions[key].path, "symbol": functions[key].qualname}
                        for key in path[1:-1]
                    ],
                    "sink": sink,
                    "call_path": list(path),
                    "call_evidence": list(call_evidence),
                    "semantic_markers": merged_markers(path, functions),
                    "mechanical_confidence": "high" if len(path) == 1 else "medium",
                    "review_status": REVIEW_REQUIRED,
                }
            for call in next_calls:
                destination = call["destination"]
                if destination in path:
                    continue
                queue.append((
                    destination,
                    tuple(call["tainted_parameters"]),
                    path + (destination,),
                    call_evidence + ({
                        "caller": function_key, "callee": destination,
                        "target": call["target"], "line_start": call["line_start"],
                        "line_end": call["line_end"],
                    },),
                ))
                if destination not in functions:
                    material = {"source_id": seed.source_id, **call}
                    finding_id = digest(material)
                    unresolved[finding_id] = {
                        "finding_id": finding_id, **material,
                        "review_status": REVIEW_REQUIRED,
                    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "human_accepted": False,
        "runtime_integrated": False,
        "python_files_scanned": len(sources),
        "function_count": len(functions),
        "candidate_edges": [candidates[key] for key in sorted(candidates)],
        "unresolved_boundary_calls": [unresolved[key] for key in sorted(unresolved)],
        "parse_errors": parse_errors,
        "summary": {
            "candidate_edge_count": len(candidates),
            "unresolved_boundary_call_count": len(unresolved),
            "parse_error_count": len(parse_errors),
            "source_seed_count": len(seeds),
            "analyzed_state_count": analyzed_state_count,
        },
    }
    report["report_digest"] = digest(report)
    return report


def reviewed(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value != REVIEW_REQUIRED


def validate_exact_review_set(expected: set[str], values: Any, id_field: str, fields: set[str], label: str, errors: list[str]) -> set[str]:
    if not isinstance(values, list):
        errors.append(f"{label} must be a list")
        return set()
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get(id_field), str):
            errors.append(f"{label}[{index}] is malformed")
            continue
        identifier = value[id_field]
        if identifier in seen:
            errors.append(f"duplicate {label} decision: {identifier}")
        seen.add(identifier)
        if identifier not in expected:
            errors.append(f"stale {label} decision: {identifier}")
        for field in fields:
            if not reviewed(value.get(field)):
                errors.append(f"{label} {identifier} field {field} requires review")
    if expected - seen:
        errors.append(f"missing {label} decisions: {len(expected - seen)}")
    return seen


def validate_decisions(report: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        errors.append("decision schema_version mismatch")
    if payload.get("candidate_report_digest") != report.get("report_digest"):
        errors.append("candidate_report_digest mismatch")
    edge_expected = {item["edge_id"] for item in report["candidate_edges"]}
    edge_values = payload.get("edge_decisions", [])
    edge_seen = validate_exact_review_set(edge_expected, edge_values, "edge_id", DECISION_FIELDS - {"edge_id"}, "edge", errors)
    if isinstance(edge_values, list):
        for value in edge_values:
            if isinstance(value, dict) and value.get("decision") not in DECISIONS:
                errors.append(f"invalid edge decision: {value.get('edge_id')}")
    unresolved_expected = {item["finding_id"] for item in report["unresolved_boundary_calls"]}
    unresolved_seen = validate_exact_review_set(unresolved_expected, payload.get("unresolved_call_decisions", []), "finding_id", REVIEW_FIELDS, "unresolved_call", errors)
    parse_expected = {item["finding_id"] for item in report["parse_errors"]}
    parse_seen = validate_exact_review_set(parse_expected, payload.get("parse_error_decisions", []), "finding_id", REVIEW_FIELDS, "parse_error", errors)
    result: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_report_digest": report["report_digest"],
        "valid": not errors,
        "errors": errors,
        "edge_decision_count": len(edge_seen),
        "unresolved_call_decision_count": len(unresolved_seen),
        "parse_error_decision_count": len(parse_seen),
        "eligible_for_human_review": not errors,
        "human_accepted": False,
        "authority": AUTHORITY,
    }
    result["validation_digest"] = digest(result)
    return result


def build_output(root: Path, decisions: Path | None = None) -> dict[str, Any]:
    report = extract_candidates(root)
    if decisions is None:
        report["decision_validation"] = {
            "status": "not_requested", "eligible_for_human_review": False,
            "human_accepted": False, "authority": AUTHORITY,
        }
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
