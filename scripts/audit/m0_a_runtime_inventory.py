#!/usr/bin/env python3
"""M0-A runtime, dependency, mutation, and test inventory.

The audit is read-only by default. It scans tracked Python files, emits a
canonical JSON document to stdout, and never commits or creates generated
artifacts. Operators may use --output for an ephemeral CI artifact only.
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

SCHEMA_VERSION = "1.0.0-m0-a"
EXCLUDED_PARTS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
TOP_LEVEL_ENTRYPOINT_NAMES = {
    "main",
    "repl",
    "cli",
    "run",
    "serve",
    "worker",
    "build_full_engine",
    "build_minimal_engine",
}
METHOD_ENTRYPOINT_NAMES = {
    "run",
    "start",
    "start_background",
    "serve_forever",
    "step",
    "chat_stream",
    "proactive_stream",
    "save",
    "load",
}
CONSTRUCTOR_SUFFIXES = (
    "Adapter",
    "Client",
    "Controller",
    "Engine",
    "Loop",
    "Manager",
    "Model",
    "Pipeline",
    "Registry",
    "Runtime",
    "Server",
    "Store",
    "Tracker",
    "Wrapper",
)
WRITE_CALL_SUFFIXES = {
    "Path.write_bytes",
    "Path.write_text",
    "gzip.open",
    "json.dump",
    "os.makedirs",
    "os.mkdir",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "os.unlink",
    "pickle.dump",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    "shutil.copytree",
    "shutil.move",
    "sqlite3.connect",
    "yaml.dump",
}
WRITE_METHOD_NAMES = {
    "commit",
    "dump",
    "flush",
    "mkdir",
    "persist",
    "rename",
    "rmdir",
    "save",
    "touch",
    "unlink",
    "write",
    "write_bytes",
    "write_text",
}
DB_WRITE_METHOD_NAMES = {"execute", "executemany"}
DB_RECEIVER_HINTS = ("conn", "connection", "cursor", "database", "db", "sqlite")
MUTATION_METHOD_NAMES = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "popitem",
    "remove",
    "set",
    "setdefault",
    "update",
}
EXECUTION_BOUNDARY_CALLS = {
    "asyncio.run",
    "multiprocessing.Process",
    "subprocess.Popen",
    "threading.Thread",
    "uvicorn.run",
}
LEGACY_AUTHORITY_MARKERS = (
    "docs/EVE_DESIGN_v3.md",
    "docs/EVE_DESIGN_v3_1.md",
    "docs/EVE_IMPLEMENTATION_STATUS_v3_1.md",
    "docs/EVE_DEPENDENCY_MAP_v3_1.md",
)

MANUAL_OVERRIDES: dict[tuple[str, str], tuple[str, str, bool]] = {
    ("main.py", "build_full_engine"): (
        "ACTIVE_RUNTIME_COMPOSITION_ROOT",
        "high",
        False,
    ),
    ("main.py", "build_minimal_engine"): (
        "ACTIVE_MINIMAL_RUNTIME_COMPOSITION_ROOT",
        "high",
        False,
    ),
    ("main.py", "repl"): (
        "ACTIVE_INTERACTIVE_ENTRYPOINT",
        "high",
        False,
    ),
    ("adapters/live_loop.py", "LiveLoop._run"): (
        "ACTIVE_BACKGROUND_STATE_MUTATION_LOOP",
        "high",
        False,
    ),
    ("adapters/live_loop.py", "LiveLoop._do_autosave"): (
        "ACTIVE_AUTOSAVE_WRITE_PATH",
        "high",
        False,
    ),
    ("adapters/live_loop.py", "LiveLoop.start"): (
        "ACTIVE_BACKGROUND_THREAD_START",
        "high",
        False,
    ),
    ("adapters/persistence_adapter.py", "PersistenceAdapter.save"): (
        "ACTIVE_PERSISTENCE_WRITE_PATH",
        "high",
        False,
    ),
    ("adapters/persistence_adapter.py", "PersistenceAdapter.load"): (
        "ACTIVE_PERSISTENCE_RESTORE_PATH",
        "high",
        False,
    ),
    ("language/streaming.py", "StreamingEngine.__init__"): (
        "ACTIVE_ENGINE_DEPENDENCY_CONSTRUCTION",
        "high",
        False,
    ),
    ("language/streaming.py", "StreamingEngine.chat_stream"): (
        "ACTIVE_CHAT_STATE_TRANSITION_PATH",
        "high",
        False,
    ),
    ("core/autonomous.py", "AutonomousLoop.step"): (
        "ACTIVE_AUTONOMOUS_STATE_TRANSITION_PATH",
        "high",
        False,
    ),
}


def _node_end(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))


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


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_kind(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return "attribute_assignment"
    if isinstance(node, ast.Subscript):
        return "subscript_assignment"
    if isinstance(node, (ast.Tuple, ast.List)):
        kinds = {_target_kind(item) for item in node.elts}
        kinds.discard(None)
        if kinds:
            return "+".join(sorted(kinds))
    return None


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left = _dotted_name(test.left)
    right = _constant_string(test.comparators[0])
    reverse_left = _dotted_name(test.comparators[0])
    reverse_right = _constant_string(test.left)
    return (left == "__name__" and right == "__main__") or (
        reverse_left == "__name__" and reverse_right == "__main__"
    )


def _write_mode_from_open(call: ast.Call) -> str | None:
    target = _dotted_name(call.func)
    if target not in {"open", "builtins.open", "gzip.open", "Path.open"} and not target.endswith(".open"):
        return None
    mode_node: ast.AST | None = call.args[1] if len(call.args) >= 2 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    mode = _constant_string(mode_node) or "r"
    return mode if any(flag in mode for flag in "wax+") else None


def _looks_like_constructor(target: str) -> bool:
    leaf = target.rsplit(".", 1)[-1]
    return bool(leaf) and (leaf[0].isupper() or leaf.endswith(CONSTRUCTOR_SUFFIXES))


def _git_tracked_python_files(root: Path) -> list[Path]:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    paths: list[Path] = []
    for value in raw.split(b"\0"):
        if not value:
            continue
        relative = Path(os.fsdecode(value))
        if not any(part in EXCLUDED_PARTS for part in relative.parts):
            paths.append(root / relative)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def iter_python_files(root: Path) -> Iterator[Path]:
    tracked = _git_tracked_python_files(root)
    if tracked:
        yield from tracked
        return
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        yield path


def _local_import_roots(root: Path, paths: Iterable[Path]) -> set[str]:
    roots: set[str] = set()
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) == 1:
            roots.add(relative.stem)
        else:
            roots.add(relative.parts[0])
    return roots


def _classification_for(
    category: str,
    path: str,
    callable_name: str,
    evidence: str,
) -> tuple[str, str, bool]:
    override = MANUAL_OVERRIDES.get((path, callable_name))
    if override is not None:
        return override
    if category == "entrypoint":
        if evidence == "module_main_guard":
            return "ACTIVE_MODULE_ENTRYPOINT", "high", False
        return "RUNTIME_ENTRYPOINT_CANDIDATE", "medium", True
    if category == "import":
        return "IMPORT_DEPENDENCY", "high", False
    if category == "dependency_construction":
        return "RUNTIME_DEPENDENCY_CONSTRUCTION_CANDIDATE", "medium", True
    if category == "mutation":
        return "IN_MEMORY_MUTATION_SITE", "high", False
    if category == "direct_write":
        return "FILESYSTEM_OR_PERSISTENCE_WRITE_SITE", "high", False
    if category == "execution_boundary":
        return "THREAD_PROCESS_OR_SERVER_BOUNDARY", "high", False
    if category == "parse_error":
        return "UNRESOLVED_PARSE_ERROR", "high", True
    return "UNRESOLVED", "low", True


class InventoryVisitor(ast.NodeVisitor):
    def __init__(self, path: str, local_roots: set[str]) -> None:
        self.path = path
        self.local_roots = local_roots
        self.scope: list[str] = []
        self.findings: list[dict[str, Any]] = []

    @property
    def callable_name(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def add(
        self,
        category: str,
        node: ast.AST,
        evidence: str,
        detector: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        classification, confidence, unresolved = _classification_for(
            category, self.path, self.callable_name, evidence
        )
        finding: dict[str, Any] = {
            "category": category,
            "path": self.path,
            "line_start": int(getattr(node, "lineno", 1)),
            "line_end": _node_end(node),
            "callable": self.callable_name,
            "mechanical_evidence": evidence,
            "detector": detector,
            "manual_classification": classification,
            "confidence": confidence,
            "unresolved": unresolved,
            "manual_only": False,
        }
        if details:
            finding["details"] = details
        self.findings.append(finding)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            self.add(
                "import",
                node,
                f"import {alias.name}",
                "ast.Import",
                details={"module": alias.name, "local": root in self.local_roots},
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0] if module else ""
        names = [alias.name for alias in node.names]
        self.add(
            "import",
            node,
            f"from {'.' * node.level}{module} import {','.join(names)}",
            "ast.ImportFrom",
            details={
                "module": module,
                "level": node.level,
                "names": names,
                "local": bool(node.level) or root in self.local_roots,
            },
        )
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_top_level = not self.scope
        if (is_top_level and node.name in TOP_LEVEL_ENTRYPOINT_NAMES) or (
            not is_top_level and node.name in METHOD_ENTRYPOINT_NAMES
        ):
            self.scope.append(node.name)
            self.add(
                "entrypoint",
                node,
                f"callable_name={node.name}",
                type(node).__name__,
                details={"async": isinstance(node, ast.AsyncFunctionDef)},
            )
            self.scope.pop()
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node):
            self.add(
                "entrypoint",
                node,
                "module_main_guard",
                "ast.If __name__ == '__main__'",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted_name(node.func)
        if target and _looks_like_constructor(target):
            self.add(
                "dependency_construction",
                node,
                f"constructor_call={target}",
                "ast.Call constructor heuristic",
                details={"target": target},
            )

        write_mode = _write_mode_from_open(node)
        if write_mode is not None:
            self.add(
                "direct_write",
                node,
                f"open_write_mode={write_mode}",
                "ast.Call open mode",
                details={"target": target, "mode": write_mode},
            )
        elif target in WRITE_CALL_SUFFIXES or any(target.endswith(f".{suffix}") for suffix in WRITE_CALL_SUFFIXES):
            self.add(
                "direct_write",
                node,
                f"write_call={target}",
                "ast.Call exact write target",
                details={"target": target},
            )
        else:
            leaf = target.rsplit(".", 1)[-1] if target else ""
            if leaf in WRITE_METHOD_NAMES:
                self.add(
                    "direct_write",
                    node,
                    f"write_like_method={target}",
                    "ast.Call write method heuristic",
                    details={"target": target},
                )

        if target in EXECUTION_BOUNDARY_CALLS or any(
            target.endswith(f".{name}") for name in EXECUTION_BOUNDARY_CALLS
        ):
            self.add(
                "execution_boundary",
                node,
                f"execution_call={target}",
                "ast.Call execution boundary",
                details={"target": target},
            )
        self.generic_visit(node)

    def _record_assignment_targets(self, node: ast.AST, targets: Iterable[ast.AST]) -> None:
        for target in targets:
            kind = _target_kind(target)
            if kind:
                self.add(
                    "mutation",
                    node,
                    kind,
                    type(node).__name__,
                    details={"target": _dotted_name(target)},
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_assignment_targets(node, node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_assignment_targets(node, [node.target])
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_assignment_targets(node, [node.target])
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._record_assignment_targets(node, node.targets)
        self.generic_visit(node)


def _first_matching_line(source: str, markers: Iterable[str]) -> tuple[int, str] | None:
    for line_number, line in enumerate(source.splitlines(), start=1):
        for marker in markers:
            if marker in line:
                return line_number, marker
    return None


def classify_test(path: str, source: str) -> dict[str, Any]:
    legacy_match = _first_matching_line(source, LEGACY_AUTHORITY_MARKERS)
    if legacy_match is not None:
        line, marker = legacy_match
        classification = "REWRITE"
        reason = (
            "The test references a superseded v3/v3.1 authority document and must be "
            "rewritten against active v4 authority without weakening the behavioral assertion."
        )
        confidence = "high"
        evidence = f"superseded_authority_reference={marker}"
    else:
        line = 1
        classification = "KEEP"
        reason = (
            "Conservative M0-A preservation rule: retain executable behavioral evidence until a "
            "later milestone supplies file:line evidence for rewrite or retirement."
        )
        confidence = "medium"
        evidence = "no_superseded_authority_reference_detected"
    return {
        "category": "test_classification",
        "path": path,
        "line_start": line,
        "line_end": line,
        "callable": "<module>",
        "mechanical_evidence": evidence,
        "detector": "M0-A conservative test policy",
        "manual_classification": classification,
        "manual_reason": reason,
        "confidence": confidence,
        "unresolved": False,
        "manual_only": False,
    }


def _is_test_path(relative: Path) -> bool:
    name = relative.name
    return (
        "tests" in relative.parts
        and (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")
    )


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = list(iter_python_files(root))
    local_roots = _local_import_roots(root, paths)
    findings: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []

    for path in paths:
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = path.read_text(encoding="utf-8", errors="replace")
        if _is_test_path(relative):
            tests.append(classify_test(relative_text, source))
        try:
            tree = ast.parse(source, filename=relative_text, type_comments=True)
        except (SyntaxError, ValueError) as exc:
            line = int(getattr(exc, "lineno", 1) or 1)
            classification, confidence, unresolved = _classification_for(
                "parse_error", relative_text, "<module>", str(exc)
            )
            parse_errors.append(
                {
                    "category": "parse_error",
                    "path": relative_text,
                    "line_start": line,
                    "line_end": line,
                    "callable": "<module>",
                    "mechanical_evidence": str(exc),
                    "detector": "ast.parse",
                    "manual_classification": classification,
                    "confidence": confidence,
                    "unresolved": unresolved,
                    "manual_only": False,
                }
            )
            continue
        visitor = InventoryVisitor(relative_text, local_roots)
        visitor.visit(tree)
        findings.extend(visitor.findings)

    all_entries = findings + tests + parse_errors
    all_entries.sort(
        key=lambda item: (
            item["path"],
            int(item["line_start"]),
            item["category"],
            item["mechanical_evidence"],
        )
    )
    category_counts = Counter(entry["category"] for entry in all_entries)
    classification_counts = Counter(entry["manual_classification"] for entry in all_entries)
    unresolved_count = sum(bool(entry["unresolved"]) for entry in all_entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "root": root.as_posix(),
        "scope": {
            "tracked_python_only_when_git_available": True,
            "generated_json_committed": False,
            "runtime_activation_performed": False,
            "source_mutation_performed": False,
        },
        "summary": {
            "python_files_scanned": len(paths),
            "entries": len(all_entries),
            "test_files_classified": len(tests),
            "parse_errors": len(parse_errors),
            "unresolved_entries": unresolved_count,
            "category_counts": dict(sorted(category_counts.items())),
            "classification_counts": dict(sorted(classification_counts.items())),
        },
        "entries": all_entries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--fail-on-parse-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_repository(args.root)
    payload: dict[str, Any] = report["summary"] if args.summary_only else report
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.fail_on_parse_error and report["summary"]["parse_errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
