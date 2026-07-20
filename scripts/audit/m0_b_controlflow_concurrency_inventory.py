#!/usr/bin/env python3
"""Generate the EVE M0-B control-flow, failure, clock, and concurrency inventory.

The audit is read-only unless ``--output`` is explicitly supplied. Generated
JSON is intended for stdout or an ephemeral validation artifact and must not be
committed.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import tarfile
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "1.0.0-m0-b"
AUDIT_SNAPSHOT_SHA = "eea70c286e947cbc180db9565bfa5ddc062d1ac3"

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

GATE_TOKENS = {
    "allow",
    "allowed",
    "authorization",
    "authorized",
    "consent",
    "disabled",
    "dry_run",
    "enable",
    "enabled",
    "enforcement",
    "gate",
    "guard",
    "opt_in",
    "permission",
    "ready",
    "safe",
    "trusted",
    "validated",
    "validation",
}

BYPASS_TOKENS = {
    "bypass",
    "force",
    "forced",
    "override",
    "skip_check",
    "skip_checks",
    "skip_validation",
    "unsafe",
}

OUTPUT_TARGETS = {
    "print",
    "builtins.print",
    "sys.stdout.write",
    "sys.stderr.write",
    "logging.debug",
    "logging.info",
    "logging.warning",
    "logging.error",
    "logging.exception",
    "logging.critical",
}
OUTPUT_LEAVES = {
    "broadcast",
    "emit",
    "notify",
    "publish",
    "reply",
    "respond",
    "send",
    "sendall",
}

CLOCK_TARGETS = {
    "asyncio.sleep",
    "date.today",
    "datetime.date.today",
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
    "datetime.now",
    "datetime.utcnow",
    "time.monotonic",
    "time.monotonic_ns",
    "time.perf_counter",
    "time.perf_counter_ns",
    "time.process_time",
    "time.sleep",
    "time.time",
    "time.time_ns",
}
CLOCK_LEAVES = {
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "sleep",
    "time",
    "time_ns",
    "today",
    "utcnow",
}

QUEUE_CONSTRUCTORS = {
    "asyncio.Queue",
    "collections.deque",
    "multiprocessing.JoinableQueue",
    "multiprocessing.Queue",
    "queue.LifoQueue",
    "queue.PriorityQueue",
    "queue.Queue",
    "queue.SimpleQueue",
}
QUEUE_METHODS = {
    "empty",
    "full",
    "get",
    "get_nowait",
    "join",
    "popleft",
    "put",
    "put_nowait",
    "qsize",
    "task_done",
}
QUEUE_RECEIVER_HINTS = ("queue", "deque", "buffer", "mailbox", "inbox", "outbox")

CONCURRENCY_TARGETS = {
    "asyncio.create_task",
    "asyncio.gather",
    "asyncio.run",
    "asyncio.shield",
    "asyncio.to_thread",
    "asyncio.wait",
    "asyncio.wait_for",
    "concurrent.futures.ProcessPoolExecutor",
    "concurrent.futures.ThreadPoolExecutor",
    "multiprocessing.Pool",
    "multiprocessing.Process",
    "threading.Barrier",
    "threading.BoundedSemaphore",
    "threading.Condition",
    "threading.Event",
    "threading.Lock",
    "threading.RLock",
    "threading.Semaphore",
    "threading.Thread",
}
CONCURRENCY_METHODS = {"acquire", "cancel", "join", "release", "result", "start", "wait"}
CONCURRENCY_RECEIVER_HINTS = (
    "barrier",
    "condition",
    "event",
    "executor",
    "future",
    "lock",
    "pool",
    "process",
    "semaphore",
    "task",
    "thread",
)

NONDETERMINISTIC_EXACT_TARGETS = {
    "hash",
    "builtins.hash",
    "id",
    "builtins.id",
    "os.urandom",
    "uuid.uuid1",
    "uuid.uuid4",
}
ORDER_DEPENDENCY_TARGETS = {
    "glob.glob",
    "glob.iglob",
    "os.listdir",
}
ORDER_DEPENDENCY_LEAVES = {"glob", "iglob", "iterdir", "listdir", "rglob"}

RANDOM_PREFIXES = (
    "random.",
    "secrets.",
    "np.random.",
    "numpy.random.",
)
RANDOM_EXEMPT_SUFFIXES = (".seed",)

TEST_REWRITE_CALLS = {
    "asyncio.sleep",
    "time.sleep",
    "uuid.uuid1",
    "uuid.uuid4",
    "os.urandom",
}

MANUAL_OVERRIDES: dict[tuple[str, str, str], tuple[str, str, bool]] = {
    ("main.py", "build_full_engine", "gate"): (
        "ACTIVE_RUNTIME_COMPOSITION_GATE",
        "high",
        False,
    ),
    ("main.py", "repl", "output"): (
        "ACTIVE_INTERACTIVE_OUTPUT_SURFACE",
        "high",
        False,
    ),
    ("adapters/live_loop.py", "LiveLoop._run", "clock"): (
        "ACTIVE_BACKGROUND_CLOCK_DEPENDENCY",
        "high",
        False,
    ),
    ("adapters/live_loop.py", "LiveLoop.start", "concurrency"): (
        "ACTIVE_DAEMON_THREAD_BOUNDARY",
        "high",
        False,
    ),
    ("language/streaming.py", "StreamingEngine.chat_stream", "exception"): (
        "ACTIVE_CHAT_FAILURE_HANDLING_PATH",
        "high",
        True,
    ),
    ("core/autonomous.py", "AutonomousLoop.step", "output"): (
        "ACTIVE_AUTONOMOUS_OUTPUT_SURFACE",
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


def _call_leaf(target: str) -> str:
    return target.rsplit(".", 1)[-1] if target else ""


def _expression_text(node: ast.AST) -> str:
    try:
        rendered = ast.unparse(node)
    except Exception:
        rendered = type(node).__name__
    rendered = " ".join(rendered.split())
    return rendered[:240]


def _identifier_tokens(node: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tokens.add(child.id.lower())
        elif isinstance(child, ast.Attribute):
            tokens.add(child.attr.lower())
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            value = child.value.strip().lower()
            if value and len(value) <= 80:
                tokens.add(value)
    return tokens


def _exception_type(node: ast.AST | None) -> str:
    if node is None:
        return "<bare>"
    if isinstance(node, ast.Tuple):
        return "|".join(filter(None, (_dotted_name(item) for item in node.elts)))
    return _dotted_name(node) or _expression_text(node)


def _call_is_signal(call: ast.Call) -> bool:
    target = _dotted_name(call.func)
    leaf = _call_leaf(target).lower()
    if target in OUTPUT_TARGETS or leaf in OUTPUT_LEAVES:
        return True
    return any(
        marker in leaf
        for marker in ("error", "warn", "exception", "diagnostic", "telemetry", "record")
    )


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    for child in ast.walk(handler):
        if isinstance(child, ast.Raise):
            return False
        if isinstance(child, ast.Call) and _call_is_signal(child):
            return False
    return True


_SNAPSHOT_SOURCE_CACHE: dict[tuple[str, str], dict[str, str] | None] = {}


def _git_snapshot_sources(root: Path) -> dict[str, str] | None:
    """Return Python source text from the completed audit snapshot."""
    key = (str(root.resolve()), AUDIT_SNAPSHOT_SHA)
    if key in _SNAPSHOT_SOURCE_CACHE:
        return _SNAPSHOT_SOURCE_CACHE[key]
    try:
        archive = subprocess.check_output(
            ["git", "-C", str(root), "archive", "--format=tar", AUDIT_SNAPSHOT_SHA],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        _SNAPSHOT_SOURCE_CACHE[key] = None
        return None
    sources: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        for member in handle.getmembers():
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            extracted = handle.extractfile(member)
            if extracted is not None:
                sources[Path(member.name).as_posix()] = extracted.read().decode(
                    "utf-8", errors="replace"
                )
    _SNAPSHOT_SOURCE_CACHE[key] = sources
    return sources


def _git_tracked_python_files(root: Path) -> list[Path]:
    sources = _git_snapshot_sources(root)
    if sources is None:
        return []
    return [
        root / Path(value)
        for value in sorted(sources)
        if not any(part in EXCLUDED_PARTS for part in Path(value).parts)
    ]


def _read_source(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    sources = _git_snapshot_sources(root)
    if sources is not None and relative in sources:
        return sources[relative]
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")

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


def _classification_for(
    category: str,
    path: str,
    callable_name: str,
    *,
    broad: bool = False,
    silent: bool = False,
    external_order: bool = False,
) -> tuple[str, str, bool]:
    override = MANUAL_OVERRIDES.get((path, callable_name, category))
    if override is not None:
        return override
    if category == "gate":
        return "RUNTIME_GATE_CANDIDATE", "medium", True
    if category == "bypass":
        return "BYPASS_OR_OVERRIDE_CANDIDATE", "high", True
    if category == "output":
        return "OBSERVABLE_OUTPUT_SURFACE", "high", False
    if category == "exception":
        if broad and silent:
            return "SILENT_BROAD_EXCEPTION_PATH", "high", True
        if silent:
            return "SILENT_SPECIFIC_EXCEPTION_PATH", "high", True
        if broad:
            return "BROAD_EXCEPTION_PATH", "high", True
        return "SPECIFIC_EXCEPTION_PATH", "high", False
    if category == "clock":
        return "CLOCK_OR_DELAY_DEPENDENCY", "high", True
    if category == "queue":
        return "QUEUE_OR_BUFFER_BOUNDARY", "high", True
    if category == "concurrency":
        return "THREAD_PROCESS_TASK_OR_LOCK_BOUNDARY", "high", True
    if category == "nondeterminism":
        if external_order:
            return "EXTERNAL_ORDER_DEPENDENCY_CANDIDATE", "medium", True
        return "NONDETERMINISTIC_INPUT_CANDIDATE", "high", True
    if category == "parse_error":
        return "UNRESOLVED_PARSE_ERROR", "high", True
    return "UNRESOLVED", "low", True


class ControlFlowVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
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
        broad: bool = False,
        silent: bool = False,
        external_order: bool = False,
    ) -> None:
        classification, confidence, unresolved = _classification_for(
            category,
            self.path,
            self.callable_name,
            broad=broad,
            silent=silent,
            external_order=external_order,
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

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
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

    def _record_predicate(self, node: ast.AST, predicate: ast.AST, detector: str) -> None:
        tokens = _identifier_tokens(predicate)
        expression = _expression_text(predicate)
        gate_hits = sorted(tokens & GATE_TOKENS)
        bypass_hits = sorted(tokens & BYPASS_TOKENS)
        if gate_hits:
            self.add(
                "gate",
                node,
                f"gate_predicate={expression}",
                detector,
                details={"tokens": gate_hits},
            )
        if bypass_hits:
            self.add(
                "bypass",
                node,
                f"bypass_predicate={expression}",
                detector,
                details={"tokens": bypass_hits},
            )

    def visit_If(self, node: ast.If) -> None:
        self._record_predicate(node, node.test, "ast.If predicate")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record_predicate(node, node.test, "ast.IfExp predicate")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._record_predicate(node, node.test, "ast.Assert predicate")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        exception = _exception_type(node.type)
        broad = exception in {"<bare>", "Exception", "BaseException", "builtins.Exception"}
        silent = _handler_is_silent(node)
        self.add(
            "exception",
            node,
            f"except={exception};silent={str(silent).lower()}",
            "ast.ExceptHandler",
            details={"exception": exception, "broad": broad, "silent": silent},
            broad=broad,
            silent=silent,
        )
        self.generic_visit(node)

    def _record_call(self, node: ast.Call, target: str) -> None:
        leaf = _call_leaf(target)
        lower_target = target.lower()
        lower_leaf = leaf.lower()

        if target in OUTPUT_TARGETS or lower_leaf in OUTPUT_LEAVES:
            self.add(
                "output",
                node,
                f"output_call={target}",
                "ast.Call output target",
                details={"target": target},
            )

        if target in CLOCK_TARGETS or (
            lower_leaf in CLOCK_LEAVES
            and any(prefix in lower_target for prefix in ("time", "date", "clock", "timer"))
        ):
            self.add(
                "clock",
                node,
                f"clock_call={target}",
                "ast.Call clock target",
                details={"target": target, "delay": lower_leaf in {"sleep", "wait"}},
            )

        receiver = lower_target.rsplit(".", 1)[0] if "." in lower_target else ""
        if target in QUEUE_CONSTRUCTORS or (
            lower_leaf in QUEUE_METHODS
            and any(hint in receiver for hint in QUEUE_RECEIVER_HINTS)
        ):
            self.add(
                "queue",
                node,
                f"queue_call={target}",
                "ast.Call queue target",
                details={"target": target},
            )

        concurrency_method = (
            lower_leaf in CONCURRENCY_METHODS
            and any(hint in receiver for hint in CONCURRENCY_RECEIVER_HINTS)
        )
        if target in CONCURRENCY_TARGETS or concurrency_method:
            self.add(
                "concurrency",
                node,
                f"concurrency_call={target}",
                "ast.Call concurrency target",
                details={"target": target},
            )

        random_target = lower_target.startswith(RANDOM_PREFIXES) and not lower_target.endswith(
            RANDOM_EXEMPT_SUFFIXES
        )
        exact_nondeterministic = target in NONDETERMINISTIC_EXACT_TARGETS
        external_order = target in ORDER_DEPENDENCY_TARGETS or (
            lower_leaf in ORDER_DEPENDENCY_LEAVES
            and any(part in receiver for part in ("path", "dir", "glob", "os"))
        )
        if random_target or exact_nondeterministic or external_order:
            self.add(
                "nondeterminism",
                node,
                f"nondeterministic_call={target}",
                "ast.Call nondeterminism target",
                details={
                    "target": target,
                    "random_source": random_target,
                    "external_order": external_order,
                },
                external_order=external_order,
            )

        target_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                lower_target.replace("-", "_").replace(".", "_"),
            )
        )
        bypass_hits = target_tokens & BYPASS_TOKENS
        keyword_hits: list[str] = []
        for keyword in node.keywords:
            if keyword.arg and keyword.arg.lower() in BYPASS_TOKENS:
                if isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    continue
                keyword_hits.append(keyword.arg.lower())
        if bypass_hits or keyword_hits:
            self.add(
                "bypass",
                node,
                f"bypass_call={target}",
                "ast.Call bypass name or keyword",
                details={
                    "target": target,
                    "name_hits": sorted(bypass_hits),
                    "keyword_hits": sorted(keyword_hits),
                },
            )

    def visit_Call(self, node: ast.Call) -> None:
        self._record_call(node, _dotted_name(node.func))
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self.add(
            "output",
            node,
            "yield_output",
            "ast.Yield",
            details={"value": _expression_text(node.value) if node.value else None},
        )
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.add(
            "output",
            node,
            "yield_from_output",
            "ast.YieldFrom",
            details={"value": _expression_text(node.value)},
        )
        self.generic_visit(node)


class TestSignalVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.random_calls: list[tuple[int, str]] = []
        self.seed_calls: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted_name(node.func)
        lower = target.lower()
        if target in TEST_REWRITE_CALLS:
            self.calls.append((int(getattr(node, "lineno", 1)), target))
        if lower.startswith(RANDOM_PREFIXES):
            if lower.endswith(RANDOM_EXEMPT_SUFFIXES):
                self.seed_calls.append((int(getattr(node, "lineno", 1)), target))
            else:
                self.random_calls.append((int(getattr(node, "lineno", 1)), target))
        self.generic_visit(node)


def classify_test(path: str, tree: ast.AST) -> dict[str, Any]:
    visitor = TestSignalVisitor()
    visitor.visit(tree)

    classification = "KEEP"
    reason = (
        "Preserve executable evidence unless exact M0-B clock or unseeded "
        "nondeterminism evidence requires deterministic rewriting."
    )
    line = 1
    evidence = "conservative_default_keep"
    confidence = "high"
    unresolved = False

    if visitor.calls:
        line, target = sorted(visitor.calls)[0]
        classification = "REWRITE"
        reason = (
            "The test directly depends on a real delay or nondeterministic source; "
            "rewrite it around an injected clock or deterministic source while "
            "preserving the behavioral assertion."
        )
        evidence = f"m0_b_test_call={target}"
    elif visitor.random_calls and not visitor.seed_calls:
        line, target = sorted(visitor.random_calls)[0]
        classification = "REWRITE"
        reason = (
            "The test uses a random source without an explicit seed in the same "
            "file; rewrite it to deterministic input while preserving coverage."
        )
        evidence = f"unseeded_test_random_call={target}"

    return {
        "category": "test_classification",
        "path": path,
        "line_start": line,
        "line_end": line,
        "callable": "<test-file>",
        "mechanical_evidence": evidence,
        "detector": "AST M0-B test policy",
        "manual_classification": classification,
        "manual_reason": reason,
        "confidence": confidence,
        "unresolved": unresolved,
        "manual_only": False,
    }


def _parse_file(root: Path, path: Path) -> tuple[ast.AST | None, str | None]:
    source = _read_source(root, path)
    try:
        return ast.parse(source, filename=path.relative_to(root).as_posix()), None
    except SyntaxError as exc:
        message = exc.msg or "syntax error"
        return None, f"{message} at line {exc.lineno or 1}"


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = list(iter_python_files(root))
    entries: list[dict[str, Any]] = []

    for path in paths:
        relative = path.relative_to(root).as_posix()
        tree, parse_error = _parse_file(root, path)
        if tree is None:
            classification, confidence, unresolved = _classification_for(
                "parse_error", relative, "<module>"
            )
            entries.append(
                {
                    "category": "parse_error",
                    "path": relative,
                    "line_start": 1,
                    "line_end": 1,
                    "callable": "<module>",
                    "mechanical_evidence": parse_error or "syntax error",
                    "detector": "ast.parse",
                    "manual_classification": classification,
                    "confidence": confidence,
                    "unresolved": unresolved,
                    "manual_only": False,
                }
            )
            continue

        visitor = ControlFlowVisitor(relative)
        visitor.visit(tree)
        entries.extend(visitor.findings)
        if relative.startswith("tests/") or "/tests/" in relative:
            entries.append(classify_test(relative, tree))

    entries.sort(
        key=lambda item: (
            item["path"],
            int(item["line_start"]),
            item["category"],
            item["mechanical_evidence"],
        )
    )
    category_counts = Counter(entry["category"] for entry in entries)
    classification_counts = Counter(
        entry["manual_classification"] for entry in entries
    )
    test_counts = Counter(
        entry["manual_classification"]
        for entry in entries
        if entry["category"] == "test_classification"
    )
    summary = {
        "python_files_scanned": len(paths),
        "total_entries": len(entries),
        "category_counts": dict(sorted(category_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "test_classification_counts": {
            key: test_counts.get(key, 0) for key in ("KEEP", "REWRITE", "RETIRE")
        },
        "unresolved_entries": sum(bool(entry["unresolved"]) for entry in entries),
        "silent_exception_paths": sum(
            entry["manual_classification"].startswith("SILENT_")
            for entry in entries
        ),
        "broad_exception_paths": sum(
            bool(entry.get("details", {}).get("broad"))
            for entry in entries
            if entry["category"] == "exception"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "root": ".",
        "summary": summary,
        "entries": entries,
        "scope": {
            "tracked_python_only_when_git_available": True,
            "generated_json_committed": False,
            "runtime_activation_performed": False,
            "source_mutation_performed": False,
            "test_behavior_changed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = audit_repository(args.root)
    payload: Any = report["summary"] if args.summary_only else report
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
