#!/usr/bin/env python3
"""Generate the EVE M0-C persistence, state, and hormone-to-drive inventory.

The audit is read-only unless ``--output`` is explicitly supplied. Generated
JSON is intended for stdout or an ephemeral validation artifact and must not be
committed.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "1.0.0-m0-c"

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

FORMAT_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".jsonl.gz", "jsonl_gzip"),
    (".json.gz", "json_gzip"),
    (".pickle.gz", "pickle_gzip"),
    (".pkl.gz", "pickle_gzip"),
    (".sqlite3", "sqlite"),
    (".sqlite", "sqlite"),
    (".jsonl", "jsonl"),
    (".pickle", "pickle"),
    (".pkl", "pickle"),
    (".json", "json"),
    (".db", "database"),
    (".npy", "numpy"),
    (".npz", "numpy"),
    (".safetensors", "safetensors"),
    (".ckpt", "checkpoint"),
    (".checkpoint", "checkpoint"),
    (".pth", "torch_checkpoint"),
    (".pt", "torch_checkpoint"),
    (".joblib", "joblib"),
    (".yaml", "yaml"),
    (".yml", "yaml"),
    (".csv", "csv"),
    (".tsv", "tsv"),
    (".gz", "gzip"),
)

EXACT_IO_CALLS: dict[str, tuple[str, str]] = {
    "json.dump": ("write", "json"),
    "json.load": ("read", "json"),
    "json.dumps": ("serialize", "json"),
    "json.loads": ("deserialize", "json"),
    "pickle.dump": ("write", "pickle"),
    "pickle.load": ("read", "pickle"),
    "pickle.dumps": ("serialize", "pickle"),
    "pickle.loads": ("deserialize", "pickle"),
    "yaml.dump": ("write", "yaml"),
    "yaml.safe_dump": ("write", "yaml"),
    "yaml.load": ("read", "yaml"),
    "yaml.safe_load": ("read", "yaml"),
    "numpy.save": ("write", "numpy"),
    "numpy.savez": ("write", "numpy"),
    "numpy.savez_compressed": ("write", "numpy"),
    "numpy.load": ("read", "numpy"),
    "np.save": ("write", "numpy"),
    "np.savez": ("write", "numpy"),
    "np.savez_compressed": ("write", "numpy"),
    "np.load": ("read", "numpy"),
    "torch.save": ("write", "torch_checkpoint"),
    "torch.load": ("read", "torch_checkpoint"),
    "joblib.dump": ("write", "joblib"),
    "joblib.load": ("read", "joblib"),
    "shelve.open": ("read_write", "shelve"),
    "sqlite3.connect": ("read_write", "sqlite"),
    "gzip.open": ("container", "gzip"),
    "bz2.open": ("container", "bz2"),
    "lzma.open": ("container", "lzma"),
    "csv.reader": ("read", "csv"),
    "csv.DictReader": ("read", "csv"),
    "csv.writer": ("write", "csv"),
    "csv.DictWriter": ("write", "csv"),
    "os.replace": ("replace", "filesystem"),
    "os.rename": ("replace", "filesystem"),
    "shutil.copy": ("copy", "filesystem"),
    "shutil.copy2": ("copy", "filesystem"),
    "shutil.copyfile": ("copy", "filesystem"),
    "shutil.copytree": ("copy", "filesystem"),
    "shutil.move": ("replace", "filesystem"),
}

PATH_METHODS: dict[str, tuple[str, str]] = {
    "write_text": ("write", "text"),
    "write_bytes": ("write", "bytes"),
    "read_text": ("read", "text"),
    "read_bytes": ("read", "bytes"),
    "replace": ("replace", "filesystem"),
    "rename": ("replace", "filesystem"),
    "unlink": ("delete", "filesystem"),
    "touch": ("write", "filesystem"),
    "mkdir": ("create", "filesystem"),
}

GENERIC_IO_METHODS = {
    "checkpoint",
    "dump",
    "export",
    "flush",
    "import_state",
    "load",
    "load_checkpoint",
    "persist",
    "restore",
    "rollback",
    "save",
    "save_checkpoint",
    "snapshot",
    "write",
}

GENERIC_IO_RECEIVER_HINTS = {
    "artifact",
    "autosave",
    "cache",
    "checkpoint",
    "database",
    "db",
    "debug",
    "embedding",
    "export",
    "index",
    "memory",
    "persistence",
    "repository",
    "sidecar",
    "state",
    "store",
    "vector",
    "vocab",
}

STATE_DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "episodic_memory": (
        "episodic_memory",
        "episode_memory",
        "episode_store",
        "episodes",
        "episodic",
    ),
    "semantic_memory": (
        "semantic_memory",
        "concept_memory",
        "semantic_store",
        "knowledge_store",
    ),
    "self_model": (
        "self_model",
        "self_state",
        "integrated_self",
        "identity_state",
        "identity",
        "preferences",
    ),
    "relationships": (
        "relationship",
        "relationships",
        "relationship_state",
        "user_presence",
        "social_memory",
        "familiarity",
        "trust_state",
    ),
    "affect_hormones": (
        "affect",
        "affective",
        "emotion_state",
        "emotions",
        "hormone",
        "hormones",
        "hormone_state",
        "mood",
        "cortisol",
        "dopamine",
        "oxytocin",
        "serotonin",
        "adrenaline",
        "noradrenaline",
    ),
    "goals": (
        "goal",
        "goals",
        "goal_state",
        "goal_manager",
        "intention_state",
        "plan_state",
    ),
    "learned_parameters": (
        "learned_parameter",
        "learned_parameters",
        "learning_state",
        "model_state",
        "weights",
        "weight_state",
        "optimizer_state",
        "statistics",
        "running_stats",
    ),
    "vectors": (
        "vector",
        "vectors",
        "vector_store",
        "embedding",
        "embeddings",
        "embedding_store",
        "faiss",
    ),
    "vocabularies": (
        "vocab",
        "vocabulary",
        "vocabularies",
        "lexicon",
        "lexical_store",
        "token_map",
    ),
    "checkpoints": (
        "checkpoint",
        "checkpoints",
        "ckpt",
        "rollback_state",
        "state_snapshot",
    ),
    "autosave": (
        "autosave",
        "autosave_path",
        "autosave_target",
        "auto_save",
    ),
    "debug_exports": (
        "debug_export",
        "debug_snapshot",
        "state_debug",
        "diagnostic_export",
    ),
    "operator_artifacts": (
        "operator_artifact",
        "operator_evidence",
        "validation_artifact",
        "audit_artifact",
        "artifact_path",
        "report_path",
    ),
}

HORMONE_TOKENS = {
    "adrenaline",
    "affect",
    "cortisol",
    "dopamine",
    "emotion",
    "hormone",
    "hormones",
    "mood",
    "noradrenaline",
    "oxytocin",
    "serotonin",
}

DRIVE_TOKENS = {
    "action_tendency",
    "allostasis",
    "allostatic",
    "desire",
    "drive",
    "drives",
    "homeostasis",
    "motivation",
    "motive",
    "need",
    "needs",
    "priority",
    "urge",
    "urgency",
}

LEGACY_TEST_FORMAT_MARKERS = {
    "pickle",
    "sidecar",
    "sqlite",
    "eve.ckpt",
    ".ckpt",
    ".pickle",
    ".pkl",
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


def _symbol_tokens(value: str) -> set[str]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", camel_split.replace("-", "_"))
        if token
    }


def _normalized_symbol(value: str) -> str:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")


def _domains_for_symbol(value: str) -> list[str]:
    normalized = _normalized_symbol(value)
    if not normalized:
        return []
    domains: list[str] = []
    for domain, patterns in STATE_DOMAIN_PATTERNS.items():
        if any(
            normalized == pattern
            or normalized.startswith(f"{pattern}_")
            or normalized.endswith(f"_{pattern}")
            or f"_{pattern}_" in f"_{normalized}_"
            for pattern in patterns
        ):
            domains.append(domain)
    return domains


def _format_for_path(value: str) -> str | None:
    lowered = value.strip().lower().split("?", 1)[0].split("#", 1)[0]
    for suffix, format_name in FORMAT_SUFFIXES:
        if lowered.endswith(suffix):
            return format_name
    return None


def _path_like(value: str) -> bool:
    candidate = value.strip()
    if not candidate or len(candidate) > 512:
        return False
    if "\n" in candidate or "\r" in candidate:
        return False
    if _format_for_path(candidate) is not None:
        return True
    if candidate in {"/", "\\"} or "://" in candidate:
        return False
    if any(character.isspace() for character in candidate):
        return False
    if "/" not in candidate and "\\" not in candidate:
        return False
    persistence_path_tokens = {
        "artifact",
        "artifacts",
        "autosave",
        "cache",
        "checkpoint",
        "checkpoints",
        "ckpt",
        "database",
        "db",
        "debug",
        "embedding",
        "embeddings",
        "export",
        "manifest",
        "memory",
        "model",
        "operator",
        "report",
        "seed",
        "seeds",
        "sidecar",
        "snapshot",
        "state",
        "store",
        "subset",
        "validation",
        "vector",
        "vectors",
        "vocab",
        "vocabulary",
    }
    return bool(_symbol_tokens(candidate) & persistence_path_tokens)


def _write_mode_from_open(call: ast.Call) -> str | None:
    target = _dotted_name(call.func)
    if target not in {"open", "builtins.open", "gzip.open", "bz2.open", "lzma.open", "Path.open"} and not target.endswith(".open"):
        return None
    mode_node: ast.AST | None = call.args[1] if len(call.args) >= 2 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    return _constant_string(mode_node) or "r"


def _operation_from_mode(mode: str) -> str:
    if any(flag in mode for flag in "wax+"):
        return "read_write" if "+" in mode else "write"
    return "read"


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = _dotted_name(node)
        return [name] if name else []
    if isinstance(node, ast.Subscript):
        return [_dotted_name(node.value)] if _dotted_name(node.value) else []
    if isinstance(node, (ast.Tuple, ast.List)):
        result: list[str] = []
        for item in node.elts:
            result.extend(_target_names(item))
        return result
    return []


def _identifier_tokens(node: ast.AST) -> set[str]:
    tokens: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            tokens.update(_symbol_tokens(child.id))
        elif isinstance(child, ast.Attribute):
            tokens.update(_symbol_tokens(child.attr))
        elif isinstance(child, ast.arg):
            tokens.update(_symbol_tokens(child.arg))
    return tokens


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
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
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


def _classification_for(
    category: str,
    *,
    operation: str | None = None,
    heuristic: bool = False,
) -> tuple[str, str, bool]:
    if category == "persistence_io":
        if operation in {"write", "read_write", "replace", "copy", "delete", "create"}:
            return "PERSISTENCE_WRITE_OR_MUTATION_SITE", "medium" if heuristic else "high", True
        if operation == "read":
            return "PERSISTENCE_READ_OR_RESTORE_SITE", "medium" if heuristic else "high", True
        return "SERIALIZATION_OR_STORAGE_SITE", "medium" if heuristic else "high", True
    if category == "artifact_path":
        return "PERSISTENCE_ARTIFACT_PATH_CANDIDATE", "high", True
    if category == "state_domain":
        return "PERSISTENCE_INTENDED_STATE_CANDIDATE", "medium", True
    if category == "hormone_state":
        return "LEGACY_HORMONE_OR_AFFECT_STATE_SITE", "medium", True
    if category == "drive_state":
        return "DRIVE_OR_NEED_STATE_SITE", "medium", True
    if category == "hormone_drive_bridge":
        return "HORMONE_TO_DRIVE_MIGRATION_CANDIDATE", "high", True
    if category == "parse_error":
        return "UNRESOLVED_PARSE_ERROR", "high", True
    return "UNRESOLVED", "low", True


class PersistenceVisitor(ast.NodeVisitor):
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
        operation: str | None = None,
        heuristic: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        classification, confidence, unresolved = _classification_for(
            category,
            operation=operation,
            heuristic=heuristic,
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

    def _add_symbol_state(self, node: ast.AST, symbol: str, detector: str) -> None:
        domains = _domains_for_symbol(symbol)
        for domain in domains:
            self.add(
                "state_domain",
                node,
                f"state_symbol={symbol};domain={domain}",
                detector,
                details={"symbol": symbol, "domain": domain},
            )
        tokens = _symbol_tokens(symbol)
        hormone_hits = sorted(tokens & HORMONE_TOKENS)
        drive_hits = sorted(tokens & DRIVE_TOKENS)
        if hormone_hits:
            self.add(
                "hormone_state",
                node,
                f"hormone_symbol={symbol}",
                detector,
                details={"symbol": symbol, "tokens": hormone_hits},
            )
        if drive_hits:
            self.add(
                "drive_state",
                node,
                f"drive_symbol={symbol}",
                detector,
                details={"symbol": symbol, "tokens": drive_hits},
            )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        tokens = _identifier_tokens(node)
        hormone_hits = sorted(tokens & HORMONE_TOKENS)
        drive_hits = sorted(tokens & DRIVE_TOKENS)
        self.scope.append(node.name)
        if hormone_hits and drive_hits:
            self.add(
                "hormone_drive_bridge",
                node,
                f"hormone_tokens={','.join(hormone_hits)};drive_tokens={','.join(drive_hits)}",
                "AST callable token intersection",
                details={"hormone_tokens": hormone_hits, "drive_tokens": drive_hits},
            )
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self._add_symbol_state(node, node.name, "ast.ClassDef state symbol")
        self.generic_visit(node)
        self.scope.pop()

    def _visit_assignment(self, node: ast.AST, targets: Iterable[ast.AST]) -> None:
        for target in targets:
            for name in _target_names(target):
                self._add_symbol_state(node, name, f"{type(node).__name__} target")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._visit_assignment(node, node.targets)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_assignment(node, [node.target])

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._visit_assignment(node, [node.target])

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            value = _constant_string(key)
            if value is not None:
                self._add_symbol_state(key, value, "ast.Dict string key")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _path_like(node.value):
            format_name = _format_for_path(node.value) or "path_or_target"
            self.add(
                "artifact_path",
                node,
                f"artifact_path={node.value}",
                "ast.Constant path/format",
                details={"value": node.value, "format": format_name},
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted_name(node.func)
        leaf = target.rsplit(".", 1)[-1] if target else ""
        receiver = target.rsplit(".", 1)[0] if "." in target else ""

        mode = _write_mode_from_open(node)
        if mode is not None:
            operation = _operation_from_mode(mode)
            container_format = "file"
            if target.startswith("gzip") or target.endswith("gzip.open"):
                container_format = "gzip"
            elif target.startswith("bz2") or target.endswith("bz2.open"):
                container_format = "bz2"
            elif target.startswith("lzma") or target.endswith("lzma.open"):
                container_format = "lzma"
            self.add(
                "persistence_io",
                node,
                f"open_call={target};mode={mode}",
                "ast.Call open mode",
                operation=operation,
                details={"target": target, "mode": mode, "format": container_format, "operation": operation},
            )
        elif target in EXACT_IO_CALLS:
            operation, format_name = EXACT_IO_CALLS[target]
            self.add(
                "persistence_io",
                node,
                f"persistence_call={target}",
                "ast.Call exact persistence target",
                operation=operation,
                details={"target": target, "format": format_name, "operation": operation},
            )
        elif leaf in PATH_METHODS:
            operation, format_name = PATH_METHODS[leaf]
            self.add(
                "persistence_io",
                node,
                f"path_method={target}",
                "ast.Call path-like method",
                operation=operation,
                details={"target": target, "format": format_name, "operation": operation},
            )
        else:
            target_tokens = _symbol_tokens(target)
            receiver_tokens = _symbol_tokens(receiver)
            io_leaf = leaf.lower() in GENERIC_IO_METHODS
            receiver_match = bool(receiver_tokens & GENERIC_IO_RECEIVER_HINTS)
            if io_leaf and receiver_match:
                operation = "read" if leaf.lower().startswith(("load", "restore", "import")) else "write"
                self.add(
                    "persistence_io",
                    node,
                    f"persistence_method_candidate={target}",
                    "ast.Call persistence receiver heuristic",
                    operation=operation,
                    heuristic=True,
                    details={"target": target, "operation": operation, "receiver_tokens": sorted(receiver_tokens)},
                )
            if target_tokens & (HORMONE_TOKENS | DRIVE_TOKENS):
                self._add_symbol_state(node, target, "ast.Call state symbol")

        self.generic_visit(node)


class TestPersistenceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.signals: list[tuple[int, str, bool]] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for marker in sorted(LEGACY_TEST_FORMAT_MARKERS):
                if marker in lowered:
                    self.signals.append((int(getattr(node, "lineno", 1)), f"legacy_format={marker}", True))
                    break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted_name(node.func)
        if target in EXACT_IO_CALLS:
            operation, format_name = EXACT_IO_CALLS[target]
            legacy = format_name in {"pickle", "sqlite"}
            self.signals.append((int(getattr(node, "lineno", 1)), f"persistence_call={target};operation={operation}", legacy))
        else:
            mode = _write_mode_from_open(node)
            if mode is not None:
                self.signals.append((int(getattr(node, "lineno", 1)), f"open_mode={mode}", False))
        self.generic_visit(node)


def classify_test(path: str, tree: ast.AST) -> dict[str, Any]:
    visitor = TestPersistenceVisitor()
    visitor.visit(tree)
    line = 1
    evidence = "conservative_default_keep"
    confidence = "high"
    unresolved = False
    reason = (
        "Preserve executable persistence and state evidence until a later migration "
        "patch provides exact behavior-preserving rewrite or retirement evidence."
    )
    if visitor.signals:
        line, evidence, legacy = sorted(visitor.signals)[0]
        if legacy:
            confidence = "medium"
            unresolved = True
            reason = (
                "Preserve this legacy-format persistence assertion as migration evidence; "
                "manual review must decide whether the future implementation needs a "
                "behavior-preserving rewrite."
            )
    return {
        "category": "test_classification",
        "path": path,
        "line_start": line,
        "line_end": line,
        "callable": "<test-file>",
        "mechanical_evidence": evidence,
        "detector": "AST M0-C test policy",
        "manual_classification": "KEEP",
        "manual_reason": reason,
        "confidence": confidence,
        "unresolved": unresolved,
        "manual_only": False,
    }


def _parse_file(root: Path, path: Path) -> tuple[ast.AST | None, str | None, int]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")
    try:
        return ast.parse(source, filename=path.relative_to(root).as_posix()), None, 1
    except SyntaxError as exc:
        return None, exc.msg or "syntax error", int(exc.lineno or 1)


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = list(iter_python_files(root))
    entries: list[dict[str, Any]] = []

    for path in paths:
        relative = path.relative_to(root).as_posix()
        tree, parse_error, parse_line = _parse_file(root, path)
        if tree is None:
            classification, confidence, unresolved = _classification_for("parse_error")
            entries.append(
                {
                    "category": "parse_error",
                    "path": relative,
                    "line_start": parse_line,
                    "line_end": parse_line,
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

        visitor = PersistenceVisitor(relative)
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
    classification_counts = Counter(entry["manual_classification"] for entry in entries)
    domain_counts = Counter(
        entry.get("details", {}).get("domain")
        for entry in entries
        if entry["category"] == "state_domain" and entry.get("details", {}).get("domain")
    )
    format_counts = Counter(
        entry.get("details", {}).get("format")
        for entry in entries
        if entry["category"] in {"persistence_io", "artifact_path"}
        and entry.get("details", {}).get("format")
    )
    operation_counts = Counter(
        entry.get("details", {}).get("operation")
        for entry in entries
        if entry["category"] == "persistence_io"
        and entry.get("details", {}).get("operation")
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
        "state_domain_counts": dict(sorted(domain_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "persistence_operation_counts": dict(sorted(operation_counts.items())),
        "test_classification_counts": {
            key: test_counts.get(key, 0) for key in ("KEEP", "REWRITE", "RETIRE")
        },
        "unresolved_entries": sum(bool(entry["unresolved"]) for entry in entries),
        "hormone_state_sites": category_counts.get("hormone_state", 0),
        "drive_state_sites": category_counts.get("drive_state", 0),
        "hormone_drive_bridge_candidates": category_counts.get("hormone_drive_bridge", 0),
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
            "persistence_activation_performed": False,
            "source_mutation_performed": False,
            "test_behavior_changed": False,
            "hormone_drive_migration_performed": False,
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
