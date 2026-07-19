#!/usr/bin/env python3
"""Generate the EVE M0-D component, life-loop, and module-disposition audit.

The command performs static analysis only. It does not import production runtime
modules, start loops, load vectors/models, or mutate repository state. JSON is
written to stdout unless ``--output`` is explicitly supplied for ephemeral CI
use.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "1.0.0-m0-d"
BASELINE_SHA = "fe10cd954bdf445400ea6aa9708dd214ed761114"

EXCLUDED_PARTS = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "__pycache__", "build", "dist", "node_modules", "venv",
}
NON_RUNTIME_ROOTS = {
    ".codex", ".github", "docs", "eve_v3_autonomous_handoff", "scripts", "tests",
}
RUNTIME_PACKAGE_ROOTS = {"adapters", "cognition", "core", "korean", "language", "utils"}
TOP_LEVEL_RUNTIME_FILES = {"main.py"}
DISPOSITIONS = {"KEEP", "WRAP", "REWRITE", "EXPERIMENTAL", "DEPRECATE", "REMOVE"}
LOOP_TAXONOMY_ORDER = (
    "Vital", "Cognitive", "Goal", "Activity", "Learning", "Memory", "Social",
    "Expression", "no-v4-equivalent",
)
LOOP_TAXONOMY = set(LOOP_TAXONOMY_ORDER)

LEARN_METHOD_TOKENS = {
    "adapt", "adjust", "consolidate", "embed", "encode", "fit", "learn",
    "observe", "promote", "stabilize", "train", "transform", "update", "vectorize",
}
VECTOR_TOKENS = {
    "embedding", "embeddings", "fasttext", "feature", "matrix", "numpy", "pmi",
    "representation", "svd", "tensor", "vector", "vectors", "vocab", "vocabulary",
    "weight", "weights",
}
NUMERIC_CALL_LEAVES = {
    "array", "asarray", "dot", "einsum", "empty", "eye", "load", "loadtxt",
    "matmul", "memmap", "ones", "save", "savez", "savez_compressed", "savetxt",
    "svd", "zeros",
}
ARTIFACT_CALL_LEAVES = {
    "load", "loads", "read", "read_bytes", "read_text", "restore", "save", "savetxt",
    "write", "write_bytes", "write_text",
}
LOOP_NAME_TOKENS = {
    "autonomous", "daemon", "dmn", "live_loop", "loop", "proactive", "spontaneous",
    "tick", "worker",
}
CLOCK_CALLS = {
    "asyncio.sleep", "datetime.datetime.now", "datetime.now", "time.monotonic",
    "time.perf_counter", "time.sleep", "time.time",
}
CONCURRENCY_CALL_LEAVES = {
    "Event", "Lock", "Process", "Queue", "RLock", "Semaphore", "Task", "Thread",
    "create_task", "join", "run_in_executor", "start", "wait",
}

MANUAL_DISPOSITION_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "main.py": (
        "REWRITE", "high",
        "Active composition root mixes construction, command dispatch, automatic background start, and persistence boundaries that must be separated into v4 capabilities and event-driven services.",
    ),
    "language/streaming.py": (
        "REWRITE", "high",
        "Active chat funnel combines raw input handling, state mutation, learning, history, and expression; v4 requires structural separation between quarantined source text, cognition, and expression.",
    ),
    "adapters/live_loop.py": (
        "REWRITE", "high",
        "Active daemon loop combines clocks, hormone mutation, autonomy, proactive output, queues, and autosave; v4 requires explicit loop taxonomy, event provenance, lifecycle ownership, and isolated persistence.",
    ),
    "core/autonomous.py": (
        "REWRITE", "high",
        "Active autonomous step combines need detection, state transition, environment mutation, curiosity, history, and proactive expression; it must be decomposed into v4 goal/activity/learning/expression loops.",
    ),
    "adapters/persistence_adapter.py": (
        "REWRITE", "high",
        "Current legacy persistence plus gzip/pickle sidecar conflicts with the future append-only SQLite event log and validated snapshot architecture.",
    ),
    "adapters/hormone_adapter.py": (
        "REWRITE", "high",
        "Current hormone representation is a legacy affect substrate; v4 requires core drives, appraisal, and derived emotion with continuity-preserving migration.",
    ),
    "adapters/allostatic_adapter.py": (
        "WRAP", "high",
        "Allostatic coupling is potentially reusable as a bounded Vital-loop compatibility projection but must not remain an implicit hormone-to-agency bridge.",
    ),
    "adapters/urge_adapter.py": (
        "WRAP", "high",
        "Urge computation may inform v4 drives but currently consumes hormone state and feeds proactive behavior; retain only behind an explicit compatibility boundary.",
    ),
}

FROZEN_PR_RECOMMENDATIONS: list[dict[str, Any]] = [
    {"pr": 109, "recommendation": "REWRITE-AS-V4-CONTRACT", "reason": "Deterministic read-only conclusion-candidate evidence is useful, but the branch was authored against the superseded v3 schema ladder and an M0-prebaseline main. Preserve strict-JSON, canonical-ID, forbidden-field, tamper, and no-side-effect tests, then restate the contract under v4 observation/provenance, quarantine, and expression boundaries.", "preserve": "Adapter validation matrix, canonical-ID rules, hostile-input tests, recursive forbidden-field tests, tamper tests, and downstream read-only plan invariants.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 97, "recommendation": "CLOSE-PRESERVE-EVIDENCE", "reason": "The branch is an obsolete pre-fix duplicate of non-visual virtual-situation work later merged and corrected on main, including deterministic ID repair. Merging it would reintroduce an older authority and duplicate files.", "preserve": "Fail-closed situation validation cases, entity/relationship fixtures, read-only handoff-plan assertions, and historical evidence of the deterministic-ID defect.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 86, "recommendation": "REWRITE-AS-V4-CONTRACT", "reason": "Memory replay provenance and boundary tests remain relevant, but the contract uses v3 memory/fact/hormone terminology and predates the v4 event, source-claim, and forgetting architecture.", "preserve": "Replay source classes, confidence/boundary matrices, no-mutation assertions, origin/fact-status tests, and operator-report fixtures.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 84, "recommendation": "REWRITE-AS-V4-CONTRACT", "reason": "Cross-modal preflight evidence is compatible with v4's candidate-only observation boundary, but the branch predates v4 provenance, model-version, capability, and quarantine requirements.", "preserve": "Modality compatibility matrix, identity-resolution prohibition, fail-closed cases, deterministic behavior, and no-side-effect tests.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 82, "recommendation": "REWRITE-AS-V4-CONTRACT", "reason": "The multimodal event candidate contract is useful source-boundary evidence but is governed by v3.1 assumptions and predates v4's explicit source store, model/tool version provenance, and expression isolation.", "preserve": "Supported modality/event matrices, mixed-boundary cases, no-fact/no-identity assertions, and quarantine/AGP/fallback safety tests.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 11, "recommendation": "ABSORB-INTO-M1", "reason": "The branch proposes operator-controlled persistence activation and ephemeral runtime mutation, which cannot merge during M0 and conflicts with the v4 event-store direction. Its safety proofs should become M1 requirements without preserving the activation implementation.", "preserve": "Operator approval guard, checkpoint-before-mutation ordering, before/after debug evidence, rollback verification, and protected-state non-mutation tests.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 7, "recommendation": "ABSORB-INTO-M1", "reason": "The branch combines vector restoration, manual validation, AGP proof expansion, and persistence approval/decision logic on an obsolete baseline. Preserve its controls as M1 bounded learned-subsystem and persistence requirements; do not merge the mixed bundle.", "preserve": "Vector manifest/checksum/shape/dtype gates, ignored-artifact boundary, approval decision records, AGP proof tests, and fail-closed hard-stop behavior.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 4, "recommendation": "ABSORB-INTO-M1", "reason": "The read-only medium-vector restoration audit contains useful external-seed provenance and artifact validation evidence, but its fixed 30k fastText workflow is a v3 implementation decision rather than a v4 bounded learned-subsystem contract.", "preserve": "Manifest provenance, SHA-256, shape/dtype verification, no-download/no-copy default, no-binary-commit policy, and missing-artifact fail-closed tests.", "manual_only": True, "confidence": "high", "unresolved": False},
    {"pr": 1, "recommendation": "CLOSE-PRESERVE-EVIDENCE", "reason": "The split Round96 source-package restore workflow addresses a historical checkout artifact that no longer governs the current repository. Retain hash-verification and hard-stop evidence in history, not in v4 runtime or governance.", "preserve": "Manifest hash verification, missing-part failure behavior, safe extraction checks, and reports documenting the historical package blocker.", "manual_only": True, "confidence": "high", "unresolved": False},
]


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


def _symbol_tokens(value: str) -> set[str]:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return {part for part in normalized.split("_") if part}


def _git_tracked_python_files(root: Path) -> list[Path]:
    try:
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"], stderr=subprocess.DEVNULL)
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


def _is_runtime_module(relative: Path) -> bool:
    if not relative.parts or relative.parts[0] in NON_RUNTIME_ROOTS:
        return False
    if relative.name.startswith("test_") or relative.name.endswith("_test.py"):
        return False
    if len(relative.parts) == 1:
        return relative.name in TOP_LEVEL_RUNTIME_FILES or relative.name.startswith("eve_")
    return relative.parts[0] in RUNTIME_PACKAGE_ROOTS


def _module_name(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _load_audit_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cross_reference_reports(root: Path) -> dict[str, dict[str, Any]]:
    sources = {
        "M0-A": root / "scripts/audit/m0_a_runtime_inventory.py",
        "M0-B": root / "scripts/audit/m0_b_controlflow_concurrency_inventory.py",
        "M0-C": root / "scripts/audit/m0_c_persistence_state_inventory.py",
    }
    reports: dict[str, dict[str, Any]] = {}
    for index, (label, path) in enumerate(sources.items()):
        module = _load_audit_module(path, f"_eve_{label.lower().replace('-', '_')}_{index}")
        reports[label] = module.audit_repository(root)
    return reports


def _reference_index(reports: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    documents = {
        "M0-A": "docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md",
        "M0-B": "docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md",
        "M0-C": "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md",
    }
    for label, report in reports.items():
        for entry in report.get("entries", []):
            if entry.get("category") in {"test_classification", "parse_error"}:
                continue
            indexed[str(entry.get("path", ""))].append({
                "audit": label,
                "document": documents[label],
                "path": entry.get("path"),
                "line_start": entry.get("line_start"),
                "line_end": entry.get("line_end"),
                "callable": entry.get("callable"),
                "category": entry.get("category"),
                "classification": entry.get("manual_classification"),
                "mechanical_evidence": entry.get("mechanical_evidence"),
                "unresolved": bool(entry.get("unresolved")),
            })
    for values in indexed.values():
        values.sort(key=lambda item: (item["audit"], int(item["line_start"] or 1), str(item["category"]), str(item["mechanical_evidence"])))
    return indexed


class ComponentVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.entries: list[dict[str, Any]] = []
        self.imported_numpy_aliases: set[str] = set()
        self.imported_modules: list[tuple[str, int]] = []
        self.life_candidates: dict[tuple[str, int], dict[str, Any]] = {}
        self.life_stack: list[tuple[str, int]] = []

    @property
    def symbol(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def add(self, node: ast.AST, detection: str, evidence: str, classification: str, *, confidence: str = "medium", unresolved: bool = True, details: dict[str, Any] | None = None) -> None:
        entry: dict[str, Any] = {
            "category": "component_evidence", "path": self.path,
            "line_start": int(getattr(node, "lineno", 1)), "line_end": _node_end(node),
            "symbol": self.symbol, "detection": detection, "evidence": evidence,
            "classification": classification, "confidence": confidence,
            "unresolved": unresolved, "manual_only": False,
        }
        if details:
            entry["details"] = details
        self.entries.append(entry)

    def _life_candidate(self, node: ast.AST, name: str, kind: str) -> tuple[str, int] | None:
        tokens = _symbol_tokens(name)
        if not (tokens & LOOP_NAME_TOKENS or name.lower() in {"run", "_run", "step", "tick"}):
            return None
        key = (self.symbol, int(getattr(node, "lineno", 1)))
        self.life_candidates.setdefault(key, {
            "path": self.path, "line_start": int(getattr(node, "lineno", 1)),
            "line_end": _node_end(node), "symbol": self.symbol, "kind": kind,
            "name": name, "clock_calls": [], "concurrency_calls": [], "output_calls": [],
        })
        return key

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported_modules.append((alias.name, int(getattr(node, "lineno", 1))))
            if alias.name == "numpy":
                self.imported_numpy_aliases.add(alias.asname or "numpy")
                self.add(node, "ast.Import", f"numpy_import={alias.asname or 'numpy'}", "NUMERIC_REPRESENTATION_DEPENDENCY", confidence="high", unresolved=False)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self.imported_modules.append((module, int(getattr(node, "lineno", 1))))
        if module == "numpy":
            for alias in node.names:
                self.imported_numpy_aliases.add(alias.asname or alias.name)
            self.add(node, "ast.ImportFrom", f"numpy_from_import={','.join(alias.name for alias in node.names)}", "NUMERIC_REPRESENTATION_DEPENDENCY", confidence="high", unresolved=False)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        tokens = _symbol_tokens(node.name)
        matched = sorted(tokens & LEARN_METHOD_TOKENS)
        if matched:
            self.add(node, type(node).__name__, f"adaptive_method_tokens={','.join(matched)}", "ADAPTIVE_OR_LEARNING_METHOD_CANDIDATE", details={"method_name": node.name})
        vector_matches = sorted(tokens & VECTOR_TOKENS)
        if vector_matches:
            self.add(node, type(node).__name__, f"representation_method_tokens={','.join(vector_matches)}", "VECTOR_OR_NUMERIC_METHOD_CANDIDATE", details={"method_name": node.name})
        life_key = self._life_candidate(node, node.name, "callable")
        if life_key is not None:
            self.life_stack.append(life_key)
        self.generic_visit(node)
        if life_key is not None:
            self.life_stack.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        tokens = _symbol_tokens(node.name)
        vector_matches = sorted(tokens & VECTOR_TOKENS)
        if vector_matches:
            self.add(node, "ast.ClassDef", f"representation_class_tokens={','.join(vector_matches)}", "VECTOR_OR_NUMERIC_COMPONENT_CANDIDATE", details={"class_name": node.name})
        life_key = self._life_candidate(node, node.name, "class")
        if life_key is not None:
            self.life_stack.append(life_key)
        self.generic_visit(node)
        if life_key is not None:
            self.life_stack.pop()
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        target = _dotted_name(node.func)
        leaf = target.rsplit(".", 1)[-1] if target else ""
        root = target.split(".", 1)[0] if target else ""
        tokens = _symbol_tokens(target)
        if root in self.imported_numpy_aliases or (root in {"numpy", "np"} and leaf in NUMERIC_CALL_LEAVES):
            self.add(node, "ast.Call numeric", f"numeric_call={target}", "NUMERIC_REPRESENTATION_OPERATION", confidence="high", unresolved=False, details={"target": target})
        if leaf in ARTIFACT_CALL_LEAVES and tokens & VECTOR_TOKENS:
            self.add(node, "ast.Call artifact", f"representation_artifact_call={target}", "VECTOR_OR_VOCAB_ARTIFACT_IO_CANDIDATE", details={"target": target})
        learn_matches = sorted(tokens & LEARN_METHOD_TOKENS)
        if learn_matches:
            self.add(node, "ast.Call adaptive", f"adaptive_call_tokens={','.join(learn_matches)} target={target}", "ADAPTIVE_STATE_TRANSITION_CANDIDATE", details={"target": target})
        if self.life_stack:
            candidate = self.life_candidates[self.life_stack[-1]]
            if target in CLOCK_CALLS or leaf in {"sleep", "monotonic", "time", "wait"}:
                candidate["clock_calls"].append({"line": int(getattr(node, "lineno", 1)), "target": target})
            if leaf in CONCURRENCY_CALL_LEAVES:
                candidate["concurrency_calls"].append({"line": int(getattr(node, "lineno", 1)), "target": target})
            if leaf in {"emit", "publish", "send", "respond", "speak"} or any(token in target.lower() for token in ("proactive", "speech", "output")):
                candidate["output_calls"].append({"line": int(getattr(node, "lineno", 1)), "target": target})
        self.generic_visit(node)

    def _assignment_target_names(self, target: ast.AST) -> Iterable[str]:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Attribute):
            yield _dotted_name(target)
        elif isinstance(target, ast.Subscript):
            yield _dotted_name(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                yield from self._assignment_target_names(item)

    def _record_assignment(self, node: ast.AST, targets: Iterable[ast.AST]) -> None:
        for target in targets:
            for name in self._assignment_target_names(target):
                matches = sorted(_symbol_tokens(name) & VECTOR_TOKENS)
                if matches:
                    self.add(node, type(node).__name__, f"representation_state_assignment={name} tokens={','.join(matches)}", "NUMERIC_OR_LEARNED_STATE_CANDIDATE", details={"target": name})

    def visit_Assign(self, node: ast.Assign) -> None:
        self._record_assignment(node, node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_assignment(node, [node.target])
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_assignment(node, [node.target])
        self.generic_visit(node)


def _resolve_import_to_path(module: str, current: Path, module_paths: dict[str, str]) -> str | None:
    if not module:
        return None
    if module in module_paths:
        return module_paths[module]
    root = module.split(".", 1)[0]
    if root in module_paths:
        return module_paths[root]
    current_module = _module_name(current)
    if "." in current_module:
        combined = f"{current_module.rsplit('.', 1)[0]}.{module}"
        if combined in module_paths:
            return module_paths[combined]
    return None


def _reachable_modules(module_paths: dict[str, str], imports_by_path: dict[str, list[tuple[str, int]]]) -> set[str]:
    roots = {"main.py", "language/streaming.py"} & set(imports_by_path)
    graph: dict[str, set[str]] = defaultdict(set)
    for path_text, imports in imports_by_path.items():
        current = Path(path_text)
        for module, _line in imports:
            target = _resolve_import_to_path(module, current, module_paths)
            if target:
                graph[path_text].add(target)
    seen: set[str] = set()
    queue: deque[str] = deque(sorted(roots))
    while queue:
        item = queue.popleft()
        if item in seen:
            continue
        seen.add(item)
        for child in sorted(graph.get(item, set())):
            if child not in seen:
                queue.append(child)
    return seen


def _taxonomy_for(path: str, symbol: str) -> list[str]:
    value = f"{path} {symbol}".lower()
    categories: list[str] = []
    if any(token in value for token in ("hormone", "allostat", "homeostat", "vital", "live_loop")):
        categories.append("Vital")
    if any(token in value for token in ("dmn", "reason", "cognit", "think", "workspace")):
        categories.append("Cognitive")
    if any(token in value for token in ("goal", "intent", "plan", "urge", "need")):
        categories.append("Goal")
    if any(token in value for token in ("autonomous", "activity", "action", "outing", "live_loop")):
        categories.append("Activity")
    if any(token in value for token in ("learn", "adapt", "train", "curiosity", "stabil")):
        categories.append("Learning")
    if any(token in value for token in ("memory", "consolid", "replay")):
        categories.append("Memory")
    if any(token in value for token in ("social", "relationship", "user_presence")):
        categories.append("Social")
    if any(token in value for token in ("speech", "proactive", "spontaneous", "stream", "expression", "dmn")):
        categories.append("Expression")
    if not categories:
        categories.append("no-v4-equivalent")
    return [category for category in LOOP_TAXONOMY_ORDER if category in categories]


def _entry_ref(entry: dict[str, Any], document: str) -> dict[str, Any]:
    return {"document": document, "path": entry.get("path"), "line_start": entry.get("line_start"), "line_end": entry.get("line_end"), "symbol": entry.get("callable", entry.get("symbol", "<module>")), "detection": entry.get("detector", entry.get("detection")), "evidence": entry.get("mechanical_evidence", entry.get("evidence")), "classification": entry.get("manual_classification", entry.get("classification"))}


def _build_life_loops(life_candidates: list[dict[str, Any]], reference_index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in life_candidates:
        path, symbol = candidate["path"], candidate["symbol"]
        references = [ref for ref in reference_index.get(path, []) if ref.get("callable") == symbol and ref.get("category") in {"mutation", "direct_write", "clock", "queue", "concurrency", "output", "exception", "bypass", "entrypoint", "execution_boundary"}]
        triggers: list[str] = []
        if candidate["clock_calls"] or any(ref.get("category") == "clock" for ref in references):
            triggers.append("clock_or_delay")
        if candidate["concurrency_calls"] or any(ref.get("category") in {"queue", "concurrency", "execution_boundary"} for ref in references):
            triggers.append("thread_queue_or_task_lifecycle")
        name = candidate["name"].lower()
        if name in {"start", "run", "_run", "step", "tick"}:
            triggers.append("explicit_callable_invocation")
        if "proactive" in name or "spontaneous" in name or "dmn" in name:
            triggers.append("internal_proactive_or_dmn_condition")
        if not triggers:
            triggers.append("call_graph_unresolved")
        mutation_refs = [ref for ref in references if ref.get("category") in {"mutation", "direct_write"}]
        evidence_refs = [{"document": ref["document"], "path": ref["path"], "line_start": ref["line_start"], "line_end": ref["line_end"], "symbol": ref["callable"], "detection": ref["category"], "evidence": ref["mechanical_evidence"], "classification": ref["classification"]} for ref in references[:12]]
        results.append({
            "category": "life_loop", "path": path, "line_start": candidate["line_start"], "line_end": candidate["line_end"], "symbol": symbol,
            "detection": "AST callable/class name plus M0-A/M0-B cross-reference", "evidence": f"life_loop_candidate={candidate['name']}",
            "classification": "LIFE_LOOP_ASSESSMENT", "confidence": "high" if references else "medium", "unresolved": not bool(references), "manual_only": False,
            "trigger": sorted(set(triggers)), "mutates": [f"{ref['path']}:{ref['line_start']}-{ref['line_end']} {ref['mechanical_evidence']}" for ref in mutation_refs[:12]],
            "mutation_reference_count": len(mutation_refs), "v4_loop_taxonomy": _taxonomy_for(path, symbol),
            "clock_calls": candidate["clock_calls"], "concurrency_calls": candidate["concurrency_calls"], "output_calls": candidate["output_calls"], "evidence_references": evidence_refs,
        })
    results.sort(key=lambda item: (item["path"], item["line_start"], item["symbol"]))
    return results


def _module_disposition(path: str, reachable: bool, parse_error: dict[str, Any] | None, component_entries: list[dict[str, Any]], references: list[dict[str, Any]]) -> tuple[str, str, bool, str]:
    override = MANUAL_DISPOSITION_OVERRIDES.get(path)
    if override is not None:
        disposition, confidence, reason = override
        return disposition, confidence, False, reason
    if parse_error is not None or Path(path).name.startswith("eve_foundation_v"):
        return "DEPRECATE", "high", False, "Tracked legacy foundation/snapshot module is not safely analyzable or is explicitly versioned legacy; preserve for historical/migration evidence but exclude from future runtime authority."
    categories = Counter(ref.get("category") for ref in references)
    classifications = {str(ref.get("classification")) for ref in references}
    component_classes = {entry["classification"] for entry in component_entries}
    has_persistence = bool(categories.get("persistence_io") or categories.get("direct_write"))
    has_concurrency = bool(categories.get("concurrency") or categories.get("queue") or categories.get("execution_boundary"))
    has_external_boundary = any(token in path.lower() for token in ("server", "web", "voice", "vision", "sensory", "device"))
    has_adaptive = any("ADAPTIVE" in item or "LEARN" in item for item in component_classes)
    has_numeric = any("VECTOR" in item or "NUMERIC" in item for item in component_classes)
    has_hormone = bool(categories.get("hormone_state") or categories.get("hormone_drive_bridge"))
    has_active = any(value.startswith("ACTIVE_") for value in classifications)
    if reachable and (has_concurrency or has_external_boundary or has_persistence):
        return "WRAP", "medium", True, "Reachable module crosses persistence, concurrency, device/network, or observable-effect boundaries; retain behavior only behind explicit v4 capability, lifecycle, provenance, and rollback wrappers pending M1 design."
    if reachable and has_hormone:
        return "REWRITE", "medium", True, "Reachable module is coupled to legacy hormone/affect state; v4 requires drive/appraisal/derived-emotion semantics and a continuity-preserving migration contract."
    if reachable and (has_adaptive or has_numeric):
        return "EXPERIMENTAL", "medium", True, "Reachable learned or numeric representation component lacks a complete v4 bounded-subsystem provenance/evaluation/version/rollback decision; preserve as experimental evidence with default no-load."
    if reachable or has_active:
        return "KEEP", "medium", True, "Module is reachable from the active composition graph or has active M0 evidence and no stronger rewrite/wrap trigger; retain while M1 assigns explicit v4 ownership and contracts."
    if has_adaptive or has_numeric:
        return "EXPERIMENTAL", "medium", True, "Unreached neural/vector/adaptive candidate has mechanical evidence but no active-runtime proof; preserve for bounded evaluation rather than promoting or deleting it."
    return "EXPERIMENTAL", "low", True, "Static analysis found no active-root reachability and insufficient evidence for deletion or deprecation; preserve as experimental/unresolved until reviewer disposition."


def _build_conflicts() -> list[dict[str, Any]]:
    return [
        {"id": "event-log-vs-direct-mutation", "assumption": "Meaningful state transitions are represented by replayable events with causal provenance.", "runtime_reality": "The active chat, live, autonomous, and persistence funnels perform distributed direct mutation and writes without one event-kernel boundary.", "evidence": ["docs/EVE_DESIGN_v4.md:29-35", "docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md:78-86"], "unresolved": True},
        {"id": "sqlite-event-store-vs-pickle-sidecar", "assumption": "M1/M2 persistence uses append-only SQLite events and validated snapshots.", "runtime_reality": "Current active persistence combines legacy state with gzip/pickle sidecars, explicit operator save/load, and automatic autosave.", "evidence": ["docs/EVE_DESIGN_v4.md:37-39", "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:126-147"], "unresolved": True},
        {"id": "affect-migration-plan-missing", "assumption": "M0 proposes migration from 26 hormones toward core drives, appraisal, and derived emotion while preserving continuity.", "runtime_reality": "M0-C inventories 1,777 hormone/affect sites, 386 drive/need sites, and 54 bridge candidates, but contains no concrete migration phases, state mapping, compatibility projection, event/snapshot migration, rollback, or acceptance criteria.", "evidence": ["docs/EVE_DESIGN_v4.md:45-47", "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:157-176", "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:191-203"], "unresolved": True},
        {"id": "speech-output-vs-life-continuity", "assumption": "Timer ticks, hormone decay, and proactive speech are not proof of life; continuity depends on state, goals, learning, and resumable activity.", "runtime_reality": "Current LiveLoop, AutonomousLoop, and DMN/proactive paths converge heavily on timed output and speech while lifecycle/state ownership is distributed.", "evidence": ["docs/EVE_DESIGN_v4.md:57-59", "docs/audit/M0_B_GATE_FAILURE_CLOCK_CONCURRENCY_MAP.md:90-96", "docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md:80-86"], "unresolved": True},
        {"id": "source-quarantine-vs-chat-funnel", "assumption": "Raw external text is confined to a quarantined source store and expression cannot read it.", "runtime_reality": "The active StreamingEngine chat funnel receives input, mutates context/learning/history, and produces expression inside one module boundary; structural source-store isolation is not demonstrated by M0 evidence.", "evidence": ["docs/EVE_DESIGN_v4.md:25-27", "docs/audit/M0_A_RUNTIME_ENTRYPOINT_AND_MUTATION_MAP.md:78-79"], "unresolved": True},
        {"id": "bounded-learned-subsystems-vs-distributed-numeric-state", "assumption": "Learned subsystems require provenance, confidence, capability, evaluation, versioning, rollback, and default no-load.", "runtime_reality": "Numeric/vector/adaptive state and update methods are distributed across many modules and artifact formats; a single bounded activation/version/rollback contract is not present.", "evidence": ["docs/EVE_DESIGN_v4.md:17-19", "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:62-80", "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md:82-124"], "unresolved": True},
    ]


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    all_paths = list(iter_python_files(root))
    runtime_paths = [path for path in all_paths if _is_runtime_module(path.relative_to(root))]
    reports = _cross_reference_reports(root)
    references_by_path = _reference_index(reports)
    component_entries: list[dict[str, Any]] = []
    life_candidates: list[dict[str, Any]] = []
    imports_by_path: dict[str, list[tuple[str, int]]] = {}
    module_paths: dict[str, str] = {}
    parse_errors: dict[str, dict[str, Any]] = {}
    for path in runtime_paths:
        relative = path.relative_to(root)
        path_text = relative.as_posix()
        module_paths[_module_name(relative)] = path_text
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=path_text, type_comments=True)
        except (SyntaxError, ValueError) as exc:
            line = int(getattr(exc, "lineno", 1) or 1)
            parse_errors[path_text] = {"category": "parse_error", "path": path_text, "line_start": line, "line_end": line, "symbol": "<module>", "detection": "ast.parse", "evidence": str(exc), "classification": "UNRESOLVED_PARSE_ERROR", "confidence": "high", "unresolved": True, "manual_only": False}
            imports_by_path[path_text] = []
            continue
        visitor = ComponentVisitor(path_text)
        visitor.visit(tree)
        component_entries.extend(visitor.entries)
        life_candidates.extend(visitor.life_candidates.values())
        imports_by_path[path_text] = visitor.imported_modules
    reachable = _reachable_modules(module_paths, imports_by_path)
    components_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in component_entries:
        components_by_path[entry["path"]].append(entry)
    module_dispositions: list[dict[str, Any]] = []
    for path in sorted(imports_by_path):
        refs = references_by_path.get(path, [])
        disposition, confidence, unresolved, reason = _module_disposition(path, path in reachable, parse_errors.get(path), components_by_path.get(path, []), refs)
        if disposition not in DISPOSITIONS:
            raise AssertionError(f"invalid disposition {disposition} for {path}")
        evidence_references = [{"document": ref["document"], "path": ref["path"], "line_start": ref["line_start"], "line_end": ref["line_end"], "symbol": ref["callable"], "detection": ref["category"], "evidence": ref["mechanical_evidence"], "classification": ref["classification"]} for ref in refs[:12]]
        for entry in components_by_path.get(path, [])[:8]:
            evidence_references.append(_entry_ref(entry, "scripts/audit/m0_d_component_inventory.py"))
        if parse_errors.get(path):
            evidence_references.append(_entry_ref(parse_errors[path], "scripts/audit/m0_d_component_inventory.py"))
        module_dispositions.append({
            "category": "module_disposition", "path": path, "line_start": 1, "line_end": 1, "symbol": "<module>",
            "detection": "AST import reachability plus M0-A/B/C and M0-D evidence",
            "evidence": f"reachable_from_active_root={path in reachable}; references={len(refs)}; component_evidence={len(components_by_path.get(path, []))}",
            "classification": disposition, "confidence": confidence, "unresolved": unresolved,
            "manual_only": path in MANUAL_DISPOSITION_OVERRIDES, "reason": reason,
            "reachable_from_active_root": path in reachable, "evidence_references": evidence_references,
        })
    life_loops = _build_life_loops(life_candidates, references_by_path)
    unresolved_items = [{"category": "governance_gap", "path": "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md", "line_start": 157, "line_end": 176, "symbol": "Hormone-to-drive migration inventory", "detection": "manual governing-document comparison", "evidence": "M0-C inventories bridge candidates but provides no concrete migration plan required by EVE v4 section 10.", "classification": "M0_C_REQUIRED_MIGRATION_PLAN_ABSENT", "confidence": "high", "unresolved": True, "manual_only": True}]
    unresolved_items.extend(parse_errors.values())
    component_entries.sort(key=lambda item: (item["path"], int(item["line_start"]), item["classification"], item["evidence"]))
    module_dispositions.sort(key=lambda item: item["path"])
    disposition_counts = Counter(item["classification"] for item in module_dispositions)
    component_counts = Counter(item["classification"] for item in component_entries)
    life_taxonomy_counts: Counter[str] = Counter()
    for item in life_loops:
        life_taxonomy_counts.update(item["v4_loop_taxonomy"])
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "root": root.as_posix(),
        "scope": {"static_analysis_only": True, "runtime_execution_performed": False, "loop_start_performed": False, "vector_or_model_load_performed": False, "production_source_mutation_performed": False, "frozen_pr_mutation_performed": False, "generated_json_committed": False},
        "summary": {
            "tracked_python_files": len(all_paths), "runtime_modules_classified": len(module_dispositions),
            "component_evidence_entries": len(component_entries), "life_loop_entries": len(life_loops),
            "module_disposition_counts": {category: disposition_counts.get(category, 0) for category in sorted(DISPOSITIONS)},
            "component_classification_counts": dict(sorted(component_counts.items())),
            "life_loop_taxonomy_counts": dict(sorted(life_taxonomy_counts.items())),
            "unresolved_module_dispositions": sum(bool(item["unresolved"]) for item in module_dispositions),
            "unresolved_items": len(unresolved_items), "parse_errors": len(parse_errors),
            "frozen_pr_recommendations": len(FROZEN_PR_RECOMMENDATIONS), "v4_runtime_conflicts": len(_build_conflicts()),
        },
        "component_entries": component_entries,
        "life_loops": life_loops,
        "module_dispositions": module_dispositions,
        "frozen_pr_recommendations": FROZEN_PR_RECOMMENDATIONS,
        "v4_runtime_conflicts": _build_conflicts(),
        "unresolved_items": unresolved_items,
        "source_audit_summaries": {label: report["summary"] for label, report in sorted(reports.items())},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_repository(args.root)
    payload: dict[str, Any] = report["summary"] if args.summary_only else report
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":")) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
