"""Rounds122-126 legacy import blocker recovery helpers.

The helpers are deterministic, read-only report builders for the post-NO-GO
validation recovery loop.  They diagnose legacy root import blockers and record
compatibility/validation outcomes without enabling production persistence,
runtime mapping by default, enforcement, AGP bypass, or vector artifacts.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

ROUND122_LEGACY_ROOT_IMPORT_BLOCKER_DIAGNOSIS_VERSION = "v3_round122_legacy_root_import_blocker_diagnosis"
ROUND123_SPREADING_ACTIVATION_COMPAT_SHIM_VERSION = "v3_round123_spreading_activation_import_compatibility_shim"
ROUND124_COLLECT_ONLY_RECOVERY_VERIFICATION_VERSION = "v3_round124_collect_only_recovery_verification"
ROUND125_BROADER_VALIDATION_FAILURE_TAXONOMY_VERSION = "v3_round125_broader_validation_failure_taxonomy"
ROUND126_NEXT_GO_NO_GO_REFRESH_VERSION = "v3_round126_next_go_no_go_refresh_after_blocker_isolation"

ROUND137_DMN_IMPORT_BLOCKER_DIAGNOSIS_VERSION = "v3_round137_dmn_import_blocker_diagnosis"
ROUND138_DMN_COMPAT_SHIM_VERSION = "v3_round138_dmn_import_compatibility_shim"
ROUND139_COLLECT_ONLY_AFTER_DMN_ISOLATION_VERSION = "v3_round139_collect_only_after_dmn_isolation"
ROUND140_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION = "v3_round140_broader_validation_taxonomy_refresh"
ROUND141_GO_NO_GO_REFRESH_AFTER_DMN_ISOLATION_VERSION = "v3_round141_go_no_go_refresh_after_dmn_isolation"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _root_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.py") if path.is_file())


def _imports_module(path: Path, module_name: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name or alias.name.startswith(module_name + "."):
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module == module_name or (node.module or "").startswith(module_name + "."):
                return True
    return False


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _module_available(module_name: str, root: str | Path | None = None) -> bool:
    if root is not None:
        base = Path(root)
        if (base / f"{module_name}.py").exists() or (base / module_name / "__init__.py").exists():
            return True
    return importlib.util.find_spec(module_name) is not None


def build_round122_legacy_root_import_blocker_diagnosis(
    *,
    repo_root: str | Path = ".",
    module_name: str = "spreading_activation",
    source_round121_isolation: dict[str, Any] | None = None,
    module_available: bool | None = None,
) -> dict[str, Any]:
    """Diagnose root-level files importing the legacy module path."""

    root = Path(repo_root)
    import_sites = [
        _relative(path, root)
        for path in _root_python_files(root)
        if path.name != f"{module_name}.py" and _imports_module(path, module_name)
    ]
    available = _module_available(module_name, root) if module_available is None else bool(module_available)
    blocker_active = bool(import_sites) and not available
    return {
        "diagnosis_version": ROUND122_LEGACY_ROOT_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
        "round": 122,
        "source_round": (source_round121_isolation or {}).get("round"),
        "diagnosis_status": "legacy_root_import_blocker_active" if blocker_active else "legacy_root_import_sites_identified_module_resolvable",
        "module_name": module_name,
        "module_available_at_root": available,
        "root_import_sites": import_sites,
        "root_import_site_count": len(import_sites),
        "primary_blocker": f"missing {module_name} compatibility import" if blocker_active else None,
        "recommended_round123_action": "add_minimal_compatibility_shim" if blocker_active else "verify_existing_compatibility_shim",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round123_spreading_activation_import_compatibility_shim_decision(
    *,
    source_round122_diagnosis: dict[str, Any] | None = None,
    shim_path: str | Path = "spreading_activation.py",
    legacy_source_path: str | Path = "legacy/eve_modules/spreading_activation.py",
    import_check_passed: bool = False,
    behavior_source: str = "legacy_reexport_only",
) -> dict[str, Any]:
    """Record the compatibility-shim or isolation decision."""

    shim = Path(shim_path)
    legacy = Path(legacy_source_path)
    source = source_round122_diagnosis or {}
    shim_text = shim.read_text(encoding="utf-8") if shim.exists() else ""
    forbidden_fake_markers = ["class SpreadingActivation", "dummy", "random", "vectors.npy"]
    fake_markers_present = sorted(marker for marker in forbidden_fake_markers if marker in shim_text)
    shim_is_minimal_reexport = shim.exists() and legacy.exists() and "legacy.eve_modules.spreading_activation" in shim_text and not fake_markers_present
    decision = "minimal_compatibility_shim_applied" if shim_is_minimal_reexport and import_check_passed else "isolation_plan_required"
    return {
        "compatibility_version": ROUND123_SPREADING_ACTIVATION_COMPAT_SHIM_VERSION,
        "round": 123,
        "source_round": source.get("round"),
        "decision_status": decision,
        "shim_path": str(shim_path).replace("\\", "/"),
        "legacy_source_path": str(legacy_source_path).replace("\\", "/"),
        "shim_exists": shim.exists(),
        "legacy_source_exists": legacy.exists(),
        "import_check_passed": bool(import_check_passed),
        "shim_is_minimal_reexport": shim_is_minimal_reexport,
        "behavior_source": behavior_source,
        "fake_behavior_markers_present": fake_markers_present,
        "isolation_plan": [] if decision == "minimal_compatibility_shim_applied" else [
            "keep root collect-only partial",
            "quarantine or port legacy root tests in a separate validation hygiene round",
        ],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round124_collect_only_recovery_verification(
    *,
    source_round122_diagnosis: dict[str, Any] | None = None,
    source_round123_decision: dict[str, Any] | None = None,
    collect_command: str = "pytest --collect-only -q",
    return_code: int | None = None,
    collected_tests: int | None = None,
    remaining_errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record collect-only recovery after the spreading_activation shim."""

    errors = list(remaining_errors or [])
    spreading_errors = [err for err in errors if err.get("missing_import") == "spreading_activation"]
    status = "collect_only_recovered" if return_code == 0 else "collect_only_partial_new_blockers_after_spreading_activation_recovery"
    return {
        "collect_recovery_version": ROUND124_COLLECT_ONLY_RECOVERY_VERIFICATION_VERSION,
        "round": 124,
        "source_rounds": [r for r in [(source_round122_diagnosis or {}).get("round"), (source_round123_decision or {}).get("round")] if r is not None],
        "collect_recovery_status": status,
        "collect_command": collect_command,
        "return_code": return_code,
        "collected_tests": collected_tests,
        "spreading_activation_import_errors_remaining": len(spreading_errors),
        "remaining_error_count": len(errors),
        "remaining_errors": errors,
        "critical_blocker_improved": len(spreading_errors) == 0,
        "broader_validation_status": "collect_only_passed" if return_code == 0 else "blocked_partial",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round125_broader_validation_failure_taxonomy(
    *,
    source_round124_collect_recovery: dict[str, Any] | None = None,
    validation_items: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify broader validation as pass/partial/blocked without masking failures."""

    collect = source_round124_collect_recovery or {}
    items = list(validation_items or [])
    categories: dict[str, list[dict[str, Any]]] = {
        "compile_checks": [],
        "focused_round122_124_tests": [],
        "collect_only": [],
        "broader_validation": [],
    }
    for item in items:
        categories.setdefault(str(item.get("category", "broader_validation")), []).append(item)
    blocked = [item for item in items if str(item.get("status")) in {"blocked", "partial", "blocked_partial", "fail"}]
    if collect.get("broader_validation_status") == "blocked_partial":
        blocked.append({"category": "collect_only", "status": "blocked_partial", "reason": "root collection still has non-spreading legacy import blockers"})
    return {
        "taxonomy_version": ROUND125_BROADER_VALIDATION_FAILURE_TAXONOMY_VERSION,
        "round": 125,
        "source_round": collect.get("round"),
        "taxonomy_status": "broader_validation_partial_or_blocked" if blocked else "broader_validation_green",
        "validation_categories": categories,
        "blocked_or_partial_items": blocked,
        "primary_remaining_blocker_family": "legacy_root_import_compatibility" if blocked else None,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round126_next_go_no_go_refresh(
    *,
    source_round124_collect_recovery: dict[str, Any] | None = None,
    source_round125_taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh the next recommendation after blocker isolation/recovery."""

    collect = source_round124_collect_recovery or {}
    taxonomy = source_round125_taxonomy or {}
    collect_green = collect.get("collect_recovery_status") == "collect_only_recovered"
    broader_green = taxonomy.get("taxonomy_status") == "broader_validation_green"
    recommendation = "GO" if collect_green and broader_green else "NO-GO"
    blockers = []
    if not collect_green:
        blockers.append("collect_only_still_partial_or_blocked")
    if not broader_green:
        blockers.append("broader_validation_partial_or_blocked")
    return {
        "go_no_go_refresh_version": ROUND126_NEXT_GO_NO_GO_REFRESH_VERSION,
        "round": 126,
        "source_rounds": [r for r in [collect.get("round"), taxonomy.get("round")] if r is not None],
        "refresh_status": "next_go_no_go_refreshed",
        "final_recommendation": recommendation,
        "recommendation_reason": "Keep NO-GO unless collect-only and critical blockers improve." if recommendation == "NO-GO" else "Collect-only and broader validation are green.",
        "remaining_blockers": blockers,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def write_round_report(path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, report)
    return {
        "path": str(path),
        "json_written": True,
        "runtime_mapping_enabled": False,
        "enforcement_enabled": False,
        "vectors_npy_committed": False,
        "agp_bypass_used": False,
    }



ROUND127_WORKING_MEMORY_IMPORT_BLOCKER_DIAGNOSIS_VERSION = "v3_round127_working_memory_import_blocker_diagnosis"
ROUND128_WORKING_MEMORY_COMPAT_SHIM_VERSION = "v3_round128_working_memory_import_compatibility_shim"
ROUND129_COLLECT_ONLY_AFTER_WORKING_MEMORY_VERIFICATION_VERSION = "v3_round129_collect_only_after_working_memory_verification"
ROUND130_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION = "v3_round130_broader_validation_taxonomy_refresh"
ROUND131_GO_NO_GO_REFRESH_AFTER_WORKING_MEMORY_VERSION = "v3_round131_go_no_go_refresh_after_working_memory_isolation"


def build_round127_working_memory_import_blocker_diagnosis(
    *,
    repo_root: str | Path = ".",
    source_round126_refresh: dict[str, Any] | None = None,
    module_available: bool | None = None,
) -> dict[str, Any]:
    """Diagnose the next legacy root blocker: ``working_memory`` imports."""

    diagnosis = build_round122_legacy_root_import_blocker_diagnosis(
        repo_root=repo_root,
        module_name="working_memory",
        module_available=module_available,
    )
    blocker_active = diagnosis["diagnosis_status"] == "legacy_root_import_blocker_active"
    return {
        "diagnosis_version": ROUND127_WORKING_MEMORY_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
        "round": 127,
        "source_round": (source_round126_refresh or {}).get("round"),
        "diagnosis_status": "working_memory_import_blocker_active" if blocker_active else "working_memory_import_sites_identified_module_resolvable",
        "module_name": "working_memory",
        "module_available_at_root": diagnosis["module_available_at_root"],
        "root_import_sites": diagnosis["root_import_sites"],
        "root_import_site_count": diagnosis["root_import_site_count"],
        "adapter_import_sites": [
            "adapters/activation_adapter.py",
            "adapters/memory_adapter.py",
            "adapters/nl_adapter.py",
        ],
        "retained_legacy_candidate": "legacy/eve_modules/working_memory.py",
        "retained_legacy_class": "WorkingMemory",
        "primary_blocker": "missing working_memory compatibility import" if blocker_active else None,
        "recommended_round128_action": "add_minimal_compatibility_shim" if blocker_active else "verify_existing_compatibility_shim",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round128_working_memory_import_compatibility_shim_decision(
    *,
    source_round127_diagnosis: dict[str, Any] | None = None,
    shim_path: str | Path = "working_memory.py",
    legacy_source_path: str | Path = "legacy/eve_modules/working_memory.py",
    import_check_passed: bool = False,
    behavior_source: str = "legacy_reexport_only",
) -> dict[str, Any]:
    """Record the working-memory shim decision without faking behavior."""

    shim = Path(shim_path)
    legacy = Path(legacy_source_path)
    source = source_round127_diagnosis or {}
    shim_text = shim.read_text(encoding="utf-8") if shim.exists() else ""
    forbidden_fake_markers = ["class WorkingMemory", "dummy", "random", "vectors.npy"]
    fake_markers_present = sorted(marker for marker in forbidden_fake_markers if marker in shim_text)
    shim_is_minimal_reexport = (
        shim.exists()
        and legacy.exists()
        and "legacy.eve_modules.working_memory" in shim_text
        and "WorkingMemory" in shim_text
        and not fake_markers_present
    )
    decision = "minimal_compatibility_shim_applied" if shim_is_minimal_reexport and import_check_passed else "isolation_plan_required"
    return {
        "compatibility_version": ROUND128_WORKING_MEMORY_COMPAT_SHIM_VERSION,
        "round": 128,
        "source_round": source.get("round"),
        "decision_status": decision,
        "shim_path": str(shim_path).replace("\\", "/"),
        "legacy_source_path": str(legacy_source_path).replace("\\", "/"),
        "shim_exists": shim.exists(),
        "legacy_source_exists": legacy.exists(),
        "import_check_passed": bool(import_check_passed),
        "shim_is_minimal_reexport": shim_is_minimal_reexport,
        "behavior_source": behavior_source,
        "reexported_symbols": ["WMSlot", "WorkingMemory"] if shim_is_minimal_reexport else [],
        "fake_behavior_markers_present": fake_markers_present,
        "isolation_plan": [] if decision == "minimal_compatibility_shim_applied" else [
            "keep root collect-only partial",
            "do not fake WorkingMemory behavior",
            "plan separate legacy test isolation if retained implementation is unavailable",
        ],
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round129_collect_only_after_working_memory_verification(
    *,
    source_round127_diagnosis: dict[str, Any] | None = None,
    source_round128_decision: dict[str, Any] | None = None,
    collect_command: str = "pytest --collect-only -q",
    return_code: int | None = None,
    collected_tests: int | None = None,
    remaining_errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record collect-only after working-memory compatibility recovery."""

    errors = list(remaining_errors or [])
    working_memory_errors = [err for err in errors if err.get("missing_import") == "working_memory"]
    system_exit_errors = [err for err in errors if err.get("error_family") == "legacy_collection_side_effect_system_exit"]
    recovered = return_code == 0
    if recovered:
        status = "collect_only_recovered"
    elif not working_memory_errors and system_exit_errors:
        status = "collect_only_partial_new_legacy_side_effect_blocker_after_working_memory_recovery"
    else:
        status = "collect_only_partial_after_working_memory_recovery"
    return {
        "collect_recovery_version": ROUND129_COLLECT_ONLY_AFTER_WORKING_MEMORY_VERIFICATION_VERSION,
        "round": 129,
        "source_rounds": [r for r in [(source_round127_diagnosis or {}).get("round"), (source_round128_decision or {}).get("round")] if r is not None],
        "collect_recovery_status": status,
        "collect_command": collect_command,
        "return_code": return_code,
        "collected_tests": collected_tests,
        "working_memory_import_errors_remaining": len(working_memory_errors),
        "remaining_error_count": len(errors),
        "remaining_errors": errors,
        "critical_blocker_improved": len(working_memory_errors) == 0,
        "next_blocker_family": "legacy_collection_side_effect_system_exit" if system_exit_errors else (errors[0].get("error_family") if errors else None),
        "broader_validation_status": "collect_only_passed" if recovered else "blocked_partial",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round130_broader_validation_taxonomy_refresh(
    *,
    source_round129_collect_recovery: dict[str, Any] | None = None,
    validation_items: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refresh broader validation taxonomy after working-memory isolation."""

    collect = source_round129_collect_recovery or {}
    items = list(validation_items or [])
    categories: dict[str, list[dict[str, Any]]] = {
        "compile_checks": [],
        "focused_round127_129_tests": [],
        "collect_only": [],
        "broader_validation": [],
    }
    for item in items:
        categories.setdefault(str(item.get("category", "broader_validation")), []).append(item)
    blocked = [item for item in items if str(item.get("status")) in {"blocked", "partial", "blocked_partial", "fail"}]
    if collect.get("broader_validation_status") == "blocked_partial":
        blocked.append({
            "category": "collect_only",
            "status": "blocked_partial",
            "reason": collect.get("next_blocker_family") or "collect-only remains partial",
        })
    return {
        "taxonomy_version": ROUND130_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
        "round": 130,
        "source_round": collect.get("round"),
        "taxonomy_status": "broader_validation_partial_or_blocked" if blocked else "broader_validation_green",
        "validation_categories": categories,
        "blocked_or_partial_items": blocked,
        "primary_remaining_blocker_family": collect.get("next_blocker_family") if blocked else None,
        "working_memory_blocker_recovered": collect.get("working_memory_import_errors_remaining") == 0,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round131_go_no_go_refresh_after_working_memory(
    *,
    source_round129_collect_recovery: dict[str, Any] | None = None,
    source_round130_taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh recommendation after the working-memory import blocker round set."""

    collect = source_round129_collect_recovery or {}
    taxonomy = source_round130_taxonomy or {}
    collect_green = collect.get("collect_recovery_status") == "collect_only_recovered"
    broader_green = taxonomy.get("taxonomy_status") == "broader_validation_green"
    critical_improved = collect.get("critical_blocker_improved") is True
    recommendation = "GO" if collect_green and broader_green and critical_improved else "NO-GO"
    blockers = []
    if not collect_green:
        blockers.append("collect_only_still_partial_or_blocked")
    if not broader_green:
        blockers.append("broader_validation_partial_or_blocked")
    if collect.get("next_blocker_family"):
        blockers.append(str(collect["next_blocker_family"]))
    return {
        "go_no_go_refresh_version": ROUND131_GO_NO_GO_REFRESH_AFTER_WORKING_MEMORY_VERSION,
        "round": 131,
        "source_rounds": [r for r in [collect.get("round"), taxonomy.get("round")] if r is not None],
        "refresh_status": "go_no_go_refreshed_after_working_memory_isolation",
        "critical_blocker_improved": critical_improved,
        "final_recommendation": recommendation,
        "recommendation_reason": "Keep NO-GO unless collect-only and critical blockers improve." if recommendation == "NO-GO" else "Collect-only, focused tests, and broader validation are green.",
        "remaining_blockers": blockers,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


ROUND132_NATURAL_LANG_V2_SYSTEM_EXIT_DIAGNOSIS_VERSION = "v3_round132_natural_lang_v2_system_exit_diagnosis"
ROUND133_COLLECTION_SIDE_EFFECT_ISOLATION_VERSION = "v3_round133_collection_side_effect_isolation"
ROUND134_COLLECT_ONLY_AFTER_SYSTEM_EXIT_ISOLATION_VERSION = "v3_round134_collect_only_after_system_exit_isolation"
ROUND135_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION = "v3_round135_broader_validation_taxonomy_refresh"
ROUND136_GO_NO_GO_REFRESH_AFTER_SYSTEM_EXIT_VERSION = "v3_round136_go_no_go_refresh_after_system_exit_isolation"


def _is_main_guard_node(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _module_level_call_lines(path: Path, call_name: str) -> list[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    lines: list[int] = []

    def visit_module_statement(node: ast.AST) -> None:
        if _is_main_guard_node(node):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        target = node.value if isinstance(node, ast.Expr) else node
        if isinstance(target, ast.Call):
            func = target.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                dotted = f"{func.value.id}.{func.attr}"
                if dotted == call_name:
                    lines.append(node.lineno)
            elif isinstance(func, ast.Name) and func.id == call_name:
                lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit_module_statement(child)

    for node in tree.body:
        visit_module_statement(node)
    return sorted(set(lines))


def _has_main_guard(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return True
    return False


def build_round132_natural_lang_v2_system_exit_diagnosis(
    *,
    test_path: str | Path = "test_natural_lang_v2.py",
    source_round129_collect_recovery: dict[str, Any] | None = None,
    collect_return_code: int | None = None,
    observed_system_exit_line: int | None = None,
) -> dict[str, Any]:
    """Diagnose legacy collection-time ``SystemExit`` without mutating runtime state."""

    path = Path(test_path)
    sys_exit_lines = _module_level_call_lines(path, "sys.exit") if path.exists() else []
    has_main_guard = _has_main_guard(path) if path.exists() else False
    blocker_active = (bool(sys_exit_lines) and not has_main_guard) or observed_system_exit_line is not None
    return {
        "diagnosis_version": ROUND132_NATURAL_LANG_V2_SYSTEM_EXIT_DIAGNOSIS_VERSION,
        "round": 132,
        "source_round": (source_round129_collect_recovery or {}).get("round"),
        "diagnosis_status": "collection_time_system_exit_active" if blocker_active else "collection_time_system_exit_isolated_or_absent",
        "test_path": str(test_path).replace("\\", "/"),
        "test_file_exists": path.exists(),
        "observed_collect_return_code": collect_return_code,
        "observed_system_exit_line": observed_system_exit_line,
        "module_level_sys_exit_lines": sys_exit_lines,
        "main_guard_present": has_main_guard,
        "root_cause": "module-level validation body calls sys.exit during pytest import" if blocker_active else "legacy validation execution is isolated from pytest import",
        "recommended_round133_action": "move execution behind main guard and expose pytest-safe validation wrapper" if blocker_active else "verify collection-safe wrapper",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round133_collection_side_effect_isolation_decision(
    *,
    test_path: str | Path = "test_natural_lang_v2.py",
    source_round132_diagnosis: dict[str, Any] | None = None,
    import_check_passed: bool = False,
    legacy_script_exit_preserved: bool = False,
    pytest_behavior_test_present: bool = False,
) -> dict[str, Any]:
    """Record the collection-safe isolation decision while preserving test intent."""

    path = Path(test_path)
    sys_exit_lines = _module_level_call_lines(path, "sys.exit") if path.exists() else []
    has_main_guard = _has_main_guard(path) if path.exists() else False
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    wrapper_present = "def run_natural_language_v2_validation" in text
    pytest_test_present = pytest_behavior_test_present or "def test_natural_language_v2_validation_behavior" in text
    isolated = path.exists() and has_main_guard and wrapper_present and pytest_test_present and not sys_exit_lines and import_check_passed
    return {
        "isolation_version": ROUND133_COLLECTION_SIDE_EFFECT_ISOLATION_VERSION,
        "round": 133,
        "source_round": (source_round132_diagnosis or {}).get("round"),
        "isolation_status": "collection_side_effect_isolated_test_intent_preserved" if isolated else "collection_side_effect_isolation_incomplete",
        "test_path": str(test_path).replace("\\", "/"),
        "import_check_passed": bool(import_check_passed),
        "legacy_script_exit_preserved": bool(legacy_script_exit_preserved),
        "main_guard_present": has_main_guard,
        "module_level_sys_exit_lines_remaining": sys_exit_lines,
        "collection_safe_wrapper_present": wrapper_present,
        "pytest_behavior_test_present": pytest_test_present,
        "test_intent_preserved": pytest_test_present and legacy_script_exit_preserved,
        "weakening_action_used": False,
        "skip_or_xfail_added": False,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round134_collect_only_after_system_exit_isolation(
    *,
    source_round132_diagnosis: dict[str, Any] | None = None,
    source_round133_isolation: dict[str, Any] | None = None,
    collect_command: str = "pytest --collect-only -q",
    return_code: int | None = None,
    collected_tests: int | None = None,
    remaining_errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record collect-only recovery after isolating the legacy SystemExit."""

    errors = list(remaining_errors or [])
    system_exit_errors = [err for err in errors if err.get("error_family") == "legacy_collection_side_effect_system_exit"]
    recovered = return_code == 0
    return {
        "collect_recovery_version": ROUND134_COLLECT_ONLY_AFTER_SYSTEM_EXIT_ISOLATION_VERSION,
        "round": 134,
        "source_rounds": [r for r in [(source_round132_diagnosis or {}).get("round"), (source_round133_isolation or {}).get("round")] if r is not None],
        "collect_recovery_status": "collect_only_recovered_after_system_exit_isolation" if recovered else "collect_only_partial_after_system_exit_isolation",
        "collect_command": collect_command,
        "return_code": return_code,
        "collected_tests": collected_tests,
        "system_exit_errors_remaining": len(system_exit_errors),
        "remaining_error_count": len(errors),
        "remaining_errors": errors,
        "critical_blocker_improved": len(system_exit_errors) == 0,
        "broader_validation_status": "collect_only_passed" if recovered else "blocked_partial",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round135_broader_validation_taxonomy_refresh(
    *,
    source_round134_collect_recovery: dict[str, Any] | None = None,
    validation_items: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refresh validation taxonomy after collection-time side-effect isolation."""

    collect = source_round134_collect_recovery or {}
    items = list(validation_items or [])
    categories: dict[str, list[dict[str, Any]]] = {
        "compile_checks": [],
        "focused_round132_134_tests": [],
        "collect_only": [],
        "legacy_behavior_tests": [],
        "broader_validation": [],
    }
    for item in items:
        categories.setdefault(str(item.get("category", "broader_validation")), []).append(item)
    blocked = [item for item in items if str(item.get("status")) in {"blocked", "partial", "blocked_partial", "fail"}]
    if collect.get("broader_validation_status") == "blocked_partial":
        blocked.append({"category": "collect_only", "status": "blocked_partial", "reason": "collect-only still blocked"})
    legacy_failures = [item for item in items if item.get("category") == "legacy_behavior_tests" and item.get("status") == "fail"]
    return {
        "taxonomy_version": ROUND135_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
        "round": 135,
        "source_round": collect.get("round"),
        "taxonomy_status": "broader_validation_partial_or_blocked" if blocked else "broader_validation_green",
        "validation_categories": categories,
        "blocked_or_partial_items": blocked,
        "system_exit_blocker_recovered": collect.get("system_exit_errors_remaining") == 0,
        "legacy_behavior_failures_preserved": legacy_failures,
        "primary_remaining_blocker_family": "legacy_behavior_failure" if legacy_failures else ("collect_only" if collect.get("broader_validation_status") == "blocked_partial" else None),
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round136_go_no_go_refresh_after_system_exit(
    *,
    source_round134_collect_recovery: dict[str, Any] | None = None,
    source_round135_taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh production-persistence recommendation after collect-only recovery."""

    collect = source_round134_collect_recovery or {}
    taxonomy = source_round135_taxonomy or {}
    collect_improved = collect.get("critical_blocker_improved") is True
    collect_green = collect.get("collect_recovery_status") == "collect_only_recovered_after_system_exit_isolation"
    broader_green = taxonomy.get("taxonomy_status") == "broader_validation_green"
    recommendation = "GO" if collect_green and broader_green and collect_improved else "NO-GO"
    blockers = []
    if not collect_green:
        blockers.append("collect_only_still_partial_or_blocked")
    if not broader_green:
        blockers.append("broader_validation_partial_or_blocked")
    if taxonomy.get("primary_remaining_blocker_family"):
        blockers.append(str(taxonomy["primary_remaining_blocker_family"]))
    return {
        "go_no_go_refresh_version": ROUND136_GO_NO_GO_REFRESH_AFTER_SYSTEM_EXIT_VERSION,
        "round": 136,
        "source_rounds": [r for r in [collect.get("round"), taxonomy.get("round")] if r is not None],
        "refresh_status": "go_no_go_refreshed_after_system_exit_isolation",
        "critical_blocker_improved": collect_improved,
        "collect_only_green": collect_green,
        "final_recommendation": recommendation,
        "recommendation_reason": "Keep NO-GO: production persistence remains disabled unless collect-only and broader validation are green." if recommendation == "NO-GO" else "Collect-only and broader validation are green.",
        "remaining_blockers": blockers,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }



def build_round137_dmn_import_blocker_diagnosis(
    *,
    repo_root: str | Path = ".",
    source_round136_refresh: dict[str, Any] | None = None,
    module_available: bool | None = None,
) -> dict[str, Any]:
    """Diagnose legacy root ``dmn`` import blockers without enabling runtime features."""

    root = Path(repo_root)
    module_name = "dmn"
    import_sites = [
        _relative(path, root)
        for path in _root_python_files(root)
        if path.name != f"{module_name}.py" and _imports_module(path, module_name)
    ]
    available = _module_available(module_name, root) if module_available is None else bool(module_available)
    legacy_source = root / "legacy" / "eve_modules" / "dmn.py"
    blocker_active = bool(import_sites) and not available
    return {
        "diagnosis_version": ROUND137_DMN_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
        "round": 137,
        "source_round": (source_round136_refresh or {}).get("round"),
        "diagnosis_status": "dmn_import_blocker_active" if blocker_active else "dmn_import_sites_identified_module_resolvable",
        "module_name": module_name,
        "module_available_at_root": available,
        "root_import_sites": import_sites,
        "root_import_site_count": len(import_sites),
        "retained_legacy_candidate": "legacy/eve_modules/dmn.py" if legacy_source.exists() else None,
        "retained_legacy_candidate_exists": legacy_source.exists(),
        "required_symbol": "DefaultModeNetwork",
        "primary_blocker": "missing dmn compatibility import" if blocker_active else None,
        "recommended_round138_action": "add_minimal_compatibility_shim" if blocker_active and legacy_source.exists() else ("hard_stop_isolation_plan_required" if blocker_active else "verify_existing_compatibility_shim"),
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round138_dmn_import_compatibility_shim_decision(
    *,
    source_round137_diagnosis: dict[str, Any] | None = None,
    shim_path: str | Path = "dmn.py",
    legacy_source_path: str | Path = "legacy/eve_modules/dmn.py",
    import_check_passed: bool = False,
    behavior_source: str = "legacy_reexport_only",
) -> dict[str, Any]:
    """Record the DMN compatibility shim or hard-stop isolation decision."""

    shim = Path(shim_path)
    legacy = Path(legacy_source_path)
    shim_text = shim.read_text(encoding="utf-8") if shim.exists() else ""
    forbidden_fake_markers = ["class DefaultModeNetwork", "dummy", "random", "vectors.npy"]
    fake_markers_present = sorted(marker for marker in forbidden_fake_markers if marker in shim_text)
    shim_is_minimal_reexport = shim.exists() and legacy.exists() and "legacy.eve_modules.dmn" in shim_text and not fake_markers_present
    status = "minimal_compatibility_shim_applied" if shim_is_minimal_reexport and import_check_passed else "isolation_plan_required"
    return {
        "compatibility_version": ROUND138_DMN_COMPAT_SHIM_VERSION,
        "round": 138,
        "source_round": (source_round137_diagnosis or {}).get("round"),
        "decision_status": status,
        "shim_path": str(shim_path),
        "legacy_source_path": str(legacy_source_path),
        "retained_legacy_source_exists": legacy.exists(),
        "shim_is_minimal_reexport": shim_is_minimal_reexport,
        "reexported_symbols": ["DefaultModeNetwork"] if shim_is_minimal_reexport else [],
        "import_check_passed": import_check_passed,
        "behavior_source": behavior_source if shim_is_minimal_reexport else "no_runtime_behavior_applied",
        "fake_behavior_markers_present": fake_markers_present,
        "isolation_plan": None if status == "minimal_compatibility_shim_applied" else "Hard stop: locate retained legacy dmn implementation before restoring root import path.",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round139_collect_only_after_dmn_isolation(
    *,
    source_round137_diagnosis: dict[str, Any] | None = None,
    source_round138_decision: dict[str, Any] | None = None,
    collect_command: str = "pytest --collect-only -q",
    return_code: int | None = None,
    collected_tests: int | None = None,
    remaining_errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record collect-only recovery after DMN import isolation."""

    errors = list(remaining_errors or [])
    dmn_errors = [err for err in errors if err.get("missing_import") == "dmn" or err.get("error_family") == "legacy_root_dmn_import_blocker"]
    recovered = return_code == 0
    return {
        "collect_recovery_version": ROUND139_COLLECT_ONLY_AFTER_DMN_ISOLATION_VERSION,
        "round": 139,
        "source_rounds": [r for r in [(source_round137_diagnosis or {}).get("round"), (source_round138_decision or {}).get("round")] if r is not None],
        "collect_recovery_status": "collect_only_recovered_after_dmn_isolation" if recovered else "collect_only_partial_after_dmn_isolation",
        "collect_command": collect_command,
        "return_code": return_code,
        "collected_tests": collected_tests,
        "dmn_import_errors_remaining": len(dmn_errors),
        "remaining_error_count": len(errors),
        "remaining_errors": errors,
        "critical_blocker_improved": len(dmn_errors) == 0,
        "broader_validation_status": "collect_only_passed" if recovered else "blocked_partial",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round140_broader_validation_taxonomy_refresh(
    *,
    source_round139_collect_recovery: dict[str, Any] | None = None,
    validation_items: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refresh broader validation taxonomy after DMN isolation."""

    collect = source_round139_collect_recovery or {}
    items = list(validation_items or [])
    categories: dict[str, list[dict[str, Any]]] = {
        "compile_checks": [],
        "focused_round137_139_tests": [],
        "collect_only": [],
        "legacy_behavior_tests": [],
        "broader_validation": [],
    }
    for item in items:
        categories.setdefault(str(item.get("category", "broader_validation")), []).append(item)
    blocked = [item for item in items if str(item.get("status")) in {"blocked", "partial", "blocked_partial", "fail"}]
    if collect.get("broader_validation_status") == "blocked_partial":
        blocked.append({"category": "collect_only", "status": "blocked_partial", "reason": "collect-only still blocked"})
    legacy_failures = [item for item in items if item.get("category") == "legacy_behavior_tests" and item.get("status") == "fail"]
    primary = None
    if collect.get("broader_validation_status") == "blocked_partial":
        primary = "collect_only"
    elif legacy_failures:
        primary = "legacy_behavior_failure"
    elif blocked:
        primary = "broader_validation_partial_or_blocked"
    return {
        "taxonomy_version": ROUND140_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
        "round": 140,
        "source_round": collect.get("round"),
        "taxonomy_status": "broader_validation_partial_or_blocked" if blocked else "broader_validation_green",
        "validation_categories": categories,
        "blocked_or_partial_items": blocked,
        "dmn_blocker_recovered": collect.get("dmn_import_errors_remaining") == 0,
        "collect_only_green": collect.get("collect_recovery_status") == "collect_only_recovered_after_dmn_isolation",
        "legacy_behavior_failures_preserved": legacy_failures,
        "primary_remaining_blocker_family": primary,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round141_go_no_go_refresh_after_dmn_isolation(
    *,
    source_round139_collect_recovery: dict[str, Any] | None = None,
    source_round140_taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh production-persistence recommendation after DMN isolation."""

    collect = source_round139_collect_recovery or {}
    taxonomy = source_round140_taxonomy or {}
    collect_green = collect.get("collect_recovery_status") == "collect_only_recovered_after_dmn_isolation"
    collect_improved = collect.get("critical_blocker_improved") is True
    broader_green = taxonomy.get("taxonomy_status") == "broader_validation_green"
    recommendation = "GO" if collect_green and broader_green and collect_improved else "NO-GO"
    blockers = []
    if not collect_green:
        blockers.append("collect_only_still_partial_or_blocked")
    if not broader_green:
        blockers.append("broader_validation_partial_or_blocked")
    if taxonomy.get("primary_remaining_blocker_family"):
        blockers.append(str(taxonomy["primary_remaining_blocker_family"]))
    return {
        "go_no_go_refresh_version": ROUND141_GO_NO_GO_REFRESH_AFTER_DMN_ISOLATION_VERSION,
        "round": 141,
        "source_rounds": [r for r in [collect.get("round"), taxonomy.get("round")] if r is not None],
        "refresh_status": "go_no_go_refreshed_after_dmn_isolation",
        "critical_blocker_improved": collect_improved,
        "collect_only_green": collect_green,
        "final_recommendation": recommendation,
        "recommendation_reason": "Keep NO-GO: production persistence remains disabled unless collect-only and broader validation are green." if recommendation == "NO-GO" else "Collect-only and broader validation are green.",
        "remaining_blockers": blockers,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


__all__ = [name for name in globals() if name.startswith("ROUND") or name.startswith("build_") or name == "write_round_report"]


ROUND142_DIGITAL_SOMATIC_IMPORT_BLOCKER_DIAGNOSIS_VERSION = "v3_round142_digital_somatic_import_blocker_diagnosis"
ROUND143_DIGITAL_SOMATIC_COMPAT_SHIM_VERSION = "v3_round143_digital_somatic_import_compatibility_shim"
ROUND144_COLLECT_ONLY_AFTER_DIGITAL_SOMATIC_ISOLATION_VERSION = "v3_round144_collect_only_after_digital_somatic_isolation"
ROUND145_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION = "v3_round145_broader_validation_taxonomy_refresh"
ROUND146_GO_NO_GO_REFRESH_AFTER_DIGITAL_SOMATIC_ISOLATION_VERSION = "v3_round146_go_no_go_refresh_after_digital_somatic_isolation"


def build_round142_digital_somatic_import_blocker_diagnosis(
    *,
    repo_root: str | Path = ".",
    source_round141_refresh: dict[str, Any] | None = None,
    module_available: bool | None = None,
) -> dict[str, Any]:
    """Diagnose legacy root ``digital_somatic`` import blockers."""

    root = Path(repo_root)
    module_name = "digital_somatic"
    import_sites = [
        _relative(path, root)
        for path in _root_python_files(root)
        if path.name != f"{module_name}.py" and _imports_module(path, module_name)
    ]
    adapter_import_sites = [
        _relative(path, root)
        for path in sorted((root / "adapters").glob("*.py"))
        if path.is_file() and _imports_module(path, module_name)
    ] if (root / "adapters").exists() else []
    available = _module_available(module_name, root) if module_available is None else bool(module_available)
    legacy_source = root / "legacy" / "eve_modules" / "digital_somatic.py"
    blocker_active = bool(import_sites) and not available
    return {
        "diagnosis_version": ROUND142_DIGITAL_SOMATIC_IMPORT_BLOCKER_DIAGNOSIS_VERSION,
        "round": 142,
        "source_round": (source_round141_refresh or {}).get("round"),
        "diagnosis_status": "digital_somatic_import_blocker_active" if blocker_active else "digital_somatic_import_sites_identified_module_resolvable",
        "module_name": module_name,
        "module_available_at_root": available,
        "root_import_sites": import_sites,
        "root_import_site_count": len(import_sites),
        "adapter_import_sites": adapter_import_sites,
        "retained_legacy_candidate": "legacy/eve_modules/digital_somatic.py" if legacy_source.exists() else None,
        "retained_legacy_candidate_exists": legacy_source.exists(),
        "required_symbol": "DigitalSomatic",
        "primary_blocker": "missing digital_somatic compatibility import" if blocker_active else None,
        "recommended_round143_action": "add_minimal_compatibility_shim" if blocker_active and legacy_source.exists() else ("hard_stop_isolation_plan_required" if blocker_active else "verify_existing_compatibility_shim"),
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round143_digital_somatic_import_compatibility_shim_decision(
    *,
    source_round142_diagnosis: dict[str, Any] | None = None,
    shim_path: str | Path = "digital_somatic.py",
    legacy_source_path: str | Path = "legacy/eve_modules/digital_somatic.py",
    import_check_passed: bool = False,
    behavior_source: str = "legacy_reexport_only",
) -> dict[str, Any]:
    """Record the DigitalSomatic compatibility-shim or isolation decision."""

    shim = Path(shim_path)
    legacy = Path(legacy_source_path)
    shim_text = shim.read_text(encoding="utf-8") if shim.exists() else ""
    forbidden_fake_markers = ["class DigitalSomatic", "dummy", "random", "vectors.npy"]
    fake_markers_present = sorted(marker for marker in forbidden_fake_markers if marker in shim_text)
    shim_is_minimal_reexport = shim.exists() and legacy.exists() and "legacy.eve_modules.digital_somatic" in shim_text and not fake_markers_present
    decision = "minimal_compatibility_shim_applied" if shim_is_minimal_reexport and import_check_passed else "isolation_plan_required"
    return {
        "compatibility_version": ROUND143_DIGITAL_SOMATIC_COMPAT_SHIM_VERSION,
        "round": 143,
        "source_round": (source_round142_diagnosis or {}).get("round"),
        "decision_status": decision,
        "shim_path": str(shim_path),
        "legacy_source_path": str(legacy_source_path),
        "retained_legacy_source_exists": legacy.exists(),
        "reexported_symbols": ["DigitalSomatic"] if shim_is_minimal_reexport else [],
        "shim_is_minimal_reexport": shim_is_minimal_reexport,
        "behavior_source": behavior_source if shim_is_minimal_reexport else "none",
        "fake_behavior_markers_present": fake_markers_present,
        "import_check_passed": import_check_passed,
        "isolation_plan": None if decision == "minimal_compatibility_shim_applied" else {
            "required_action": "restore retained legacy/eve_modules/digital_somatic.py before adding root compatibility import",
            "hard_stop_reason": "DigitalSomatic behavior must not be faked or replaced with dummy vectors",
        },
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round144_collect_only_after_digital_somatic_isolation(
    *,
    source_round142_diagnosis: dict[str, Any] | None = None,
    source_round143_decision: dict[str, Any] | None = None,
    collect_command: str = "python -m pytest --collect-only -q",
    return_code: int = 1,
    collected_tests: int | None = None,
    remaining_errors: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record collect-only recovery after DigitalSomatic import isolation."""

    errors = list(remaining_errors or [])
    ds_errors = [err for err in errors if err.get("missing_import") == "digital_somatic" or err.get("error_family") == "legacy_root_digital_somatic_import_blocker"]
    recovered = return_code == 0
    return {
        "collect_recovery_version": ROUND144_COLLECT_ONLY_AFTER_DIGITAL_SOMATIC_ISOLATION_VERSION,
        "round": 144,
        "source_rounds": [r for r in [(source_round142_diagnosis or {}).get("round"), (source_round143_decision or {}).get("round")] if r is not None],
        "collect_recovery_status": "collect_only_recovered_after_digital_somatic_isolation" if recovered else "collect_only_partial_after_digital_somatic_isolation",
        "collect_command": collect_command,
        "return_code": return_code,
        "collected_tests": collected_tests,
        "digital_somatic_import_errors_remaining": len(ds_errors),
        "remaining_error_count": len(errors),
        "remaining_errors": errors,
        "critical_blocker_improved": len(ds_errors) == 0,
        "broader_validation_status": "collect_only_passed" if recovered else "blocked_partial",
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round145_broader_validation_taxonomy_refresh(
    *,
    source_round144_collect_recovery: dict[str, Any] | None = None,
    validation_items: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refresh broader validation taxonomy after DigitalSomatic isolation."""

    collect = source_round144_collect_recovery or {}
    items = list(validation_items or [])
    categories: dict[str, list[dict[str, Any]]] = {
        "compile_checks": [],
        "focused_round142_144_tests": [],
        "collect_only": [],
        "legacy_behavior_tests": [],
        "broader_validation": [],
    }
    for item in items:
        categories.setdefault(str(item.get("category", "broader_validation")), []).append(item)
    blocked = [item for item in items if str(item.get("status")) in {"blocked", "partial", "blocked_partial", "fail"}]
    if collect.get("broader_validation_status") == "blocked_partial":
        blocked.append({"category": "collect_only", "status": "blocked_partial", "reason": "collect-only still blocked"})
    legacy_failures = [item for item in items if item.get("category") == "legacy_behavior_tests" and item.get("status") == "fail"]
    primary = None
    if collect.get("broader_validation_status") == "blocked_partial":
        primary = "collect_only"
    elif legacy_failures:
        primary = "legacy_behavior_failure"
    elif blocked:
        primary = "broader_validation_partial_or_blocked"
    return {
        "taxonomy_version": ROUND145_BROADER_VALIDATION_TAXONOMY_REFRESH_VERSION,
        "round": 145,
        "source_round": collect.get("round"),
        "taxonomy_status": "broader_validation_partial_or_blocked" if blocked else "broader_validation_green",
        "validation_categories": categories,
        "blocked_or_partial_items": blocked,
        "digital_somatic_blocker_recovered": collect.get("digital_somatic_import_errors_remaining") == 0,
        "collect_only_green": collect.get("collect_recovery_status") == "collect_only_recovered_after_digital_somatic_isolation",
        "legacy_behavior_failures_preserved": legacy_failures,
        "primary_remaining_blocker_family": primary,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }


def build_round146_go_no_go_refresh_after_digital_somatic_isolation(
    *,
    source_round144_collect_recovery: dict[str, Any] | None = None,
    source_round145_taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh production-persistence recommendation after DigitalSomatic isolation."""

    collect = source_round144_collect_recovery or {}
    taxonomy = source_round145_taxonomy or {}
    collect_green = collect.get("collect_recovery_status") == "collect_only_recovered_after_digital_somatic_isolation"
    collect_improved = collect.get("critical_blocker_improved") is True
    broader_green = taxonomy.get("taxonomy_status") == "broader_validation_green"
    recommendation = "GO" if collect_green and broader_green and collect_improved else "NO-GO"
    blockers = []
    if not collect_green:
        blockers.append("collect_only_still_partial_or_blocked")
    if not broader_green:
        blockers.append("broader_validation_partial_or_blocked")
    if taxonomy.get("primary_remaining_blocker_family"):
        blockers.append(str(taxonomy["primary_remaining_blocker_family"]))
    return {
        "go_no_go_refresh_version": ROUND146_GO_NO_GO_REFRESH_AFTER_DIGITAL_SOMATIC_ISOLATION_VERSION,
        "round": 146,
        "source_rounds": [r for r in [collect.get("round"), taxonomy.get("round")] if r is not None],
        "refresh_status": "go_no_go_refreshed_after_digital_somatic_isolation",
        "critical_blocker_improved": collect_improved,
        "collect_only_green": collect_green,
        "final_recommendation": recommendation,
        "recommendation_reason": "Keep NO-GO: production persistence remains disabled unless collect-only and broader validation are green." if recommendation == "NO-GO" else "Collect-only and broader validation are green.",
        "remaining_blockers": blockers,
        "production_persistence_enabled": False,
        "runtime_mapping_enabled_default": False,
        "enforcement_enabled_default": False,
        "agp_bypass_used": False,
        "vectors_npy_committed": False,
        "read_only": True,
    }

__all__ = [name for name in globals() if name.startswith("ROUND") or name.startswith("build_") or name == "write_round_report"]
