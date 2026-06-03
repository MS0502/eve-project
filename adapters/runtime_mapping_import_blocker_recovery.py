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


__all__ = [name for name in globals() if name.startswith("ROUND") or name.startswith("build_") or name == "write_round_report"]
