"""Memory and exact-path preflight for M3-C-R."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.m3_c_o_private_device_goal_dual_read_operator import (
    build_private_path_binding,
    require_single_use_private_paths,
)
from core.m3_c_r_contracts import (
    EXPECTED_PACKAGE_DIGEST,
    MINIMUM_AVAILABLE_MIB,
    PREFLIGHT_SCHEMA,
    M3CRResumableOperatorError,
    capture_or_reuse_local_pin,
    load_canonical_mapping,
    private_paths,
    require_review_confirmation,
    validate_immutable_inputs,
    write_idempotent_canonical,
)


def available_memory_bytes() -> tuple[int, str]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024, "proc_meminfo_memavailable_v1"
        except (OSError, ValueError, IndexError):
            pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        pages = page_size = -1
    if (
        isinstance(pages, int)
        and isinstance(page_size, int)
        and pages > 0
        and page_size > 0
    ):
        return pages * page_size, "sysconf_available_pages_v1"
    raise M3CRResumableOperatorError("no real available-memory source is usable")


def require_memory_headroom(
    *, minimum_available_mib: int = MINIMUM_AVAILABLE_MIB
) -> dict[str, Any]:
    if minimum_available_mib < MINIMUM_AVAILABLE_MIB:
        raise M3CRResumableOperatorError(
            f"minimum available memory cannot be below {MINIMUM_AVAILABLE_MIB} MiB"
        )
    available, source = available_memory_bytes()
    threshold = minimum_available_mib * 1024 * 1024
    result = {
        "available_bytes": available,
        "available_mib": available // (1024 * 1024),
        "memory_source": source,
        "minimum_available_bytes": threshold,
        "minimum_available_mib": minimum_available_mib,
        "sufficient": available >= threshold,
    }
    if not result["sufficient"]:
        raise M3CRResumableOperatorError(
            "insufficient available memory; stopped before engine construction"
        )
    return result


def expected_execution_preflight(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    require_review_confirmation(private_root)
    pin_receipt = capture_or_reuse_local_pin(private_root)
    package, _, forbidden = validate_immutable_inputs(private_root)
    binding = build_private_path_binding(
        package_path=paths["package"],
        working_store_path=paths["working_store"],
        baseline_backup_path=paths["baseline_backup"],
        separate_restore_path=paths["separate_restore"],
        forbidden_existing_path_digests=forbidden,
    )
    if binding.binding_digest != package.authorization.path_binding_digest:
        raise M3CRResumableOperatorError(
            "concrete phone paths differ from the package path binding"
        )
    if not paths["operator_receipt"].exists():
        require_single_use_private_paths(
            package_path=paths["package"],
            working_store_path=paths["working_store"],
            baseline_backup_path=paths["baseline_backup"],
            separate_restore_path=paths["separate_restore"],
            path_binding=binding,
        )
    return {
        "authorization_capture_receipt_digest": pin_receipt[
            "capture_receipt_digest"
        ],
        "engine_construction_authorized_stage": 4,
        "forbidden_prior_path_digest_count": len(forbidden),
        "legacy_goal_authority_transfer_authorized": False,
        "m3_e_authority_open": False,
        "memory_preflight": require_memory_headroom(),
        "package_digest": package.package_digest,
        "path_binding_digest": binding.binding_digest,
        "pin_stage_engine_load": False,
        "raw_private_text_or_path_output": False,
        "schema_version": PREFLIGHT_SCHEMA,
    }


def record_execution_preflight(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    if paths["preflight"].exists():
        existing = load_canonical_mapping(
            paths["preflight"], field="execution preflight"
        )
        if (
            existing.get("schema_version") != PREFLIGHT_SCHEMA
            or existing.get("package_digest") != EXPECTED_PACKAGE_DIGEST
        ):
            raise M3CRResumableOperatorError("execution preflight mismatch")
        return existing
    preflight = expected_execution_preflight(private_root)
    write_idempotent_canonical(paths["preflight"], preflight)
    return preflight


def require_execution_preflight(private_root: str | Path) -> dict[str, Any]:
    paths = private_paths(private_root)
    value = load_canonical_mapping(
        paths["preflight"], field="execution preflight"
    )
    expected = {
        "engine_construction_authorized_stage": 4,
        "legacy_goal_authority_transfer_authorized": False,
        "m3_e_authority_open": False,
        "package_digest": EXPECTED_PACKAGE_DIGEST,
        "pin_stage_engine_load": False,
        "raw_private_text_or_path_output": False,
        "schema_version": PREFLIGHT_SCHEMA,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise M3CRResumableOperatorError("execution preflight field mismatch")
    return value
