#!/usr/bin/env python3
"""Stage 4: run the single engine-loading M3-C goal dual-read window."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_c_o_private_device_goal_dual_read_operator import (  # noqa: E402
    build_private_path_binding,
    execute_private_device_goal_dual_read_window,
    require_single_use_private_paths,
    verify_active_operator_authorization,
)
from core.m3_c_p_private_device_goal_window_authorization_pin import (  # noqa: E402
    binding_from_private_package,
)
from core.m3_c_r_resumable_phone_goal_window import (  # noqa: E402
    canonical_json,
    consume_local_pin,
    existing_completed_operator_receipt,
    local_reviewed_operator_session,
    private_paths,
    read_forbidden_digests,
    refuse_partial_execution,
    require_execution_preflight,
    require_memory_headroom,
    validate_immutable_inputs,
    write_idempotent_canonical,
)


def _clean_exact_head(expected_head: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual != expected_head:
        raise SystemExit(
            f"repository head mismatch: expected {expected_head}, got {actual}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty.strip():
        raise SystemExit("stage 4 requires a clean exact checkout")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--private-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launch_head = _clean_exact_head(args.expected_head)
    paths = private_paths(args.private_root)

    completed = existing_completed_operator_receipt(paths["root"])
    if completed is not None:
        print(
            canonical_json(
                {
                    "engine_loaded": False,
                    "operator_execution": completed,
                    "resumed_completed_stage": True,
                    "stage": 4,
                }
            )
        )
        return 0

    require_execution_preflight(paths["root"])
    refuse_partial_execution(paths["root"])
    memory = require_memory_headroom()
    package, _, _ = validate_immutable_inputs(paths["root"])
    forbidden = read_forbidden_digests(paths["forbidden"])
    path_binding = build_private_path_binding(
        package_path=paths["package"],
        working_store_path=paths["working_store"],
        baseline_backup_path=paths["baseline_backup"],
        separate_restore_path=paths["separate_restore"],
        forbidden_existing_path_digests=forbidden,
    )
    if path_binding.binding_digest != package.authorization.path_binding_digest:
        raise SystemExit("stage 4 path binding differs from reviewed package")
    require_single_use_private_paths(
        package_path=paths["package"],
        working_store_path=paths["working_store"],
        baseline_backup_path=paths["baseline_backup"],
        separate_restore_path=paths["separate_restore"],
        path_binding=path_binding,
    )
    binding = binding_from_private_package(package)

    with local_reviewed_operator_session(paths["root"], binding):
        verify_active_operator_authorization(package.authorization)
        from main import build_full_engine

        engine = build_full_engine()
        goal_adapter = engine.goal_adapter
        if goal_adapter.production_origin_shadow_tap is not None:
            raise SystemExit(
                "default engine unexpectedly contains a goal shadow tap"
            )
        receipt, window_receipt = execute_private_device_goal_dual_read_window(
            package,
            goal_adapter=goal_adapter,
            path_binding=path_binding,
            working_store_path=paths["working_store"],
            baseline_backup_path=paths["baseline_backup"],
            separate_restore_path=paths["separate_restore"],
            launch_repository_head=launch_head,
        )

    public_receipt = {
        "engine_load_count": 1,
        "memory_preflight": memory,
        "operator_receipt": receipt.to_mapping(),
        "operator_receipt_digest": receipt.receipt_digest,
        "private_path_plaintext_public": False,
        "raw_private_text_public": False,
        "window_receipt": window_receipt.to_mapping(),
    }
    write_idempotent_canonical(paths["operator_receipt"], public_receipt)
    consume_local_pin(paths["root"])
    print(
        canonical_json(
            {
                "engine_loaded": True,
                "operator_execution": public_receipt,
                "stage": 4,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
