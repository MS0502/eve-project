from __future__ import annotations

import json
from pathlib import Path

from core.m2_e_window_driver import CHAOS_PHASES, WINDOW_BASELINE_SHA
from scripts.audit.m2_e_window_driver import (
    create_portability,
    run_chaos,
    verify_portability,
)


def test_synthetic_hard_kill_matrix_corruption_and_disk_pressure(tmp_path: Path):
    evidence = run_chaos(workspace=tmp_path / "chaos", repetitions=1)
    assert evidence["machine_passed"] is True, json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, indent=2
    )
    assert evidence["baseline_sha"] == WINDOW_BASELINE_SHA
    assert evidence["synthetic_store_only"] is True
    assert evidence["legacy_runtime_authoritative"] is True
    assert evidence["cutover_authorized"] is False
    assert evidence["m3_authority_open"] is False
    assert len(evidence["kill_matrix"]) == len(CHAOS_PHASES)
    assert {item["phase"] for item in evidence["kill_matrix"]} == set(CHAOS_PHASES)
    assert all(item["checks"]["recovery_digest_match"] for item in evidence["kill_matrix"])
    assert evidence["corruption"]["passed"] is True
    assert evidence["disk_pressure"]["passed"] is True
    assert len(evidence["evidence_digest"]) == 64


def test_portable_backup_restores_and_replays_from_detached_artifact(tmp_path: Path):
    manifest, backup = create_portability(workspace=tmp_path / "source")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    result = verify_portability(
        backup=backup,
        manifest_path=manifest_path,
        workspace=tmp_path / "target",
    )
    assert result["machine_passed"] is True
    assert result["recovered_digest"] == manifest["expected_recovery_digest"]
    assert result["checks"]["backup_digest_match"] is True
    assert result["checks"]["replay_digest_match"] is True
    assert len(result["verification_digest"]) == 64
