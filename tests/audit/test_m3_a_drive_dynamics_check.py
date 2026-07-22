from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/audit/m3_a_drive_dynamics_check.py"
DESIGN = REPO_ROOT / "docs/audit/M3_A_DRIVE_DYNAMICS_DESIGN.md"
AFFECT_PLAN = REPO_ROOT / "docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md"
STATUS = REPO_ROOT / "docs/EVE_IMPLEMENTATION_STATUS_v4.md"

EXPECTED_DRIVES = (
    "energy", "safety", "affiliation", "curiosity",
    "agency", "coherence", "competence", "expression",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("m3_a_drive_dynamics_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


@pytest.fixture(scope="module")
def report(module):
    return module.audit_repository(REPO_ROOT)


def test_required_files_exist():
    assert all(path.is_file() for path in {DESIGN, SCRIPT, Path(__file__), AFFECT_PLAN, STATUS})


def test_schema_baseline_and_static_scope(report):
    assert report["schema_version"] == "eve.m3-a.drive-dynamics-check.v1"
    assert report["baseline_sha"] == "7697c1047bbf081295a01f630d63d8a3ad5c69b0"
    assert report["validation_errors"] == []
    assert report["scope"] == {
        "standard_library_only": True,
        "static_document_analysis_only": True,
        "runtime_import_performed": False,
        "runtime_execution_performed": False,
        "production_state_read_performed": False,
        "production_state_write_performed": False,
        "event_emission_performed": False,
        "projection_implementation_performed": False,
        "scheduler_integration_performed": False,
        "goal_integration_performed": False,
        "persistence_integration_performed": False,
        "cutover_authorized": False,
        "m3_e_authority_granted": False,
        "integration_eligible_before_cutover": False,
    }


def test_drive_catalog_and_continuous_parameters(report):
    params = report["drive_parameters"]
    assert tuple(row["drive"] for row in params) == EXPECTED_DRIVES
    assert len(params) == 8
    for row in params:
        assert row["ruling"] == "RESOLVED"
        assert row["open_question"] == "—"
        assert row["tau_seconds"] > 0
        assert 0 <= row["floor"] < row["ceiling"] <= 1
        assert row["floor"] <= row["baseline"] <= row["ceiling"]
        assert row["gain"] > 0
        assert row["max_slew_per_second"] > 0
    assert set(report["baseline_states"]) == set(EXPECTED_DRIVES)


def test_four_semantic_states_per_drive(report):
    states = report["semantic_states"]
    assert len(states) == 32
    assert Counter(row["drive"] for row in states) == Counter({drive: 4 for drive in EXPECTED_DRIVES})
    for drive in EXPECTED_DRIVES:
        rows = sorted((row for row in states if row["drive"] == drive), key=lambda row: row["ordinal"])
        assert [row["ordinal"] for row in rows] == [0, 1, 2, 3]
        assert len({row["state"] for row in rows}) == 4
        assert all(row["ruling"] == "RESOLVED" for row in rows)


def test_named_transition_catalog_is_exactly_bidirectional_and_adjacent(report):
    states_by_drive = {
        drive: [row["state"] for row in sorted(
            (value for value in report["semantic_states"] if value["drive"] == drive),
            key=lambda value: value["ordinal"],
        )]
        for drive in EXPECTED_DRIVES
    }
    expected = set()
    for drive, states in states_by_drive.items():
        for left, right in zip(states, states[1:]):
            expected.add((drive, left, right))
            expected.add((drive, right, left))
    transitions = report["named_transitions"]
    actual = {(row["drive"], row["from_state"], row["to_state"]) for row in transitions}
    assert len(transitions) == 48
    assert actual == expected
    assert len({row["transition_id"] for row in transitions}) == 48
    assert all(row["predicate_version"] == "eve.m3-a.named-transition-predicate.v1" for row in transitions)
    assert all(row["ruling"] == "RESOLVED" for row in transitions)


def test_hysteresis_width_and_cooldown_are_resolved_per_named_edge(report):
    states = {
        drive: {row["state"]: row["ordinal"] for row in report["semantic_states"] if row["drive"] == drive}
        for drive in EXPECTED_DRIVES
    }
    by_edge = {(row["drive"], row["from_state"], row["to_state"]): row for row in report["named_transitions"]}
    for drive in EXPECTED_DRIVES:
        ordered = [state for state, _ in sorted(states[drive].items(), key=lambda item: item[1])]
        for left, right in zip(ordered, ordered[1:]):
            up = by_edge[(drive, left, right)]
            down = by_edge[(drive, right, left)]
            assert up["direction"] == "up" and down["direction"] == "down"
            assert up["operator"] == ">=" and down["operator"] == "<="
            assert up["threshold"] > down["threshold"]
            assert up["hysteresis_width"] == pytest.approx(up["threshold"] - down["threshold"])
            assert down["hysteresis_width"] == pytest.approx(up["threshold"] - down["threshold"])
            assert up["cooldown_seconds"] == down["cooldown_seconds"]
            assert up["cooldown_seconds"] > 0


def test_candidate_lifecycle_is_complete_and_has_no_self_loop(report, module):
    lifecycle = report["candidate_lifecycle"]
    edges = {(row["from_state"], row["to_state"]) for row in lifecycle}
    assert edges == module.EXPECTED_LIFECYCLE_EDGES
    assert all(left != right for left, right in edges)
    assert all(row["trigger"] and row["authority_effect"] for row in lifecycle)


def test_affect_plan_all_63_axes_land_or_preserve(report):
    summary = report["summary"]
    assert summary["affect_axes"] == 63
    assert summary["mapped_axes"] == 59
    assert summary["proposed_drop_axes"] == 4
    assert summary["source_unresolved_axes"] == 0
    assert summary["covered_axes"] == 63
    assert summary["drive_target_landings"] >= 59
    assert summary["appraisal_target_landings"] >= 59
    assert summary["emotion_target_landings"] >= 59
    rows = report["axis_landings"]
    assert len(rows) == 63 and all(row["covered"] for row in rows)
    for row in rows:
        if row["status"] == "MAPPED":
            assert any(target.startswith("drive::") for target in row["targets"])
            assert all(target.startswith(("drive::", "appraisal::", "emotion::")) for target in row["targets"])
        else:
            assert row["status"] == "PROPOSED-DROP"
            assert row["targets"] == []
            assert row["preservation"] not in {"", "—", "-"}


def test_a9_continuous_sample_and_duplicate_proof_text():
    text = DESIGN.read_text(encoding="utf-8")
    required = (
        "Continuous sampling emits zero events.",
        "Only one adjacent edge is eligible per drive per logical step.",
        "A drive has at most one non-terminal candidate.",
        "Candidate identity is fixed to `next_state_epoch`",
        "A reverse candidate therefore requires a prior accepted transition",
        "at most one adjacent state",
        "monotonic `Δt`",
    )
    assert all(token in text for token in required)


def test_checker_imports_standard_library_only():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    allowed = {"__future__", "argparse", "json", "math", "re", "sys", "collections", "pathlib", "typing"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", 1)[0])
    assert imported <= allowed
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not (called & {"open", "exec", "eval", "__import__"})


def test_cli_is_deterministic_and_fail_on_unresolved_is_green():
    command = [sys.executable, str(SCRIPT), "--root", str(REPO_ROOT)]
    first = subprocess.run(command, check=False, text=True, capture_output=True)
    second = subprocess.run(command, check=False, text=True, capture_output=True)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert json.loads(first.stdout)["summary"]["validation_errors"] == 0
    strict = subprocess.run(command + ["--fail-on-unresolved", "--summary-only"], check=False, text=True, capture_output=True)
    assert strict.returncode == 0
    summary = json.loads(strict.stdout)
    assert summary["unresolved_rulings"] == 0
    assert summary["source_unresolved_axes"] == 0


def test_status_records_design_only_parallel_boundary():
    text = STATUS.read_text(encoding="utf-8")
    required = (
        "M3-A drive-dynamics design status",
        "documentation-only",
        "63-axis Affect Migration Plan",
        "48 bidirectional named transitions",
        "no runtime integration",
        "integration eligibility only after persistence cutover",
    )
    assert all(token in text for token in required)
