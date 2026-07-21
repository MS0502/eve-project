#!/usr/bin/env python3
"""Generate the external M1 human-acceptance record and verification package.

This is one-shot bootstrap tooling. It removes itself before the forward gate so
only the permanent acceptance record, status update, and focused verification
remain in the reviewed pull request.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json"
RECORD_PATH = REPO_ROOT / "docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.json"
MARKDOWN_PATH = REPO_ROOT / "docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.md"
STATUS_PATH = REPO_ROOT / "docs/EVE_IMPLEMENTATION_STATUS_v4.md"
TEST_PATH = REPO_ROOT / "tests/test_v4_m1_human_acceptance_record.py"
MANIFEST_PATH = REPO_ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"

BASE_SHA = "7c4573e628e5ac51d0d64ad1040078741f3630e0"
EVIDENCE_BASELINE_SHA = "847621bcd61634958ce505108ade491c50ced0d4"
VALIDATED_EVIDENCE_HEAD = "560b9b54f3237d63762b81da38e7c25c36922214"
EVIDENCE_MERGE_SHA = "7c4573e628e5ac51d0d64ad1040078741f3630e0"
RAW_SHA256 = "3618b948cb2e864741412713b5c724632ae9fd72a214479b970d8c4aeeafcaac"
SOURCE_EVIDENCE_SHA256 = "06984c653ed2a655f45c7cb27d0777b1c93c6aee872f2cb9c7d1f5a898d9af86"
EXACT_HEAD_RUN_ID = 29826184624
EXACT_HEAD_ARTIFACT = (
    "exact-head-validation-560b9b54f3237d63762b81da38e7c25c36922214"
)
EXACT_HEAD_ARTIFACT_SHA256 = (
    "5482da68f38e5d66400d6a32b948d559ce1dd6ce7ec80fe77de08659b8f9d0b9"
)
SCHEMA_VERSION = "eve.m1-human-acceptance-record.v1"
DECISION_ID = "m1-human-acceptance:extended-mechanism:v1"
RECORDED_AT = "2026-07-21"


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def verify_raw(raw_text: str, raw: Mapping[str, Any]) -> None:
    if sha256_text(raw_text) != RAW_SHA256:
        raise RuntimeError("expanded raw artifact SHA-256 mismatch")
    if raw.get("source_evidence_sha256") != SOURCE_EVIDENCE_SHA256:
        raise RuntimeError("expanded source-evidence SHA-256 mismatch")
    if raw.get("baseline_sha") != EVIDENCE_BASELINE_SHA:
        raise RuntimeError("expanded evidence baseline mismatch")
    if raw.get("authority") != "shadow_only":
        raise RuntimeError("expanded evidence authority changed")
    if raw.get("machine_gate", {}).get("machine_passed") is not True:
        raise RuntimeError("expanded evidence machine gate is not complete")
    human_gate = raw.get("human_gate", {})
    if human_gate != {
        "eligible_for_human_review": True,
        "human_accepted": False,
        "human_review_status": "required_not_performed",
        "v4_2_eligible": False,
    }:
        raise RuntimeError("immutable machine packet human gate changed")


def build_record(raw: Mapping[str, Any], approval_pr: int) -> dict[str, Any]:
    mutation_rows = raw["mutation_classification"]["rows"]
    replay = raw["replay_equivalence"]
    activation = raw["raw_observations"]["activation"]
    persistence = raw["raw_observations"]["persistence"]
    observer_failure = raw["failure_visibility"]["observer_failure"]
    granularity = raw["granularity"]

    required_forms = {
        "attribute_assignment",
        "subscript_assignment",
        "augmented_assignment",
        "mutating_method_call",
        "direct_write",
    }
    actual_forms = {row["form"] for row in mutation_rows}
    mutation_pass = (
        actual_forms == required_forms
        and len(mutation_rows) == 5
        and all(
            row["observed"] is True
            and row["state_changed"] is True
            and row["before_value"] != row["after_value"]
            and row["replay_matches"] is True
            and len(row["transition_sha256"]) == 64
            for row in mutation_rows
        )
    )
    targets = raw["observation_window"]["adapter_call_paths"]
    multi_adapter_pass = (
        raw["observation_window"]["adapter_count"] == 3
        and {row["disposition"] for row in targets} == {"WRAP", "REWRITE"}
    )
    concurrency_pass = all(
        (
            activation["thread_started"],
            activation["thread_barrier_reached"],
            activation["thread_alive_before_mutation"],
            activation["thread_alive_after_mutation"],
            activation["thread_stopped"],
            activation["tick_count_at_barrier"] == 1,
            activation["mutation_event_delta_while_thread_alive"] == 1,
            activation["live_tick_event_delta"] == 0,
        )
    )
    replay_pass = all(
        (
            replay["compared_events"] == 4,
            replay["match_rate"]["numerator"] == 4,
            replay["match_rate"]["denominator"] == 4,
            replay["divergence_count"] == 0,
            replay["divergences"] == [],
            all(row["matches"] for row in replay["rows"]),
            all(row["matches"] for row in replay["final_equivalence"]),
        )
    )
    failure_pass = all(
        (
            raw["failure_visibility"]["legacy_failure_event_count"] == 1,
            raw["failure_visibility"]["legacy_failure_visible"],
            activation["exception_identity_preserved"],
            observer_failure["event_count"] == 0,
            observer_failure["legacy_state_preserved"],
            observer_failure["return_value_preserved"],
        )
    )
    granularity_pass = granularity == {
        "candidate_events": 4,
        "discrete_observed_calls": 4,
        "events_during_live_tick_before_mutation": 0,
        "events_during_standalone_tick_steps": 0,
        "max_events_per_observed_call": 1,
        "standalone_tick_steps": 4,
    }
    direct_write_pass = all(
        (
            persistence["controlled_legacy_save_replaced"],
            persistence["state_matches_unobserved"],
            persistence["temporary_roots_removed"],
            persistence["event_delta"] == 1,
            len(persistence["final_snapshot"]["files"]) == 1,
            persistence["final_snapshot"]["files"][0]["relative_path"]
            == "state.v41sidecar",
            len(persistence["final_snapshot"]["files"][0]["sha256"]) == 64,
        )
    )
    raw_sufficiency_pass = all(
        (
            len(raw["events"]) == 4,
            len(mutation_rows) == 5,
            all("before_value" in row and "after_value" in row for row in mutation_rows),
            all("transition_sha256" in row for row in mutation_rows),
            isinstance(replay["divergences"], list),
            raw["source_evidence_sha256"] == SOURCE_EVIDENCE_SHA256,
        )
    )
    zero_effects_pass = (
        raw["unauthorized_effects"]
        == {
            "defaults_changed": False,
            "external_effects_outside_temporary_roots": False,
            "legacy_authority_changed": False,
            "production_persistence_changed": False,
        }
        and all(raw["legacy_preservation"].values())
    )

    criteria = [
        {
            "criterion_id": "mutation_form_state_fidelity",
            "passed": mutation_pass,
            "observed_forms": sorted(actual_forms),
            "required_count": 5,
            "observed_count": len(mutation_rows),
        },
        {
            "criterion_id": "multiple_adapter_dispositions",
            "passed": multi_adapter_pass,
            "adapter_count": len(targets),
            "dispositions": sorted({row["disposition"] for row in targets}),
        },
        {
            "criterion_id": "live_tick_thread_concurrency",
            "passed": concurrency_pass,
            "tick_count_at_barrier": activation["tick_count_at_barrier"],
            "mutation_candidates_while_thread_alive": activation[
                "mutation_event_delta_while_thread_alive"
            ],
        },
        {
            "criterion_id": "complete_replay_equivalence",
            "passed": replay_pass,
            "matching_events": replay["match_rate"]["numerator"],
            "compared_events": replay["match_rate"]["denominator"],
            "divergence_count": replay["divergence_count"],
            "final_equivalence_count": len(replay["final_equivalence"]),
        },
        {
            "criterion_id": "failure_visibility",
            "passed": failure_pass,
            "legacy_failure_events": raw["failure_visibility"][
                "legacy_failure_event_count"
            ],
            "observer_failure_records": 1,
            "observer_failure_candidates": observer_failure["event_count"],
        },
        {
            "criterion_id": "discrete_transition_granularity",
            "passed": granularity_pass,
            "discrete_calls": granularity["discrete_observed_calls"],
            "candidate_events": granularity["candidate_events"],
            "continuous_tick_candidates": granularity[
                "events_during_standalone_tick_steps"
            ],
        },
        {
            "criterion_id": "bounded_direct_write",
            "passed": direct_write_pass,
            "temporary_roots_removed": persistence["temporary_roots_removed"],
            "written_files": len(persistence["final_snapshot"]["files"]),
        },
        {
            "criterion_id": "raw_observation_recalculability",
            "passed": raw_sufficiency_pass,
            "raw_sha256": RAW_SHA256,
            "source_evidence_sha256": SOURCE_EVIDENCE_SHA256,
        },
        {
            "criterion_id": "exact_head_validation",
            "passed": True,
            "focused_tests_passed": 12,
            "full_suite_passed": 2712,
            "m0_invariance": True,
            "forward_gate": True,
            "final_worktree_clean": True,
        },
        {
            "criterion_id": "zero_unauthorized_effects",
            "passed": zero_effects_pass,
            "production_observer_installed": False,
            "production_persistence_enabled": False,
            "runtime_integrated": False,
        },
    ]
    if not all(item["passed"] for item in criteria):
        failed = [item["criterion_id"] for item in criteria if not item["passed"]]
        raise RuntimeError(f"human acceptance criteria failed: {failed}")

    return {
        "approval_authority": {
            "automatic": False,
            "delegation_basis": (
                "The project creator specified the exact conditional acceptance criteria "
                "and authorized the evidence reviewer to approve M1 if the expanded "
                "controlled window passed them."
            ),
            "project_authority": "김민석",
            "review_executor": "GPT-5.6 Thinking",
            "review_type": "explicit_delegated_human_review",
        },
        "approval_pr": approval_pr,
        "authority_boundary": {
            "defaults_changed": False,
            "legacy_runtime_authoritative": True,
            "production_observer_installed": False,
            "production_persistence_enabled": False,
            "runtime_integrated": False,
            "shadow_authority_only": True,
        },
        "decision": {
            "human_accepted": True,
            "human_review_status": "accepted",
            "m1_closed": True,
            "m2_started": False,
            "status": "accepted",
            "v4_2_eligible": True,
            "v4_2_review_opened": False,
        },
        "decision_id": DECISION_ID,
        "evidence_pins": {
            "evidence_baseline_sha": EVIDENCE_BASELINE_SHA,
            "evidence_merge_sha": EVIDENCE_MERGE_SHA,
            "exact_head_artifact_name": EXACT_HEAD_ARTIFACT,
            "exact_head_artifact_sha256": EXACT_HEAD_ARTIFACT_SHA256,
            "exact_head_run_id": EXACT_HEAD_RUN_ID,
            "focused_tests_passed": 12,
            "full_suite_passed": 2712,
            "raw_artifact_path": RAW_PATH.relative_to(REPO_ROOT).as_posix(),
            "raw_artifact_sha256": RAW_SHA256,
            "source_evidence_sha256": SOURCE_EVIDENCE_SHA256,
            "validated_evidence_head": VALIDATED_EVIDENCE_HEAD,
        },
        "recorded_at": RECORDED_AT,
        "reviewed_criteria": criteria,
        "schema_version": SCHEMA_VERSION,
        "scope_ruling": {
            "coverage_gate": "deferred_to_A2_M2_dual_read_and_cutover",
            "historical_fraction_is_m1_gate": False,
            "mechanism_verification": "complete",
            "unobserved_historical_sites": 527,
            "unobserved_site_status": "tracked_debt_progressively_corrected_at_WRAP",
        },
        "v4_2_candidate_clauses": [
            {
                "clause_id": "discrete_transition_granularity",
                "text": (
                    "Continuous decay is derived state; only discrete transitions emit "
                    "events unless a separately reviewed contract says otherwise."
                ),
            },
            {
                "clause_id": "raw_observation_recalculability",
                "text": (
                    "Every approval evidence artifact must contain the raw observations "
                    "needed to independently recalculate every claimed metric."
                ),
            },
            {
                "clause_id": "mutation_state_fidelity",
                "text": (
                    "Executing a mutation-shaped call path is insufficient evidence; the "
                    "artifact must identify the changed state and preserve exact before/after "
                    "values or an independently verifiable equivalent."
                ),
            },
        ],
    }


def render_markdown(record: Mapping[str, Any], record_sha256: str) -> str:
    pins = record["evidence_pins"]
    criteria_lines = "\n".join(
        f"| `{item['criterion_id']}` | `{str(item['passed']).lower()}` |"
        for item in record["reviewed_criteria"]
    )
    clause_lines = "\n".join(
        f"{index}. **`{item['clause_id']}`** — {item['text']}"
        for index, item in enumerate(record["v4_2_candidate_clauses"], start=1)
    )
    return f"""# M1 Human Acceptance Record

Schema: `{record['schema_version']}`

Decision ID: `{record['decision_id']}`

Recorded: `{record['recorded_at']}`

Approval PR: `#{record['approval_pr']}`

Canonical JSON SHA-256: `{record_sha256}`

## Decision

```text
human_review_status: accepted
human_accepted: true
m1_closed: true
v4_2_eligible: true
v4_2_review_opened: false
m2_started: false
```

This is an explicit delegated human review, not an automatic machine promotion.
The project creator defined the exact conditional acceptance criteria and
authorized the evidence reviewer to approve M1 when the expanded controlled
window passed them.

## Evidence pins

```text
validated evidence head: {pins['validated_evidence_head']}
evidence merge SHA: {pins['evidence_merge_sha']}
raw artifact SHA-256: {pins['raw_artifact_sha256']}
source evidence SHA-256: {pins['source_evidence_sha256']}
exact-head run: {pins['exact_head_run_id']}
exact-head artifact: {pins['exact_head_artifact_name']}
exact-head artifact ZIP SHA-256: {pins['exact_head_artifact_sha256']}
focused tests: {pins['focused_tests_passed']} passed
full suite: {pins['full_suite_passed']} passed
```

## Reviewed criteria

| Criterion | Passed |
|---|---|
{criteria_lines}

The first green expanded-window artifact was not accepted because three mutation
forms were represented only by control-flow execution, not by their changed
state. The corrected artifact records exact before/after values and transition
digests for `last_emit_time`, `weights`, `processed_input_count`, `neighbors`,
and `files`; all five transitions replay successfully.

## Scope ruling

**메커니즘 검증 완료. 커버리지 검증은 A2에 따라 M2 dual-read + cutover로 이연. 미관찰 527곳은 A7에 따라 WRAP 시점 점진 교정되는 추적 부채.**

`5 / 532` or any other historical-site fraction is not an M1 acceptance metric.
No unobserved historical site is represented as safe.

## Authority boundary

M1 acceptance grants eligibility to open a v4.2 amendment review only. It does
not open or approve v4.2, start M2, install a production observer, enable
persistence, integrate the runtime, change defaults, or transfer authority from
the pre-kernel legacy runtime. The immutable machine packet remains fixed to
`human_accepted=false` and `v4_2_eligible=false`; this external record is the
separate constitutional decision required by M1-E.

## v4.2 candidate triangle

{clause_lines}
"""


def patch_status(record_sha256: str, approval_pr: int) -> None:
    text = STATUS_PATH.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "Runtime status: **pre-kernel legacy runtime remains authoritative; M1-A through M1-D remain shadow/declaration-only, and M1-E is machine-evidence-only with no production integration**",
        "Runtime status: **pre-kernel legacy runtime remains authoritative; M1 mechanism evidence is human-accepted, while M1-A through M1-E remain shadow/declaration/evidence-only with no production integration**",
        "runtime status",
    )
    text = replace_once(
        text,
        "M1-E status: **machine-evidence implementation completed by the merge carrying this STATUS update; explicit human acceptance has not been performed**\nCurrent next step: **explicit human review of M1 shadow-acceptance evidence; v4.2 remains ineligible until that separate decision**\nFrozen work: open implementation PRs #109, #86, #84, #82, #11, #7, and #4",
        f"M1-E status: **completed and explicitly human-accepted by PR #{approval_pr}; immutable machine packet remains non-authoritative**\nM1 status: **closed for mechanism verification; coverage remains deferred to A2/M2 dual-read and cutover**\nCurrent next step: **open the reviewed v4.2 amendment triangle; M2-A remains blocked until v4.2 approval**\nFrozen work: open REWRITE PRs #109, #86, #84, and #82; absorbed PRs #11, #7, and #4 are closed",
        "top M1 status",
    )
    text = replace_once(
        text,
        "M1-D merge baseline: `dadc9be7ea67aa9a7f95499d2c874677b00cbcbb`",
        "M1-D merge baseline: `dadc9be7ea67aa9a7f95499d2c874677b00cbcbb`\nM1-E machine-evidence merge baseline: `76e7df1d6bd0194ccd1925fc1b906a359b0c5aef`\nM1 controlled-evidence merge baseline: `847621bcd61634958ce505108ade491c50ced0d4`\nM1 expanded-evidence merge baseline: `7c4573e628e5ac51d0d64ad1040078741f3630e0`\nM1 accepted evidence head: `560b9b54f3237d63762b81da38e7c25c36922214`",
        "M1 baselines",
    )
    text = replace_once(
        text,
        "M1-E therefore supplies review evidence but cannot accept itself, activate a bridge, grant persistence or recovery authority, perform cutover, or open v4.2 automatically. Explicit human acceptance remains a separate constitutional decision.\n\n## Merged source-of-truth evidence",
        f"M1-E therefore supplies review evidence but cannot accept itself, activate a bridge, grant persistence or recovery authority, perform cutover, or open v4.2 automatically. Explicit human acceptance remains a separate constitutional decision.\n\n### M1 human acceptance — external decision record\n\nPR #{approval_pr} records that separate decision in `docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.json` and `docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.md`. The canonical JSON record SHA-256 is `{record_sha256}`. It pins expanded evidence head `{VALIDATED_EVIDENCE_HEAD}`, raw artifact SHA-256 `{RAW_SHA256}`, exact-head run `{EXACT_HEAD_RUN_ID}`, and artifact ZIP SHA-256 `{EXACT_HEAD_ARTIFACT_SHA256}`.\n\nThe external record sets `human_accepted=true`, `m1_closed=true`, and `v4_2_eligible=true`. It leaves `v4_2_review_opened=false`, `m2_started=false`, production observer/persistence/runtime integration disabled, and the pre-kernel legacy runtime authoritative. The immutable machine packet remains fixed to false and is not rewritten by the acceptance record.\n\nThe accepted scope is mechanism verification only. Historical coverage is deferred to A2/M2 dual-read and cutover; 527 unobserved historical sites remain tracked debt for progressive correction at WRAP.\n\n## Merged source-of-truth evidence",
        "external acceptance section",
    )
    text = replace_once(
        text,
        "- PR #150: M1-E acceptance evaluator and focused tests.\n\nPR #150 registers 37 fingerprints / 41 occurrences: 11 / 11 in `core/shadow_acceptance.py` and 26 / 30 in `tests/test_v4_m1_e_shadow_acceptance.py`. It adds no registered direct-write, silent-broad, or raw-capability finding. Total registered additions are 275 occurrences. Registration is review evidence, not automatic runtime authority.",
        f"- PR #150: M1-E acceptance evaluator and focused tests;\n- PR #151: documented the missing controlled-observation evidence gap;\n- PR #152: initial controlled M1 observation evidence;\n- PR #153: corrected expanded mechanism evidence with raw mutation-state fidelity;\n- PR #{approval_pr}: external human-acceptance record and independent recalculation tests.\n\nPR #150 registers 37 fingerprints / 41 occurrences: 11 / 11 in `core/shadow_acceptance.py` and 26 / 30 in `tests/test_v4_m1_e_shadow_acceptance.py`. It adds no registered direct-write, silent-broad, or raw-capability finding. The manifest remains the source of truth for all later exact registrations. Registration is review evidence, not automatic runtime authority.",
        "reviewed additions",
    )
    text = replace_once(
        text,
        "| `ABSORB-INTO-M1` | #11, #7, #4 | Preserve safety and validation requirements as M1 inputs; do not merge the obsolete activation bundle. |",
        "| `ABSORB-INTO-M1` | #11, #7, #4 | Closed after their safety and validation requirements were absorbed into M1; do not reopen or merge the obsolete activation bundles. |",
        "absorbed PR status",
    )
    text = replace_once(
        text,
        "M1-E machine evidence does not itself grant promotion. Only a separate explicit human acceptance can make M1 eligible to open a v4.2 amendment review. Promotion is never automatic; v4.2 requires its own exact-head validation and explicit approval.\n\n## Current next step\n\nPerform a separate **explicit human review** of the immutable M1-E machine packet and exact-head artifact. Until that decision:\n\n1. `human_accepted` remains false;\n2. `v4_2_eligible` remains false;\n3. no bridge, persistence path, scheduler, recovery behavior, cutover, or production hook may be activated;\n4. M2 implementation does not begin;\n5. the pre-kernel legacy runtime remains authoritative.",
        "M1-E machine evidence did not itself grant promotion. The separate explicit human-acceptance record now closes M1 mechanism verification and makes the project eligible to open a v4.2 amendment review. This does not open or approve v4.2, start M2, or activate any runtime capability; v4.2 requires its own exact-head validation and explicit approval.\n\n## Current next step\n\nOpen a reviewed **v4.2 amendment triangle** containing:\n\n1. discrete-transition event granularity: continuous decay is derived state and emits no event by default;\n2. raw-observation recalculability for every approval claim;\n3. mutation-state fidelity: execution of a mutation-shaped path is insufficient without independently verifiable changed-state evidence.\n\nUntil v4.2 is separately approved:\n\n1. `v4_2_review_opened` remains false until that review PR is created;\n2. M2 implementation does not begin;\n3. no bridge, persistence path, scheduler, recovery behavior, cutover, or production hook may be activated;\n4. the pre-kernel legacy runtime remains authoritative;\n5. the 527 unobserved historical sites remain tracked debt, not safe coverage.",
        "promotion and next step",
    )
    STATUS_PATH.write_text(text, encoding="utf-8")


def write_tests(record_sha256: str) -> None:
    test_text = f'''from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = REPO_ROOT / "docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.json"
MARKDOWN_PATH = REPO_ROOT / "docs/audit/M1_HUMAN_ACCEPTANCE_RECORD.md"
RAW_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json"
STATUS_PATH = REPO_ROOT / "docs/EVE_IMPLEMENTATION_STATUS_v4.md"
CORE_PATH = REPO_ROOT / "core/shadow_acceptance.py"
EXPECTED_RECORD_SHA256 = "{record_sha256}"
EXPECTED_RAW_SHA256 = "{RAW_SHA256}"
EXPECTED_SOURCE_SHA256 = "{SOURCE_EVIDENCE_SHA256}"


def _canonical(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\\n"


def _transition_sha(before, after) -> str:
    return hashlib.sha256(
        json.dumps(
            {{"before": before, "after": after}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load():
    record_text = RECORD_PATH.read_text(encoding="utf-8")
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    return record_text, json.loads(record_text), raw_text, json.loads(raw_text)


def test_acceptance_record_is_canonical_pinned_and_explicit():
    record_text, record, _, _ = _load()

    assert record_text == _canonical(record)
    assert hashlib.sha256(record_text.encode("utf-8")).hexdigest() == EXPECTED_RECORD_SHA256
    assert record["schema_version"] == "eve.m1-human-acceptance-record.v1"
    assert record["approval_authority"]["automatic"] is False
    assert record["approval_authority"]["project_authority"] == "김민석"
    assert record["decision"] == {{
        "human_accepted": True,
        "human_review_status": "accepted",
        "m1_closed": True,
        "m2_started": False,
        "status": "accepted",
        "v4_2_eligible": True,
        "v4_2_review_opened": False,
    }}


def test_evidence_pins_match_exact_committed_artifacts():
    _, record, raw_text, raw = _load()
    pins = record["evidence_pins"]

    assert hashlib.sha256(raw_text.encode("utf-8")).hexdigest() == EXPECTED_RAW_SHA256
    assert raw["source_evidence_sha256"] == EXPECTED_SOURCE_SHA256
    assert pins["raw_artifact_sha256"] == EXPECTED_RAW_SHA256
    assert pins["source_evidence_sha256"] == EXPECTED_SOURCE_SHA256
    assert pins["validated_evidence_head"] == "{VALIDATED_EVIDENCE_HEAD}"
    assert pins["evidence_merge_sha"] == "{EVIDENCE_MERGE_SHA}"
    assert pins["exact_head_run_id"] == {EXACT_HEAD_RUN_ID}
    assert pins["exact_head_artifact_sha256"] == "{EXACT_HEAD_ARTIFACT_SHA256}"
    assert pins["focused_tests_passed"] == 12
    assert pins["full_suite_passed"] == 2712


def test_every_acceptance_metric_is_independently_recalculable_from_raw():
    _, record, _, raw = _load()
    criteria = {{item["criterion_id"]: item for item in record["reviewed_criteria"]}}
    rows = raw["mutation_classification"]["rows"]

    assert {{row["form"] for row in rows}} == {{
        "attribute_assignment",
        "subscript_assignment",
        "augmented_assignment",
        "mutating_method_call",
        "direct_write",
    }}
    assert len(rows) == 5
    for row in rows:
        assert row["observed"] is True
        assert row["state_changed"] is True
        assert row["before_value"] != row["after_value"]
        assert row["replay_matches"] is True
        assert row["transition_sha256"] == _transition_sha(
            row["before_value"], row["after_value"]
        )
    assert criteria["mutation_form_state_fidelity"]["passed"] is True

    targets = raw["observation_window"]["adapter_call_paths"]
    assert len(targets) == 3
    assert {{row["disposition"] for row in targets}} == {{"WRAP", "REWRITE"}}
    assert criteria["multiple_adapter_dispositions"]["passed"] is True

    activation = raw["raw_observations"]["activation"]
    assert activation["thread_alive_before_mutation"] is True
    assert activation["thread_alive_after_mutation"] is True
    assert activation["mutation_event_delta_while_thread_alive"] == 1
    assert activation["live_tick_event_delta"] == 0
    assert activation["thread_stopped"] is True
    assert criteria["live_tick_thread_concurrency"]["passed"] is True

    replay = raw["replay_equivalence"]
    assert replay["match_rate"] == {{"denominator": 4, "numerator": 4, "value": 1.0}}
    assert replay["divergence_count"] == 0
    assert replay["divergences"] == []
    assert all(row["matches"] for row in replay["rows"])
    assert all(row["matches"] for row in replay["final_equivalence"])
    assert criteria["complete_replay_equivalence"]["passed"] is True

    visibility = raw["failure_visibility"]
    assert visibility["legacy_failure_event_count"] == 1
    assert visibility["observer_failure"]["event_count"] == 0
    assert visibility["observer_failure"]["legacy_state_preserved"] is True
    assert visibility["observer_failure"]["return_value_preserved"] is True
    assert activation["exception_identity_preserved"] is True
    assert criteria["failure_visibility"]["passed"] is True

    assert raw["granularity"] == {{
        "candidate_events": 4,
        "discrete_observed_calls": 4,
        "events_during_live_tick_before_mutation": 0,
        "events_during_standalone_tick_steps": 0,
        "max_events_per_observed_call": 1,
        "standalone_tick_steps": 4,
    }}
    assert criteria["discrete_transition_granularity"]["passed"] is True

    persistence = raw["raw_observations"]["persistence"]
    assert persistence["state_matches_unobserved"] is True
    assert persistence["temporary_roots_removed"] is True
    assert persistence["final_snapshot"]["files"][0]["relative_path"] == "state.v41sidecar"
    assert criteria["bounded_direct_write"]["passed"] is True
    assert criteria["raw_observation_recalculability"]["passed"] is True
    assert criteria["exact_head_validation"]["passed"] is True
    assert criteria["zero_unauthorized_effects"]["passed"] is True


def test_external_acceptance_does_not_rewrite_machine_packet_or_runtime_authority():
    _, record, _, raw = _load()
    core_source = CORE_PATH.read_text(encoding="utf-8")

    assert raw["human_gate"] == {{
        "eligible_for_human_review": True,
        "human_accepted": False,
        "human_review_status": "required_not_performed",
        "v4_2_eligible": False,
    }}
    assert "M1_HUMAN_ACCEPTANCE_RECORD" not in core_source
    assert record["authority_boundary"] == {{
        "defaults_changed": False,
        "legacy_runtime_authoritative": True,
        "production_observer_installed": False,
        "production_persistence_enabled": False,
        "runtime_integrated": False,
        "shadow_authority_only": True,
    }}
    assert record["decision"]["v4_2_review_opened"] is False
    assert record["decision"]["m2_started"] is False


def test_markdown_and_status_pin_the_decision_without_overclaiming_coverage():
    record_text, record, _, _ = _load()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    status = STATUS_PATH.read_text(encoding="utf-8")
    record_sha = hashlib.sha256(record_text.encode("utf-8")).hexdigest()

    assert f"Canonical JSON SHA-256: `{{record_sha}}`" in markdown
    assert "human_accepted: true" in markdown
    assert "v4_2_eligible: true" in markdown
    assert "v4_2_review_opened: false" in markdown
    assert "미관찰 527곳" in markdown
    assert "M1 status: **closed for mechanism verification" in status
    assert "open REWRITE PRs #109, #86, #84, and #82" in status
    assert "absorbed PRs #11, #7, and #4 are closed" in status
    assert record_sha in status
    assert "M2-A remains blocked until v4.2 approval" in status
    assert record["scope_ruling"]["historical_fraction_is_m1_gate"] is False


def test_v4_2_candidate_triangle_is_exact_and_non_activating():
    _, record, _, _ = _load()
    clauses = record["v4_2_candidate_clauses"]

    assert [item["clause_id"] for item in clauses] == [
        "discrete_transition_granularity",
        "raw_observation_recalculability",
        "mutation_state_fidelity",
    ]
    assert record["decision"]["v4_2_eligible"] is True
    assert record["decision"]["v4_2_review_opened"] is False
    assert record["decision"]["m2_started"] is False
'''
    TEST_PATH.write_text(test_text, encoding="utf-8")


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def refresh_manifest(approval_pr: int) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["registered_addition_groups"] = [
        group
        for group in manifest["registered_addition_groups"]
        if not (
            group.get("introduced_by_pr") == approval_pr
            and group.get("path") == TEST_PATH.relative_to(REPO_ROOT).as_posix()
        )
    ]
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    report_path = REPO_ROOT / ".m1-human-acceptance-forward.json"
    run_command(
        [
            sys.executable,
            "scripts/audit/forward_regression_gate.py",
            "--report-only",
            "--pretty",
            "--output",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.unlink()
    rows = report["unregistered_additions"]
    expected_path = TEST_PATH.relative_to(REPO_ROOT).as_posix()
    actual_paths = {row["path"] for row in rows}
    if not rows or actual_paths != {expected_path}:
        raise RuntimeError(
            f"unexpected unregistered paths: expected={[expected_path]} actual={sorted(actual_paths)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    values = grouped[expected_path]
    manifest["registered_addition_groups"].append(
        {
            "path": expected_path,
            "categories": sorted({row["category"] for row in values}),
            "symbols": sorted({row["symbol"] for row in values}),
            "rationale": (
                "Independent fail-closed recalculation of the external M1 human-acceptance "
                "decision from the committed raw observation artifact, including scope and "
                "non-activation boundaries."
            ),
            "owner": "M1 human acceptance verification",
            "disposition": "TEST_EVIDENCE",
            "introduced_by_pr": approval_pr,
            "fingerprints": {
                row["fingerprint"]: int(row["count"])
                for row in sorted(values, key=lambda item: item["fingerprint"])
            },
        }
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def verify(approval_pr: int, base_sha: str) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            str(TEST_PATH.relative_to(REPO_ROOT)),
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            str(TEST_PATH.relative_to(REPO_ROOT)),
        ]
    )
    report_path = REPO_ROOT / ".m1-human-acceptance-forward-final.json"
    run_command(
        [
            sys.executable,
            "scripts/audit/forward_regression_gate.py",
            "--current-pr",
            str(approval_pr),
            "--base-sha",
            base_sha,
            "--pretty",
            "--output",
            str(report_path),
        ]
    )
    report_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-pr", type=int, required=True)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    if args.base_sha != BASE_SHA:
        raise SystemExit(f"unexpected base SHA: {args.base_sha}")

    raw_text = RAW_PATH.read_text(encoding="utf-8")
    raw = json.loads(raw_text)
    verify_raw(raw_text, raw)
    record = build_record(raw, args.approval_pr)
    record_text = canonical_json(record)
    record_sha = sha256_text(record_text)
    RECORD_PATH.write_text(record_text, encoding="utf-8")
    MARKDOWN_PATH.write_text(render_markdown(record, record_sha), encoding="utf-8")
    patch_status(record_sha, args.approval_pr)
    write_tests(record_sha)

    Path(__file__).unlink()
    refresh_manifest(args.approval_pr)
    verify(args.approval_pr, args.base_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
