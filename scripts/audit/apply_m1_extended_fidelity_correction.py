#!/usr/bin/env python3
"""Apply the one-shot PR #153 mutation-state fidelity correction.

This file is temporary bootstrap tooling. The workflow removes it before the
corrected evidence commit is promoted back to PR #153.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_PATH = REPO_ROOT / "scripts/audit/m1_extended_controlled_observation_campaign.py"
TESTS_PATH = REPO_ROOT / "tests/test_v4_m1_extended_controlled_observation_campaign.py"
REGISTRATION_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_REGISTRATION.md"
RAW_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json"
EVIDENCE_PATH = REPO_ROOT / "docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_EVIDENCE.md"
MANIFEST_PATH = REPO_ROOT / "docs/audit/FORWARD_ADDITIONS_MANIFEST.json"
TARGET_PR = 153
BASE_SHA = "847621bcd61634958ce505108ade491c50ced0d4"
REFRESHED_PATHS = {
    "scripts/audit/m1_extended_controlled_observation_campaign.py",
    "tests/test_v4_m1_extended_controlled_observation_campaign.py",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{label}: start marker missing")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: end marker missing")
    return text[:start] + replacement.rstrip() + "\n\n\n" + text[end:]


def patch_campaign() -> None:
    text = CAMPAIGN_PATH.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''import adapters.persistence_adapter as persistence_module
from adapters.activation_adapter import ActivationAdapter
from adapters.live_loop import LiveLoop
from adapters.persistence_adapter import PersistenceAdapter
from core.event_kernel import InMemoryEventKernel, SHADOW_AUTHORITY, canonical_json_object
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    SUCCESS_EVENT_TYPE,
    LegacyFunnelShadowObserver,
    ShadowTarget,
)
from core.shadow_projection import (
    ActivationLearnPairShadowState,
    ShadowProjectionError,
    reduce_activation_learn_pair,
)
''',
        '''import adapters.live_loop as live_loop_module
import adapters.persistence_adapter as persistence_module
from adapters.activation_adapter import ActivationAdapter
from adapters.live_loop import LiveLoop
from adapters.persistence_adapter import PersistenceAdapter
from core.event_kernel import InMemoryEventKernel, SHADOW_AUTHORITY, canonical_json_object
from core.shadow_observer import (
    ACTIVATION_LEARN_PAIR_TARGET,
    FAILURE_EVENT_TYPE,
    SUCCESS_EVENT_TYPE,
    LegacyFunnelShadowObserver,
    ShadowTarget,
)
''',
        "imports",
    )
    text = replace_once(
        text,
        '''    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": [list(item) for item in self.calls],
            "learned": [list(item) for item in self.learned],
        }
''',
        '''    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": [list(item) for item in self.calls],
            "learned": [list(item) for item in self.learned],
            "neighbors": [
                [category, sorted(values)]
                for category, values in sorted(self.inner.neighbors.items())
            ],
            "weights": [
                [left, right, float(weight)]
                for (left, right), weight in sorted(self.inner.weights.items())
            ],
        }
''',
        "activation snapshot",
    )
    text = replace_once(
        text,
        '''def _live_snapshot(loop: LiveLoop, emissions: list[str]) -> dict[str, Any]:
    return {
        "emissions": list(emissions),
        "processed_input_count": int(loop.processed_input_count),
        "queue_size": int(loop._user_input_queue.qsize()),
    }
''',
        '''def _live_snapshot(loop: LiveLoop, emissions: list[str]) -> dict[str, Any]:
    return {
        "emissions": list(emissions),
        "last_emit_time": float(loop._last_emit_time),
        "processed_input_count": int(loop.processed_input_count),
        "queue_size": int(loop._user_input_queue.qsize()),
    }
''',
        "live snapshot",
    )
    text = replace_once(
        text,
        '''    observed_result = observer.observe_call(
        LIVE_LOOP_DRAIN_TARGET.target_id,
        event_id=event_id,
        correlation_id=CORRELATION_ID,
        causation_id=causation_id,
        legacy_callable=observed._drain_user_inputs,
        before_snapshot=before,
        after_snapshot=after,
    )
    baseline_result = baseline._drain_user_inputs()
''',
        '''    original_time = live_loop_module.time.time
    live_loop_module.time.time = lambda: 4242.0
    try:
        observed_result = observer.observe_call(
            LIVE_LOOP_DRAIN_TARGET.target_id,
            event_id=event_id,
            correlation_id=CORRELATION_ID,
            causation_id=causation_id,
            legacy_callable=observed._drain_user_inputs,
            before_snapshot=before,
            after_snapshot=after,
        )
        baseline_result = baseline._drain_user_inputs()
    finally:
        live_loop_module.time.time = original_time
''',
        "live controlled time",
    )
    text = replace_once(
        text,
        '''    observed_result = observer.observe_call(
        LIVE_LOOP_DRAIN_TARGET.target_id,
        event_id="m1-extended:observer-failure:001",
        correlation_id=CORRELATION_ID,
        legacy_callable=observed._drain_user_inputs,
        before_snapshot=broken_before,
        after_snapshot=lambda: _live_snapshot(observed, observed_emissions),
    )
    baseline_result = baseline._drain_user_inputs()
''',
        '''    original_time = live_loop_module.time.time
    live_loop_module.time.time = lambda: 4343.0
    try:
        observed_result = observer.observe_call(
            LIVE_LOOP_DRAIN_TARGET.target_id,
            event_id="m1-extended:observer-failure:001",
            correlation_id=CORRELATION_ID,
            legacy_callable=observed._drain_user_inputs,
            before_snapshot=broken_before,
            after_snapshot=lambda: _live_snapshot(observed, observed_emissions),
        )
        baseline_result = baseline._drain_user_inputs()
    finally:
        live_loop_module.time.time = original_time
''',
        "observer failure controlled time",
    )
    text = replace_once(
        text,
        '''    if observed_outcome != baseline_outcome:
        raise ExtendedCampaignError("activation failure outcome diverged from baseline")

    events_before_ticks = len(kernel.events())
''',
        '''    if observed_outcome != baseline_outcome:
        raise ExtendedCampaignError("activation failure outcome diverged from baseline")

    event_boundary_snapshot = observed_ledger.snapshot()
    baseline_event_boundary_snapshot = baseline_ledger.snapshot()
    if event_boundary_snapshot != baseline_event_boundary_snapshot:
        raise ExtendedCampaignError(
            "activation event-boundary state diverged from unobserved baseline"
        )

    events_before_ticks = len(kernel.events())
''',
        "event boundary snapshot",
    )
    text = replace_once(
        text,
        '''        "event_ids": [event_id_success, event_id_failure],
        "exception_identity_preserved": propagated_identity,
        "final_snapshot": final,
''',
        '''        "event_boundary_snapshot": event_boundary_snapshot,
        "event_ids": [event_id_success, event_id_failure],
        "exception_identity_preserved": propagated_identity,
        "final_snapshot": final,
''',
        "event boundary result",
    )

    replay_block = '''def _generic_replay_event(
    state: Mapping[str, Any],
    event: Any,
    target: ShadowTarget,
    expected_sequence: int,
    expected_causation_id: str | None,
) -> tuple[dict[str, Any], list[str]]:
    mismatches: list[str] = []
    payload = event.payload
    expected_target = {
        "callable": target.callable_name,
        "disposition": target.module_disposition,
        "module_path": target.module_path,
        "target_id": target.target_id,
    }
    expected_context = {
        "arguments_captured": False,
        "legacy_result_captured": False,
        "observation_phase": "after_the_fact",
        "source_evidence_range": target.evidence_range,
    }
    if event.authority != SHADOW_AUTHORITY:
        mismatches.append("authority")
    if event.producer != "core.shadow_observer":
        mismatches.append("producer")
    if event.producer_version != "1.0.0":
        mismatches.append("producer_version")
    if event.schema_version != "eve.event-envelope.v1":
        mismatches.append("schema_version")
    if event.correlation_id != CORRELATION_ID:
        mismatches.append("correlation_id")
    if event.causation_id != expected_causation_id:
        mismatches.append("causation_id")
    if event.causal_context != expected_context:
        mismatches.append("causal_context")
    if event.stream_id != target.stream_id:
        mismatches.append("stream_id")
    if event.sequence != expected_sequence:
        mismatches.append("sequence")
    if payload.get("target") != expected_target:
        mismatches.append("target_metadata")
    outcome = payload.get("legacy_outcome")
    if not isinstance(outcome, Mapping):
        mismatches.append("legacy_outcome")
    else:
        expected_type = (
            SUCCESS_EVENT_TYPE if outcome.get("succeeded") is True else FAILURE_EVENT_TYPE
        )
        if event.event_type != expected_type:
            mismatches.append("event_type")
        if outcome.get("succeeded") is True and outcome.get("error_type") is not None:
            mismatches.append("success_error_type")
        if outcome.get("succeeded") is False and not isinstance(
            outcome.get("error_type"), str
        ):
            mismatches.append("failure_error_type")
    if payload.get("before") != dict(state):
        mismatches.append("before_snapshot")
    after = payload.get("after")
    if not isinstance(after, Mapping):
        mismatches.append("after_snapshot")
        return dict(state), mismatches
    return dict(after), mismatches


def _replay_all(
    events: tuple[Any, ...],
    initial_by_target: Mapping[str, Mapping[str, Any]],
    final_by_target: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    target_by_id = {target.target_id: target for target in EXTENDED_TARGETS}
    target_by_stream = {target.stream_id: target for target in EXTENDED_TARGETS}
    states = {
        target_id: dict(snapshot)
        for target_id, snapshot in initial_by_target.items()
    }
    sequences = {target_id: 0 for target_id in states}
    rows: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    previous_event_id: str | None = None
    for event in events:
        target = target_by_stream.get(event.stream_id)
        mismatch_codes: list[str] = []
        if target is None:
            mismatch_codes.append("unknown_stream")
            target_id = "unknown"
        else:
            target_id = target.target_id
            sequences[target_id] += 1
            states[target_id], mismatch_codes = _generic_replay_event(
                states[target_id],
                event,
                target,
                sequences[target_id],
                previous_event_id,
            )
        row = {
            "event_digest": event.digest,
            "event_id": event.event_id,
            "matches": not mismatch_codes,
            "mismatch_codes": mismatch_codes,
            "sequence": event.sequence,
            "stream_id": event.stream_id,
            "target_id": target_id,
        }
        rows.append(row)
        if mismatch_codes:
            divergences.append(row)
        previous_event_id = event.event_id

    final_rows = []
    for target_id in sorted(target_by_id):
        actual = dict(states[target_id])
        expected = dict(final_by_target[target_id])
        final_rows.append(
            {
                "actual_digest": _sha(actual, f"replay-final:{target_id}:actual"),
                "expected_digest": _sha(
                    expected,
                    f"replay-final:{target_id}:expected",
                ),
                "matches": actual == expected,
                "target_id": target_id,
            }
        )
    matching = sum(1 for row in rows if row["matches"])
    return {
        "compared_events": len(rows),
        "divergence_count": len(divergences),
        "divergences": divergences,
        "final_equivalence": final_rows,
        "match_rate": {
            "denominator": len(rows),
            "numerator": matching,
            "value": matching / len(rows),
        },
        "rows": rows,
    }
'''
    text = replace_between(
        text,
        "def _generic_replay_event(",
        "def _mutation_form_rows(",
        replay_block,
        "replay functions",
    )

    mutation_block = '''def _mutation_form_rows(
    replay: Mapping[str, Any],
    events: tuple[Any, ...],
) -> list[dict[str, Any]]:
    replay_matches = {
        row["event_id"]: bool(row["matches"])
        for row in replay["rows"]
    }
    event_by_id = {event.event_id: event for event in events}
    rows = [
        {
            "form": "attribute_assignment",
            "path": "adapters/live_loop.py",
            "line_range": "101-105",
            "call_path": "LiveLoop._drain_user_inputs -> LiveLoop._handle_user_input",
            "target_id": LIVE_LOOP_DRAIN_TARGET.target_id,
            "event_ids": ["m1-extended:event:live-drain:001"],
            "state_field": "last_emit_time",
        },
        {
            "form": "subscript_assignment",
            "path": "legacy/eve_modules/spreading_activation.py",
            "line_range": "239-243",
            "call_path": "ActivationAdapter.learn_pair -> SpreadingActivation.learn_pair",
            "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            "event_ids": ["m1-extended:event:activation:001"],
            "state_field": "weights",
        },
        {
            "form": "augmented_assignment",
            "path": "adapters/live_loop.py",
            "line_range": "68-77",
            "call_path": "LiveLoop._drain_user_inputs",
            "target_id": LIVE_LOOP_DRAIN_TARGET.target_id,
            "event_ids": ["m1-extended:event:live-drain:001"],
            "state_field": "processed_input_count",
        },
        {
            "form": "mutating_method_call",
            "path": "legacy/eve_modules/spreading_activation.py",
            "line_range": "241-243",
            "call_path": "ActivationAdapter.learn_pair -> SpreadingActivation.learn_pair",
            "target_id": ACTIVATION_LEARN_PAIR_TARGET.target_id,
            "event_ids": ["m1-extended:event:activation:001"],
            "state_field": "neighbors",
        },
        {
            "form": "direct_write",
            "path": "adapters/persistence_adapter.py",
            "line_range": "65-74",
            "call_path": "PersistenceAdapter.save",
            "target_id": PERSISTENCE_SAVE_TARGET.target_id,
            "event_ids": ["m1-extended:event:persistence-save:001"],
            "state_field": "files",
        },
    ]
    for row in rows:
        event_id = row["event_ids"][0]
        event = event_by_id[event_id]
        field = row["state_field"]
        before_value = event.payload["before"][field]
        after_value = event.payload["after"][field]
        state_changed = before_value != after_value
        row["before_value"] = before_value
        row["after_value"] = after_value
        row["state_changed"] = state_changed
        row["transition_sha256"] = _sha(
            {"before": before_value, "after": after_value},
            f"mutation-transition:{row['form']}",
        )
        row["observed"] = state_changed
        row["replay_matches"] = all(
            replay_matches.get(candidate_id, False)
            for candidate_id in row["event_ids"]
        )
    return rows
'''
    text = replace_between(
        text,
        "def _mutation_form_rows(",
        "def run_extended_controlled_observation_campaign(",
        mutation_block,
        "mutation rows",
    )
    text = replace_once(
        text,
        '    mutation_forms = _mutation_form_rows(replay)\n',
        '    mutation_forms = _mutation_form_rows(replay, events)\n',
        "mutation row call",
    )
    text = replace_once(
        text,
        '        ACTIVATION_LEARN_PAIR_TARGET.target_id: activation["final_snapshot"],\n',
        '        ACTIVATION_LEARN_PAIR_TARGET.target_id: activation["event_boundary_snapshot"],\n',
        "replay event-boundary final",
    )
    text = replace_once(
        text,
        '''    form_lines = "\n".join(
        f"| `{row['form']}` | `{row['path']}:{row['line_range']}` | "
        f"`{row['target_id']}` | `{str(row['replay_matches']).lower()}` |"
        for row in forms
    )
''',
        '''    form_lines = "\n".join(
        f"| `{row['form']}` | `{row['path']}:{row['line_range']}` | "
        f"`{row['target_id']}` | `{row['state_field']}` | "
        f"`{str(row['state_changed']).lower()}` | "
        f"`{str(row['replay_matches']).lower()}` |"
        for row in forms
    )
''',
        "render form lines",
    )
    text = replace_once(
        text,
        '''| M0-A form | Executed source | Observed target | Replay match |
|---|---|---|---|
{form_lines}

All five required forms were executed at least once and tied to raw before/after
events. The rows are mechanism evidence, not a claim that all historical mutation
sites are covered or safe.
''',
        '''| M0-A form | Executed source | Observed target | State field | Changed | Replay match |
|---|---|---|---|---|---|
{form_lines}

All five required forms were executed at least once. Each row identifies the
mutated state field and records its exact raw before/after values plus a transition
digest. The rows are mechanism evidence, not a claim that all historical mutation
sites are covered or safe.
''',
        "render mutation section",
    )
    CAMPAIGN_PATH.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS_PATH.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    assert all(row["observed"] is True for row in rows)
    assert all(row["replay_matches"] is True for row in rows)
    assert all(row["event_ids"] for row in rows)
''',
        '''    assert all(row["observed"] is True for row in rows)
    assert all(row["state_changed"] is True for row in rows)
    assert all(row["before_value"] != row["after_value"] for row in rows)
    assert all(len(row["transition_sha256"]) == 64 for row in rows)
    assert all(row["replay_matches"] is True for row in rows)
    assert all(row["event_ids"] for row in rows)
    assert {row["form"]: row["state_field"] for row in rows} == {
        "attribute_assignment": "last_emit_time",
        "subscript_assignment": "weights",
        "augmented_assignment": "processed_input_count",
        "mutating_method_call": "neighbors",
        "direct_write": "files",
    }
''',
        "mutation assertions",
    )
    inserted = '''def test_each_mutation_form_snapshot_contains_the_changed_state(evidence):
    events = {event["event_id"]: event for event in evidence["events"]}
    live = events["m1-extended:event:live-drain:001"]["payload"]
    activation = events["m1-extended:event:activation:001"]["payload"]
    persistence = events["m1-extended:event:persistence-save:001"]["payload"]

    assert live["before"]["last_emit_time"] == 0.0
    assert live["after"]["last_emit_time"] == 4242.0
    assert live["before"]["processed_input_count"] == 0
    assert live["after"]["processed_input_count"] == 1
    assert activation["before"]["weights"] == []
    assert activation["after"]["weights"] == [["alpha", "beta", 0.25]]
    assert activation["before"]["neighbors"] == []
    assert activation["after"]["neighbors"] == [
        ["alpha", ["beta"]],
        ["beta", ["alpha"]],
    ]
    assert persistence["before"]["files"] == []
    assert persistence["after"]["files"][0]["relative_path"] == "state.v41sidecar"


'''
    text = replace_once(
        text,
        "def test_source_rows_correspond_to_real_ast_mutation_shapes(evidence):\n",
        inserted + "def test_source_rows_correspond_to_real_ast_mutation_shapes(evidence):\n",
        "state snapshot test insertion",
    )
    TESTS_PATH.write_text(text, encoding="utf-8")


def patch_registration() -> None:
    text = REGISTRATION_PATH.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''The Markdown report pins the raw JSON SHA-256. Focused tests regenerate the
campaign, require byte-identical canonical JSON, recalculate event/replay/failure
and granularity totals, and require the committed report to equal the renderer
output for that raw hash.
''',
        '''The Markdown report pins the raw JSON SHA-256. Every mutation-form row names
its changed state field and stores exact before/after values plus a transition
digest. Focused tests regenerate the campaign, require byte-identical canonical
JSON, recalculate event/replay/failure and granularity totals, and require the
committed report to equal the renderer output for that raw hash.
''',
        "registration fidelity contract",
    )
    REGISTRATION_PATH.write_text(text, encoding="utf-8")


def run_command(args: list[str]) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def regenerate_artifacts_and_pins() -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "scripts.audit.m1_extended_controlled_observation_campaign",
            "--raw-output",
            str(RAW_PATH.relative_to(REPO_ROOT)),
            "--evidence-output",
            str(EVIDENCE_PATH.relative_to(REPO_ROOT)),
        ]
    )
    raw_text = RAW_PATH.read_text(encoding="utf-8")
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    source_sha = json.loads(raw_text)["source_evidence_sha256"]
    registration = REGISTRATION_PATH.read_text(encoding="utf-8")
    registration, raw_count = re.subn(
        r"raw artifact SHA-256: [0-9a-f]{64}",
        f"raw artifact SHA-256: {raw_sha}",
        registration,
    )
    registration, source_count = re.subn(
        r"source evidence SHA-256: [0-9a-f]{64}",
        f"source evidence SHA-256: {source_sha}",
        registration,
    )
    if raw_count != 1 or source_count != 1:
        raise RuntimeError("artifact pin replacement failed")
    REGISTRATION_PATH.write_text(registration, encoding="utf-8")


def refresh_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["registered_addition_groups"] = [
        group
        for group in manifest["registered_addition_groups"]
        if not (
            group.get("introduced_by_pr") == TARGET_PR
            and group.get("path") in REFRESHED_PATHS
        )
    ]
    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = REPO_ROOT / ".m1-forward-unregistered.json"
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
    actual_paths = {row["path"] for row in rows}
    if not rows or actual_paths != REFRESHED_PATHS:
        raise RuntimeError(
            "unexpected unregistered paths: "
            f"expected={sorted(REFRESHED_PATHS)} actual={sorted(actual_paths)}"
        )
    metadata = {
        "scripts/audit/m1_extended_controlled_observation_campaign.py": {
            "rationale": (
                "Expanded disconnected M1 mechanism observation with mutation-state "
                "fidelity, multi-adapter coverage, live concurrency, strict replay, "
                "failure visibility, and canonical raw evidence."
            ),
            "owner": "M1 extended controlled observation evidence",
            "disposition": "M1_EXTENDED_CONTROLLED_EVIDENCE",
        },
        "tests/test_v4_m1_extended_controlled_observation_campaign.py": {
            "rationale": (
                "Focused fail-closed verification and independent metric and mutation-state "
                "recomputation for the extended controlled M1 artifact."
            ),
            "owner": "M1 extended controlled observation verification",
            "disposition": "TEST_EVIDENCE",
        },
    }
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    for path in sorted(grouped):
        values = grouped[path]
        manifest["registered_addition_groups"].append(
            {
                "path": path,
                "categories": sorted({row["category"] for row in values}),
                "symbols": sorted({row["symbol"] for row in values}),
                **metadata[path],
                "introduced_by_pr": TARGET_PR,
                "fingerprints": {
                    row["fingerprint"]: int(row["count"])
                    for row in sorted(values, key=lambda item: item["fingerprint"])
                },
            }
        )
    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def verify(base_sha: str) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "scripts/audit/m1_extended_controlled_observation_campaign.py",
            "tests/test_v4_m1_extended_controlled_observation_campaign.py",
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "tests/test_v4_m1_extended_controlled_observation_campaign.py",
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/audit/forward_regression_gate.py",
            "--current-pr",
            str(TARGET_PR),
            "--base-sha",
            base_sha,
            "--pretty",
            "--output",
            ".m1-forward-corrected.json",
        ]
    )
    (REPO_ROOT / ".m1-forward-corrected.json").unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", default=BASE_SHA)
    args = parser.parse_args()
    if args.base_sha != BASE_SHA:
        raise SystemExit("unexpected base SHA")
    patch_campaign()
    patch_tests()
    patch_registration()
    regenerate_artifacts_and_pins()
    refresh_manifest()
    verify(args.base_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
