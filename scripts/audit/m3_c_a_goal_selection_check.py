#!/usr/bin/env python3
"""Static standard-library checker for the documentation-only M3-C-A design."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "eve.m3-c-a.goal-selection-check.v1"
BASELINE_SHA = "a9f70ef78b06744eba01a0b35c60371b10eaf672"
AUTHORIZATION_DIGEST = "3844e4d0a836924eb881048d45d98d89d5041f87d15a836686119a2d8487efbf"
ACTIVE_STORE_ROLE = "authoritative_persistence_substrate_for_v4_native_subsystems"
DESIGN_PATH = Path("docs/audit/M3_C_A_GOAL_GENERATION_SELECTION_DESIGN.md")
M3A_PATH = Path("docs/audit/M3_A_DRIVE_DYNAMICS_DESIGN.md")
AFFECT_PATH = Path("docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md")
CUTOVER_PATH = Path("core/m2_e_cutover_activation.py")
ALLOWED_DRIVES = (
    "energy", "safety", "affiliation", "curiosity",
    "agency", "coherence", "competence", "expression",
)
COUNTERFACTUAL_COLUMN_DRIVES = (
    "energy", "safety", "curiosity", "affiliation",
    "agency", "coherence", "competence", "expression",
)
EXPECTED_POLICY = {
    "proposal enter threshold": 0.20,
    "proposal exit threshold": 0.10,
    "selection minimum score": 0.30,
    "initial winner margin": 0.08,
    "switch margin": 0.12,
    "selection cooldown seconds": 30.0,
}
EXPECTED_LIFECYCLE = {
    ("absent", "proposed"),
    ("proposed", "validated"),
    ("proposed", "rejected"),
    ("proposed", "expired"),
    ("validated", "eligible"),
    ("validated", "rejected"),
    ("eligible", "selected"),
    ("eligible", "withdrawn"),
    ("selected", "superseded"),
    ("selected", "expired"),
    ("rejected", "absent"),
    ("expired", "absent"),
    ("withdrawn", "absent"),
    ("superseded", "absent"),
}
TABLES = {
    "drive": ("<!-- BEGIN M3C DRIVE INTEGRATION TABLE -->", "<!-- END M3C DRIVE INTEGRATION TABLE -->", 6),
    "candidate": ("<!-- BEGIN M3C CANDIDATE FIELD TABLE -->", "<!-- END M3C CANDIDATE FIELD TABLE -->", 5),
    "policy": ("<!-- BEGIN M3C POLICY TABLE -->", "<!-- END M3C POLICY TABLE -->", 5),
    "lifecycle": ("<!-- BEGIN M3C LIFECYCLE TABLE -->", "<!-- END M3C LIFECYCLE TABLE -->", 5),
    "cf_candidate": ("<!-- BEGIN M3C COUNTERFACTUAL CANDIDATE TABLE -->", "<!-- END M3C COUNTERFACTUAL CANDIDATE TABLE -->", 7),
    "cf_drive": ("<!-- BEGIN M3C COUNTERFACTUAL DRIVE TABLE -->", "<!-- END M3C COUNTERFACTUAL DRIVE TABLE -->", 11),
    "affect": ("<!-- BEGIN AFFECT AXIS MAPPING TABLE -->", "<!-- END AFFECT AXIS MAPPING TABLE -->", 12),
}
REQUIRED_TEXT = (
    "## Authority and scope",
    "## Deterministic scoring",
    "## A9 no-duplicate proof",
    "## Authoritative persistence design",
    "## Counterfactual verification — does affect-derived state change actual choice?",
    "## M3-E boundary",
    BASELINE_SHA,
    AUTHORIZATION_DIGEST,
    ACTIVE_STORE_ROLE,
    "legacy goal authority:       unchanged until its own later migration gate",
    "M3-E affect cutover:         false",
    "Continuous drive values and continuously recomputed scores are **not events**.",
    "Selection is `argmax(score_g)` with lexical `candidate_id` as the deterministic tie-break.",
    "No runtime implementation.",
)


def _clean(value: str) -> str:
    return value.strip().strip("`").strip()


def _number(value: str) -> float:
    result = float(_clean(value))
    if not math.isfinite(result):
        raise AssertionError("non-finite numeric design value")
    return result


def _table(text: str, name: str) -> list[list[str]]:
    begin, end, width = TABLES[name]
    if begin not in text or end not in text:
        raise AssertionError(f"missing table markers: {name}")
    body = text.split(begin, 1)[1].split(end, 1)[0]
    rows: list[list[str]] = []
    header = True
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != width:
            raise AssertionError(f"{name}: expected {width} columns, got {len(cells)}")
        if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
            continue
        if header:
            header = False
            continue
        rows.append(cells)
    if not rows:
        raise AssertionError(f"empty table: {name}")
    return rows


def _ruling(cells: list[str], ruling_index: int, question_index: int | None = None) -> int:
    ruling = _clean(cells[ruling_index])
    if ruling not in {"RESOLVED", "UNRESOLVED"}:
        raise AssertionError(f"invalid ruling: {ruling}")
    if ruling == "UNRESOLVED" and question_index is not None:
        question = _clean(cells[question_index])
        if question in {"", "—", "-"}:
            raise AssertionError("unresolved row without open question")
    return int(ruling == "UNRESOLVED")


def _candidate_id(candidate: str) -> str:
    material = f"eve.m3-c-a.goal-candidate.v1|{candidate}|0|counterfactual-evidence|eve.m3-c-a.goal-score.v1"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _drive_term(candidate: dict[str, Any], z: dict[str, float]) -> float:
    q = candidate["confidence"]
    numerator = sum(q * candidate["alignments"][drive] * z[drive] for drive in ALLOWED_DRIVES)
    denominator = sum(q * abs(candidate["alignments"][drive]) for drive in ALLOWED_DRIVES)
    return numerator / max(1.0, denominator)


def _score(candidate: dict[str, Any], z: dict[str, float]) -> float:
    # Counterfactual non-drive attributes are fixed by the design document.
    raw = 0.30 * 0.30 + 0.30 * _drive_term(candidate, z)
    return max(-1.0, min(1.0, raw))


def _counterfactual(design: str) -> tuple[list[dict[str, Any]], int]:
    unresolved = 0
    candidates = []
    for cells in _table(design, "cf_candidate"):
        unresolved += _ruling(cells, 6)
        other = _number(cells[4])
        candidates.append({
            "candidate": _clean(cells[0]),
            "alignments": {
                "energy": _number(cells[1]),
                "safety": _number(cells[2]),
                "curiosity": _number(cells[3]),
                "affiliation": other,
                "agency": other,
                "coherence": other,
                "competence": other,
                "expression": other,
            },
            "confidence": _number(cells[5]),
        })

    conditions = []
    for cells in _table(design, "cf_drive"):
        unresolved += _ruling(cells, 10)
        z = {
            drive: _number(cells[index + 1])
            for index, drive in enumerate(COUNTERFACTUAL_COLUMN_DRIVES)
        }
        conditions.append({
            "condition": _clean(cells[0]),
            "z": z,
            "expected_winner": _clean(cells[9]),
        })

    results = []
    for condition in conditions:
        scored = sorted(
            (
                _score(candidate, condition["z"]),
                _candidate_id(candidate["candidate"]),
                candidate["candidate"],
            )
            for candidate in candidates
        )
        scored.reverse()
        # Reverse sorting would also reverse the tie key, so explicitly re-sort by
        # descending score and ascending candidate-id as the design requires.
        scored.sort(key=lambda row: (-row[0], row[1]))
        winner_score, _, winner = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        results.append({
            "condition": condition["condition"],
            "winner": winner,
            "expected_winner": condition["expected_winner"],
            "winner_score": winner_score,
            "runner_up_score": runner_up,
            "margin": winner_score - runner_up,
        })
    return results, unresolved


def validate() -> dict[str, Any]:
    design = DESIGN_PATH.read_text(encoding="utf-8")
    m3a = M3A_PATH.read_text(encoding="utf-8")
    affect = AFFECT_PATH.read_text(encoding="utf-8")
    cutover = CUTOVER_PATH.read_text(encoding="utf-8")
    errors: list[str] = []
    unresolved = 0

    for token in REQUIRED_TEXT:
        if token not in design:
            errors.append(f"missing design token: {token}")

    drive_rows = _table(design, "drive")
    design_drives = []
    for cells in drive_rows:
        unresolved += _ruling(cells, 4, 5)
        design_drives.append(_clean(cells[0]))
        if _clean(cells[1]) != "eve.m3-a.drive-dynamics.v1":
            errors.append(f"drive source mismatch: {_clean(cells[0])}")
        if _clean(cells[3]) != "none":
            errors.append(f"drive grants direct action authority: {_clean(cells[0])}")
    if tuple(design_drives) != ALLOWED_DRIVES:
        errors.append("drive set/order mismatch")
    if "integration eligibility exists only after an explicit persistence cutover" not in m3a:
        errors.append("M3-A cutover eligibility precedent missing")

    candidate_rows = _table(design, "candidate")
    candidate_fields = set()
    for cells in candidate_rows:
        unresolved += _ruling(cells, 3, 4)
        candidate_fields.add(_clean(cells[0]))
    expected_fields = {
        "semantic_goal_id", "decision_epoch", "evidence_digest", "base_value",
        "expected_value", "urgency", "continuity", "cost", "risk",
        "drive_alignment[8]", "drive_confidence[8]",
    }
    if candidate_fields != expected_fields:
        errors.append("candidate field catalog mismatch")

    policy_rows = _table(design, "policy")
    policy = {}
    for cells in policy_rows:
        unresolved += _ruling(cells, 3, 4)
        policy[_clean(cells[0])] = _number(cells[1])
    if set(policy) != set(EXPECTED_POLICY):
        errors.append("policy item set mismatch")
    for item, expected in EXPECTED_POLICY.items():
        if item in policy and not math.isclose(policy[item], expected, abs_tol=1e-12):
            errors.append(f"policy mismatch: {item}")
    if policy.get("proposal exit threshold", 99) >= policy.get("proposal enter threshold", -99):
        errors.append("proposal hysteresis invalid")
    if policy.get("switch margin", 0) <= policy.get("initial winner margin", 0):
        errors.append("switch margin must exceed initial winner margin")
    if policy.get("selection cooldown seconds", 0) <= 0:
        errors.append("selection cooldown must be positive")

    lifecycle_rows = _table(design, "lifecycle")
    lifecycle = {(_clean(cells[0]), _clean(cells[1])) for cells in lifecycle_rows}
    if lifecycle != EXPECTED_LIFECYCLE or len(lifecycle_rows) != len(EXPECTED_LIFECYCLE):
        errors.append("lifecycle edge mismatch")
    if any(_clean(cells[3]) != "yes" for cells in lifecycle_rows):
        errors.append("lifecycle contains non-event transition row")

    affect_rows = _table(affect, "affect")
    affect_status = Counter(_clean(cells[2]) for cells in affect_rows)
    if len(affect_rows) != 63 or len({_clean(cells[0]) for cells in affect_rows}) != 63:
        errors.append("affect axis count/uniqueness mismatch")
    if affect_status != Counter({"MAPPED": 59, "PROPOSED-DROP": 4}):
        errors.append("affect reviewer ruling counts mismatch")
    for cells in affect_rows:
        status = _clean(cells[2])
        targets = [] if _clean(cells[3]) in {"", "—", "-"} else [part.strip() for part in _clean(cells[3]).split(";") if part.strip()]
        if status == "MAPPED" and any(target not in ALLOWED_DRIVES for target in targets):
            errors.append(f"affect target outside M3-A drive set: {_clean(cells[0])}")
        if status == "PROPOSED-DROP" and targets:
            errors.append(f"dropped axis unexpectedly targets drive: {_clean(cells[0])}")

    for token in (
        AUTHORIZATION_DIGEST,
        ACTIVE_STORE_ROLE,
        "m3_authority_open=True",
        "legacy_domain_authority_transfer_authorized: bool = False",
        "m3_e_affect_cutover_authorized: bool = False",
        "legacy_persistence_path_changed: bool = False",
    ):
        if token not in cutover:
            errors.append(f"missing cutover boundary token: {token}")

    counterfactual, cf_unresolved = _counterfactual(design)
    unresolved += cf_unresolved
    if {row["condition"] for row in counterfactual} != {
        "strain_mapped_affect", "recovered_exploration"
    }:
        errors.append("counterfactual condition set mismatch")
    for row in counterfactual:
        if row["winner"] != row["expected_winner"]:
            errors.append(f"counterfactual winner mismatch: {row['condition']}")
        if row["winner_score"] + 1e-12 < EXPECTED_POLICY["selection minimum score"]:
            errors.append(f"counterfactual winner below selection minimum: {row['condition']}")
        if row["margin"] + 1e-12 < EXPECTED_POLICY["initial winner margin"]:
            errors.append(f"counterfactual winner margin too small: {row['condition']}")
    if len({row["winner"] for row in counterfactual}) != 2:
        errors.append("counterfactual did not flip selected goal")

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_sha": BASELINE_SHA,
        "authorization_digest": AUTHORIZATION_DIGEST,
        "active_store_role": ACTIVE_STORE_ROLE,
        "drive_count": len(drive_rows),
        "candidate_field_count": len(candidate_rows),
        "policy_count": len(policy_rows),
        "lifecycle_edge_count": len(lifecycle_rows),
        "affect_axis_count": len(affect_rows),
        "affect_status_counts": dict(sorted(affect_status.items())),
        "unresolved_count": unresolved,
        "counterfactual": counterfactual,
        "errors": errors,
        "pass": not errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate()
    output = result
    if args.summary_only:
        output = {
            "schema_version": result["schema_version"],
            "pass": result["pass"],
            "drive_count": result["drive_count"],
            "affect_status_counts": result["affect_status_counts"],
            "unresolved_count": result["unresolved_count"],
            "counterfactual": result["counterfactual"],
            "errors": result["errors"],
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    if not result["pass"]:
        return 1
    if args.fail_on_unresolved and result["unresolved_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
