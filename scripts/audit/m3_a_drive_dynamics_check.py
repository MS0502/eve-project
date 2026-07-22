#!/usr/bin/env python3
"""Static standard-library checker for the documentation-only EVE M3-A design."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "eve.m3-a.drive-dynamics-check.v1"
BASELINE_SHA = "7697c1047bbf081295a01f630d63d8a3ad5c69b0"
DESIGN_PATH = Path("docs/audit/M3_A_DRIVE_DYNAMICS_DESIGN.md")
AFFECT_PLAN_PATH = Path("docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md")
ALLOWED_DRIVES = ("energy", "safety", "affiliation", "curiosity", "agency", "coherence", "competence", "expression")
ALLOWED_RULINGS = {"RESOLVED", "UNRESOLVED"}
PREDICATE_VERSION = "eve.m3-a.named-transition-predicate.v1"
MARKERS = {
    "drive": ("<!-- BEGIN M3A DRIVE TABLE -->", "<!-- END M3A DRIVE TABLE -->", 9),
    "state": ("<!-- BEGIN M3A STATE TABLE -->", "<!-- END M3A STATE TABLE -->", 5),
    "boundary": ("<!-- BEGIN M3A BOUNDARY TABLE -->", "<!-- END M3A BOUNDARY TABLE -->", 6),
    "lifecycle": ("<!-- BEGIN M3A LIFECYCLE TABLE -->", "<!-- END M3A LIFECYCLE TABLE -->", 4),
    "affect": ("<!-- BEGIN AFFECT AXIS MAPPING TABLE -->", "<!-- END AFFECT AXIS MAPPING TABLE -->", 12),
}
REQUIRED = (
    "## Authority and scope", "## Continuous dynamics", "## Drive parameter rulings",
    "## Semantic state catalog", "## Named transition catalog", "## Candidate lifecycle",
    "## A9 no-duplicate proof", "## A9 compliance matrix", "## Affect Migration Plan landing",
    "## Reviewer-ruling regime", "## Verification", "## Acceptance criteria",
    "## Explicit non-goals", "## Changed-file boundary", BASELINE_SHA,
    "a_d       = clip(", "target_d  = clip(",
    "relaxed_d = target_d+(x_d-target_d)*exp(-Δt/tau_d)", "x_next_d  = clip(",
    "Continuous sampling emits zero events.", "monotonic elapsed time",
    "at most one adjacent state", "integration eligibility exists only after an explicit persistence cutover",
    "No runtime integration", "M3-E authority",
)
EXPECTED_LIFECYCLE_EDGES = {
    ("absent", "proposed"), ("proposed", "validated"), ("proposed", "rejected"),
    ("proposed", "expired"), ("validated", "emitted"), ("validated", "rejected"),
    ("emitted", "absent"), ("rejected", "absent"), ("expired", "absent"),
}


def _clean(value: str) -> str:
    return value.strip().strip("`").strip()


def _list(value: str) -> list[str]:
    value = _clean(value)
    return [] if value in {"", "—", "-"} else [part.strip() for part in value.split(";") if part.strip()]


def _table(text: str, name: str) -> list[list[str]]:
    begin, end, count = MARKERS[name]
    if begin not in text or end not in text:
        raise AssertionError(f"missing table: {name}")
    rows, header = [], False
    for raw in text.split(begin, 1)[1].split(end, 1)[0].splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != count:
            raise AssertionError(f"{name}: expected {count} cells")
        if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
            continue
        if not header:
            header = True
            continue
        rows.append(cells)
    if not rows:
        raise AssertionError(f"empty table: {name}")
    return rows


def _num(value: str, field: str) -> float:
    try:
        result = float(_clean(value))
    except ValueError as exc:
        raise AssertionError(f"{field}: not numeric") from exc
    if not math.isfinite(result):
        raise AssertionError(f"{field}: not finite")
    return result


def _int(value: str, field: str) -> int:
    value = _clean(value)
    if not re.fullmatch(r"-?[0-9]+", value):
        raise AssertionError(f"{field}: not integer")
    return int(value)


def _ruling(value: str, question: str, field: str) -> tuple[str, str]:
    value, question = _clean(value), _clean(question)
    if value not in ALLOWED_RULINGS:
        raise AssertionError(f"{field}: bad ruling")
    if value == "UNRESOLVED" and question in {"", "—", "-"}:
        raise AssertionError(f"{field}: unresolved without question")
    return value, question


def parse_drives(text: str) -> list[dict[str, Any]]:
    out = []
    for c in _table(text, "drive"):
        drive = _clean(c[0])
        ruling, question = _ruling(c[7], c[8], drive)
        out.append({"drive": drive, "baseline": _num(c[1], "baseline"), "tau_seconds": _int(c[2], "tau"),
                    "floor": _num(c[3], "floor"), "ceiling": _num(c[4], "ceiling"),
                    "gain": _num(c[5], "gain"), "max_slew_per_second": _num(c[6], "slew"),
                    "ruling": ruling, "open_question": question})
    return out


def parse_states(text: str) -> list[dict[str, Any]]:
    out = []
    for c in _table(text, "state"):
        drive, names, meanings = _clean(c[0]), _list(c[1]), _list(c[2])
        ruling, question = _ruling(c[3], c[4], drive)
        if len(names) != 4 or len(meanings) != 4:
            raise AssertionError(f"{drive}: need four states and meanings")
        out.extend({"drive": drive, "ordinal": i, "state": state, "meaning": meaning,
                    "ruling": ruling, "open_question": question}
                   for i, (state, meaning) in enumerate(zip(names, meanings)))
    return out


def parse_boundaries(text: str, states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = {drive: [r["state"] for r in sorted((x for x in states if x["drive"] == drive), key=lambda x: x["ordinal"])]
             for drive in ALLOWED_DRIVES}
    out = []
    for c in _table(text, "boundary"):
        drive = _clean(c[0])
        ruling, question = _ruling(c[4], c[5], drive)
        if len(names.get(drive, [])) != 4:
            raise AssertionError(f"{drive}: missing states")
        for i, cell in enumerate(c[1:4]):
            parts = _clean(cell).split("/")
            if len(parts) != 3:
                raise AssertionError(f"{drive}: bad boundary cell")
            down, up, cooldown = _num(parts[0], "down"), _num(parts[1], "up"), _int(parts[2], "cooldown")
            lower, upper = names[drive][i], names[drive][i + 1]
            out.append({"drive": drive, "lower_state": lower, "upper_state": upper,
                        "down_threshold": down, "up_threshold": up, "hysteresis_width": up - down,
                        "cooldown_seconds": cooldown,
                        "up_transition_id": f"m3a.{drive}.{lower}_to_{upper}.v1",
                        "down_transition_id": f"m3a.{drive}.{upper}_to_{lower}.v1",
                        "up_candidate": f"drive.{drive}.{upper}_candidate",
                        "down_candidate": f"drive.{drive}.{lower}_candidate",
                        "ruling": ruling, "open_question": question})
    return out


def expand_transitions(boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in boundaries:
        common = {"drive": row["drive"], "hysteresis_width": row["hysteresis_width"],
                  "cooldown_seconds": row["cooldown_seconds"], "predicate_version": PREDICATE_VERSION,
                  "ruling": row["ruling"], "open_question": row["open_question"]}
        out.extend((
            {**common, "transition_id": row["up_transition_id"], "from_state": row["lower_state"],
             "to_state": row["upper_state"], "direction": "up", "operator": ">=",
             "threshold": row["up_threshold"], "candidate_type": row["up_candidate"]},
            {**common, "transition_id": row["down_transition_id"], "from_state": row["upper_state"],
             "to_state": row["lower_state"], "direction": "down", "operator": "<=",
             "threshold": row["down_threshold"], "candidate_type": row["down_candidate"]},
        ))
    return out


def parse_lifecycle(text: str) -> list[dict[str, str]]:
    return [{"from_state": _clean(c[0]), "to_state": _clean(c[1]),
             "trigger": _clean(c[2]), "authority_effect": _clean(c[3])}
            for c in _table(text, "lifecycle")]


def parse_affect(text: str) -> list[dict[str, Any]]:
    return [{"axis": _clean(c[0]), "source_family": _clean(c[1]), "status": _clean(c[2]),
             "target_drives": _list(c[3]), "appraisals": _list(c[4]), "emotions": _list(c[5]),
             "preservation": _clean(c[10]), "open_question": _clean(c[11])}
            for c in _table(text, "affect")]


def validate_design(text: str, drives: list[dict[str, Any]], states: list[dict[str, Any]],
                    boundaries: list[dict[str, Any]], transitions: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    errors = [f"missing token: {item}" for item in REQUIRED if item not in text]
    if tuple(r["drive"] for r in drives) != ALLOWED_DRIVES:
        errors.append("drive set/order mismatch")
    by_drive = {r["drive"]: r for r in drives}
    for row in drives:
        if row["tau_seconds"] <= 0 or row["gain"] <= 0 or row["max_slew_per_second"] <= 0:
            errors.append(f"{row['drive']}: non-positive parameter")
        if not (0 <= row["floor"] < row["ceiling"] <= 1 and row["floor"] <= row["baseline"] <= row["ceiling"]):
            errors.append(f"{row['drive']}: invalid bounds/baseline")

    state_names, state_ord = {}, {}
    for drive in ALLOWED_DRIVES:
        rows = sorted((r for r in states if r["drive"] == drive), key=lambda r: r["ordinal"])
        if len(rows) != 4 or [r["ordinal"] for r in rows] != [0, 1, 2, 3]:
            errors.append(f"{drive}: state catalog mismatch")
        state_names[drive] = [r["state"] for r in rows]
        state_ord[drive] = {r["state"]: r["ordinal"] for r in rows}

    expected = {(drive, values[i], values[i + 1]) for drive, values in state_names.items() for i in range(3)}
    actual = {(r["drive"], r["lower_state"], r["upper_state"]) for r in boundaries}
    if len(boundaries) != 24 or actual != expected:
        errors.append("boundary catalog mismatch")
    if len(transitions) != 48 or len({r["transition_id"] for r in transitions}) != 48:
        errors.append("transition catalog mismatch")

    ups = {drive: [] for drive in ALLOWED_DRIVES}
    for row in boundaries:
        drive = row["drive"]
        if drive not in by_drive or state_ord[drive].get(row["upper_state"]) != state_ord[drive].get(row["lower_state"], -9) + 1:
            errors.append(f"{drive}: non-adjacent/unknown boundary")
            continue
        width = row["up_threshold"] - row["down_threshold"]
        if width <= 0 or not math.isclose(width, row["hysteresis_width"], abs_tol=1e-9):
            errors.append(f"{drive}: hysteresis mismatch")
        if row["cooldown_seconds"] <= 0:
            errors.append(f"{drive}: cooldown not positive")
        if not by_drive[drive]["floor"] <= row["down_threshold"] < row["up_threshold"] <= by_drive[drive]["ceiling"]:
            errors.append(f"{drive}: threshold out of bounds")
        ups[drive].append(row["up_threshold"])
    baseline_states = {}
    for drive, thresholds in ups.items():
        if thresholds != sorted(thresholds):
            errors.append(f"{drive}: thresholds unordered")
        if len(thresholds) == 3:
            baseline_states[drive] = state_names[drive][sum(by_drive[drive]["baseline"] >= t for t in thresholds)]
    return errors, baseline_states


def validate_lifecycle(rows: list[dict[str, str]]) -> list[str]:
    edges = {(r["from_state"], r["to_state"]) for r in rows}
    errors = [] if edges == EXPECTED_LIFECYCLE_EDGES and len(edges) == len(rows) else ["lifecycle edge mismatch"]
    if any(r["trigger"] in {"", "—"} or r["authority_effect"] in {"", "—"} for r in rows):
        errors.append("incomplete lifecycle row")
    return errors


def validate_affect(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    errors = []
    if len(rows) != 63 or len({r["axis"] for r in rows}) != 63:
        errors.append("axis count/uniqueness mismatch")
    if Counter(r["status"] for r in rows) != Counter({"MAPPED": 59, "PROPOSED-DROP": 4}):
        errors.append("axis status counts mismatch")
    if Counter(r["source_family"] for r in rows) != Counter({"legacy_mutable_hormone": 26, "read_only_affect_registry": 37}):
        errors.append("axis source counts mismatch")
    landings = []
    for row in rows:
        invalid = sorted(set(row["target_drives"]) - set(ALLOWED_DRIVES))
        targets = ([f"drive::{v}" for v in row["target_drives"]]
                   + [f"appraisal::{v}" for v in row["appraisals"]]
                   + [f"emotion::{v}" for v in row["emotions"]])
        if row["status"] == "MAPPED":
            covered = bool(row["target_drives"]) and bool(targets) and not invalid
        elif row["status"] == "PROPOSED-DROP":
            covered = not targets and row["preservation"] not in {"", "—", "-"}
        elif row["status"] == "UNRESOLVED":
            covered = False
            if row["open_question"] in {"", "—", "-"}:
                errors.append(f"{row['axis']}: unresolved without question")
        else:
            covered = False
        if invalid or not covered:
            errors.append(f"{row['axis']}: invalid landing/drop")
        landings.append({"axis": row["axis"], "source_family": row["source_family"],
                         "status": row["status"], "targets": targets,
                         "preservation": row["preservation"], "covered": covered})
    if sum(r["covered"] for r in landings) != 63:
        errors.append("not all axes covered")
    return errors, landings


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
    affect_text = (root / AFFECT_PLAN_PATH).read_text(encoding="utf-8")
    drives, states = parse_drives(design_text), parse_states(design_text)
    boundaries = parse_boundaries(design_text, states)
    transitions, lifecycle, affect = expand_transitions(boundaries), parse_lifecycle(design_text), parse_affect(affect_text)
    errors, baseline_states = validate_design(design_text, drives, states, boundaries, transitions)
    errors.extend(validate_lifecycle(lifecycle))
    affect_errors, landings = validate_affect(affect)
    errors.extend(affect_errors)
    rulings, statuses = Counter(r["ruling"] for r in [*drives, *states, *boundaries]), Counter(r["status"] for r in affect)
    targets = Counter(t.split("::", 1)[0] for row in landings for t in row["targets"])
    return {
        "schema_version": SCHEMA_VERSION, "baseline_sha": BASELINE_SHA, "root": ".",
        "summary": {"drives": len(drives), "semantic_states": len(states),
                    "transition_boundaries": len(boundaries), "named_transitions": len(transitions),
                    "candidate_lifecycle_edges": len(lifecycle),
                    "resolved_rulings": rulings.get("RESOLVED", 0),
                    "unresolved_rulings": rulings.get("UNRESOLVED", 0),
                    "affect_axes": len(affect), "mapped_axes": statuses.get("MAPPED", 0),
                    "proposed_drop_axes": statuses.get("PROPOSED-DROP", 0),
                    "source_unresolved_axes": statuses.get("UNRESOLVED", 0),
                    "covered_axes": sum(r["covered"] for r in landings),
                    "drive_target_landings": targets.get("drive", 0),
                    "appraisal_target_landings": targets.get("appraisal", 0),
                    "emotion_target_landings": targets.get("emotion", 0),
                    "validation_errors": len(errors)},
        "drive_parameters": drives, "baseline_states": baseline_states,
        "semantic_states": states, "transition_boundaries": boundaries,
        "named_transitions": transitions, "candidate_lifecycle": lifecycle,
        "axis_landings": landings, "validation_errors": errors,
        "scope": {"standard_library_only": True, "static_document_analysis_only": True,
                  "runtime_import_performed": False, "runtime_execution_performed": False,
                  "production_state_read_performed": False, "production_state_write_performed": False,
                  "event_emission_performed": False, "projection_implementation_performed": False,
                  "scheduler_integration_performed": False, "goal_integration_performed": False,
                  "persistence_integration_performed": False, "cutover_authorized": False,
                  "m3_e_authority_granted": False, "integration_eligible_before_cutover": False},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = audit_repository(args.root)
    except (AssertionError, OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if report["validation_errors"]:
        for error in report["validation_errors"]:
            print(error, file=sys.stderr)
        return 2
    if args.fail_on_unresolved and (report["summary"]["unresolved_rulings"] or report["summary"]["source_unresolved_axes"]):
        print("unresolved M3-A or Affect Plan rulings remain", file=sys.stderr)
        return 3
    payload: Any = report["summary"] if args.summary_only else report
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                indent=2 if args.pretty else None,
                                separators=None if args.pretty else (",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
