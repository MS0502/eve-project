#!/usr/bin/env python3
"""Deterministic audit harness for the bounded M3-B shadow affect projection."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.m3_b_affect_projection import (
    ALLOWED_DRIVES,
    DRIVE_SPECS,
    M3_A_MERGE_SHA,
    NEGATIVE_TARGET_PAIRS,
    PROJECTION_SCHEMA_VERSION,
    AxisMapping,
    AxisObservation,
    DriveShadowPrior,
    ShadowAffectProjection,
    project_shadow_affect,
)

SCHEMA_VERSION = "eve.m3-b.affect-shadow-check.v1"
AFFECT_PLAN_PATH = Path("docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md")
M3_A_DESIGN_PATH = Path("docs/audit/M3_A_DRIVE_DYNAMICS_DESIGN.md")
TABLE_BEGIN = "<!-- BEGIN AFFECT AXIS MAPPING TABLE -->"
TABLE_END = "<!-- END AFFECT AXIS MAPPING TABLE -->"
EXPECTED_SOURCE_COUNTS = Counter(
    {"legacy_mutable_hormone": 26, "read_only_affect_registry": 37}
)
EXPECTED_STATUS_COUNTS = Counter({"MAPPED": 59, "PROPOSED-DROP": 4})
EXPECTED_DROP_AXES = ("estrogen", "testosterone", "prolactin", "progesterone")
EXPECTED_NEGATIVE_PAIR_COUNT = 35
SYNTHETIC_VALUES = (0.05, 0.25, 0.50, 0.75, 0.95)
SYNTHETIC_ELAPSED_SECONDS = 120


class AuditError(ValueError):
    """Raised when source design material cannot be parsed safely."""


def _clean(value: str) -> str:
    return value.strip().strip("`").strip()


def _list(value: str) -> tuple[str, ...]:
    cleaned = _clean(value)
    if cleaned in {"", "—", "-"}:
        return ()
    return tuple(part.strip() for part in cleaned.split(";") if part.strip())


def _table(text: str) -> list[list[str]]:
    if TABLE_BEGIN not in text or TABLE_END not in text:
        raise AuditError("affect mapping table markers are missing")
    rows: list[list[str]] = []
    header_seen = False
    body = text.split(TABLE_BEGIN, 1)[1].split(TABLE_END, 1)[0]
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 12:
            raise AuditError("affect mapping row must contain 12 columns")
        if all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells):
            continue
        if not header_seen:
            header_seen = True
            continue
        rows.append(cells)
    if not rows:
        raise AuditError("affect mapping table is empty")
    return rows


def parse_mappings(text: str) -> tuple[AxisMapping, ...]:
    mappings: list[AxisMapping] = []
    for cells in _table(text):
        axis = _clean(cells[0])
        status = _clean(cells[2])
        open_question = _clean(cells[11])
        if open_question not in {"", "—", "-"}:
            raise AuditError(f"{axis}: unresolved source-plan question")
        mappings.append(
            AxisMapping(
                axis=axis,
                source_family=_clean(cells[1]),
                status=status,
                target_drives=_list(cells[3]),
                appraisals=_list(cells[4]),
                emotions=_list(cells[5]),
                confidence_ruling=_clean(cells[8]),
                preservation=_clean(cells[10]),
            )
        )
    return tuple(mappings)


def _source_digest(axis: str, source_family: str) -> str:
    return hashlib.sha256(f"m3-b-source\x1f{source_family}\x1f{axis}".encode("utf-8")).hexdigest()


def synthetic_observations(mappings: Iterable[AxisMapping]) -> tuple[AxisObservation, ...]:
    rows: list[AxisObservation] = []
    for index, mapping in enumerate(mappings):
        rows.append(
            AxisObservation(
                axis=mapping.axis,
                source_family=mapping.source_family,
                value=SYNTHETIC_VALUES[index % len(SYNTHETIC_VALUES)],
                baseline=0.50,
                floor=0.00,
                ceiling=1.00,
                confidence=1.00 - ((index % 4) * 0.10),
                source_snapshot_id="m3-b:synthetic:complete-63-axis:v1",
                source_schema_version="eve.m3-b.synthetic-source.v1",
                source_integrity_digest=_source_digest(mapping.axis, mapping.source_family),
                source_metadata=(
                    ("axis_ordinal", str(index)),
                    ("evidence_mode", "synthetic_replay_only"),
                ),
            )
        )
    return tuple(rows)


def baseline_priors(*, pending_drive: str | None = None) -> tuple[DriveShadowPrior, ...]:
    rows: list[DriveShadowPrior] = []
    for drive in ALLOWED_DRIVES:
        prior = DriveShadowPrior.baseline(drive)
        rows.append(
            DriveShadowPrior(
                drive=prior.drive,
                value=prior.value,
                named_state=prior.named_state,
                state_epoch=0,
                seconds_since_transition=max(boundary[2] for boundary in DRIVE_SPECS[drive].boundaries),
                pending_candidate_id="a" * 64 if drive == pending_drive else None,
            )
        )
    return tuple(rows)


def _mapping_summary(mappings: Sequence[AxisMapping]) -> dict[str, Any]:
    target_pairs = [
        (mapping.axis, drive)
        for mapping in mappings
        for drive in mapping.target_drives
    ]
    return {
        "axis_count": len(mappings),
        "axis_unique_count": len({mapping.axis for mapping in mappings}),
        "confidence_rulings": dict(sorted(Counter(mapping.confidence_ruling for mapping in mappings).items())),
        "drop_axes": [mapping.axis for mapping in mappings if mapping.status == "PROPOSED-DROP"],
        "negative_target_pair_count": len(NEGATIVE_TARGET_PAIRS),
        "source_family_counts": dict(sorted(Counter(mapping.source_family for mapping in mappings).items())),
        "status_counts": dict(sorted(Counter(mapping.status for mapping in mappings).items())),
        "target_pair_count": len(target_pairs),
        "target_pair_unique_count": len(set(target_pairs)),
    }


def _projection_summary(projection: ShadowAffectProjection) -> dict[str, Any]:
    mapped = [row for row in projection.axis_projections if row.status == "MAPPED"]
    dropped = [row for row in projection.axis_projections if row.status == "PROPOSED-DROP"]
    drive_rows = projection.drive_projections
    return {
        "axis_projection_count": len(projection.axis_projections),
        "candidate_count": sum(row.candidate is not None for row in drive_rows),
        "contribution_count": sum(len(row.contributions) for row in mapped),
        "drive_projection_count": len(drive_rows),
        "drop_contribution_count": sum(len(row.contributions) for row in dropped),
        "drop_preserved_count": sum(
            row.original_value is not None
            and row.observation_digest is not None
            and row.source_integrity_digest is not None
            for row in dropped
        ),
        "event_emission_count": sum(row.event_emitted for row in drive_rows),
        "mapped_axis_count": len(mapped),
        "missing_axis_count": len(projection.missing_axes),
        "named_state_mutation_count": sum(row.named_state_mutated for row in drive_rows),
        "projection_digest": projection.digest,
        "source_saturation_count": sum(row.saturated for row in projection.axis_projections),
        "target_saturation_count": sum(row.saturated for row in drive_rows),
    }


def validate_catalog(mappings: Sequence[AxisMapping]) -> list[str]:
    errors: list[str] = []
    summary = _mapping_summary(mappings)
    if summary["axis_count"] != 63 or summary["axis_unique_count"] != 63:
        errors.append("axis count/uniqueness mismatch")
    if Counter(mapping.source_family for mapping in mappings) != EXPECTED_SOURCE_COUNTS:
        errors.append("source-family counts mismatch")
    if Counter(mapping.status for mapping in mappings) != EXPECTED_STATUS_COUNTS:
        errors.append("mapping-status counts mismatch")
    drops = tuple(mapping.axis for mapping in mappings if mapping.status == "PROPOSED-DROP")
    if drops != EXPECTED_DROP_AXES:
        errors.append("proposed-drop axis catalog/order mismatch")
    target_pairs = {
        (mapping.axis, drive)
        for mapping in mappings
        for drive in mapping.target_drives
    }
    invalid_negative = sorted(NEGATIVE_TARGET_PAIRS - target_pairs)
    if invalid_negative:
        errors.append(f"negative polarity pairs are not source-plan targets: {invalid_negative}")
    if len(NEGATIVE_TARGET_PAIRS) != EXPECTED_NEGATIVE_PAIR_COUNT:
        errors.append("negative polarity decision count mismatch")
    if tuple(DRIVE_SPECS) != ALLOWED_DRIVES:
        errors.append("M3-A drive spec order mismatch")
    return errors


def validate_projection(projection: ShadowAffectProjection) -> list[str]:
    errors: list[str] = []
    summary = _projection_summary(projection)
    if summary["axis_projection_count"] != 63 or summary["mapped_axis_count"] != 59:
        errors.append("projection coverage mismatch")
    if summary["drop_preserved_count"] != 4 or summary["drop_contribution_count"] != 0:
        errors.append("proposed-drop preservation/behavior boundary mismatch")
    if summary["drive_projection_count"] != 8:
        errors.append("drive projection count mismatch")
    if summary["missing_axis_count"] != 0:
        errors.append("complete projection contains missing axes")
    if summary["event_emission_count"] or summary["named_state_mutation_count"]:
        errors.append("M3-B projection mutated named state or emitted an event")
    if (
        projection.authority != "shadow_only"
        or not projection.legacy_runtime_authoritative
        or projection.persistence_accessed
        or projection.event_append_performed
        or projection.live_behavior_changed
        or projection.cutover_authorized
        or projection.m3_authority_open
    ):
        errors.append("M3-B authority boundary violated")
    for row in projection.drive_projections:
        spec = DRIVE_SPECS[row.drive]
        if not spec.floor <= row.next_value <= spec.ceiling:
            errors.append(f"{row.drive}: projected value outside bounds")
        if abs(row.next_value - row.previous_value) > spec.max_slew_per_second * projection.elapsed_seconds + 1e-12:
            errors.append(f"{row.drive}: slew bound exceeded")
        if row.named_state_retained not in spec.states:
            errors.append(f"{row.drive}: retained state is invalid")
        if row.candidate is not None and (
            not row.candidate.diagnostic_only
            or row.candidate.event_append_authorized
            or row.candidate.authority != "shadow_only"
        ):
            errors.append(f"{row.drive}: diagnostic candidate gained authority")
    return errors


def audit_repository(root: Path = ROOT) -> dict[str, Any]:
    affect_path = root / AFFECT_PLAN_PATH
    m3_a_path = root / M3_A_DESIGN_PATH
    mappings = parse_mappings(affect_path.read_text(encoding="utf-8"))
    observations = synthetic_observations(mappings)
    priors = baseline_priors()
    first = project_shadow_affect(
        mappings=mappings,
        observations=observations,
        priors=priors,
        elapsed_seconds=SYNTHETIC_ELAPSED_SECONDS,
        strict=True,
    )
    second = project_shadow_affect(
        mappings=mappings,
        observations=observations,
        priors=priors,
        elapsed_seconds=SYNTHETIC_ELAPSED_SECONDS,
        strict=True,
    )
    errors = validate_catalog(mappings) + validate_projection(first)
    if first.to_mapping() != second.to_mapping() or first.digest != second.digest:
        errors.append("same input/version did not produce byte-equivalent projection material")
    m3_a_text = m3_a_path.read_text(encoding="utf-8")
    required_m3_a_tokens = (
        M3_A_MERGE_SHA,
        "eve.m3-a.drive-dynamics.v1",
        "eve.m3-a.named-transition-predicate.v1",
        "Continuous sampling emits zero events.",
    )
    # The design records its original baseline rather than merge SHA, so the merge
    # pin is carried here and in the report; semantic version tokens must exist in
    # the source document itself.
    if any(token not in m3_a_text for token in required_m3_a_tokens[1:]):
        errors.append("M3-A source contract tokens are missing")
    report: dict[str, Any] = {
        "authority": "shadow_only",
        "cutover_authorized": False,
        "deterministic_replay_equal": first.digest == second.digest,
        "errors": errors,
        "legacy_runtime_authoritative": True,
        "live_behavior_changed": False,
        "m3_a_merge_sha": M3_A_MERGE_SHA,
        "m3_authority_open": False,
        "mapping_summary": _mapping_summary(mappings),
        "observation_mode": "synthetic_complete_63_axis_replay",
        "persistence_accessed": False,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "projection_summary": _projection_summary(first),
        "schema_version": SCHEMA_VERSION,
        "strict_complete_coverage": True,
    }
    digest_material = dict(report)
    report["report_digest"] = hashlib.sha256(
        json.dumps(digest_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def _write_json(report: Mapping[str, Any], output: Path | None, *, pretty: bool) -> None:
    text = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    ) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_repository()
    if args.summary_only:
        summary = {
            "axis_count": report["mapping_summary"]["axis_count"],
            "drive_projection_count": report["projection_summary"]["drive_projection_count"],
            "errors": report["errors"],
            "mapped": report["mapping_summary"]["status_counts"].get("MAPPED", 0),
            "proposed_drop": report["mapping_summary"]["status_counts"].get("PROPOSED-DROP", 0),
            "projection_digest": report["projection_summary"]["projection_digest"],
            "report_digest": report["report_digest"],
        }
        _write_json(summary, args.output, pretty=args.pretty)
    else:
        _write_json(report, args.output, pretty=args.pretty)
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
