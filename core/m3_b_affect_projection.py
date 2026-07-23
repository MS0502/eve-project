"""Bounded read-only M3-B affect projection contracts.

The module consumes caller-supplied immutable observations and returns an immutable
shadow projection. Import and construction perform no I/O, install no observer,
append no event, access no persistence, and grant no behavioral or cutover authority.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from core.event_kernel import SHADOW_AUTHORITY, canonical_json_object

PROJECTION_SCHEMA_VERSION = "eve.m3-b.affect-shadow-projection.v1"
OBSERVATION_SCHEMA_VERSION = "eve.m3-b.affect-axis-observation.v1"
MAPPING_SCHEMA_VERSION = "eve.m3-b.affect-axis-mapping.v1"
DRIVE_PROJECTION_SCHEMA_VERSION = "eve.m3-b.drive-shadow-projection.v1"
CANDIDATE_SCHEMA_VERSION = "eve.m3-a.transition-candidate.v1"
PARAMETER_VERSION = "eve.m3-a.drive-dynamics.v1"
PREDICATE_VERSION = "eve.m3-a.named-transition-predicate.v1"
M3_A_MERGE_SHA = "6d581ba1cf11ffbefafe77beabd8f669102909d0"

ALLOWED_DRIVES = (
    "energy",
    "safety",
    "affiliation",
    "curiosity",
    "agency",
    "coherence",
    "competence",
    "expression",
)
ALLOWED_SOURCE_FAMILIES = {"legacy_mutable_hormone", "read_only_affect_registry"}
ALLOWED_STATUSES = {"MAPPED", "PROPOSED-DROP"}
CONFIDENCE_CAPS = {"high": 1.0, "medium": 0.75, "low": 0.50}

# All mapped target pairs are positive unless listed here. Deficit, pressure,
# inhibition, threat, and risk axes lower the achieved/readiness state of the
# named target while their appraisal labels remain available for later review.
NEGATIVE_TARGET_PAIRS = frozenset(
    {
        ("norepinephrine", "safety"),
        ("adenosine", "energy"),
        ("cortisol", "safety"),
        ("melatonin", "energy"),
        ("ghrelin", "energy"),
        ("fatigue_pressure", "energy"),
        ("recovery_need", "energy"),
        ("recovery_need", "safety"),
        ("stress_load", "safety"),
        ("stability_need", "coherence"),
        ("stability_need", "safety"),
        ("overload_risk", "energy"),
        ("overload_risk", "safety"),
        ("threat_pressure", "safety"),
        ("uncertainty_pressure", "coherence"),
        ("self_protection", "safety"),
        ("boundary_defense", "safety"),
        ("trust_risk", "affiliation"),
        ("trust_risk", "safety"),
        ("exposure_risk", "safety"),
        ("social_pain", "affiliation"),
        ("social_pain", "safety"),
        ("loneliness_pressure", "affiliation"),
        ("belonging_need", "affiliation"),
        ("rejection_sensitivity", "affiliation"),
        ("rejection_sensitivity", "safety"),
        ("learning_pressure", "competence"),
        ("memory_consolidation_pressure", "coherence"),
        ("prediction_error_pressure", "coherence"),
        ("competence_drive", "competence"),
        ("agency_pressure", "agency"),
        ("expression_inhibition", "expression"),
        ("expression_inhibition", "safety"),
        ("conflict_avoidance", "affiliation"),
        ("conflict_avoidance", "safety"),
    }
)


class AffectProjectionError(ValueError):
    """Raised when projection input violates the fixed M3-B contract."""


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AffectProjectionError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AffectProjectionError(f"{field} must be finite")
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    text = canonical_json_object(value, field=field)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_digest(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AffectProjectionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise AffectProjectionError(f"{field} must be a bounded non-empty string")
    return value


def _ordered_unique(values: Iterable[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise AffectProjectionError(f"{field} entries must be non-empty strings")
    if len(set(result)) != len(result):
        raise AffectProjectionError(f"{field} entries must be unique")
    return result


@dataclass(frozen=True, slots=True)
class AxisObservation:
    axis: str
    source_family: str
    value: float
    baseline: float
    floor: float
    ceiling: float
    confidence: float
    source_snapshot_id: str
    source_schema_version: str
    source_integrity_digest: str
    source_metadata: tuple[tuple[str, str], ...] = ()
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    acquisition_mode: str = "caller_supplied_read_only"

    def __post_init__(self) -> None:
        _require_identifier(self.axis, "axis")
        if self.source_family not in ALLOWED_SOURCE_FAMILIES:
            raise AffectProjectionError("unsupported source family")
        value = _finite(self.value, "value")
        baseline = _finite(self.baseline, "baseline")
        floor = _finite(self.floor, "floor")
        ceiling = _finite(self.ceiling, "ceiling")
        confidence = _finite(self.confidence, "confidence")
        if not floor < baseline < ceiling:
            raise AffectProjectionError("observation requires floor < baseline < ceiling")
        if not floor <= value <= ceiling:
            raise AffectProjectionError("observation value must remain within declared bounds")
        if not 0.0 <= confidence <= 1.0:
            raise AffectProjectionError("confidence must be within [0,1]")
        _require_identifier(self.source_snapshot_id, "source_snapshot_id")
        _require_identifier(self.source_schema_version, "source_schema_version")
        _require_digest(self.source_integrity_digest, "source_integrity_digest")
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise AffectProjectionError("unsupported observation schema")
        if self.authority != SHADOW_AUTHORITY or self.acquisition_mode != "caller_supplied_read_only":
            raise AffectProjectionError("observation cannot grant authority or live acquisition")
        keys: set[str] = set()
        for key, item in self.source_metadata:
            _require_identifier(key, "source_metadata.key")
            if not isinstance(item, str) or len(item) > 1024:
                raise AffectProjectionError("source metadata values must be bounded strings")
            if key in keys:
                raise AffectProjectionError("source metadata keys must be unique")
            keys.add(key)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "floor", floor)
        object.__setattr__(self, "ceiling", ceiling)
        object.__setattr__(self, "confidence", confidence)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "acquisition_mode": self.acquisition_mode,
            "authority": self.authority,
            "axis": self.axis,
            "baseline": self.baseline,
            "ceiling": self.ceiling,
            "confidence": self.confidence,
            "floor": self.floor,
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "source_integrity_digest": self.source_integrity_digest,
            "source_metadata": [list(item) for item in self.source_metadata],
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "value": self.value,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping(), "axis_observation")


@dataclass(frozen=True, slots=True)
class AxisMapping:
    axis: str
    source_family: str
    status: str
    target_drives: tuple[str, ...]
    appraisals: tuple[str, ...]
    emotions: tuple[str, ...]
    confidence_ruling: str
    preservation: str
    schema_version: str = MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.axis, "axis")
        if self.source_family not in ALLOWED_SOURCE_FAMILIES:
            raise AffectProjectionError("unsupported mapping source family")
        if self.status not in ALLOWED_STATUSES:
            raise AffectProjectionError("unsupported mapping status")
        drives = _ordered_unique(self.target_drives, "target_drives")
        appraisals = _ordered_unique(self.appraisals, "appraisals")
        emotions = _ordered_unique(self.emotions, "emotions")
        if any(drive not in ALLOWED_DRIVES for drive in drives):
            raise AffectProjectionError("mapping contains unknown drive")
        if self.confidence_ruling not in CONFIDENCE_CAPS:
            raise AffectProjectionError("unsupported confidence ruling")
        _require_identifier(self.preservation, "preservation")
        if self.schema_version != MAPPING_SCHEMA_VERSION:
            raise AffectProjectionError("unsupported mapping schema")
        if self.status == "MAPPED" and (not drives or not (appraisals or emotions)):
            raise AffectProjectionError("mapped axis requires targets and semantic landing")
        if self.status == "PROPOSED-DROP" and (drives or appraisals or emotions):
            raise AffectProjectionError("proposed-drop axis cannot have future targets")
        object.__setattr__(self, "target_drives", drives)
        object.__setattr__(self, "appraisals", appraisals)
        object.__setattr__(self, "emotions", emotions)

    def polarity(self, drive: str) -> int:
        if drive not in self.target_drives:
            raise AffectProjectionError("polarity requested for unmapped drive")
        return -1 if (self.axis, drive) in NEGATIVE_TARGET_PAIRS else 1

    @property
    def confidence_cap(self) -> float:
        return CONFIDENCE_CAPS[self.confidence_ruling]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "appraisals": list(self.appraisals),
            "axis": self.axis,
            "confidence_cap": self.confidence_cap,
            "confidence_ruling": self.confidence_ruling,
            "emotions": list(self.emotions),
            "polarities": [[drive, self.polarity(drive)] for drive in self.target_drives],
            "preservation": self.preservation,
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "status": self.status,
            "target_drives": list(self.target_drives),
        }


@dataclass(frozen=True, slots=True)
class DriveSpec:
    drive: str
    baseline: float
    tau_seconds: int
    floor: float
    ceiling: float
    gain: float
    max_slew_per_second: float
    states: tuple[str, str, str, str]
    boundaries: tuple[tuple[float, float, int], tuple[float, float, int], tuple[float, float, int]]

    def __post_init__(self) -> None:
        if self.drive not in ALLOWED_DRIVES:
            raise AffectProjectionError("unknown drive spec")
        if self.tau_seconds <= 0 or self.gain <= 0 or self.max_slew_per_second <= 0:
            raise AffectProjectionError("drive dynamics must be positive")
        if not 0 <= self.floor < self.ceiling <= 1 or not self.floor <= self.baseline <= self.ceiling:
            raise AffectProjectionError("invalid drive bounds")
        if len(set(self.states)) != 4:
            raise AffectProjectionError("drive states must be unique")
        previous_up = -math.inf
        for down, up, cooldown in self.boundaries:
            if not self.floor <= down < up <= self.ceiling or cooldown <= 0 or up <= previous_up:
                raise AffectProjectionError("invalid drive boundary")
            previous_up = up


DRIVE_SPECS: dict[str, DriveSpec] = {
    "energy": DriveSpec("energy", 0.60, 300, 0.02, 0.98, 0.38, 0.020, ("depleted", "guarded", "available", "abundant"), ((0.20, 0.28, 30), (0.45, 0.55, 45), (0.72, 0.82, 90))),
    "safety": DriveSpec("safety", 0.62, 180, 0.02, 0.98, 0.42, 0.015, ("threatened", "guarded", "secure", "resilient"), ((0.22, 0.30, 20), (0.48, 0.58, 60), (0.74, 0.84, 120))),
    "affiliation": DriveSpec("affiliation", 0.35, 1800, 0.01, 0.95, 0.30, 0.005, ("withdrawn", "receptive", "connected", "affiliative"), ((0.17, 0.25, 120), (0.42, 0.52, 300), (0.68, 0.80, 600))),
    "curiosity": DriveSpec("curiosity", 0.32, 240, 0.01, 0.97, 0.45, 0.020, ("quiet", "attentive", "exploring", "absorbed"), ((0.16, 0.24, 30), (0.40, 0.50, 60), (0.66, 0.78, 120))),
    "agency": DriveSpec("agency", 0.50, 420, 0.01, 0.97, 0.40, 0.015, ("constrained", "deliberative", "self_directed", "assertive"), ((0.20, 0.28, 45), (0.46, 0.56, 90), (0.70, 0.82, 180))),
    "coherence": DriveSpec("coherence", 0.68, 1200, 0.03, 0.99, 0.28, 0.005, ("fragmented", "reconciling", "coherent", "integrated"), ((0.22, 0.32, 60), (0.50, 0.60, 180), (0.74, 0.86, 600))),
    "competence": DriveSpec("competence", 0.42, 1800, 0.01, 0.96, 0.30, 0.005, ("uncertain", "practicing", "capable", "mastering"), ((0.18, 0.26, 60), (0.44, 0.54, 180), (0.68, 0.80, 600))),
    "expression": DriveSpec("expression", 0.25, 90, 0.00, 0.95, 0.50, 0.030, ("silent", "forming", "ready", "expressive"), ((0.14, 0.22, 15), (0.38, 0.48, 30), (0.62, 0.74, 90))),
}


@dataclass(frozen=True, slots=True)
class DriveShadowPrior:
    drive: str
    value: float
    named_state: str
    state_epoch: int = 0
    seconds_since_transition: int = 0
    pending_candidate_id: str | None = None

    def __post_init__(self) -> None:
        if self.drive not in DRIVE_SPECS:
            raise AffectProjectionError("unknown drive prior")
        spec = DRIVE_SPECS[self.drive]
        value = _finite(self.value, "prior.value")
        if not spec.floor <= value <= spec.ceiling:
            raise AffectProjectionError("prior value outside drive bounds")
        if self.named_state not in spec.states:
            raise AffectProjectionError("prior named state is invalid")
        if isinstance(self.state_epoch, bool) or not isinstance(self.state_epoch, int) or self.state_epoch < 0:
            raise AffectProjectionError("state_epoch must be a non-negative integer")
        if (
            isinstance(self.seconds_since_transition, bool)
            or not isinstance(self.seconds_since_transition, int)
            or self.seconds_since_transition < 0
        ):
            raise AffectProjectionError("seconds_since_transition must be a non-negative integer")
        if self.pending_candidate_id is not None:
            _require_digest(self.pending_candidate_id, "pending_candidate_id")
        object.__setattr__(self, "value", value)

    @classmethod
    def baseline(cls, drive: str) -> "DriveShadowPrior":
        spec = DRIVE_SPECS[drive]
        ordinal = sum(spec.baseline >= boundary[1] for boundary in spec.boundaries)
        return cls(drive=drive, value=spec.baseline, named_state=spec.states[ordinal])


@dataclass(frozen=True, slots=True)
class DriveContribution:
    axis: str
    drive: str
    normalized_value: float
    polarity: int
    confidence: float
    contribution: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "confidence": self.confidence,
            "contribution": self.contribution,
            "drive": self.drive,
            "normalized_value": self.normalized_value,
            "polarity": self.polarity,
        }


@dataclass(frozen=True, slots=True)
class AxisProjection:
    axis: str
    status: str
    observation_digest: str | None
    original_value: float | None
    original_baseline: float | None
    original_floor: float | None
    original_ceiling: float | None
    normalized_value: float | None
    saturated: bool
    calibrated_confidence: float
    contributions: tuple[DriveContribution, ...]
    appraisals: tuple[str, ...]
    emotions: tuple[str, ...]
    preservation: str
    source_snapshot_id: str | None
    source_schema_version: str | None
    source_integrity_digest: str | None
    source_metadata: tuple[tuple[str, str], ...]
    missing_input: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "appraisals": list(self.appraisals),
            "axis": self.axis,
            "calibrated_confidence": self.calibrated_confidence,
            "contributions": [item.to_mapping() for item in self.contributions],
            "emotions": list(self.emotions),
            "missing_input": self.missing_input,
            "normalized_value": self.normalized_value,
            "observation_digest": self.observation_digest,
            "original_baseline": self.original_baseline,
            "original_ceiling": self.original_ceiling,
            "original_floor": self.original_floor,
            "original_value": self.original_value,
            "preservation": self.preservation,
            "saturated": self.saturated,
            "source_integrity_digest": self.source_integrity_digest,
            "source_metadata": [list(item) for item in self.source_metadata],
            "source_schema_version": self.source_schema_version,
            "source_snapshot_id": self.source_snapshot_id,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticTransitionCandidate:
    candidate_id: str
    transition_id: str
    drive: str
    from_state: str
    to_state: str
    direction: str
    threshold: float
    next_state_epoch: int
    predicate_version: str = PREDICATE_VERSION
    parameter_version: str = PARAMETER_VERSION
    authority: str = SHADOW_AUTHORITY
    diagnostic_only: bool = True
    event_append_authorized: bool = False

    def __post_init__(self) -> None:
        _require_digest(self.candidate_id, "candidate_id")
        if self.drive not in DRIVE_SPECS:
            raise AffectProjectionError("candidate drive is unknown")
        if self.direction not in {"up", "down"}:
            raise AffectProjectionError("candidate direction is invalid")
        if self.authority != SHADOW_AUTHORITY or not self.diagnostic_only or self.event_append_authorized:
            raise AffectProjectionError("M3-B candidate cannot emit or gain authority")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "candidate_id": self.candidate_id,
            "diagnostic_only": self.diagnostic_only,
            "direction": self.direction,
            "drive": self.drive,
            "event_append_authorized": self.event_append_authorized,
            "from_state": self.from_state,
            "next_state_epoch": self.next_state_epoch,
            "parameter_version": self.parameter_version,
            "predicate_version": self.predicate_version,
            "threshold": self.threshold,
            "to_state": self.to_state,
            "transition_id": self.transition_id,
        }


@dataclass(frozen=True, slots=True)
class DriveShadowProjection:
    drive: str
    previous_value: float
    aggregate_input: float
    target_value: float
    relaxed_value: float
    next_value: float
    named_state_retained: str
    state_epoch_retained: int
    contribution_count: int
    total_confidence: float
    slew_limited: bool
    saturated: bool
    pending_candidate_retained: bool
    candidate: DiagnosticTransitionCandidate | None
    schema_version: str = DRIVE_PROJECTION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    named_state_mutated: bool = False
    event_emitted: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "aggregate_input": self.aggregate_input,
            "authority": self.authority,
            "candidate": None if self.candidate is None else self.candidate.to_mapping(),
            "contribution_count": self.contribution_count,
            "drive": self.drive,
            "event_emitted": self.event_emitted,
            "named_state_mutated": self.named_state_mutated,
            "named_state_retained": self.named_state_retained,
            "next_value": self.next_value,
            "pending_candidate_retained": self.pending_candidate_retained,
            "previous_value": self.previous_value,
            "relaxed_value": self.relaxed_value,
            "saturated": self.saturated,
            "schema_version": self.schema_version,
            "slew_limited": self.slew_limited,
            "state_epoch_retained": self.state_epoch_retained,
            "target_value": self.target_value,
            "total_confidence": self.total_confidence,
        }


@dataclass(frozen=True, slots=True)
class ShadowAffectProjection:
    axis_projections: tuple[AxisProjection, ...]
    drive_projections: tuple[DriveShadowProjection, ...]
    elapsed_seconds: int
    mapping_digest: str
    observations_digest: str
    missing_axes: tuple[str, ...]
    proposed_drop_axes: tuple[str, ...]
    schema_version: str = PROJECTION_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    legacy_runtime_authoritative: bool = True
    persistence_accessed: bool = False
    event_append_performed: bool = False
    live_behavior_changed: bool = False
    cutover_authorized: bool = False
    m3_authority_open: bool = False

    def __post_init__(self) -> None:
        if self.authority != SHADOW_AUTHORITY:
            raise AffectProjectionError("projection authority must remain shadow_only")
        if not self.legacy_runtime_authoritative:
            raise AffectProjectionError("legacy runtime must remain authoritative")
        if (
            self.persistence_accessed
            or self.event_append_performed
            or self.live_behavior_changed
            or self.cutover_authorized
            or self.m3_authority_open
        ):
            raise AffectProjectionError("M3-B projection cannot promote or mutate live state")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axis_projections": [item.to_mapping() for item in self.axis_projections],
            "cutover_authorized": self.cutover_authorized,
            "drive_projections": [item.to_mapping() for item in self.drive_projections],
            "elapsed_seconds": self.elapsed_seconds,
            "event_append_performed": self.event_append_performed,
            "legacy_runtime_authoritative": self.legacy_runtime_authoritative,
            "live_behavior_changed": self.live_behavior_changed,
            "m3_authority_open": self.m3_authority_open,
            "mapping_digest": self.mapping_digest,
            "missing_axes": list(self.missing_axes),
            "observations_digest": self.observations_digest,
            "persistence_accessed": self.persistence_accessed,
            "proposed_drop_axes": list(self.proposed_drop_axes),
            "schema_version": self.schema_version,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping(), "m3_b_shadow_projection")


def normalize_observation(observation: AxisObservation) -> tuple[float, bool]:
    if observation.value >= observation.baseline:
        denominator = observation.ceiling - observation.baseline
    else:
        denominator = observation.baseline - observation.floor
    raw = (observation.value - observation.baseline) / denominator
    normalized = _clip(raw, -1.0, 1.0)
    return normalized, not math.isclose(raw, normalized, abs_tol=1e-12)


def _axis_projection(mapping: AxisMapping, observation: AxisObservation | None) -> AxisProjection:
    if observation is None:
        return AxisProjection(
            axis=mapping.axis,
            status=mapping.status,
            observation_digest=None,
            original_value=None,
            original_baseline=None,
            original_floor=None,
            original_ceiling=None,
            normalized_value=None,
            saturated=False,
            calibrated_confidence=0.0,
            contributions=(),
            appraisals=mapping.appraisals,
            emotions=mapping.emotions,
            preservation=mapping.preservation,
            source_snapshot_id=None,
            source_schema_version=None,
            source_integrity_digest=None,
            source_metadata=(),
            missing_input=True,
        )
    if observation.axis != mapping.axis or observation.source_family != mapping.source_family:
        raise AffectProjectionError(f"{mapping.axis}: observation does not match mapping")
    normalized, saturated = normalize_observation(observation)
    calibrated = observation.confidence * mapping.confidence_cap
    contributions = tuple(
        DriveContribution(
            axis=mapping.axis,
            drive=drive,
            normalized_value=normalized,
            polarity=mapping.polarity(drive),
            confidence=calibrated,
            contribution=normalized * mapping.polarity(drive),
        )
        for drive in mapping.target_drives
    )
    return AxisProjection(
        axis=mapping.axis,
        status=mapping.status,
        observation_digest=observation.digest,
        original_value=observation.value,
        original_baseline=observation.baseline,
        original_floor=observation.floor,
        original_ceiling=observation.ceiling,
        normalized_value=normalized,
        saturated=saturated,
        calibrated_confidence=calibrated,
        contributions=contributions,
        appraisals=mapping.appraisals,
        emotions=mapping.emotions,
        preservation=mapping.preservation,
        source_snapshot_id=observation.source_snapshot_id,
        source_schema_version=observation.source_schema_version,
        source_integrity_digest=observation.source_integrity_digest,
        source_metadata=observation.source_metadata,
        missing_input=False,
    )


def _candidate(prior: DriveShadowPrior, next_value: float) -> DiagnosticTransitionCandidate | None:
    if prior.pending_candidate_id is not None:
        return None
    spec = DRIVE_SPECS[prior.drive]
    ordinal = spec.states.index(prior.named_state)
    direction: str | None = None
    destination: str | None = None
    threshold: float | None = None
    cooldown = 0
    if ordinal < 3:
        _down, up, required = spec.boundaries[ordinal]
        if next_value >= up:
            direction, destination, threshold, cooldown = "up", spec.states[ordinal + 1], up, required
    if direction is None and ordinal > 0:
        down, _up, required = spec.boundaries[ordinal - 1]
        if next_value <= down:
            direction, destination, threshold, cooldown = "down", spec.states[ordinal - 1], down, required
    if direction is None or prior.seconds_since_transition < cooldown or destination is None or threshold is None:
        return None
    next_epoch = prior.state_epoch + 1
    transition_id = f"m3a.{prior.drive}.{prior.named_state}_to_{destination}.v1"
    material = "\x1f".join(
        (
            CANDIDATE_SCHEMA_VERSION,
            prior.drive,
            prior.named_state,
            destination,
            str(next_epoch),
            PREDICATE_VERSION,
            PARAMETER_VERSION,
        )
    )
    candidate_id = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return DiagnosticTransitionCandidate(
        candidate_id=candidate_id,
        transition_id=transition_id,
        drive=prior.drive,
        from_state=prior.named_state,
        to_state=destination,
        direction=direction,
        threshold=threshold,
        next_state_epoch=next_epoch,
    )


def _drive_projection(
    prior: DriveShadowPrior,
    contributions: tuple[DriveContribution, ...],
    *,
    elapsed_seconds: int,
) -> DriveShadowProjection:
    spec = DRIVE_SPECS[prior.drive]
    total_confidence = sum(item.confidence for item in contributions)
    numerator = sum(item.confidence * item.contribution for item in contributions)
    aggregate = _clip(numerator / max(1.0, total_confidence), -1.0, 1.0)
    unclipped_target = spec.baseline + spec.gain * aggregate
    target = _clip(unclipped_target, spec.floor, spec.ceiling)
    relaxed = target + (prior.value - target) * math.exp(-elapsed_seconds / spec.tau_seconds)
    delta = relaxed - prior.value
    limit = spec.max_slew_per_second * elapsed_seconds
    bounded_delta = _clip(delta, -limit, limit)
    next_value = _clip(prior.value + bounded_delta, spec.floor, spec.ceiling)
    candidate = _candidate(prior, next_value)
    return DriveShadowProjection(
        drive=prior.drive,
        previous_value=prior.value,
        aggregate_input=aggregate,
        target_value=target,
        relaxed_value=relaxed,
        next_value=next_value,
        named_state_retained=prior.named_state,
        state_epoch_retained=prior.state_epoch,
        contribution_count=len(contributions),
        total_confidence=total_confidence,
        slew_limited=not math.isclose(delta, bounded_delta, abs_tol=1e-12),
        saturated=not math.isclose(unclipped_target, target, abs_tol=1e-12),
        pending_candidate_retained=prior.pending_candidate_id is not None,
        candidate=candidate,
    )


def project_shadow_affect(
    *,
    mappings: Iterable[AxisMapping],
    observations: Iterable[AxisObservation],
    priors: Iterable[DriveShadowPrior],
    elapsed_seconds: int,
    strict: bool = True,
) -> ShadowAffectProjection:
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, int) or elapsed_seconds < 0:
        raise AffectProjectionError("elapsed_seconds must be a non-negative integer")
    mapping_rows = tuple(mappings)
    observation_rows = tuple(observations)
    prior_rows = tuple(priors)
    if len({row.axis for row in mapping_rows}) != len(mapping_rows):
        raise AffectProjectionError("mapping axes must be unique")
    if len({row.axis for row in observation_rows}) != len(observation_rows):
        raise AffectProjectionError("observation axes must be unique")
    if tuple(row.drive for row in prior_rows) != ALLOWED_DRIVES:
        raise AffectProjectionError("priors must contain all eight drives in canonical order")
    mapping_by_axis = {row.axis: row for row in mapping_rows}
    unknown = sorted(row.axis for row in observation_rows if row.axis not in mapping_by_axis)
    if unknown:
        raise AffectProjectionError(f"observations contain unknown axes: {unknown}")
    observation_by_axis = {row.axis: row for row in observation_rows}
    missing = tuple(row.axis for row in mapping_rows if row.axis not in observation_by_axis)
    if strict and missing:
        raise AffectProjectionError(f"strict projection missing axes: {list(missing)}")
    axis_rows = tuple(_axis_projection(row, observation_by_axis.get(row.axis)) for row in mapping_rows)
    by_drive: dict[str, list[DriveContribution]] = {drive: [] for drive in ALLOWED_DRIVES}
    for row in axis_rows:
        by_drive_values = row.contributions
        for contribution in by_drive_values:
            by_drive[contribution.drive].append(contribution)
    drive_rows = tuple(
        _drive_projection(
            prior,
            tuple(by_drive[prior.drive]),
            elapsed_seconds=elapsed_seconds,
        )
        for prior in prior_rows
    )
    mapping_digest = _digest(
        {"mappings": [row.to_mapping() for row in mapping_rows], "schema_version": MAPPING_SCHEMA_VERSION},
        "m3_b_mapping_catalog",
    )
    observations_digest = _digest(
        {"observations": [row.to_mapping() for row in observation_rows], "schema_version": OBSERVATION_SCHEMA_VERSION},
        "m3_b_observations",
    )
    return ShadowAffectProjection(
        axis_projections=axis_rows,
        drive_projections=drive_rows,
        elapsed_seconds=elapsed_seconds,
        mapping_digest=mapping_digest,
        observations_digest=observations_digest,
        missing_axes=missing,
        proposed_drop_axes=tuple(row.axis for row in mapping_rows if row.status == "PROPOSED-DROP"),
    )
