"""Read-only M3-B bridge for the existing ActiveInference prediction-error trace.

This module does not install a runtime hook and does not register a production source
verifier. It can inspect an already-completed AIAdapter/ActiveInference prediction +
error pair, freeze that material into an immutable source snapshot, and convert the
snapshot into the existing detached ``prediction_error_pressure`` source-binding
record. The resulting detached evidence is not a production observation by itself.

A later PR must supply a trusted production-runtime provenance anchor and the reviewed
production verifier registration before this bridge can contribute retained real
observation coverage.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_registry_observation_evidence import RegistryAxisPositiveConfidenceEvidence
from core.m3_b_validated_learning_source_binding import (
    ValidatedLearningRawRecord,
    derive_validated_learning_axis_evidence,
    validated_learning_raw_observation_digest,
)

AXIS = "prediction_error_pressure"
BRIDGE_SCHEMA_VERSION = "eve.m3-b.prediction-error-runtime-source-bridge.v1"
SOURCE_SCHEMA_VERSION = "eve.m3-b.prediction-error-runtime-source-snapshot.v1"
MODEL_VERSION = "eve.active-inference.prediction-error-trace.v1"
VERIFICATION_STATUS = "verified"
MOOD_ERROR_MAX = 3.0
CAPABILITY_SCHEMA_VERSION = "eve.m3-b.prediction-error-runtime-source-bridge-capability.v1"
RUNTIME_PROVENANCE_BLOCKER = "PREDICTION_ERROR_PRODUCTION_RUNTIME_PROVENANCE_ANCHOR_ABSENT"
PRODUCTION_VERIFIER_BLOCKER = "REGISTRY_PRODUCTION_SOURCE_VERIFIER_COVERAGE_INCOMPLETE"
POSITIVE_CONFIDENCE_BLOCKER = "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
OBSERVATION_WINDOW_BLOCKER = "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED"


class PredictionErrorRuntimeSourceBridgeError(ValueError):
    """Raised when runtime trace material cannot satisfy the exact bridge contract."""


def _identifier(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PredictionErrorRuntimeSourceBridgeError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PredictionErrorRuntimeSourceBridgeError(
            f"{field} must be a non-negative integer"
        )
    return value


def _finite(value: Any, field: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PredictionErrorRuntimeSourceBridgeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise PredictionErrorRuntimeSourceBridgeError(
            f"{field} must be finite and inside [{lower},{upper}]"
        )
    return result


def _canonical(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PredictionErrorRuntimeSourceBridgeError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Any, field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


def _freeze_mapping(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise PredictionErrorRuntimeSourceBridgeError(f"{field} must be a mapping")
    return _canonical(dict(value), field)


def _thaw_mapping(value: str, field: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PredictionErrorRuntimeSourceBridgeError(
            f"{field} is not canonical JSON object material"
        ) from exc
    if not isinstance(result, dict) or _canonical(result, field) != value:
        raise PredictionErrorRuntimeSourceBridgeError(
            f"{field} must remain a canonical JSON object"
        )
    return result


@dataclass(frozen=True, slots=True)
class PredictionErrorRuntimeSourceSnapshot:
    """Immutable copy of one already-completed ActiveInference observation trace."""

    source_instance_id: str
    logical_tick: int
    prediction_id: str
    prediction_json: str
    error_json: str
    schema_version: str = SOURCE_SCHEMA_VERSION
    bridge_schema_version: str = BRIDGE_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    runtime_source_read_only: bool = True
    production_origin_verified: bool = False
    fixture_only: bool = False
    production_verifier_registered: bool = False
    retained_real_observation: bool = False

    def __post_init__(self) -> None:
        _identifier(self.source_instance_id, "source_instance_id")
        _nonnegative_int(self.logical_tick, "logical_tick")
        _identifier(self.prediction_id, "prediction_id")
        prediction = _thaw_mapping(self.prediction_json, "prediction_json")
        error = _thaw_mapping(self.error_json, "error_json")
        if prediction.get("id") != self.prediction_id:
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction id does not match immutable source snapshot"
            )
        if prediction.get("observed") is not True:
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction-error source requires an already-observed prediction"
            )
        if error.get("prediction_id") != self.prediction_id:
            raise PredictionErrorRuntimeSourceBridgeError(
                "error record does not match the observed prediction"
            )
        expected_state = prediction.get("expected_state")
        if not isinstance(expected_state, Mapping):
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction expected_state is absent"
            )
        mood = expected_state.get("mood")
        if not isinstance(mood, Mapping):
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction expected mood is absent"
            )
        _finite(mood.get("valence"), "expected mood valence", lower=-1.0, upper=1.0)
        _finite(mood.get("arousal"), "expected mood arousal", lower=0.0, upper=1.0)
        _finite(error.get("mood_error"), "mood_error", lower=0.0, upper=MOOD_ERROR_MAX)
        _finite(error.get("pred_confidence"), "pred_confidence", lower=0.0, upper=1.0)
        if not isinstance(error.get("hormone_errors"), Mapping):
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction error hormone_errors are absent"
            )
        if self.schema_version != SOURCE_SCHEMA_VERSION:
            raise PredictionErrorRuntimeSourceBridgeError(
                "unsupported prediction-error source snapshot schema"
            )
        if self.bridge_schema_version != BRIDGE_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction-error source snapshot must remain exact shadow-only bridge material"
            )
        if self.runtime_source_read_only is not True:
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction-error bridge source read must remain read-only"
            )
        if any(
            (
                self.production_origin_verified,
                self.production_verifier_registered,
                self.retained_real_observation,
            )
        ):
            raise PredictionErrorRuntimeSourceBridgeError(
                "runtime bridge snapshot cannot claim production verification or retention"
            )

    @property
    def prediction(self) -> dict[str, Any]:
        return _thaw_mapping(self.prediction_json, "prediction_json")

    @property
    def error(self) -> dict[str, Any]:
        return _thaw_mapping(self.error_json, "error_json")

    @property
    def normalized_error(self) -> float:
        return float(self.error["mood_error"] / MOOD_ERROR_MAX)

    @property
    def predicted_value_digest(self) -> str:
        prediction = self.prediction
        return _digest(
            {
                "confidence": prediction.get("confidence"),
                "expected_outcome": prediction.get("expected_outcome"),
                "expected_state": prediction.get("expected_state"),
                "horizon": prediction.get("horizon"),
                "prediction_id": self.prediction_id,
            },
            "prediction_error_predicted_value",
        )

    @property
    def observed_value_digest(self) -> str:
        error = self.error
        return _digest(
            {
                "actual_outcome": error.get("actual_outcome"),
                "hormone_errors": error.get("hormone_errors"),
                "mood_error": error.get("mood_error"),
                "outcome_match": error.get("outcome_match"),
                "prediction_id": self.prediction_id,
            },
            "prediction_error_observed_value",
        )

    @property
    def source_integrity_digest(self) -> str:
        return _digest(
            {
                "bridge_schema_version": self.bridge_schema_version,
                "error": self.error,
                "logical_tick": self.logical_tick,
                "prediction": self.prediction,
                "prediction_id": self.prediction_id,
                "source_instance_id": self.source_instance_id,
                "source_schema_version": self.schema_version,
            },
            "prediction_error_runtime_source_snapshot",
        )

    @property
    def source_snapshot_id(self) -> str:
        return (
            f"prediction-error-runtime:{self.logical_tick}:{self.prediction_id}:"
            f"{self.source_integrity_digest[:16]}"
        )

    @property
    def raw_values(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("model_version", MODEL_VERSION),
            ("normalized_error", self.normalized_error),
            ("observed_value_digest", self.observed_value_digest),
            ("predicted_value_digest", self.predicted_value_digest),
            ("verification_status", VERIFICATION_STATUS),
        )

    def to_validated_learning_raw_record(self) -> ValidatedLearningRawRecord:
        validation_input_digest = self.source_integrity_digest
        validation_integrity_digest = _digest(
            {
                "axis": AXIS,
                "input_digest": validation_input_digest,
                "method": "exact_active_inference_prediction_error_trace_validation.v1",
                "outcome": "verified",
                "raw_values": self.raw_values,
            },
            "prediction_error_validation_trace",
        )
        appraisal_input_digest = validation_integrity_digest
        appraisal_integrity_digest = _digest(
            {
                "axis": AXIS,
                "input_digest": appraisal_input_digest,
                "method": "bounded_prediction_error_pressure_appraisal.v1",
                "normalized_error": self.normalized_error,
                "outcome": "verified",
            },
            "prediction_error_appraisal_trace",
        )
        observation_id = (
            f"prediction-error-runtime:{self.logical_tick}:"
            f"{self.source_integrity_digest[:24]}"
        )
        validation_trace_id = (
            f"prediction-error-validation:{self.logical_tick}:"
            f"{validation_integrity_digest[:16]}"
        )
        appraisal_trace_id = (
            f"prediction-error-appraisal:{self.logical_tick}:"
            f"{appraisal_integrity_digest[:16]}"
        )
        raw_observation_digest = validated_learning_raw_observation_digest(
            axis=AXIS,
            logical_tick=self.logical_tick,
            observation_id=observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.schema_version,
            source_integrity_digest=self.source_integrity_digest,
            validation_trace_id=validation_trace_id,
            validation_input_digest=validation_input_digest,
            validation_integrity_digest=validation_integrity_digest,
            appraisal_trace_id=appraisal_trace_id,
            appraisal_input_digest=appraisal_input_digest,
            appraisal_integrity_digest=appraisal_integrity_digest,
            raw_values=self.raw_values,
        )
        return ValidatedLearningRawRecord(
            axis=AXIS,
            logical_tick=self.logical_tick,
            observation_id=observation_id,
            source_instance_id=self.source_instance_id,
            source_snapshot_id=self.source_snapshot_id,
            source_schema_version=self.schema_version,
            source_integrity_digest=self.source_integrity_digest,
            validation_trace_id=validation_trace_id,
            validation_input_digest=validation_input_digest,
            validation_integrity_digest=validation_integrity_digest,
            appraisal_trace_id=appraisal_trace_id,
            appraisal_input_digest=appraisal_input_digest,
            appraisal_integrity_digest=appraisal_integrity_digest,
            raw_observation_digest=raw_observation_digest,
            raw_values=self.raw_values,
        )


def prediction_error_runtime_snapshot_from_mappings(
    *,
    prediction: Mapping[str, Any],
    error: Mapping[str, Any],
    observe_count: int,
    source_instance_id: str,
    fixture_only: bool = False,
) -> PredictionErrorRuntimeSourceSnapshot:
    """Freeze one completed runtime-shaped prediction/error pair without mutation."""

    _nonnegative_int(observe_count, "observe_count")
    if observe_count == 0:
        raise PredictionErrorRuntimeSourceBridgeError(
            "prediction-error source requires a completed observation count"
        )
    prediction_id = _identifier(prediction.get("id"), "prediction id")
    return PredictionErrorRuntimeSourceSnapshot(
        source_instance_id=_identifier(source_instance_id, "source_instance_id"),
        logical_tick=observe_count,
        prediction_id=prediction_id,
        prediction_json=_freeze_mapping(prediction, "prediction"),
        error_json=_freeze_mapping(error, "error"),
        fixture_only=bool(fixture_only),
    )


def read_prediction_error_runtime_source(
    ai_adapter: Any,
    *,
    source_instance_id: str,
    fixture_only: bool = False,
) -> PredictionErrorRuntimeSourceSnapshot | None:
    """Read the already-existing latest ActiveInference trace; never call predict/observe/tick."""

    prediction = getattr(ai_adapter, "_last_prediction", None)
    active_inference = getattr(ai_adapter, "ai", None)
    if prediction is None or active_inference is None:
        return None
    errors = getattr(active_inference, "errors", None)
    observe_count = getattr(active_inference, "observe_count", None)
    if errors is None or observe_count is None or len(errors) == 0:
        return None
    error = errors[-1]
    if not isinstance(prediction, Mapping) or not isinstance(error, Mapping):
        raise PredictionErrorRuntimeSourceBridgeError(
            "runtime prediction/error trace must use mapping records"
        )
    return prediction_error_runtime_snapshot_from_mappings(
        prediction=prediction,
        error=error,
        observe_count=observe_count,
        source_instance_id=source_instance_id,
        fixture_only=fixture_only,
    )


def derive_detached_prediction_error_evidence(
    snapshots: Sequence[PredictionErrorRuntimeSourceSnapshot],
) -> RegistryAxisPositiveConfidenceEvidence:
    """Derive existing detached evidence; this is not production verification."""

    items = tuple(snapshots)
    if not items or any(type(item) is not PredictionErrorRuntimeSourceSnapshot for item in items):
        raise PredictionErrorRuntimeSourceBridgeError(
            "snapshots must contain exact immutable prediction-error source snapshots"
        )
    if len({item.source_instance_id for item in items}) != 1:
        raise PredictionErrorRuntimeSourceBridgeError(
            "prediction-error snapshots must share one source instance"
        )
    if any(item.production_origin_verified or item.retained_real_observation for item in items):
        raise PredictionErrorRuntimeSourceBridgeError(
            "detached bridge evidence cannot contain production/retention claims"
        )
    records = tuple(item.to_validated_learning_raw_record() for item in items)
    return derive_validated_learning_axis_evidence(records)


@dataclass(frozen=True, slots=True)
class PredictionErrorRuntimeSourceBridgeCapability:
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    axis: str = AXIS
    source_bridge_present: bool = True
    runtime_hook_installed: bool = False
    trusted_production_runtime_provenance_present: bool = False
    production_source_verifier_registered: bool = False
    retained_real_observation_count: int = 0
    positive_confidence_real_observation_count: int = 0
    observation_window_eligible: bool = False
    observation_window_started: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise PredictionErrorRuntimeSourceBridgeError(
                "unsupported prediction-error bridge capability schema"
            )
        if self.axis != AXIS or self.source_bridge_present is not True:
            raise PredictionErrorRuntimeSourceBridgeError(
                "prediction-error bridge capability identity is invalid"
            )
        if any(
            (
                self.runtime_hook_installed,
                self.trusted_production_runtime_provenance_present,
                self.production_source_verifier_registered,
                bool(self.retained_real_observation_count),
                bool(self.positive_confidence_real_observation_count),
                self.observation_window_eligible,
                self.observation_window_started,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise PredictionErrorRuntimeSourceBridgeError(
                "bridge preflight cannot claim installation, production provenance, retention, window, or authority"
            )

    @property
    def blockers(self) -> tuple[str, ...]:
        return (
            RUNTIME_PROVENANCE_BLOCKER,
            PRODUCTION_VERIFIER_BLOCKER,
            POSITIVE_CONFIDENCE_BLOCKER,
            OBSERVATION_WINDOW_BLOCKER,
        )


def prediction_error_runtime_source_bridge_capability() -> PredictionErrorRuntimeSourceBridgeCapability:
    return PredictionErrorRuntimeSourceBridgeCapability()
