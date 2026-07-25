"""M3-B retained-real-observation capture preflight for all 37 registry axes.

This module closes no production capability.  It proves that all seven source-
binding groups cover the canonical 37-axis registry while the two production
components required to retain real observations are still absent.  Therefore
it cannot claim retained observations, positive-confidence real coverage, an
observation-window transition, M3-B completion, M3-C, cutover, or M3-E
runtime authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from core.event_kernel import SHADOW_AUTHORITY
from core.m3_b_agp_bounded_expression_action_source_binding import (
    EXPRESSION_ACTION_AXES,
    agp_bounded_expression_action_source_bindings,
)
from core.m3_b_appraised_survival_source_binding import (
    APPRAISED_SURVIVAL_AXES,
    appraised_survival_source_bindings,
)
from core.m3_b_long_horizon_self_identity_source_binding import (
    SELF_IDENTITY_AXES,
    long_horizon_self_identity_source_bindings,
)
from core.m3_b_operational_registry_source_binding import (
    OPERATIONAL_AXES,
    operational_registry_source_bindings,
)
from core.m3_b_quarantined_risk_source_binding import (
    RISK_DEFENSE_AXES,
    quarantined_risk_source_bindings,
)
from core.m3_b_quarantined_social_source_binding import (
    SOCIAL_RELATIONSHIP_AXES,
    quarantined_social_source_bindings,
)
from core.m3_b_registry_affect_owner import REGISTRY_AXIS_ORDER
from core.m3_b_validated_learning_source_binding import (
    LEARNING_EXPLORATION_AXES,
    validated_learning_source_bindings,
)

SCHEMA_VERSION = "eve.m3-b.retained-real-observation-capture-preflight.v1"
GROUP_SCHEMA_VERSION = "eve.m3-b.source-binding-coverage-group.v1"
COMPONENT_SCHEMA_VERSION = "eve.m3-b.required-production-capture-component.v1"
TOTAL_AXIS_COUNT = 37
RETAINED_REAL_OBSERVATION_CAPTURE_BLOCKER = (
    "REGISTRY_RETAINED_REAL_OBSERVATION_CAPTURE_ABSENT"
)
POSITIVE_CONFIDENCE_COVERAGE_BLOCKER = (
    "REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE"
)
OBSERVATION_WINDOW_NOT_STARTED_BLOCKER = "REGISTRY_OBSERVATION_WINDOW_NOT_STARTED"
PRODUCTION_CAPTURE_COMPONENT_ID = "registry_37_axis_production_capture_adapter"
RETENTION_SINK_COMPONENT_ID = "registry_immutable_retained_real_observation_sink"
PRODUCTION_CAPTURE_FUTURE_PATH = "core/m3_b_registry_production_capture_adapter.py"
RETENTION_SINK_FUTURE_PATH = "core/m3_b_registry_retained_real_observation_sink.py"


class RetainedRealObservationCapturePreflightError(ValueError):
    """Raised when the preflight attempts to overstate production authority."""


def _canonical(value: Mapping[str, Any], field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RetainedRealObservationCapturePreflightError(
            f"{field} is not canonical JSON"
        ) from exc


def _digest(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(_canonical(value, field).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceBindingCoverageGroup:
    group_id: str
    axes: tuple[str, ...]
    group_binding_count: int
    cumulative_bound_axis_count: int
    binding_set_digest: str
    schema_version: str = GROUP_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    production_capture_present: bool = False
    retained_real_observation_count: int = 0
    observation_window_started: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise RetainedRealObservationCapturePreflightError(
                "coverage group id must be non-empty"
            )
        axes = tuple(self.axes)
        if self.group_binding_count != len(axes) or len(set(axes)) != len(axes):
            raise RetainedRealObservationCapturePreflightError(
                "coverage group count must match unique axes"
            )
        if not 0 < self.cumulative_bound_axis_count <= TOTAL_AXIS_COUNT:
            raise RetainedRealObservationCapturePreflightError(
                "coverage cumulative count is outside the 37-axis registry"
            )
        if (
            not isinstance(self.binding_set_digest, str)
            or len(self.binding_set_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.binding_set_digest)
        ):
            raise RetainedRealObservationCapturePreflightError(
                "coverage group requires a lowercase SHA-256 binding-set digest"
            )
        if self.schema_version != GROUP_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RetainedRealObservationCapturePreflightError(
                "coverage group must remain exact shadow-only preflight evidence"
            )
        if (
            self.production_capture_present
            or self.retained_real_observation_count != 0
            or self.observation_window_started
        ):
            raise RetainedRealObservationCapturePreflightError(
                "source-binding coverage cannot claim production capture or retained observation"
            )
        object.__setattr__(self, "axes", axes)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "axes": list(self.axes),
            "binding_set_digest": self.binding_set_digest,
            "cumulative_bound_axis_count": self.cumulative_bound_axis_count,
            "group_binding_count": self.group_binding_count,
            "group_id": self.group_id,
            "observation_window_started": self.observation_window_started,
            "production_capture_present": self.production_capture_present,
            "retained_real_observation_count": self.retained_real_observation_count,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class RequiredProductionCaptureComponent:
    component_id: str
    future_path: str
    responsibility: str
    present: bool = False
    schema_version: str = COMPONENT_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    installed: bool = False
    enabled: bool = False

    def __post_init__(self) -> None:
        for field in ("component_id", "future_path", "responsibility"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise RetainedRealObservationCapturePreflightError(
                    f"{field} must be non-empty"
                )
        if self.schema_version != COMPONENT_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RetainedRealObservationCapturePreflightError(
                "required production component must remain preflight-only"
            )
        if self.present or self.installed or self.enabled:
            raise RetainedRealObservationCapturePreflightError(
                "this preflight cannot claim a production capture component is present"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "component_id": self.component_id,
            "enabled": self.enabled,
            "future_path": self.future_path,
            "installed": self.installed,
            "present": self.present,
            "responsibility": self.responsibility,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class RetainedRealObservationCapturePreflight:
    source_binding_groups: tuple[SourceBindingCoverageGroup, ...]
    required_production_components: tuple[RequiredProductionCaptureComponent, ...]
    schema_version: str = SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    source_binding_count: int = TOTAL_AXIS_COUNT
    source_binding_complete: bool = True
    production_capture_adapter_present: bool = False
    retention_sink_present: bool = False
    retained_real_observation_count: int = 0
    positive_confidence_real_observation_count: int = 0
    observation_window_eligible: bool = False
    observation_window_started: bool = False
    observation_window_satisfied: bool = False
    runtime_hook_installed: bool = False
    scheduler_installed: bool = False
    persistence_accessed: bool = False
    event_append_performed: bool = False
    registry_owner_mutated: bool = False
    live_affect_mutated: bool = False
    live_drive_mutated: bool = False
    named_state_mutated: bool = False
    goal_memory_self_expression_mutated: bool = False
    m3_b_complete: bool = False
    m3_c_open: bool = False
    m3_e_authority_open: bool = False
    cutover_authorized: bool = False

    def __post_init__(self) -> None:
        groups = tuple(self.source_binding_groups)
        components = tuple(self.required_production_components)
        if len(groups) != 7:
            raise RetainedRealObservationCapturePreflightError(
                "preflight requires all seven source-binding groups"
            )
        axes = tuple(axis for group in groups for axis in group.axes)
        if axes != REGISTRY_AXIS_ORDER or len(set(axes)) != TOTAL_AXIS_COUNT:
            raise RetainedRealObservationCapturePreflightError(
                "source-binding groups must cover canonical 37-axis order exactly once"
            )
        cumulative = tuple(group.cumulative_bound_axis_count for group in groups)
        if cumulative != (4, 6, 12, 19, 25, 31, 37):
            raise RetainedRealObservationCapturePreflightError(
                "source-binding cumulative coverage must be exact 4/6/12/19/25/31/37"
            )
        if self.source_binding_count != TOTAL_AXIS_COUNT or self.source_binding_complete is not True:
            raise RetainedRealObservationCapturePreflightError(
                "preflight must preserve exact 37/37 source-binding coverage"
            )
        if (
            len(components) != 2
            or tuple(component.component_id for component in components)
            != (PRODUCTION_CAPTURE_COMPONENT_ID, RETENTION_SINK_COMPONENT_ID)
            or tuple(component.future_path for component in components)
            != (PRODUCTION_CAPTURE_FUTURE_PATH, RETENTION_SINK_FUTURE_PATH)
        ):
            raise RetainedRealObservationCapturePreflightError(
                "preflight must enumerate exact future capture adapter and immutable retention sink"
            )
        if self.schema_version != SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise RetainedRealObservationCapturePreflightError(
                "retained-real-observation preflight must remain shadow-only"
            )
        if self.production_capture_adapter_present or self.retention_sink_present:
            raise RetainedRealObservationCapturePreflightError(
                "this preflight cannot claim production capture or retention components exist"
            )
        if self.retained_real_observation_count != 0 or self.positive_confidence_real_observation_count != 0:
            raise RetainedRealObservationCapturePreflightError(
                "preflight cannot fabricate retained real observations or positive-confidence coverage"
            )
        if self.observation_window_eligible or self.observation_window_started or self.observation_window_satisfied:
            raise RetainedRealObservationCapturePreflightError(
                "observation window cannot become eligible or start before retained real coverage"
            )
        if any(
            (
                self.runtime_hook_installed,
                self.scheduler_installed,
                self.persistence_accessed,
                self.event_append_performed,
                self.registry_owner_mutated,
                self.live_affect_mutated,
                self.live_drive_mutated,
                self.named_state_mutated,
                self.goal_memory_self_expression_mutated,
                self.m3_b_complete,
                self.m3_c_open,
                self.m3_e_authority_open,
                self.cutover_authorized,
            )
        ):
            raise RetainedRealObservationCapturePreflightError(
                "preflight cannot grant runtime mutation, persistence, window, cutover, or authority"
            )
        object.__setattr__(self, "source_binding_groups", groups)
        object.__setattr__(self, "required_production_components", components)

    @property
    def blockers(self) -> tuple[str, ...]:
        return (
            RETAINED_REAL_OBSERVATION_CAPTURE_BLOCKER,
            POSITIVE_CONFIDENCE_COVERAGE_BLOCKER,
            OBSERVATION_WINDOW_NOT_STARTED_BLOCKER,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "blockers": list(self.blockers),
            "cutover_authorized": self.cutover_authorized,
            "event_append_performed": self.event_append_performed,
            "goal_memory_self_expression_mutated": self.goal_memory_self_expression_mutated,
            "live_affect_mutated": self.live_affect_mutated,
            "live_drive_mutated": self.live_drive_mutated,
            "m3_b_complete": self.m3_b_complete,
            "m3_c_open": self.m3_c_open,
            "m3_e_authority_open": self.m3_e_authority_open,
            "named_state_mutated": self.named_state_mutated,
            "observation_window_eligible": self.observation_window_eligible,
            "observation_window_satisfied": self.observation_window_satisfied,
            "observation_window_started": self.observation_window_started,
            "persistence_accessed": self.persistence_accessed,
            "positive_confidence_real_observation_count": self.positive_confidence_real_observation_count,
            "production_capture_adapter_present": self.production_capture_adapter_present,
            "registry_owner_mutated": self.registry_owner_mutated,
            "required_production_components": [item.to_mapping() for item in self.required_production_components],
            "retained_real_observation_count": self.retained_real_observation_count,
            "retention_sink_present": self.retention_sink_present,
            "runtime_hook_installed": self.runtime_hook_installed,
            "scheduler_installed": self.scheduler_installed,
            "schema_version": self.schema_version,
            "source_binding_complete": self.source_binding_complete,
            "source_binding_count": self.source_binding_count,
            "source_binding_groups": [group.to_mapping() for group in self.source_binding_groups],
        }

    @property
    def preflight_digest(self) -> str:
        return _digest(self.to_mapping(), "retained_real_observation_capture_preflight")


def _coverage_group(
    group_id: str,
    axes: tuple[str, ...],
    binding_set: Any,
    cumulative: int,
) -> SourceBindingCoverageGroup:
    bindings = tuple(binding_set.bindings)
    if tuple(item.axis for item in bindings) != axes:
        raise RetainedRealObservationCapturePreflightError(
            f"{group_id} binding axes do not match their canonical group"
        )
    if len(bindings) != len(axes):
        raise RetainedRealObservationCapturePreflightError(
            f"{group_id} binding count does not match canonical group"
        )
    if getattr(binding_set, "production_capture_present", False):
        raise RetainedRealObservationCapturePreflightError(
            f"{group_id} unexpectedly claims production capture"
        )
    if getattr(binding_set, "observation_window_started", False):
        raise RetainedRealObservationCapturePreflightError(
            f"{group_id} unexpectedly claims an observation window"
        )
    return SourceBindingCoverageGroup(
        group_id=group_id,
        axes=axes,
        group_binding_count=len(bindings),
        cumulative_bound_axis_count=cumulative,
        binding_set_digest=binding_set.binding_set_digest,
    )


def retained_real_observation_capture_preflight() -> RetainedRealObservationCapturePreflight:
    """Return the current fail-closed preflight; no production capture is installed."""

    groups = (
        _coverage_group(
            "operational",
            OPERATIONAL_AXES,
            operational_registry_source_bindings(),
            4,
        ),
        _coverage_group(
            "appraised_survival",
            APPRAISED_SURVIVAL_AXES,
            appraised_survival_source_bindings(),
            6,
        ),
        _coverage_group(
            "quarantined_risk_defense",
            RISK_DEFENSE_AXES,
            quarantined_risk_source_bindings(),
            12,
        ),
        _coverage_group(
            "quarantined_social_relationship",
            SOCIAL_RELATIONSHIP_AXES,
            quarantined_social_source_bindings(),
            19,
        ),
        _coverage_group(
            "validated_learning_exploration",
            LEARNING_EXPLORATION_AXES,
            validated_learning_source_bindings(),
            25,
        ),
        _coverage_group(
            "long_horizon_self_identity",
            SELF_IDENTITY_AXES,
            long_horizon_self_identity_source_bindings(),
            31,
        ),
        _coverage_group(
            "agp_bounded_expression_action",
            EXPRESSION_ACTION_AXES,
            agp_bounded_expression_action_source_bindings(),
            37,
        ),
    )
    components = (
        RequiredProductionCaptureComponent(
            component_id=PRODUCTION_CAPTURE_COMPONENT_ID,
            future_path=PRODUCTION_CAPTURE_FUTURE_PATH,
            responsibility=(
                "Acquire exact verified source records from production-owned sources "
                "without deriving them from registry defaults, proposals, or synthetic fixtures."
            ),
        ),
        RequiredProductionCaptureComponent(
            component_id=RETENTION_SINK_COMPONENT_ID,
            future_path=RETENTION_SINK_FUTURE_PATH,
            responsibility=(
                "Append immutable retained-real-observation envelopes with source, raw, "
                "verification, and retention integrity sufficient for deterministic replay."
            ),
        ),
    )
    return RetainedRealObservationCapturePreflight(
        source_binding_groups=groups,
        required_production_components=components,
    )
