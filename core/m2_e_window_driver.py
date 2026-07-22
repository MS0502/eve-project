"""M2-E observation-window contracts for CI chaos and phone habitat tiers.

The module is pure contract/state logic. Import and construction perform no I/O,
start no thread, install no runtime hook, and grant no cutover or M3 authority.
The legacy runtime remains authoritative; all persisted window data is shadow-only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Mapping

from core.event_kernel import SHADOW_AUTHORITY, canonical_json_object

WINDOW_SCHEMA_VERSION = "eve.m2-e-window-state.v1"
WINDOW_CONFIG_SCHEMA_VERSION = "eve.m2-e-window-config.v1"
WINDOW_EVIDENCE_SCHEMA_VERSION = "eve.m2-e-window-evidence.v1"
WINDOW_BASELINE_SHA = "50a448961c8333f788b7f78fe4886cbdf7a0694e"
ACCEPTED_M2_E_HEAD = "6af18fa645a19576caa74d2f8fc8a7fee5baa139"
ACCEPTED_M2_E_PACKET_DIGEST = "fa657687cc3799e6655d5750fc75438c72b6c86e73836ffc6afde2a043f1987d"
ACCEPTED_M2_E_DECISION_DIGEST = "1c2575c7ea2b6c0b8717b6f8f49da634c1f6dfa63a4bf151b6d75e2f154a2a6a"
BOUNDED_STREAM = "shadow:legacy.activation.learn_pair"
STATE_SCHEMA_VERSION = "eve.shadow-projection.activation-learn-pair.v1"

# M2-C proves the bounded mapping is one persistent event per accepted discrete
# learn_pair call; it does not provide a wall-clock rate. The proposal fixes a
# conservative scripted workload of one discrete call every five runtime minutes.
M2_C_MEASURED_EVENTS_PER_DISCRETE_CALL = 1.0
SCRIPTED_STIMULI_PER_SIM_HOUR = 12
EVENT_CAP_PER_SIM_HOUR = 12
WINDOW_EVENT_QUOTA = 288
SIM_HOUR_SECONDS = 3600
CIRCADIAN_SIM_HOURS = 24
CIRCADIAN_RUNTIME_SECONDS = CIRCADIAN_SIM_HOURS * SIM_HOUR_SECONDS
TARGET_DAYS = 5
MAX_DAYS = 7
REQUIRED_ACTUAL_MIDNIGHTS = 3
DISK_BUDGET_BYTES = 512 * 1024 * 1024
CHAOS_KILL_REPETITIONS = 3
CHAOS_PHASES = ("idle", "mid-write", "mid-snapshot", "mid-consolidation")
HALT_REASONS = (
    "recovery_digest_mismatch",
    "unrecoverable_corruption",
    "disk_budget_exceeded",
    "a9_event_cap_exceeded",
    "a9_tick_sampling_detected",
    "unauthorized_effect_detected",
    "companion_git_exclusion_failed",
    "private_artifact_boundary_failed",
)


class WindowContractError(ValueError):
    """Raised when window evidence or state violates the fixed contract."""


def _digest(value: Mapping[str, Any], field: str) -> str:
    text = canonical_json_object(value, field=field)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_digest(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WindowContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WindowContractError(f"{field} must be a non-negative integer")
    return value


def _require_date(value: str, field: str) -> date:
    if not isinstance(value, str):
        raise WindowContractError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise WindowContractError(f"{field} must be an ISO date") from exc


@dataclass(frozen=True, slots=True)
class WindowConfig:
    event_quota: int = WINDOW_EVENT_QUOTA
    stimuli_per_sim_hour: int = SCRIPTED_STIMULI_PER_SIM_HOUR
    event_cap_per_sim_hour: int = EVENT_CAP_PER_SIM_HOUR
    target_days: int = TARGET_DAYS
    max_days: int = MAX_DAYS
    required_actual_midnights: int = REQUIRED_ACTUAL_MIDNIGHTS
    circadian_runtime_seconds: int = CIRCADIAN_RUNTIME_SECONDS
    disk_budget_bytes: int = DISK_BUDGET_BYTES
    kill_repetitions: int = CHAOS_KILL_REPETITIONS
    schema_version: str = WINDOW_CONFIG_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    cutover_authorized: bool = False
    m3_authority_open: bool = False
    legacy_runtime_authoritative: bool = True

    def __post_init__(self) -> None:
        for field in (
            "event_quota",
            "stimuli_per_sim_hour",
            "event_cap_per_sim_hour",
            "target_days",
            "max_days",
            "required_actual_midnights",
            "circadian_runtime_seconds",
            "disk_budget_bytes",
            "kill_repetitions",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise WindowContractError(f"{field} must be a positive integer")
        if self.schema_version != WINDOW_CONFIG_SCHEMA_VERSION:
            raise WindowContractError("unsupported window config schema")
        if self.authority != SHADOW_AUTHORITY:
            raise WindowContractError("window authority must remain shadow_only")
        if self.cutover_authorized or self.m3_authority_open:
            raise WindowContractError("window configuration cannot grant promotion authority")
        if not self.legacy_runtime_authoritative:
            raise WindowContractError("legacy runtime must remain authoritative")
        if self.stimuli_per_sim_hour > self.event_cap_per_sim_hour:
            raise WindowContractError("scripted rate exceeds A9 event cap")
        if self.max_days < self.target_days:
            raise WindowContractError("max_days must not precede target_days")
        expected = self.stimuli_per_sim_hour * (self.circadian_runtime_seconds // SIM_HOUR_SECONDS)
        if self.event_quota != expected:
            raise WindowContractError("quota must equal one full fixed-rate circadian cycle")

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping(), "window_config")

    def to_mapping(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__slots__}


@dataclass(frozen=True, slots=True)
class WindowState:
    window_id: str
    started_local_date: str
    last_local_date: str
    cumulative_runtime_seconds: int = 0
    event_count: int = 0
    tick_sample_events: int = 0
    divergence_count: int = 0
    unauthorized_effects: int = 0
    death_count: int = 0
    recovery_match_count: int = 0
    power_cycle_count: int = 0
    actual_midnights: int = 0
    hour_counts: tuple[tuple[int, int], ...] = ()
    expected_recovery_digest: str | None = None
    last_recovery_digest: str | None = None
    freeze_reason: str | None = None
    notification_sent: bool = False
    sealed: bool = False
    seal_reason: str | None = None
    awaiting_human_review: bool = False
    schema_version: str = WINDOW_SCHEMA_VERSION
    authority: str = SHADOW_AUTHORITY
    legacy_runtime_authoritative: bool = True
    cutover_authorized: bool = False
    m3_authority_open: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise WindowContractError("window_id must be non-empty")
        started = _require_date(self.started_local_date, "started_local_date")
        last = _require_date(self.last_local_date, "last_local_date")
        if last < started:
            raise WindowContractError("last_local_date cannot precede start")
        for field in (
            "cumulative_runtime_seconds",
            "event_count",
            "tick_sample_events",
            "divergence_count",
            "unauthorized_effects",
            "death_count",
            "recovery_match_count",
            "power_cycle_count",
            "actual_midnights",
        ):
            _require_non_negative_int(getattr(self, field), field)
        if self.recovery_match_count > self.death_count:
            raise WindowContractError("recovery matches cannot exceed deaths")
        previous_hour = -1
        total = 0
        for hour, count in self.hour_counts:
            _require_non_negative_int(hour, "hour_counts.hour")
            _require_non_negative_int(count, "hour_counts.count")
            if hour <= previous_hour:
                raise WindowContractError("hour_counts must be unique and ordered")
            previous_hour = hour
            total += count
        if total != self.event_count:
            raise WindowContractError("hour_counts must sum to event_count")
        if self.schema_version != WINDOW_SCHEMA_VERSION or self.authority != SHADOW_AUTHORITY:
            raise WindowContractError("unsupported or authoritative window state")
        if not self.legacy_runtime_authoritative or self.cutover_authorized or self.m3_authority_open:
            raise WindowContractError("window state cannot transfer authority")
        if self.freeze_reason is not None and self.freeze_reason not in HALT_REASONS:
            raise WindowContractError("unknown freeze reason")
        for field in ("expected_recovery_digest", "last_recovery_digest"):
            value = getattr(self, field)
            if value is not None:
                _require_digest(value, field)
        if self.notification_sent and self.freeze_reason is None:
            raise WindowContractError("notification evidence requires a frozen window")
        if self.sealed and not self.awaiting_human_review:
            raise WindowContractError("sealed window must await human review")
        if self.awaiting_human_review and not self.sealed:
            raise WindowContractError("unsealed window cannot await final review")

    @classmethod
    def create(cls, *, window_id: str, local_date: str) -> "WindowState":
        _require_date(local_date, "local_date")
        return cls(window_id=window_id, started_local_date=local_date, last_local_date=local_date)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WindowState":
        if not isinstance(value, Mapping):
            raise WindowContractError("window state must be a mapping")
        data = dict(value)
        if "hour_counts" in data:
            raw_counts = data["hour_counts"]
            if not isinstance(raw_counts, list) or any(
                not isinstance(item, list) or len(item) != 2 for item in raw_counts
            ):
                raise WindowContractError("hour_counts must be a JSON list of pairs")
            data["hour_counts"] = tuple((item[0], item[1]) for item in raw_counts)
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        result = {field: getattr(self, field) for field in self.__slots__}
        result["hour_counts"] = [list(item) for item in self.hour_counts]
        return result

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping(), "window_state")

    @property
    def frozen(self) -> bool:
        return self.freeze_reason is not None

    @property
    def sim_hours(self) -> float:
        return self.cumulative_runtime_seconds / SIM_HOUR_SECONDS


def freeze_shadow(state: WindowState, reason: str) -> WindowState:
    if reason not in HALT_REASONS:
        raise WindowContractError("unknown halt reason")
    if state.freeze_reason is not None:
        return state
    return replace(state, freeze_reason=reason)


def advance_runtime(state: WindowState, *, elapsed_seconds: int, local_date: str) -> WindowState:
    _require_non_negative_int(elapsed_seconds, "elapsed_seconds")
    current = _require_date(local_date, "local_date")
    previous = _require_date(state.last_local_date, "last_local_date")
    if current < previous:
        return freeze_shadow(state, "unauthorized_effect_detected")
    crossed = (current - previous).days if elapsed_seconds > 0 else 0
    return replace(
        state,
        cumulative_runtime_seconds=state.cumulative_runtime_seconds + elapsed_seconds,
        last_local_date=local_date,
        actual_midnights=state.actual_midnights + crossed,
    )


def record_power_cycle(
    state: WindowState,
    *,
    local_date: str | None = None,
) -> WindowState:
    next_date = state.last_local_date
    if local_date is not None:
        current = _require_date(local_date, "local_date")
        if current < _require_date(state.last_local_date, "last_local_date"):
            return freeze_shadow(state, "unauthorized_effect_detected")
        next_date = local_date
    return replace(
        state,
        power_cycle_count=state.power_cycle_count + 1,
        last_local_date=next_date,
    )


def record_discrete_stimulus(state: WindowState, *, config: WindowConfig) -> WindowState:
    if state.frozen or state.sealed:
        return state
    hour = state.cumulative_runtime_seconds // SIM_HOUR_SECONDS
    counts = dict(state.hour_counts)
    next_count = counts.get(hour, 0) + 1
    if next_count > config.event_cap_per_sim_hour:
        return freeze_shadow(state, "a9_event_cap_exceeded")
    counts[hour] = next_count
    return replace(
        state,
        event_count=state.event_count + 1,
        hour_counts=tuple(sorted(counts.items())),
    )


def record_tick_sample_event(state: WindowState) -> WindowState:
    updated = replace(state, tick_sample_events=state.tick_sample_events + 1)
    return freeze_shadow(updated, "a9_tick_sampling_detected")


def record_divergence(state: WindowState) -> WindowState:
    return replace(state, divergence_count=state.divergence_count + 1)


def record_unauthorized_effect(state: WindowState) -> WindowState:
    updated = replace(state, unauthorized_effects=state.unauthorized_effects + 1)
    return freeze_shadow(updated, "unauthorized_effect_detected")


def record_recovery(
    state: WindowState,
    *,
    expected_digest: str,
    recovered_digest: str,
) -> WindowState:
    _require_digest(expected_digest, "expected_digest")
    _require_digest(recovered_digest, "recovered_digest")
    matched = expected_digest == recovered_digest
    updated = replace(
        state,
        death_count=state.death_count + 1,
        recovery_match_count=state.recovery_match_count + int(matched),
        expected_recovery_digest=expected_digest,
        last_recovery_digest=recovered_digest,
    )
    return updated if matched else freeze_shadow(updated, "recovery_digest_mismatch")


def apply_watchdog(
    state: WindowState,
    *,
    config: WindowConfig,
    disk_bytes: int,
    integrity_valid: bool,
) -> WindowState:
    _require_non_negative_int(disk_bytes, "disk_bytes")
    if state.frozen:
        return state
    if not integrity_valid:
        return freeze_shadow(state, "unrecoverable_corruption")
    if disk_bytes > config.disk_budget_bytes:
        return freeze_shadow(state, "disk_budget_exceeded")
    return state


def mark_notification_sent(state: WindowState) -> WindowState:
    return state if state.notification_sent else replace(state, notification_sent=True)


def acceptance_checks(
    state: WindowState,
    *,
    config: WindowConfig,
    local_date: str | None = None,
) -> dict[str, bool]:
    max_hour_count = max((count for _hour, count in state.hour_counts), default=0)
    observed_date = state.last_local_date if local_date is None else local_date
    elapsed_calendar_days = (
        _require_date(observed_date, "local_date")
        - _require_date(state.started_local_date, "started_local_date")
    ).days + 1
    return {
        "unexplained_divergence_zero": state.divergence_count == 0,
        "all_deaths_recovered": state.death_count == state.recovery_match_count,
        "a9_tick_sampling_zero": state.tick_sample_events == 0,
        "a9_hourly_event_cap": max_hour_count <= config.event_cap_per_sim_hour,
        "unauthorized_effect_zero": state.unauthorized_effects == 0,
        "quota_met": state.event_count >= config.event_quota,
        "actual_midnights_met": state.actual_midnights >= config.required_actual_midnights,
        "circadian_cycle_met": state.cumulative_runtime_seconds >= config.circadian_runtime_seconds,
        "target_day_met": elapsed_calendar_days >= config.target_days,
        "not_frozen": not state.frozen,
        "legacy_authority_retained": state.legacy_runtime_authoritative,
        "no_automatic_cutover": not state.cutover_authorized and not state.m3_authority_open,
    }


def maybe_seal(state: WindowState, *, config: WindowConfig, local_date: str) -> WindowState:
    if state.sealed:
        return state
    current = _require_date(local_date, "local_date")
    started = _require_date(state.started_local_date, "started_local_date")
    if current < started:
        return freeze_shadow(state, "unauthorized_effect_detected")
    elapsed_days = (current - started).days + 1
    checks = acceptance_checks(state, config=config, local_date=local_date)
    complete = all(checks.values())
    if complete and elapsed_days >= config.target_days:
        return replace(
            state,
            last_local_date=local_date,
            sealed=True,
            seal_reason="quota_and_acceptance_criteria_met",
            awaiting_human_review=True,
        )
    if elapsed_days >= config.max_days:
        return replace(
            state,
            last_local_date=local_date,
            sealed=True,
            seal_reason="maximum_deadline_reached_incomplete",
            awaiting_human_review=True,
        )
    return state


def evidence_record(state: WindowState, *, config: WindowConfig) -> dict[str, Any]:
    checks = acceptance_checks(state, config=config)
    material = {
        "accepted_m2_e_decision_digest": ACCEPTED_M2_E_DECISION_DIGEST,
        "accepted_m2_e_head": ACCEPTED_M2_E_HEAD,
        "accepted_m2_e_packet_digest": ACCEPTED_M2_E_PACKET_DIGEST,
        "authority": SHADOW_AUTHORITY,
        "baseline_sha": WINDOW_BASELINE_SHA,
        "checks": checks,
        "config": config.to_mapping(),
        "config_digest": config.digest,
        "cutover_authorized": False,
        "legacy_runtime_authoritative": True,
        "m3_authority_open": False,
        "raw_private_companion_only": True,
        "schema_version": WINDOW_EVIDENCE_SCHEMA_VERSION,
        "state": state.to_mapping(),
        "state_digest": state.digest,
    }
    material["evidence_digest"] = _digest(material, "window_evidence")
    return material


def one_line_status(state: WindowState, *, config: WindowConfig) -> str:
    checks = acceptance_checks(state, config=config)
    health = "frozen" if state.frozen else "sealed" if state.sealed else "running"
    return (
        f"health={health} events={state.event_count}/{config.event_quota} "
        f"runtime_sim_hours={state.sim_hours:.2f}/{CIRCADIAN_SIM_HOURS} "
        f"midnights={state.actual_midnights}/{config.required_actual_midnights} "
        f"deaths={state.death_count} recovered={state.recovery_match_count} "
        f"divergence={state.divergence_count} unauthorized={state.unauthorized_effects} "
        f"ready={str(all(checks.values())).lower()} authority=shadow_only"
    )
