# M3-B Combined 63-Axis Observation Packet Preflight

Baseline: `aab991f5c217baf9a6f6e5d2d6115feba9000f5a`

Status: disconnected, explicit, `shadow_only` packet preflight. It combines one exact legacy 26-axis immutable capture with one already-existing detached registry 37-axis owner snapshot. It performs no projection, installs no observer or scheduler, accesses no persistence, appends no event, mutates no live state, starts no observation window, opens no M3-C, and grants no cutover or M3-E authority.

## Purpose

PR #172 supplied the immutable legacy 26-axis source envelope. PR #173 supplied the detached registry 37-axis current-value owner. This candidate proves that those two owner boundaries can produce one exact canonical 63-axis `AxisObservation` packet without mutating either source.

It also separates two claims that must not be conflated:

```text
structurally complete strict projection input  !=  observation-window evidence ready
```

The combined packet is structurally complete. The observation window is not eligible to start because the registry baseline genesis is intentionally zero-confidence and therefore not observed fact.

## Exact packet contract

`core/m3_b_observation_packet.py` defines:

```text
eve.m3-b.combined-observation-source-set.v1
eve.m3-b.combined-63-axis-observation-packet.v1
```

The packet requires caller-supplied exact source owners:

- one exact `HormoneSystem` object;
- one exact `RegistryAffectOwnerState` object;
- packet identity and sequence;
- explicit logical tick;
- explicit legacy source-instance and snapshot identities.

It does not construct a production runtime or registry owner automatically.

## Canonical 63-axis order

The packet concatenates:

1. exact `LEGACY_AXIS_ORDER` — 26 axes;
2. exact `REGISTRY_AXIS_ORDER` — 37 axes.

It requires:

- exactly 63 observations;
- exact canonical order;
- 63 unique axes;
- first 26 source family `legacy_mutable_hormone`;
- final 37 source family `read_only_affect_registry`;
- exact immutable `AxisObservation` values only.

Any missing, extra, duplicated, reordered, wrong-family, or wrong-type observation fails closed.

## Source-set evidence

The immutable source set records:

- legacy source instance identity;
- legacy source snapshot identity;
- legacy capture digest;
- legacy source-integrity digest;
- registry owner instance identity;
- registry snapshot identity;
- registry owner-state digest;
- versioned source-set schema and canonical digest.

Packet construction invokes the merged exact legacy capture, which reads the legacy object twice and proves stable container, stable 26 axis objects, and equal before/after state. The registry owner is immutable; its state digest is checked before and after packet assembly.

## Structural readiness

A successful packet exposes:

```text
axis_count = 63
legacy_axis_count = 26
registry_axis_count = 37
structurally_complete = true
strict_projection_input_ready = true
```

`strict_projection_input_ready` means the merged strict projection contract can receive one observation value for every mapped-plan axis. It does not mean that every value carries positive observed confidence, that projection has run, or that an observation window may start.

## Confidence and unknown-state boundary

The exact adult legacy capture supplies 26 positive-confidence values. Registry genesis supplies 37 structurally owned values with confidence `0.0` because registry defaults are deterministic neutral initial state, not observation evidence.

Therefore a genesis packet has:

```text
positive_confidence_count = 26
zero_confidence_count = 37
window_blockers = [REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE]
observation_window_start_eligible = false
```

After one accepted `praise` proposal touches `competence_drive` and `social_trust`, only those two registry axes gain positive confidence:

```text
positive_confidence_count = 28
zero_confidence_count = 35
observation_window_start_eligible = false
```

The packet never substitutes registry baselines as observed facts and never drops unknown axes to make readiness appear complete.

## Window start rule

Window eligibility is derived, not asserted:

```text
observation_window_start_eligible =
    exact 63-axis packet
    and zero_confidence_count == 0
    and window_blockers is empty
```

The candidate permanently fixes:

```text
observation_window_started = false
observation_window_satisfied = false
m3_b_complete = false
```

It only reports readiness. It cannot start or satisfy the window itself.

## Legacy boundary-baseline behavior

The merged v1 `AxisObservation` contract requires `floor < baseline < ceiling`. A legacy developmental profile with a baseline exactly on the source boundary remains valid immutable legacy capture evidence but cannot enter the strict v1 combined packet. Packet construction fails closed before emitting a partial packet. It does not invent epsilon, widen `[0,1]`, or substitute a different baseline.

## Authority boundary

Every packet fixes these values to false:

```text
projection_performed
observation_window_started
observation_window_satisfied
persistence_accessed
event_append_performed
live_affect_mutated
live_drive_mutated
named_state_mutated
goal_memory_self_expression_mutated
m3_b_complete
m3_c_open
m3_e_authority_open
cutover_authorized
```

Legacy runtime and persistence remain authoritative.

## Audit and focused verification

```text
python scripts/audit/m3_b_observation_packet.py --summary-only --pretty --strict
pytest -q tests/test_v4_m3_b_observation_packet.py tests/audit/test_m3_b_observation_packet.py
```

The focused suite covers:

- exact 26+37 canonical packet shape;
- exact source-family split;
- 26 positive legacy plus 37 zero-confidence registry genesis axes;
- strict-projection structural readiness distinct from window readiness;
- deterministic repeated packet equality and digests;
- unchanged legacy source and registry owner;
- positive-confidence promotion only for two validated touched registry axes;
- remaining registry confidence blocker;
- fail-closed legacy boundary baseline;
- exact source-type and identity validation;
- frozen packet and all authority/effect flags false;
- AST proof of no I/O, persistence, scheduler, event, projection, or runtime activation surface;
- recalculable audit output.

## Remaining blocker and next artifact

After this preflight, the exact blocker is:

```text
REGISTRY_POSITIVE_CONFIDENCE_COVERAGE_INCOMPLETE
```

The next required artifact is positive-confidence observed-value provenance for every remaining registry axis before any M3-B observation window starts.

This PR does not decide whether those values come from existing validated event proposals, explicit read-only sensor/appraisal owners, or another separately reviewed deterministic source contract. Defaults and proposal metadata alone are insufficient.

## Cross-chat validation reuse

Immutable prerequisites:

```text
PR #170 projection exact head:             0ccce6b0a39fd0b0030181d3bfdb7c99f01b7c6c
PR #170 exact run / artifact SHA-256:       29977749600 / 4a73b08e5ae4b143ebbd84963c7a6803a8964a5a47db95805fdef9b42df49ca1
PR #170 squash merge:                      0d755c35c994fa5b1ed3f2768c7905cda83c9a95

PR #172 legacy capture exact head:         effac41db02581c4d5521a839d6cbfbffcecad27
PR #172 exact run / focused / full:        29984825031 / 11 / 2,860
PR #172 artifact SHA-256:                  f79570993f0775382ca0a72a90692ac555758ab52d472cf72a300600fcaba481
PR #172 M2-E compatibility:                29984825029, 6/6 jobs passed
PR #172 squash merge:                      f20951351b56c7102fdcf7c00f17cbcfac792205

PR #173 registry owner exact head:         f3013486b5b5963dbe5b14e415adcd0a68134336
PR #173 exact run / focused / full:        29987091687 / 12 / 2,872
PR #173 artifact SHA-256:                  4dbe065b1965b03436e7f214a787a691b7e969aefaf45f4c37d623fe7d5359f7
PR #173 M2-E compatibility:                29987091720, 6/6 jobs passed
PR #173 squash merge / current baseline:   aab991f5c217baf9a6f6e5d2d6115feba9000f5a
```

Do not rerun prerequisite validation because work moves to another chat or because PR body/comments, Draft/Ready state, or review metadata changes. Rerun only after code-head change, artifact loss/corruption, digest mismatch, or validation-scope change.

## Intended changed-file boundary

1. `core/m3_b_observation_packet.py`
2. `scripts/audit/m3_b_observation_packet.py`
3. `tests/test_v4_m3_b_observation_packet.py`
4. `tests/audit/test_m3_b_observation_packet.py`
5. `docs/audit/M3_B_COMBINED_63_AXIS_OBSERVATION_PACKET_PREFLIGHT.md`
6. `docs/audit/forward_additions/pr-<this-pr>.json`
7. optional `docs/audit/m2_b_decision_additions/pr-<this-pr>.json` only if exact discovery adds read-capability findings
