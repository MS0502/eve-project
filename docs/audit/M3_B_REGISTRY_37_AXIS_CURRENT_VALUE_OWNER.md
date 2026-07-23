# M3-B Registry 37-Axis Current-Value Owner

Baseline: `f20951351b56c7102fdcf7c00f17cbcfac792205`

Status: disconnected `shadow_only` owner contract for the 37 axes defined by `affect_hormone_neural_rhythm_registry.py`. It does not install a runtime owner instance, scheduler, observer, persistence route, event append path, production apply permission, observation window, M3-C, cutover, or M3-E authority.

## Purpose

PR #171 proved that the registry contains definitions and proposal metadata but no current-value lifecycle owner. PR #172 separately resolved the immutable legacy 26-axis source envelope.

This candidate addresses:

```text
REGISTRY_OBSERVED_VALUE_OWNER_ABSENT
```

by introducing an explicit current-state owner whose values, provenance, cadence, replay boundary, snapshot identity, and read-only observation conversion are defined independently from registry definitions and event proposal metadata.

M3-B remains incomplete after this candidate. The next required artifact is a combined real 63-axis read-only observation packet and observation-window proof.

## Exact 37-axis ownership

`core/m3_b_registry_affect_owner.py` derives one canonical axis order from `AXIS_GROUPS` and requires exactly all 37 axes. `RegistryAffectOwnerState` owns one frozen `RegistryAxisCurrentState` per axis.

Each axis state owns:

- current value;
- baseline;
- floor and ceiling;
- confidence;
- last proposal/impulse logical tick;
- update count;
- last source kind and source identity;
- versioned axis-state schema;
- canonical SHA-256.

The owner owns:

- owner instance identity;
- logical tick;
- monotonic state sequence;
- exact canonical axis tuple;
- prior-state digest;
- last transition digest, kind, and identity;
- bounded unique proposal-ID ledger;
- genesis source identity;
- versioned owner/source schemas;
- all non-authority proof flags.

## Deterministic initial state

`create_registry_affect_owner()` materializes every registry baseline into a new explicit owner state.

This is a deterministic genesis rule, not observation evidence:

```text
genesis_is_observation_evidence = false
proposal_metadata_is_current_state = false
confidence = 0.0 for all 37 genesis axes
last_source_kind = deterministic_registry_baseline_genesis_not_observation
```

The registry definition dictionary remains detached policy data. It becomes owned current state only through explicit owner construction. The baseline supplies a deterministic neutral starting value, but it does not claim that any axis has actually been observed. A genesis snapshot therefore emits the complete 37-axis shape with confidence `0.0`, not positive evidence.

Genesis uses no wall clock, random value, process ID, filesystem state, persistence value, or runtime reference. Identical owner/genesis identities produce identical owner material and digest.

## Accepted proposal boundary

`apply_validated_registry_proposal()` accepts only an explicit caller request against an exact existing owner. It requires:

- a known event category;
- a non-empty registry-axis delta map;
- successful passage through the merged `validate_affect_event_proposal()` boundary;
- the validator's operator-authorization requirement to remain present;
- explicit operator authorization identity;
- exact expected prior owner digest;
- exactly consecutive state sequence;
- unique bounded proposal identity;
- finite positive confidence within `(0,1]`.

The function returns a new frozen owner state. It never mutates the prior owner.

Proposal metadata alone remains non-state. A proposal changes detached owner state only after all checks above pass. Only touched axes gain positive observation confidence; untouched genesis axes remain explicit unknowns at confidence `0.0`.

## Range, confidence, and saturation

For an accepted axis delta:

```text
new_value = clamp(old_value + validated_delta, floor, ceiling)
new_confidence = proposal_confidence                      if old_confidence == 0.0
                 min(old_confidence, proposal_confidence) otherwise
```

This allows the first validated source to establish confidence without letting a registry default masquerade as evidence. Later evidence cannot silently increase the retained confidence. Unknown axes, empty proposals, zero/non-finite confidence, non-finite deltas, stale owner digests, sequence gaps, duplicate proposal IDs, validator failures, and exhausted proposal-ID ledgers fail closed.

No partial state is returned.

## Cadence ownership

`advance_registry_affect_owner()` defines cadence ownership without installing a scheduler.

The caller supplies:

- a target logical tick greater than the current tick;
- cadence identity;
- exact current owner digest.

For each axis, decay begins only after its registry refractory interval and moves deterministically toward baseline:

```text
value_t = baseline + (value_0 - baseline) * (1 - decay_rate) ** active_ticks
```

The result is clamped to registry bounds. The impulse tick remains tied to the most recent accepted proposal. Cadence preserves the axis confidence and never turns an untouched zero-confidence genesis axis into observed evidence. A repeated/non-monotonic tick or stale digest fails closed.

## Replay and no-duplicate rule

An accepted proposal is bound to:

- exact prior owner digest;
- exact next sequence;
- unique proposal ID;
- operator authorization ID;
- event category and canonical ordered deltas;
- positive confidence and transition payload;
- versioned transition schema.

The transition digest covers the complete material. Reapplying the same proposal to the next state fails on duplicate ID; applying it to a different or stale state fails on digest/sequence.

This is the A9-compatible detached transition identity boundary. The owner does not append a production event and does not claim named-state transition authority.

## Read-only observation boundary

`RegistryAffectOwnerState.to_axis_observations()` returns exactly 37 immutable `AxisObservation` values.

Each observation carries:

- source family `read_only_affect_registry`;
- current owned value;
- registry baseline and bounds;
- current confidence;
- deterministic owner snapshot identity;
- source schema version;
- complete owner-state integrity digest;
- per-axis state digest;
- owner/genesis/source/update provenance metadata.

A genesis snapshot is structurally complete but all 37 confidence values are `0.0`. After an accepted proposal, only the touched axes carry positive confidence. This preserves absence/unknown state explicitly instead of substituting registry defaults as observed facts.

The conversion reads only the detached owner and performs no projection, drive mutation, named transition, event append, persistence access, or observation-window start.

## M2-B read-capability ruling

The scanner follows caller-supplied `transition_payload` through the existing proposal validator, local result inspection, and canonical SHA-256 transition evidence. PR #173 registers the exact findings append-only as `NOT_CAPABILITY_BOUNDARY`.

This path performs no external source discovery or read, retains no conversational raw text for expression, and grants no quotation, generation, speech, publication, or runtime activation capability.

## Authority boundary

Every owner state fixes these values to false:

```text
runtime_hook_installed
scheduler_installed
persistence_accessed
event_append_performed
live_affect_mutated
live_drive_mutated
goal_memory_self_expression_mutated
observation_window_started
m3_b_complete
m3_c_open
m3_e_authority_open
cutover_authorized
```

Authority remains `shadow_only`; legacy runtime and persistence remain authoritative.

## Audit and focused verification

```text
python scripts/audit/m3_b_registry_affect_owner.py --summary-only --pretty --strict
pytest -q tests/test_v4_m3_b_registry_affect_owner.py tests/audit/test_m3_b_registry_affect_owner.py
```

The focused suite covers:

- exact canonical 37-axis ownership;
- deterministic baseline genesis distinct from observation evidence;
- zero-confidence genesis and positive-confidence touched-axis promotion;
- detached definition data cannot alter owner state;
- exact 37-value immutable observation snapshot;
- validated proposal transition and unchanged prior owner;
- stale digest, sequence gap, duplicate ID, zero confidence, unknown axis, and validator rejection;
- deterministic saturation;
- explicit refractory/cadence decay without confidence promotion;
- cadence replay rejection;
- frozen owner/axis state;
- no I/O, persistence, scheduler, event, or runtime activation surface;
- recalculable deterministic audit output;
- all live and authority flags remaining false.

## Cross-chat validation reuse

Immutable prerequisites:

```text
PR #170 M3-B projection exact head:       0ccce6b0a39fd0b0030181d3bfdb7c99f01b7c6c
PR #170 exact run:                        29977749600
PR #170 artifact SHA-256:                 4a73b08e5ae4b143ebbd84963c7a6803a8964a5a47db95805fdef9b42df49ca1
PR #170 squash merge:                     0d755c35c994fa5b1ed3f2768c7905cda83c9a95

PR #171 source preflight exact head:       6c43f5ee87ed1104f4682fffcb1468701765a1aa
PR #171 exact run:                        29979778124
PR #171 artifact SHA-256:                 641c99378e5b6716708e62e144d0236c490b17f9a32c83bf9ece9108417b1ea8
PR #171 squash merge:                     97ac96b8bdf54c7fbc74d4b0346ceff49624eaec

PR #172 legacy capture exact head:         effac41db02581c4d5521a839d6cbfbffcecad27
PR #172 exact run:                        29984825031
PR #172 focused/full:                     11 / 2,860 passed
PR #172 artifact SHA-256:                 f79570993f0775382ca0a72a90692ac555758ab52d472cf72a300600fcaba481
PR #172 M2-E compatibility:               29984825029, 6/6 jobs passed
PR #172 squash merge / current baseline:   f20951351b56c7102fdcf7c00f17cbcfac792205
```

Do not rerun prerequisite validation because work moves to another chat or because PR body/comments, Draft/Ready state, or review metadata changes. Rerun only after code-head change, artifact loss/corruption, digest mismatch, or validation-scope change.

## Explicit non-goals

No production composition change, owner auto-construction, scheduler, callback, observer, wall-clock tick, persistence, SQLite, event append, legacy hormone mutation, M3-A drive mutation, named-state mutation, goal/memory/self-model/expression integration, speech, external effect, real 63-axis packet, observation-window acceptance, M3-B completion, M3-C, cutover, or M3-E authority.

## Intended changed-file boundary

1. `core/m3_b_registry_affect_owner.py`
2. `scripts/audit/m3_b_registry_affect_owner.py`
3. `tests/test_v4_m3_b_registry_affect_owner.py`
4. `tests/audit/test_m3_b_registry_affect_owner.py`
5. `docs/audit/M3_B_REGISTRY_37_AXIS_CURRENT_VALUE_OWNER.md`
6. `docs/audit/m2_b_decision_additions/pr-173.json`
7. `docs/audit/forward_additions/pr-173.json`
