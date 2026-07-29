# M3-C-B Pure Deterministic Goal-Selection Kernel

Baseline: `9a93d259a80b92791602b9032908d4188e5d655b` — PR #218 squash merge.

Status: implementation candidate, isolated and unintegrated. This slice creates a
pure v4-native scoring/selection kernel only. It does not connect to
`GoalManagement`, the live engine, event persistence, action execution,
scheduling, speech, memory, model/vector state, AGP mutation, or M3-E.

## Authority boundary

```text
M3 authority open:                    true — #215
M3-C-A design complete:               true — #217
M3-B retained real observations:      5/37 — #218
legacy goal-domain authority:         unchanged
production goal integration:          false
event append/persistence write:       false
action/scheduler/speech authority:     false
M3-E authority open:                  false
```

A selected result from this kernel is a selected **goal proposal** only. It is
not an active legacy goal and grants no downstream permission.

## Implemented contracts

The module `core/m3_c_b_goal_selection_kernel.py` implements the exact reviewed
#217 versions:

```text
eve.m3-c-b.goal-selection-kernel.v1
eve.m3-c-a.goal-candidate.v1
eve.m3-c-a.goal-score.v1
eve.m3-c-a.goal-transition-predicate.v1
eve.m3-c-a.goal-selection-receipt.v1
eve.m3-a.drive-dynamics.v1
```

### Drive input

Exactly eight `DriveSample` values are required:

```text
energy, safety, affiliation, curiosity,
agency, coherence, competence, expression
```

Every sample carries finite bounds, an in-bound value, a SHA-256 provenance
digest, exact dynamics/predicate versions, and one shared replay-carried
monotonic elapsed time. No wall clock is read.

Normalization is the #217 formula:

```text
z_d = clip((2*x_d - (L_d+U_d)) / (U_d-L_d), -1, 1)
```

### Candidate input

`GoalCandidate` accepts only a canonical internal `semantic_goal_id`, one
non-negative decision epoch, SHA-256 evidence identity, bounded candidate
attributes, and exact eight-drive alignment/confidence maps. A raw natural
language string is not a valid semantic goal identifier.

Candidate identity is:

```text
sha256(
  candidate_schema || semantic_goal_id || decision_epoch ||
  evidence_digest || scoring_policy_version
)
```

### Scoring and selection

The exact #217 score is implemented without sampling:

```text
drive_term_g = sum(q_gd*w_gd*z_d) / max(1, sum(q_gd*abs(w_gd)))

score_g = clip(
    0.30*base_value
  + 0.30*drive_term_g
  + 0.15*expected_value
  + 0.10*urgency
  + 0.10*continuity
  - 0.10*cost
  - 0.15*risk,
  -1, 1
)
```

The exact policy constants are:

```text
proposal enter:     0.20
proposal exit:      0.10
selection minimum:  0.30
initial margin:     0.08
switch margin:      0.12
switch cooldown:    30 replay-seconds
```

Ordering is descending score followed by ascending lexical `candidate_id`.
Equal-score candidates therefore have deterministic order, but an equal initial
margin still fails selection.

### Prior selection and cooldown

A prior selection must remain in the candidate set. Evidence expiry/removal is a
later lifecycle gate and cannot be silently inferred here. Replay time may not
move backwards.

A challenger switches the selected proposal only when both conditions hold:

```text
elapsed replay time >= 30 seconds
challenger score - current selected score >= 0.12
```

Otherwise the current selected proposal remains selected and
`transition_eligible=false`.

### Immutable selection receipt

The output records candidate-set and drive-sample digests, all ordered scores,
evaluated winner/comparison material, prior and resulting selected candidate,
decision kind, cooldown, margin, and exact policy versions.

Every output fixes:

```text
action_authorized=false
speech_authorized=false
persistence_write_performed=false
legacy_goal_authority_transferred=false
m3_e_authority_open=false
```

`transition_eligible=true` means only that the pure predicate found an initial
or switched selection candidate. It does not emit or persist an event.

## A9 no-continuous/no-duplicate boundary

- sample normalization and score recomputation emit no event;
- candidate identity is stable within one decision epoch;
- deterministic ordering uses no randomness;
- unchanged input plus unchanged prior state yields the same receipt digest;
- an unchanged winner returns `retained_selection` and
  `transition_eligible=false`;
- this module has no event-kernel append, SQLite, file, thread, clock, network,
  scheduler, action, or speech surface.

## Counterfactual acceptance

Focused tests reproduce the #217 causal counterfactual exactly:

```text
strain_mapped_affect  -> recover_operating_margin
recovered_exploration -> explore_information_gap
```

They also prove:

- exact constants and stable identity;
- malformed/raw-text-like material fails closed;
- lexical tie-break with insufficient equal margin;
- cooldown and switch margin both required;
- repeated unchanged evaluation is deterministic and non-transitioning;
- no downstream authority or persistence claim;
- no runtime/I/O/legacy import surface.

## Deferred gates

Not implemented here:

- semantic candidate discovery from live engine state;
- candidate lifecycle event construction or append;
- snapshot/replay persistence integration;
- legacy goal-domain migration/cutover;
- action activation, scheduler integration, or speech;
- M3-E affect authority.

Each requires a later separately reviewed slice.
