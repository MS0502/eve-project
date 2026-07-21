# M1 Human Acceptance Record

Schema: `eve.m1-human-acceptance-record.v1`

Decision ID: `m1-human-acceptance:extended-mechanism:v1`

Recorded: `2026-07-21`

Approval PR: `#158`

Canonical JSON SHA-256: `aff557da810b7faa0c9dc57bde214a9760a0d3099c8031cb6eb7a24398cf8522`

## Decision

```text
human_review_status: accepted
human_accepted: true
m1_closed: true
v4_2_eligible: true
v4_2_review_opened: false
m2_started: false
```

This is an explicit delegated human review, not an automatic machine promotion.
The project creator defined the exact conditional acceptance criteria and
authorized the evidence reviewer to approve M1 when the expanded controlled
window passed them.

## Evidence pins

```text
validated evidence head: 560b9b54f3237d63762b81da38e7c25c36922214
evidence merge SHA: 7c4573e628e5ac51d0d64ad1040078741f3630e0
raw artifact SHA-256: 3618b948cb2e864741412713b5c724632ae9fd72a214479b970d8c4aeeafcaac
source evidence SHA-256: 06984c653ed2a655f45c7cb27d0777b1c93c6aee872f2cb9c7d1f5a898d9af86
exact-head run: 29826184624
exact-head artifact: exact-head-validation-560b9b54f3237d63762b81da38e7c25c36922214
exact-head artifact ZIP SHA-256: 5482da68f38e5d66400d6a32b948d559ce1dd6ce7ec80fe77de08659b8f9d0b9
focused tests: 12 passed
full suite: 2712 passed
```

Forward registration: `4` adaptive-numeric fingerprints / `4` occurrences in the independent acceptance test, all introduced by PR `#158`.

## Reviewed criteria

| Criterion | Passed |
|---|---|
| `mutation_form_state_fidelity` | `true` |
| `multiple_adapter_dispositions` | `true` |
| `live_tick_thread_concurrency` | `true` |
| `complete_replay_equivalence` | `true` |
| `failure_visibility` | `true` |
| `discrete_transition_granularity` | `true` |
| `bounded_direct_write` | `true` |
| `raw_observation_recalculability` | `true` |
| `exact_head_validation` | `true` |
| `zero_unauthorized_effects` | `true` |

The first green expanded-window artifact was not accepted because three mutation
forms were represented only by control-flow execution, not by their changed
state. The corrected artifact records exact before/after values and transition
digests for `last_emit_time`, `weights`, `processed_input_count`, `neighbors`,
and `files`; all five transitions replay successfully.

## Scope ruling

**메커니즘 검증 완료. 커버리지 검증은 A2에 따라 M2 dual-read + cutover로 이연. 미관찰 527곳은 A7에 따라 WRAP 시점 점진 교정되는 추적 부채.**

`5 / 532` or any other historical-site fraction is not an M1 acceptance metric.
No unobserved historical site is represented as safe.

## Authority boundary

M1 acceptance grants eligibility to open a v4.2 amendment review only. It does
not open or approve v4.2, start M2, install a production observer, enable
persistence, integrate the runtime, change defaults, or transfer authority from
the pre-kernel legacy runtime. The immutable machine packet remains fixed to
`human_accepted=false` and `v4_2_eligible=false`; this external record is the
separate constitutional decision required by M1-E.

## v4.2 candidate triangle

1. **`discrete_transition_granularity`** — Continuous decay is derived state; only discrete transitions emit events unless a separately reviewed contract says otherwise.
2. **`raw_observation_recalculability`** — Every approval evidence artifact must contain the raw observations needed to independently recalculate every claimed metric.
3. **`mutation_state_fidelity`** — Executing a mutation-shaped call path is insufficient evidence; the artifact must identify the changed state and preserve exact before/after values or an independently verifiable equivalent.

This record is governance metadata only and has no executable runtime authority.
