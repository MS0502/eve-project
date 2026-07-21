# M1 Extended Controlled Observation Evidence

Campaign schema: `eve.m1-extended-controlled-observation.v1`

Campaign ID: `m1:extended-controlled-observation:mechanism:v1`

Baseline: `main` at `847621bcd61634958ce505108ade491c50ced0d4`

Raw observation artifact SHA-256: `3618b948cb2e864741412713b5c724632ae9fd72a214479b970d8c4aeeafcaac`

Status: **extended mechanism machine evidence complete; M1 human acceptance not performed**

## Boundary

This is a disconnected controlled window. It uses the existing after-the-fact
observer with a campaign-local reviewed target registry. No observer is installed
into production, no legacy authority changes, no default changes, and no
production persistence is enabled. The direct-write probe is confined to two
temporary roots and both roots are removed before the campaign returns.

## Mechanism acceptance criteria

### Mutation forms

| M0-A form | Executed source | Observed target | State field | Changed | Replay match |
|---|---|---|---|---|---|
| `attribute_assignment` | `adapters/live_loop.py:101-105` | `legacy.live_loop.drain_user_inputs` | `last_emit_time` | `true` | `true` |
| `subscript_assignment` | `legacy/eve_modules/spreading_activation.py:239-243` | `legacy.activation.learn_pair` | `weights` | `true` | `true` |
| `augmented_assignment` | `adapters/live_loop.py:68-77` | `legacy.live_loop.drain_user_inputs` | `processed_input_count` | `true` | `true` |
| `mutating_method_call` | `legacy/eve_modules/spreading_activation.py:241-243` | `legacy.activation.learn_pair` | `neighbors` | `true` | `true` |
| `direct_write` | `adapters/persistence_adapter.py:65-74` | `legacy.persistence.save` | `files` | `true` | `true` |

All five required forms were executed at least once. Each row identifies the
mutated state field and records its exact raw before/after values plus a transition
digest. The rows are mechanism evidence, not a claim that all historical mutation
sites are covered or safe.

### Multiple adapter dispositions

| Bound call path | M0-D disposition | Evidence location |
|---|---|---|
| `ActivationAdapter.learn_pair` | `WRAP` | `adapters/activation_adapter.py:103-105` |
| `LiveLoop._drain_user_inputs` | `REWRITE` | `adapters/live_loop.py:68-77` |
| `PersistenceAdapter.save` | `REWRITE` | `adapters/persistence_adapter.py:54-80` |

Observed adapter call paths: `3`.
The window includes both `WRAP` and `REWRITE` dispositions.

### Concurrency

```text
live tick thread reached barrier: true
thread alive before mutation: true
thread alive after mutation: true
tick count at barrier: 1
mutation candidates while thread alive: 1
tick candidates before mutation: 0
thread stopped and joined: true
```

### Replay and failure visibility

```text
compared events: 4
matching events: 4
match rate: 4 / 4 = 1.0
divergence count: 0
complete divergence list: []
legacy failure events: 1
observer failure records: 1
observer failure emitted candidate: 0
```

The controlled legacy exception remains visible as a failure event and the exact
exception object is re-raised. A separate observer-snapshot failure remains
visible in the observer failure ledger while the retained call still completes
and produces no candidate.

### Granularity and no amplification

```text
discrete observed calls: 4
candidate events: 4
maximum events per observed call: 1
standalone tick steps: 4
events during standalone tick steps: 0
events from live tick before discrete mutation: 0
```

The measured policy remains one candidate per discrete observed call and zero
candidates for continuous tick/decay alone. The live-loop drain contains several
low-level mutations but produces one call-boundary candidate.

## Raw-data sufficiency

`docs/audit/M1_EXTENDED_CONTROLLED_OBSERVATION_RAW.json` contains every event
envelope, before/after state, per-event replay row, complete divergence ledger,
mutation-form source row, thread-barrier observation, temporary-file name/size/hash,
legacy-failure record, observer-failure record, and final equivalence digest used
above. Every claimed metric in this report can be independently recalculated from
that artifact.

## Scope ruling carried into human review

This window tests the observer/event/replay/failure mechanism. It does not use
`5 / 532` or any other historical-site fraction as an M1 acceptance metric.
Repository-wide coverage remains an A2/M2 dual-read and cutover obligation. Any
unobserved historical site remains tracked debt and is not represented as safe.

## Gate state

```text
machine_status: extended_mechanism_evidence_complete
machine_passed: true
eligible_for_human_review: true
human_review_status: required_not_performed
human_accepted: false
v4_2_eligible: false
authority: shadow_only
production observer installed: false
production persistence enabled: false
```

This PR supplies the extended-window evidence only. A separate approval-record PR
must perform the human acceptance and artifact-hash pinning before M1 can close.
