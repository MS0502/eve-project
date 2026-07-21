# M1 Controlled Observation Evidence

Campaign schema: `eve.m1-controlled-observation-evidence.v1`

Campaign ID: `m1:controlled-observation:activation-learn-pair:v1`

Baseline: `main` at `c70936c92085d0b3bb0ebd44ec87f11952f87f32`

Status: **controlled machine evidence complete; M1 human acceptance not performed**

## Boundary

This campaign is explicit, deterministic, in-memory, and disconnected. It invokes the reviewed `ActivationAdapter.learn_pair` bound method against an isolated ledger that delegates successful calls to the retained `legacy.eve_modules.spreading_activation.SpreadingActivation` implementation.

It installs no production observer or bridge, uses no wall clock or autonomous loop, performs no persistence or external effect, changes no default, and grants no human, v4.2, M2, recovery, scheduler, or cutover authority.

## Observation window

```text
window kind: controlled logical-step window
scenarios: 3
logical steps: 18
legacy learn_pair calls: 12
deterministic tick steps: 6
decay cycles: 6
total tick dt: 6.0
candidate events: 12
  success candidates: 11
  legacy-failure candidates: 1
separate observer-failure probes: 1
production integration: false
wall-clock duration: not used
```

The seventh legacy call is a controlled failure injection before delegate mutation. All other successful calls delegate to the retained legacy spreading-activation implementation. An unobserved baseline executes the identical call/tick schedule for state comparison.

## Legacy preservation

```text
observer call order preserved: true
original controlled exception object propagated: true
observed legacy state matches unobserved baseline: true
observer-failure probe return preserved: true
observer-failure probe legacy state preserved: true
persistence behavior changed: false
defaults changed: false
external effects introduced: false
```

## Replay equivalence

```text
compared events: 12
matching events: 12
match rate: 12 / 12 = 1.0
final snapshot equivalence: true
divergence count: 0
complete divergence list: []
```

The empty divergence list is measured over this controlled 12-event window only. It is not a claim of general production equivalence.

## Silent-failure observations

Five selected active `except Exception` surfaces in `adapters/activation_adapter.py` were exercised with deterministic failing dependencies. Each failure was swallowed by the retained handler and produced the documented fallback without an outward exception.

| Candidate | Callable | Lines | Injected stage | Fallback |
|---|---|---:|---|---|
| 1 | `ActivationAdapter.ingest` | 40-43 | `sa.decay` | `None` |
| 2 | `ActivationAdapter.ingest` | 44-47 | `wm.decay` | `None` |
| 3 | `ActivationAdapter.ingest` | 50-55 | `sa.apply_hormone_modulation` | `None` |
| 4 | `ActivationAdapter.focus_category` | 88-91 | `wm.get_focus` | `None` |
| 5 | `ActivationAdapter.focus_set` | 94-97 | `wm.get_focus_set` | empty set |

```text
selected occurrences: 5
observed silent candidates: 5
M0-B frozen static denominator: 525
frozen unobserved remainder: 520
integrated pre-M0-D static denominator: 532
integrated unobserved remainder: 527
```

The historical 525 and 532 values are static AST occurrence counts. This campaign does not reinterpret the remaining 520/527 occurrences as safe, unreachable, or non-silent.

## Event rate and granularity input

```text
events per observed legacy call: 12 / 12 = 1.0
events per logical step: 12 / 18 = 0.6666666666666666
events during tick steps: 0
maximum events in one logical step: 1
tick/decay event amplification: none observed
```

The controlled result supports one candidate per registered mutation call and no event for a standalone tick under the current single-funnel observer. It does not justify instrumenting every tick, decay, or historical silent handler in v4.2.

## M1-E packet result

The campaign is evaluated through the existing M1-E packet contract:

```text
machine_status: machine_evidence_complete
machine_passed: true
eligible_for_human_review: true
human_review_status: required_not_performed
human_accepted: false
v4_2_eligible: false
authority: shadow_only
runtime_integrated: false
persistence_mode: none
unauthorized_effects_detected: false
```

## Remaining human-review question

This report closes the previously absent controlled measurements for the registered M1 stream. It does not by itself close M1. A human reviewer must decide whether the bounded scope is sufficient, and must explicitly accept or reject:

1. the 12-event controlled window as representative evidence for the single registered mutation funnel;
2. the complete zero-divergence ledger for that bounded window;
3. the deliberately narrow 5-of-525 / 5-of-532 silent-failure coverage and the explicit unobserved remainder;
4. the event granularity of one candidate per registered legacy mutation call and zero per standalone tick;
5. the separately recorded close-without-merge preservation of frozen PRs #4, #7, and #11.

Until a separate approval record is merged, M1 remains open, v4.2 is not opened, M2 does not begin, persistence and cutover remain disabled, and the pre-kernel legacy runtime remains authoritative.
