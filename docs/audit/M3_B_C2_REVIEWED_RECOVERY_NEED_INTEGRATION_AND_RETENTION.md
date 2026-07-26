# M3-B C2 — Reviewed `recovery_need` integration and sequence-four retention staging

## Baseline

- exact witness repository head: `f0edb05201671814fed131ccbb73d2cb3b8d3f59` — PR #208 squash merge
- canonical retained coverage before this activation: `3/37`
- immutable retained history before sequence four:
  1. `prediction_error_pressure` — sequence 1
  2. `energy_budget` — sequence 2
  3. `fatigue_pressure` — sequence 3
- M3-B observation window: not started
- M3-C: closed
- M3-E authority: closed
- cutover: not authorized

## Independent public-safe witness review

The operator supplied only the final public review JSON produced by the merged #208 phone witness. No private witness JSON, nonce bytes, raw process CPU/wall values, raw kernel load values, or private filesystem path is committed here.

Canonical sorted compact JSON + SHA-256 recomputation produced:

```text
axis:                recovery_need
attestation digest:  ce8cda9955a415ed05200a83fd6b3e8d4cd4028bef29f73b8b17a1d5e3ad25e1
evidence digest:     535495759c0140d875da628d2fe5cc9ffc0904d5f91fa9546a784dd51b3baa4b
public review digest:e46df034d01b13e768ce37d14261b8ed20fdec30101945bea492d97e482e4c33
confidence:          0.9999985112700722
value:               0.2559025046410264
raw record count:    3
fixture_only:        false
synthetic:           false
```

The public review also pins the expected versioned measurement identifiers:

```text
process CPU:     os_times_process_cpu_v1
queue:           kernel_loadavg_1m_capacity_comparison_v1
cooldown:        fixed_post_interaction_quiet_window_1s_v1
overload count:  loadavg_visible_cpu_capacity_breach_count_v1
recovery count:  cpu_and_queue_nonincrease_indicator_count_v1
```

All three snapshot-integrity digests are pinned. The witness remains `shadow_only` and explicitly pre-claims none of reviewed registration, runtime verifier registration, production-source verifier registration, retention, observation-window start, M3-B completion, M3-C/M3-E authority, or cutover.

## Fourth reviewed activation

`core/m3_b_c2_reviewed_recovery_need_integration.py` binds the exact reviewed material to:

- one fourth reviewed operator attestation;
- one fourth runtime-provenance verifier path;
- one fourth production-source verifier for `eve:m3-b:registry-source:recovery_need:v1`;
- one token-issued production-origin verification object;
- one token-issued retained-observation-eligible capture object.

The resulting reviewed boundary is `4/37`; retained coverage remains `3/37` until the real operator-private sequence-four append succeeds.

## Sequence-four durable retention staging

The staged event is exactly:

```text
event:          m3b:c2:retained:recovery_need:000004
sequence:       4
prior event:    m3b:c2:retained:fatigue_pressure:000003
prior envelope: f81d43bf40b4dc76130767f91b65ad2503bc70e61ef718fe3d0e446528d1a7e3
prior chain:    b73ec7ea2f5e6e4e8eda5b57b4f6464a17d94e56026718b5b2e15cbca9f2162f
```

The append fails closed unless the private SQLite stream contains exactly three prior retained events with the pinned sequence, axis, envelope digest, public-review digest, retention event type/stream/producer, `shadow_only` authority, and `retained_real_observation` classification. It additionally requires the exact sequence-three store-chain digest above before the append.

A successful future phone execution must prove ordinal 4, count `3 -> 4`, state change, exact readback of sequences 1-4, and unchanged closed authority flags. This PR does not claim that execution has happened.

## Duplicate-validation boundary

PR #208 is an immutable merged prerequisite. Its accepted evidence is reused rather than rerun:

```text
exact head:   61b2b8790ac0195e4f8212924177348ad94bcb2b
exact run:    30198670805
focused:      7 passed
full:         3,190 passed
forward gate: PASS, 0 / 0 / 0 / 0
artifact SHA: 7798899f7f24150323dd052e970139108e9fe2fcb3b2a9b9c352a5fcf5ef2a69
M2-E run:     30198670820
M2-E:         6/6 passed
merge SHA:    f0edb05201671814fed131ccbb73d2cb3b8d3f59
```

A chat/session change is not an invalidator. The exact-head reuse ledger must record #208 in this PR. Discovery/intermediate heads are not merge evidence; the full suite may run only once on the final forward-registered exact head.

## Authority boundary

This activation remains `shadow_only`. It does not start the M3-B observation window, complete M3-B, open M3-C, open M3-E authority, transfer persistence/runtime authority, or authorize cutover.
