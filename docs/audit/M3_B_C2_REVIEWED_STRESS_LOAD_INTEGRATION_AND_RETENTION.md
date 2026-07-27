# M3-B C2 — Reviewed `stress_load` integration and sequence-five retention staging

## Baseline

- exact witness repository head: `3298d3b9911c79b1551a1d8bfe83bae756880840` — PR #211 squash merge
- canonical reviewed/runtime/source/candidate coverage before this activation: `4/37`
- canonical retained real-observation coverage before this activation: `4/37`
- immutable retained history before sequence five:
  1. `prediction_error_pressure` — sequence 1
  2. `energy_budget` — sequence 2
  3. `fatigue_pressure` — sequence 3
  4. `recovery_need` — sequence 4
- M3-B observation window: not started
- M3-C: closed
- M3-E authority: closed
- cutover: not authorized

## Independent public-safe witness review

The operator executed the merged #211 phone witness exactly once and supplied only its final public-review JSON. No private witness JSON, nonce bytes, raw process CPU/wall values, raw kernel load values, or private filesystem contents are committed here.

Canonical sorted compact JSON + SHA-256 recomputation produced and matched exactly:

```text
axis:                 stress_load
attestation digest:   7191e3493c582a191db3dcd488b2452dd3b0f29774b8a3a3ffeaff3b53c525fa
evidence digest:      5bceb97155a5614de72b2b359b861db5c57eb6e892c259b56981d6003fc14680
public review digest: 1ec63bb54cfed398b0e5b93af25667474c3255d5ca50a47602974d363cf5e03a
confidence:           0.999752717989861
value:                0.29720805203604805
raw record count:     3
fixture_only:         false
synthetic:            false
```

The exact three appraisal-input digests, three appraisal-integrity digests, and three snapshot-integrity digests are pinned by the reviewed witness record.

## Two-stage appraisal provenance boundary

`stress_load` is appraisal-required and is not a hardware-direct registry axis. The reviewed witness therefore preserves the merged #211 provenance boundary exactly:

```text
new real full-engine interactions
  -> operator-private process CPU / wall / kernel load measurements
  -> deterministic versioned appraisal bridge
  -> detached verified stress_load appraisal trace
  -> canonical appraised-survival evidence
```

The public review requires all of the following simultaneously:

```text
runtime_metrics_used_as_appraisal_input:              true
appraisal_bridge_output_detached:                     true
canonical_appraised_record_runtime_polled:            false
canonical_appraised_record_hardware_direct_input:     false
raw_runtime_metrics_publicly_retained:                false
runtime_input_kind:                                   operator_private_real_runtime_metrics
appraisal_output_kind:                                detached_verified_appraisal_trace
```

This does not relabel raw hardware measurements as canonical appraisal records. Real runtime metrics establish the production-origin input; the canonical source remains the detached deterministic appraisal output required by the existing `stress_load` source contract.

The governed appraisal methods remain exactly:

```text
process CPU:       os_times_process_cpu_v1
queue:             kernel_loadavg_1m_visible_cpu_ratio_v1
controllability:   one_minus_mean_overload_and_queue_variability_v1
demand:            mean_process_cpu_and_queue_ratio_v1
overload:          max_process_cpu_and_queue_ratio_v1
uncertainty:       absolute_queue_ratio_delta_v1
appraisal policy:  eve.m3-b.phone-stress-load-appraisal-policy.v1
appraisal version: eve.m3-b.survival-appraisal-trace.v1
```

## Fifth reviewed activation

`core/m3_b_c2_reviewed_stress_load_integration.py` binds the exact reviewed material to:

- one fifth reviewed operator attestation;
- one fifth runtime-provenance verifier path;
- one fifth production-source verifier for `eve:m3-b:registry-source:stress_load:v1`;
- one token-issued production-origin runtime verification object;
- one token-issued production-source verification object;
- one token-issued retained-observation-eligible capture object.

The reviewed/runtime/source/candidate boundary becomes `5/37`; canonical retained coverage remains `4/37` until a later explicit operator-private sequence-five append actually succeeds and its public-safe receipt is separately reviewed.

## Sequence-five durable retention staging

The staged event is exactly:

```text
event:          m3b:c2:retained:stress_load:000005
sequence:       5
prior event:    m3b:c2:retained:recovery_need:000004
prior envelope: 7619663391db95dc59951a3d12bba58af1bd1e01bb3cabbb89e862b55f3f9691
prior chain:    16efec6a9f775175fc99c252411d2e0ca6b3504799c824e8e5a70cf2697f1e0f
```

The append fails closed unless the operator-private SQLite stream contains exactly the immutable sequence-1/2/3/4 history with the pinned event ids, axes, envelope digests, public-review digests, event type/stream/producer, `shadow_only` authority, `retained_real_observation` classification, and the exact sequence-four store-chain digest above.

A successful future phone execution must prove ordinal 5, count `4 -> 5`, state change, exact readback of sequences 1-5, and unchanged closed authority flags. **This PR only stages that path. It does not execute the real append and does not claim retained coverage `5/37`.**

## Witness conversational-output boundary

The three full-engine interactions used to exercise the runtime produced terse/awkward surface responses. Those utterances are not silently corrected or reclassified here. This witness governs runtime/load appraisal provenance, not conversational-quality acceptance. The exact interaction session remains immutable real witness history and is not rerun merely to obtain more natural wording. Conversational coherence is a separate quality concern and cannot be used to rewrite or cherry-pick this witness evidence.

## Duplicate-validation boundary

PR #211 is an immutable merged prerequisite. Its accepted evidence must be reused rather than rerun:

```text
exact head:   1610777e8e502feb127d4544739a2f7907c0c3aa
exact run:    30235269663
focused:      6 passed
full:         3,201 passed
forward gate: PASS 0 / 0 / 0 / 0
artifact:     exact-head-validation-1610777e8e502feb127d4544739a2f7907c0c3aa
artifact SHA: 5c6d0709c9018dac885321b424b708a13bb05aef5da652c8cd76fe556429774b
M2-E run:     30235269682
M2-E:         6/6 passed
merge SHA:    3298d3b9911c79b1551a1d8bfe83bae756880840
```

A chat/session/operator-session change is not an invalidator. The #211 full suite, M2-E compatibility matrix, and the already-completed real `stress_load` witness must not be rerun merely because work moves to another chat or PR. The exact-head reuse ledger must record #211 in this PR.

Discovery/intermediate heads are not merge evidence. If the forward gate discovers new unregistered occurrences, the discovery head must stop before the full suite. Only the final forward-registered exact head may receive the accepted full-suite validation.

## Authority boundary

This activation remains `shadow_only`. It does not execute sequence five, start the M3-B observation window, complete M3-B, open M3-C, open M3-E authority, transfer persistence/runtime authority, or authorize cutover.
