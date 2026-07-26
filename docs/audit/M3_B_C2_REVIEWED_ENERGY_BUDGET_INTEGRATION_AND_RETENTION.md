# M3-B C2 Reviewed Energy-Budget Integration and Sequence-Two Retention

Date: 2026-07-26

## Scope

This activation reviews the exact real-phone `energy_budget` witness produced by the merged PR #202 Android-compatible v2 collector. It registers the second reviewed C2 operator attestation/runtime provenance/source-verifier path and stages an operator-private durable append as retention-stream sequence 2.

It does **not** claim that the second durable append has already occurred. Until the operator executes the post-merge command and its public-safe receipt is separately reviewed and pinned, retained-real-observation coverage remains **1/37**.

## Reviewed witness pin

- witness repository head: `1161bb15d7bba0629d4862c05e8a61126cdb12c0`
- axis: `energy_budget`
- source instance: `runtime:phone-operational-energy:primary`
- source contract: `eve:m3-b:registry-source:energy_budget:v1`
- public review schema: `eve.m3-b.phone-energy-budget-public-review.v2`
- public review digest: `a2ce3d84111224e2009bf22d1e03a8f92acab0506e42515aac185ae05ff54ab4`
- attestation digest: `5413c35e912f95d90a1c0a5b0b8731a243bffc00e7b6338c1b7d9e4056e1c07f`
- evidence digest: `9d814295e3b59fb42294f3ba661866aa29c512866b946e80a3f397864974af13`
- raw records: `3`
- logical evidence tick: `2`
- confidence: `0.9993993636238185`
- fixture: `false`
- synthetic: `false`

The public digest was independently recomputed from canonical JSON with `public_review_digest` removed and matched exactly.

## Actual phone measurement methods

The reviewed witness records the exact Android-compatible methods that succeeded on the phone:

- CPU: `kernel_loadavg_1m_headroom_v1`
- memory: `proc_meminfo_available_v1`
- battery: `termux_api_battery_status_v1`

These identifiers bind the witness to actual kernel/API acquisition. The operator-private snapshots, raw CPU/memory/battery values, nonce, private companion files, SQLite database/WAL, and private paths are not repository material.

## Reviewed integration

`core/m3_b_c2_reviewed_energy_budget_integration.py`:

1. pins the exact public-safe witness and canonical digest;
2. reconstructs and verifies the public operator attestation;
3. registers an immutable energy-budget reviewed attestation entry;
4. registers an immutable runtime-provenance verifier for the reviewed phone launch;
5. registers the `energy_budget` production-source verifier for `eve:m3-b:registry-source:energy_budget:v1`;
6. verifies the positive-confidence evidence and source-manifest contract;
7. issues token-protected runtime/source/capture objects;
8. leaves retention, observation-window, M3-B completion, M3-C, M3-E, and cutover flags false.

After this integration is merged, repository-reviewed/verifiable coverage becomes 2/37, while durable retained coverage remains 1/37 until the real sequence-two append is executed and separately pinned.

## Sequence-two durable retention boundary

The first retained C2 event is immutable history:

- event id: `m3b:c2:retained:prediction_error_pressure:000001`
- sequence: `1`
- event envelope digest: `07deb0e7345db33ac7655229044c8d62e7b14198bd7d80611ace6f5352adb493`
- resulting store-chain digest: `d51406d84dc755f72bd2ab661563c75cf19244710bf98376dbe3174ff101c8ce`

`core/m3_b_c2_energy_budget_retention_activation.py` refuses to append unless the operator-private retention stream contains exactly that prior event and exact chain state. The staged second event is:

- event id: `m3b:c2:retained:energy_budget:000002`
- sequence: `2`
- causation id: `m3b:c2:retained:prediction_error_pressure:000001`

The append must prove store count `1 -> 2`, ordinal `2`, durable readback, exact prior-history continuity, and one-event chain advancement. A fresh/empty database, altered sequence-1 event, additional stream events, or duplicate sequence-2 execution fails closed.

The operator command intentionally uses the same private retention root/database as the first append. It does not create a parallel history and it can never re-append the first `prediction_error_pressure` observation.

## Authority boundary

This PR does not:

- mutate the 37-axis registry owner;
- use synthetic/CI material as real evidence;
- append the real phone retention event during CI;
- start the M3-B observation window;
- complete M3-B;
- open M3-C;
- open M3-E authority;
- authorize persistence/runtime cutover.

CI may exercise the retention code only against disposable test SQLite stores seeded with the already-pinned sequence-1 event. Such CI append results are fixtures and are never production evidence.

## Post-merge operator boundary

Only after merge may the operator run the sequence-two command on the clean exact merged head, using:

- the operator-private v2 energy-budget public-review file already produced by the phone witness; and
- the existing operator-private C2 retention root that contains the pinned first event.

The command prints one public-safe receipt. The private DB/WAL/raw witness data remain outside GitHub and chat. A later repository PR must independently verify and pin that receipt before official retained coverage advances from 1/37 to 2/37.
