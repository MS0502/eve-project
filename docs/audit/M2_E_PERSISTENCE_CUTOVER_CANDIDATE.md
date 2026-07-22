# M2-E Bounded Persistence Cutover Candidate

## Baseline and prerequisite

- Baseline: `c59095ccf75419e40107ec03fd20761ee946543d` — main after accepted M2-D PR #165.
- Accepted M2-D head: `ccf477d33b99c99302328dab1ff8e3292d9c4e91`.
- Accepted M2-D workflow: `29916248120`.
- Accepted M2-D artifact ZIP SHA-256: `c669e31928cb329dc80ee170c46cbc078a14edb78f9eb9b0311997d180e4f004`.
- Accepted M2-D packet digest: `8064f61c7dfea68a263918b764eb357f0055deb73f7df5dae24fae2a00f7e3d2`.

The accepted prerequisite is reused. It is not regenerated as a separate acceptance exercise merely because this work occurs in another chat or changes PR metadata.

## Scope

This PR defines the bounded M2-E technical and decision contracts for the single accepted stream:

```text
shadow:legacy.activation.learn_pair
```

It provides:

1. deterministic validation of the accepted M2-D packet;
2. an immutable technical-candidate packet bound to the exact M2-E head, workflow run, and expected artifact name;
3. a separate human-decision record that must pin that exact candidate packet and the independently verified artifact SHA-256;
4. a caller-invoked bounded authorization value;
5. recalculable post-cutover evidence containing store-integrity observations, canonical state/manifest evidence, replay equivalence, and rollback restoration.

## Authority boundary

The technical candidate cannot accept or promote itself. Before a separate exact-pin human decision:

- `authority=candidate_only`;
- event-store authority remains `shadow_only`;
- legacy persistence remains authoritative;
- legacy sidecars are not converted to read-only evidence;
- authoritative recovery is false;
- cutover authorization is false;
- runtime integration is false;
- production defaults remain unchanged;
- no observer, lifecycle bridge, scheduler, model, vector, affect, goal, memory, or external-effect authority is activated.

Even an accepted decision is bounded to the one reviewed stream and schema. The contract does not install a production hook or mutate defaults. A real integration patch would require separately reviewed runtime ownership and observation evidence rather than treating this value object as an automatic switch.

## Evidence requirements

The exact-head workflow must generate `m2-e-cutover-candidate.json` from the already generated M2-D packet and bind it to:

- the checked-out target head;
- the workflow run ID;
- `exact-head-validation-<head>`.

An informed human acceptance may be recorded only after the artifact ZIP has been downloaded independently and its SHA-256 verified. The decision must pin the exact head, workflow, artifact, artifact SHA-256, and candidate packet digest.

Post-cutover observation evidence must retain raw recalculable inputs rather than verdicts alone:

- integrity report fields before and after the bounded event window;
- event-count change;
- canonical before and authoritative state plus manifests and digests;
- replay-generated state using the same digest method;
- rollback-generated pre-cutover state using the same digest method;
- the exact authorization and decision digests.

Any replay mismatch, invalid integrity report, absent event advance, failed rollback, wrong candidate pin, or rejected decision fails closed.

## Two-stage promotion rule

This technical PR may become Ready and merge only after its exact technical evidence is independently reviewed and the project owner explicitly accepts that exact pin. Its merge establishes the bounded M2-E contract; it does not by itself prove that a production post-cutover observation window has occurred.

Actual authority use and any production integration remain separate, explicit, rollbackable work. No later M3 authority may be inferred from a candidate packet or from a PR being green.

## Validation reuse

Do not rerun the full suite for PR-body edits, comments, review metadata, or Draft/Ready-only changes while the exact head, workflow, artifact digest, and validation scope are unchanged. Rerun only after a head change, artifact loss or corruption, digest mismatch, or required validation-scope change.
