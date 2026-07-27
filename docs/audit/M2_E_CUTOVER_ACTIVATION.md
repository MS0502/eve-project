# M2-E Cutover Activation — v4-native authoritative persistence substrate

## Baseline

- A-1 human authorization PR: `#213`
- A-1 squash merge / A-2 exact base: `e98e007ea0418995b1056038ba2ad846ecd847de`
- canonical A-1 decision JSON SHA-256: `3844e4d0a836924eb881048d45d98d89d5041f87d15a836686119a2d8487efbf`
- sealed window digest: `5bfd2bae9a60107b5bd647eeec30b602a4d6bca922e467755f17a04c990dafbb`
- accepted window: `12/12`, events `288`, death recoveries `2`, observed midnights `4`

A-1 is immutable human authority. This PR may execute only the exact scope that A-1 authorized.

## Activated authority

After this PR merges, the digest-pinned authority resolver exposes:

```text
cutover_authorized: true
m3_authority_open: true
event_store_role: authoritative_persistence_substrate_for_v4_native_subsystems
retained_shadow_history_authoritative: true
legacy_runtime_authority: authoritative_per_domain_until_separate_domain_migration_gate
legacy_domain_authority_transfer_authorized: false
m3_e_affect_cutover_authorized: false
legacy_persistence_path_changed: false
minimum_legacy_parallel_days: 7
```

The verified 288-event shadow history is accepted as retained authoritative history by the A-1 decision. Existing immutable event envelopes are not rewritten merely to replace their historical `shadow_only` labels; promotion is represented by the higher-level digest-pinned authority decision.

## Per-domain legacy boundary

This cutover does **not** move any legacy domain to v4-native authority. Legacy runtime remains authoritative for each domain until a separately reviewed domain migration gate says otherwise.

The seven-day parallel requirement is a minimum rollback-preservation interval, not an automatic domain-transfer timer. Reaching day seven cannot transfer a legacy domain, delete a legacy persistence path, or open M3-E.

This PR intentionally does not modify any legacy persistence path.

## M3 boundary

`m3_authority_open=true` opens the M3-C authority gate after merge. It does not claim M3-C implementation is complete.

M3-E affect cutover remains false and separately gated.

The M3-B retained-real-observation program remains independent evidence work. Its `stress_load` sequence-five path is already staged by merged #212, but retained coverage remains `4/37` until the operator performs the one real private append and its receipt is reviewed.

## One-command operational rollback

Rollback is fail-closed operational control, not revocation of A-1 human approval. It lowers v4-native persistence authority back to `shadow_only`, closes `m3_authority_open`, and leaves all legacy-domain authority unchanged.

From a clean repository checkout, the default private control location is `~/.local/share/eve-m2e-window-private`:

```bash
python scripts/operator/m2_e_cutover_rollback.py
```

The command writes exactly one private `m2_e_cutover_operational_rollback.json` control record with mode `0600` under a `0700` private root. Re-running the same command is idempotent; a conflicting existing record is not overwritten.

The rollback record is cryptographically bound to the A-1 decision digest. A tampered digest, M3-E opening, legacy-domain transfer, legacy-path-change claim, or attempted human-authorization revocation fails closed.

## Restore / supersession boundary

Operational rollback does not alter the immutable A-1 artifact. A public correction, withdrawal, or revocation of the human authorization itself still requires a separate append-only supersession artifact referencing the A-1 canonical digest.

A future restore after an operational rollback must be separately reviewed; this PR does not add an automatic restore command.

## Duplicate-validation boundary

The accepted #213 exact-head evidence is a prerequisite and is not rerun because work moved to this PR or another chat. This A-2 tree receives its own exact-head validation once its forward registrations are final.

Chat/session/operator-session changes, PR body/comments/review metadata, and Draft/Ready transitions are not invalidators. Tree/head change, artifact loss/corruption, digest mismatch, required validation-scope/dependency change, or merge-ancestry break are invalidators.