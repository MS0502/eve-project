# M2-E Cutover Human Authorization Record

## Decision

**ACCEPTED — CUTOVER AUTHORIZED WITH PER-DOMAIN LEGACY AUTHORITY RETAINED.**

On 2026-07-27, project authority 김민석 sent the integrated dispatch whose transmission was explicitly defined as the cutover approval signature. This record preserves that human decision as an immutable append-only A12 artifact.

Canonical decision JSON SHA-256:

```text
3844e4d0a836924eb881048d45d98d89d5041f87d15a836686119a2d8487efbf
```

## Evidence pin

```text
seal_digest:        5bfd2bae9a60107b5bd647eeec30b602a4d6bca922e467755f17a04c990dafbb
acceptance checks:  12/12
events:             288
death recoveries:   2
observed midnights: 4
```

The operator also supplied shorthand prefixes `evidence de921e56...`, `raw 51b021c2...`, `state 8bcbea...`, and `config 2328da...`. Those are recorded in the canonical JSON strictly as reported prefixes. They are **not** expanded or represented as full SHA-256 values. The full `seal_digest` above is the exact package identity supplied for this decision.

## Approved scope

1. Promote the event kernel plus SQLite store to the authoritative persistence substrate for **v4-native subsystems**.
2. Admit the verified shadow history — 288 events, 2 death recoveries, and 4 observed midnights — as retained authoritative history.
3. Keep the legacy runtime authoritative for each legacy domain until that domain completes its own separately gated per-domain migration.
4. Open `m3_authority_open = true`.

## Explicit non-approval

This decision does **not** transfer authority for any legacy domain. Each such transfer requires its own migration gate.

This decision does **not** authorize the M3-E affect cutover. M3-E retains its separate gate.

## A12 append-only rule

The JSON and this human decision are immutable after merge. Any correction, withdrawal, revocation, or replacement must be a separate append-only supersession artifact that explicitly references the canonical decision JSON digest above and passes the same exact-head/human-review regime.

## Duplicate-validation boundary

A chat, session, operator-session, PR-body/comment/review metadata change, or Draft/Ready transition is not a validation invalidator. Accepted exact-head evidence is rerun only after a real tree/head change, artifact loss or corruption, digest mismatch, required validation-scope/dependency change, or merge-ancestry break.

This artifact records the human authorization only. The separately reviewed A-2 execution PR remains responsible for binding runtime cutover flags to this artifact digest, preserving the legacy persistence path for the seven-day parallel period, providing tested one-command rollback, and updating repository status.