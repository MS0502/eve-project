# DECISION_LOG

## Round97

Decision: implement controlled runtime mapping enable smoke as ephemeral only.

Rationale:

- Round96 proved `민석` ready for a separate enable smoke.
- The smoke must prove the runtime flag can open and close without persistence.
- Enforcement must remain disabled.
- Lexical, EveSpecific, and seed vectors remain evidence only, not AGP anchors.

Outcome:

- `민석` mapped only during the smoke.
- Rollback restored `runtime_mapping_enabled=False`.
- No hard stop.

## Round98

Decision: audit persistence readiness but do not persist runtime mapping.

Rationale:

- Round97 rollback was complete.
- Medium vectors are absent from the code-only package, so full validation is blocked/partial.
- Persistence requires operator approval and full validation or explicit partial-validation waiver.

Outcome:

- Persistence gate status is ready for operator decision.
- Persistence remains unapplied.
