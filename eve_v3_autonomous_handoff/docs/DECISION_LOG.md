# DECISION_LOG

This file records why each technical choice was made.

## D001 — Why Round96 is a precheck

Choice:

- Round96 does not activate runtime mapping.
- Round96 only prepares the gate for the next controlled smoke round.

Reason:

- Activation changes runtime behavior.
- Behavior-changing work needs checkpoint, rollback notes, tests, and audit files.
- The chat execution environment did not finish the full test suite because the process was interrupted by a time limit.

Rejected alternatives:

1. Activate immediately: too risky without full validation.
2. Only write docs: not enough progress.
3. Stop at Round95: next gate remains unclear.

Result:

- Next step is Round97 controlled runtime mapping enable smoke.

## D002 — Why docs are stored in the repo

Choice:

- Keep worklog, decisions, technical map, next actions, and operator guide in this repository.

Reason:

- The user is the planner/operator, not the technical implementer.
- Agents need persistent context to continue without repeated technical instructions.

Result:

- Future Codex sessions should read this directory first.
