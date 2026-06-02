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


## D003 — Why Round97 stopped at source-package preflight

Choice:

- Do not implement Round97 runtime enable smoke until the actual Round96 source package or expanded source tree is present in the repository.

Reason:

- Round97 is behavior-changing work and must be implemented against the real source files, not reconstructed from handoff notes.
- The required checkpoint, rollback, audit, tests, and invariant checklist cannot be truthful without the real source tree.
- Fabricating a source tree or tests would violate the autonomous prompt restrictions.

Rejected alternatives:

1. Recreate source files from documentation: unsafe and likely inaccurate.
2. Add placeholder tests for missing code: fake validation.
3. Enable runtime mapping without the Round96 baseline: violates controlled mutation gates.

Result:

- Round97 is blocked on uploading or expanding `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip`.


## D004 — Why split package restoration is codified in the repo

Choice:

- Add a dedicated `packages/` upload location, README, and deterministic restore script for the split Round96 zip.

Reason:

- The source package is too large for the text-oriented handoff path and was split into two binary parts.
- Future Codex sessions need an exact, repeatable restore process before attempting Round97 runtime behavior changes.
- SHA-256 verification prevents Codex from using a corrupted or partial source package.

Rejected alternatives:

1. Continue documenting only that the package is missing: insufficient after the operator prepared split files.
2. Guess or recreate the Round96 source: unsafe and fake validation.
3. Skip manifest verification: unsafe for a behavior-changing round.

Result:

- Round97 remains blocked only until the split binary files and manifest are uploaded, after which restoration and validation can proceed deterministically.

## D005 — Why short package filenames are accepted

Choice:

- Accept both long split-package filenames and the operator-provided short names `part01`, `part02`, and `manifest`.

Reason:

- The operator reported the upload files by short names.
- Requiring only long names could falsely block restoration even when the correct binary payloads are present.
- SHA-256 verification still protects against corrupted, swapped, or partial inputs regardless of filename style.

Rejected alternatives:

1. Require users to rename short files: unnecessary friction.
2. Trust short files without manifest verification: unsafe.
3. Proceed without seeing files in this checkout: impossible and unsafe.

Result:

- The restore workflow is filename-tolerant but still blocked in this local checkout until the binary files are visible on disk.
