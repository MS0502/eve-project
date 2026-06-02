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


## D006 — Why Round97 still cannot proceed after checkout update attempt

Choice:

- Stop before Round96 restoration and Round97 runtime mutation because the uploaded split package files are not visible in the current execution checkout.

Reason:

- Round97 depends on the real Round96 source package.
- This local repo has no configured git remote, so Codex cannot update the checkout from a PR branch.
- A direct GitHub probe is blocked in this environment.
- Proceeding without the binary parts would require fabricating or reconstructing source, which is forbidden.

Rejected alternatives:

1. Reconstruct the source package from docs: fake source state.
2. Skip SHA-256 verification: unsafe.
3. Implement Round97 against absent source files: invalid validation.

Result:

- The active hard stop remains `missing_split_package_files_in_execution_checkout`.


## D007 — Why the code-only Round96 package is preferred

Choice:

- Prefer `eve_v3_round96_code_only_no_medium_vectors.zip` over the legacy split package when it is visible in the checkout.

Reason:

- The code-only package is below the GitHub upload limit and avoids the repeated split-file visibility problem.
- It excludes only the medium fastText vector file, which is not required for the Round95~Round96 focused/adjacent mapping validation or Round97 controlled runtime mapping enable smoke.
- The manifest SHA-256 check and zip integrity test still provide deterministic package validation.

Rejected alternatives:

1. Keep relying only on the split package: repeatedly blocked by checkout visibility issues.
2. Treat code-only validation as full fastText-backed validation: inaccurate because the medium vector is intentionally absent.
3. Continue without package verification: unsafe.

Result:

- Codex should use the code-only zip first, and mark any medium-vector-dependent full validation as blocked or partial.
