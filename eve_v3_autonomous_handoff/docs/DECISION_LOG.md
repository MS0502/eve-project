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

## Round99

Decision: classify the merged PR #2 state as `blocked_partial` rather than passed.

Rationale:

- The required Round97/98 and Round92~Round98 test fixtures depend on creating an EveSpecific vector for `민석` from known fastText context words.
- `seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy` is absent, and no small/mini fallback `vectors.npy` artifact is present in this checkout.
- Without a loaded fastText subset, the commit gate correctly rejects the candidate with `insufficient_known_context`.
- Marking the validation passed would violate the hard stop against claiming full validation while medium vectors are absent.

Outcome:

- Round100 feature work was not started.
- Next selected recommendation: medium vector restoration / validation plan before AGP proof expansion or persistence approval design.

## Round100

Decision: implement an operator-supplied artifact audit and validation-tier separation instead of adding a binary vector artifact to the repository.

Rationale:

- The medium 30k `vectors.npy` file is required for honest medium/full validation, but adding it to the PR diff would violate the code-only handoff boundary.
- Creating dummy vectors or fake checksums is forbidden.
- Small 5k fallback may only be used for focused validation if the exact manifest-verified small artifact is present; it is not a medium validation substitute.

Outcome:

- Added `adapters/medium_vector_restoration.py` and focused Round100 tests.
- Runtime mapping persistence remains disabled.
- AGP proof expansion remains deferred until validation is unblocked or the operator explicitly approves a partial-validation path.

## Round101

Decision: stop the autonomous multi-round run and prepare one final integrated PR.

Rationale:

- Issue #5 requires multiple rounds on one branch when possible, with internal reports/validation JSON and one final PR only.
- Round100 already completed the only safe code-only step for the current highest-priority blocker: an operator-supplied medium vector audit path.
- The actual blocker now requires an external medium `vectors.npy` artifact or explicit partial-validation approval.
- Proceeding would require committing a binary artifact, creating fake vectors, weakening tests, or claiming blocked validation as passed, all of which are forbidden.

Outcome:

- Hard stop reason: external artifact/operator action required.
- Runtime mapping persistence remains disabled.
- AGP proof object expansion and legacy root blocker isolation remain deferred until validation substrate restoration or explicit operator approval.

## Round102

Decision: attempt the operator-supplied Release artifact restore through a deterministic temp-only helper, but keep the hard stop active in this environment.

Rationale:

- The operator supplied the medium 30k artifact as GitHub Release assets, which is a valid non-PR-diff delivery path.
- The assets must be used only as external artifacts; wrapper zips, raw parts, restored zip, and `vectors.npy` must never be staged into the PR.
- The current environment returned HTTPS CONNECT 403 for all Release asset downloads, so checksum/shape/dtype gates could not be reached.
- Claiming hard-stop release without observing the artifact audits locally would violate validation honesty.

Outcome:

- Added `adapters/medium_vector_release_restore.py` for network-enabled or manual local restore.
- Added focused fail-closed tests for the restore helper.
- Hard stop remains active until the Release assets are downloaded/available locally and the helper reports `hard_stop_released=true`.

## Round103

Decision: add an operator/manual install validation workflow instead of continuing to retry direct Release downloads from Codex.

Rationale:

- Round102 established that the Codex environment is blocked by HTTPS CONNECT 403 for GitHub Release assets.
- The operator has supplied the artifact, but actual binary restore must happen outside the PR diff in a network-enabled/manual environment.
- A single deterministic validation command reduces operator error after manual install while preserving the no-binary-commit boundary.

Outcome:

- Added `adapters/medium_vector_manual_validation.py` and focused tests.
- Current hard stop remains active in this checkout because the medium `vectors.npy` is absent.
- Runtime mapping persistence and AGP proof object expansion remain deferred until Round103 validation can run against a verified local artifact.

## Round104

Decision: accept the operator-reported Codespaces validation as the artifact-unblocked validation record for the persistence approval gate, without claiming it as local Codex execution.

Rationale:

- The Codex environment cannot download Release assets, but the operator manually restored the medium artifact in Codespaces and reported focused validation success.
- Binary safety was preserved: `vectors.npy` is gitignored, `_operator_artifacts/` is temporary, and no binary artifact enters the PR diff.
- Runtime mapping persistence is a separate mutation decision and must not be applied implicitly by a validation record.

Outcome:

- Round104 gate status is `ready_for_explicit_operator_persistence_approval`.
- Runtime mapping remains disabled by default and enforcement remains disabled.
- Round105 AGP proof object expansion is allowed as data-only work.

## Round105

Decision: expand AGP proof data only; do not call AGP verify or create anchors.

Rationale:

- Round104 has no hard stop for proof expansion.
- AGP anchoring must remain explicit-category plus SA activation; lexical vectors remain evidence only.
- Runtime mapping persistence has not been approved or applied, so proof expansion must not depend on runtime default changes.

Outcome:

- Added a read-only AGP proof object expansion.
- No AGP bypass, runtime persistence, category/memory mutation, or vector commit occurred.
