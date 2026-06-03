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

Decision: add manual medium-vector validation as a fail-closed checkpoint.

Rationale:

- Runtime mapping persistence must not proceed from an unverified, absent, fake, or checksum-mismatched medium vector artifact.

Outcome:

- Validation remains read-only and writes JSON only.
- No vector artifact is committed or installed by the validator.

## Round104

Decision: represent runtime mapping persistence approval as an explicit packet, not as an applied state change.

Rationale:

- Persistence requires operator approval after gate and vector validation evidence.

Outcome:

- Approval can become ready for decision review.
- Runtime mapping remains disabled and unpersisted.

## Round105

Decision: expand AGP proof rows while preserving the AGP anchor boundary.

Rationale:

- Runtime mapping candidates need proof that anchors remain explicit categories with SA activation, not lexical/vector shortcuts.

Outcome:

- Proof rows are read-only data.
- AGP verification is not called by the proof expansion.

## Round106

Decision: record persistence readiness without applying persistent runtime mapping.

Rationale:

- Applying persistent mapping is a separate state-changing patch and must not be smuggled into the decision packet.

Outcome:

- The decision packet may report `persistence_ready_but_not_applied`.
- Runtime mapping and enforcement remain disabled.

## Round107

Decision: add the runtime mapping persistence activation dry-run harness before any real persistence enablement.

Rationale:

- The project needs explicit checkpoint, rollback, audit-log, state-debug, and touch-plan formats before a later activation patch can safely change runtime defaults.
- Defining these formats as a dry-run preserves the disabled boundary while making future activation auditable.

Outcome:

- `runtime_mapping_enabled` remains `False` by default.
- `enforcement_enabled` remains `False` by default.
- No runtime mapping persistence is applied.
- No AGP/vector/category/concept-memory mutation is performed.

## Round108

Decision: add a guarded runtime mapping persistence activation candidate without default persistence enablement.

- Requires Round106 decision and Round107 dry-run prerequisites.
- Requires explicit operator approval token.
- Creates checkpoint before candidate mutation.
- Emits audit log and before/after state-debug exports.
- Rolls back and verifies disabled runtime/enforcement flags plus protected state surfaces.
- Keeps `runtime_mapping_enabled=False` and `enforcement_enabled=False` by default.

## Round109 decision

- Accepted an operator approval fixture only; real runtime mapping persistence remains disabled by default.
- Approval scope is `runtime_mapping_persistence_only`; explicit token allowlist is `["민석"]`.
- Rollback drill evidence is required before any later persistence enablement discussion.

## Round110 decision

- Proceeded with a limited persistence sandbox rather than production persistence.
- Allowed mutation: in-memory `runtime_mapping_enabled=True` only between checkpoint and rollback inside the sandbox runner.
- Forbidden state remains forbidden: enforcement, production persistence, AGP bypass, vector mutation, category mutation, memory mutation, and binary/operator artifacts.

## Round111 decision

- Converted the Round110 sandbox state file into a cleanup target rather than any persistent project state.
- Cleanup success requires verified Round110 checkpoint/audit/rollback evidence plus disabled runtime flags.
- The cleanup receipt is JSON-only observability, not production persistence.

## Round112 decision

- Added read-only audit replay before building any viewer/dashboard surface.
- Replay validates Round110/111 artifacts and disabled flags without re-applying sandbox mutation.
- Next recommended round is Round113 state-debug/audit replay viewer; production persistence remains blocked pending explicit operator approval.

## Round113 decision

- Added a read-only state-debug/audit replay viewer before any production persistence decision.
- Viewer output is evidence only; it does not reapply sandbox mutation.

## Round114 decision

- Isolated the broader root collection blocker as pre-existing legacy imports of missing `spreading_activation`.
- Decided not to weaken, skip, or rewrite legacy root tests in this runtime mapping PR.

## Round115 decision

- Recorded broader validation as blocked/partial where appropriate while preserving focused pass results.
- Broader validation blockers do not justify enabling production persistence.

## Round116 decision

- Added a regression guard that replays the sandbox/cleanup/replay chain before any future persistence decision.
- The guard remains JSON-only and does not create vectors or operator artifacts.

## Round117 decision

- Packaged Round113-116 evidence for operator review.
- Recommendation is `no_go_for_production_persistence_in_this_pr`; production persistence, runtime mapping default enablement, and enforcement remain disabled.

## Round118

Decision: audit production persistence readiness without enabling it.

Rationale:

- Round113-117 evidence is sufficient for operator review, but not for autonomous production persistence.
- Broader validation remains blocked/partial and must not be reported as green.

Outcome:

- Recommendation: `NO-GO`.
- Runtime mapping production persistence remains disabled.

## Round119

Decision: convert readiness evidence into a minimal risk matrix and operator checklist.

Rationale:

- Any future activation requires explicit operator approval, validation disposition, flag checks, artifact-boundary checks, and AGP boundary preservation.

Outcome:

- Required checklist items remain unsatisfied.
- Recommendation remains `NO-GO`.

## Round120

Decision: issue a final pre-activation no-go/go gate package and do not activate.

Rationale:

- Explicit operator approval is absent.
- Broader validation remains blocked/partial.
- Required checklist items are unsatisfied.

Outcome:

- Final recommendation: `NO-GO`.
- Activation action taken: `false`.

## Round121

Decision: isolate blockers instead of preparing an approval request.

Rationale:

- Round120 was `NO-GO`, so the correct next action is blocker isolation, not activation.

Outcome:

- Required blockers are listed for operator review.
- Production persistence remains disabled.
