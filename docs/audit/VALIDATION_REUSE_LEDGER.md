# Validation Reuse Ledger

Status: **cross-session evidence index and non-duplication rule**. This file does not weaken validation requirements. It prevents chat/session/shell/branch/PR-metadata changes from being mistaken for technical invalidators.

## 1. Reuse rule

Accepted exact-head evidence remains reusable when all evidence-relevant facts remain unchanged:

1. the accepted exact head/tree for the claim being reused;
2. the validation scope and named dependency/precondition set relevant to that claim;
3. the workflow/test definition used to establish that evidence, unless a later change is proven irrelevant to the reused claim;
4. required artifacts and their recorded digests remain available and match;
5. ancestry required by the acceptance record remains intact.

The following **alone are not invalidators**:

- a new ChatGPT chat or agent session;
- a new shell or workstation login;
- creation/renaming of a branch;
- PR Draft/Ready state or other PR metadata changes;
- repeating a report about an already accepted SHA;
- a new operator session that does not alter the governed tree/state.

Valid reasons to obtain new evidence include an actual changed head/tree, artifact loss/corruption/digest mismatch, a validation-scope/dependency change, an evidence-relevant workflow/test change, or ancestry break.

A reported but not directly compared SHA/run is `comparison-pending`, not automatically `revalidation-pending`.

### Task B1 reproducible validation identity

For B1 and later environment-pinned validation, the reusable identity is the SHA-256 of canonical JSON containing exactly:

```text
commit_sha
tree_sha
python_pin
requirements_lock_sha256
validation_contract_sha256
```

The Python pin is `.python-version`; the lock is `requirements-lock.txt`; the contract is `docs/audit/VALIDATION_CONTRACT.json`. Chat/session/shell/branch/PR/workflow-run metadata are excluded from identity. A metadata-only change therefore cannot create a new validation obligation.

## 2. Execution discipline

- Prefer one final validation run on the final candidate head.
- Avoid intermediate PR synchronizations that exist only to retrigger the same suite.
- If a PR-triggered exact-head workflow already validated the exact final head, do not manually dispatch the same workflow merely because the conversation changed.
- The repository's distinct M2-E window-driver remains separate evidence when its acceptance contract requires it; reuse a successful run for the same final identity rather than duplicating it.
- After merge, record exact PR head, accepted run, merge SHA, artifact identity/digest when available, and current-main comparison. Future sessions start from this ledger plus the live repository, not conversational memory alone.

## 3. Accepted historical evidence

The following records predate B1's exact interpreter/lock policy. They remain **accepted legacy evidence / environment-unpinned / not environment-reproducible** for the claims they established. B1 never retroactively invalidates them.

### PR #243 — M3-C-R resumable phone goal-window operator

```text
classification: accepted legacy evidence / environment-unpinned / not environment-reproducible
exact head: a4c8d0ec1a1767b5ccdbc105c40af94a327eb741
exact-head run: 30736080203
focused: 9 passed
full: 3,400 passed
M0: byte-identical
M2-B: valid, errors 0
forward gate: 0 / 0 / 0 / 0
M2-E run: 30736080196
M2-E: 6/6 passed
artifact SHA-256: 710e56e3c35455504e2862add82ec067756e707ba833e3cf3d476a0cc0fc25ac
squash merge: d9491f6b1dd2149338e37bb199274b63636e66f4
```

Reuse boundary: this evidence certifies the accepted #243 implementation head. Do not rerun it merely because habitat, chat/session, branch, or operator context changed.

### PR #245 — persistence integrity and semantic rebaseline preconditions

```text
classification: accepted legacy evidence / environment-unpinned / not environment-reproducible
exact head: 7582f1f38a9ed4e064942a48be93f7fbb01be580
exact-head run: 31683667893
result: all exact-head stages including full suite succeeded
M2-E run: 31683667832
result: all M2-E stages succeeded
squash merge: de3b15b6d4008555bdcf06e3ed53c62851ab3d8a
post-merge main comparison: de3b15b6d4008555bdcf06e3ed53c62851ab3d8a
parent: d9491f6b1dd2149338e37bb199274b63636e66f4
```

Reuse boundary: #245 is accepted governance evidence. It is not a substitute for Task B runtime/environment validation.

### PR #247 — workstation retarget and t=0 operational correction

```text
classification: accepted legacy evidence / environment-unpinned / not environment-reproducible
exact head: 3a1aa1e9dc52ca738e20d2791eb7fb9408f2c77c
exact-head run: 31689169525
M2-E run: 31689169521
exact-head artifact: exact-head-validation-3a1aa1e9dc52ca738e20d2791eb7fb9408f2c77c
artifact SHA-256: e6b4ff8e797bcb9ba46330dff2ca5eb792f781107124515485854e16c8c21cba
squash merge / accepted main: 0f33a91715845ce7814ad465c771bdec6df6f17b
parent: de3b15b6d4008555bdcf06e3ed53c62851ab3d8a
```

Reuse boundary: #247 is the accepted B1 baseline. Its acceptance is preserved even though its interpreter and broad dependency install were not environment-pinned.

## 4. Task B1 evidence boundary

The final B1 candidate must contain all of the following before acceptance validation starts:

1. exact `.python-version` pin;
2. runtime/development/experimental-legacy dependency split;
3. hash-pinned `requirements-lock.txt`;
4. setup-python and common environment preflight in exact-head and M2-E;
5. `VALIDATION_CONTRACT.json` and identity digest rule;
6. this ledger update.

The first successful pinned B1 exact-head on the final candidate establishes the **reproducible-from-here** boundary. If the candidate head/tree, Python pin, lock digest, or validation-contract digest changes, old evidence stays attached to the old identity and the new identity must validate. Do not rerun #243/#245/#247 for B1.

## 5. Continuity versus code validation

The M3-C-S workstation continuity witness and Task B2 physical sustained-load gate are operational/physical evidence and are not created by rerunning CI. Conversely, CI exact-head/M2-E evidence does not claim live workstation continuity or physical sustained-load proof. Keep these evidence classes separate.
