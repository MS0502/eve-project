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

## 2. Execution discipline

- Prefer one final validation run on the final candidate head.
- Avoid intermediate pushes that exist only to retrigger the same suite.
- If a PR-triggered exact-head workflow already validated the exact final head, do not manually dispatch the same workflow merely because the conversation changed.
- Do not run a separate M2-E job when the accepted exact-head workflow already generated and validated the required M2-E candidate evidence for the same scope, unless a distinct M2-E acceptance contract explicitly requires a separate run.
- After merge, record the exact PR head, accepted run, merge SHA, artifact identity/digest when available, and current-main comparison. Future sessions start from this ledger plus the live repository, not from conversational memory alone.

## 3. Accepted historical evidence

### PR #243 — M3-C-R resumable phone goal-window operator

```text
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

Reuse boundary: this evidence certifies the accepted #243 implementation head. Do not rerun it for M3-C-S merely because the habitat retarget, chat/session, or operator context changed. A code/tree change to the operator would require its own new-head evidence.

### PR #245 — persistence integrity and semantic rebaseline preconditions

```text
exact head: 7582f1f38a9ed4e064942a48be93f7fbb01be580
exact-head run: 31683667893
result: all exact-head stages including full suite succeeded
M2-E run: 31683667832
result: all M2-E stages succeeded
squash merge: de3b15b6d4008555bdcf06e3ed53c62851ab3d8a
post-merge main comparison: de3b15b6d4008555bdcf06e3ed53c62851ab3d8a
parent: d9491f6b1dd2149338e37bb199274b63636e66f4
```

Reuse boundary: #245 is a docs/governance pin. Its accepted evidence is not a substitute for runtime implementation validation required by Task B.

## 4. Current planned evidence boundaries

### M3-C-S retarget / operational correction (Task A)

Task A is docs-only. Opening its PR may trigger the repository's existing exact-head workflow once. If that PR-triggered run succeeds on the final exact head, that run is the sole new exact-head evidence required for the Task A head; do not manually retrigger it because of chat/session changes.

Previously accepted #243/#245 runs remain historical prerequisite evidence and are not rerun for Task A.

### Persistence-integrity runtime implementation (Task B)

Task B changes runtime behavior and the exact-head workflow deterministic environment. Therefore #245's run cannot certify the final Task B head.

Required discipline:

1. consolidate implementation/tests/workflow pins before opening or synchronizing the final PR head where practical;
2. accept exactly one successful exact-head full-suite run for the final Task B head;
3. if a code fix changes that head, the previous run remains evidence for the old head but cannot certify the new one;
4. do not duplicate a separate validation solely because a new chat starts;
5. after squash merge, append Task B's exact head/run/artifact/merge/main comparison here.

## 5. Continuity versus code validation

The 30-day M3-C-S workstation continuity witness is operational evidence and is not created by rerunning CI. Conversely, CI exact-head evidence does not claim that 30 days of live process/store continuity occurred. Keep these evidence classes separate.
