# Round158 — Safe handling strategy for missing vector artifacts

Round158 selected the safe handling strategy for the seed/vector artifact dependency cluster.

## Decision

Use a deterministic, read-only operator-artifact readiness gate that reports whether registered fastText subsets are ready for explicit loading.

## Rejected strategies

The following strategies remain forbidden and were not used:

- Creating dummy vectors.
- Fabricating vector checksums.
- Editing `seeds/MANIFEST.yaml` to hide absent artifacts.
- Committing `vectors.npy` or seed subset binaries.
- Skipping, xfail-ing, weakening, or deleting existing load-dependent tests.
- Enabling production persistence.
- Enabling runtime mapping by default.
- Enabling enforcement.

## Safe handling outcome

If real operator artifacts are absent, the readiness gate returns `blocked_operator_artifact_required` and `load_should_be_attempted = false`. This is skip-free: it does not hide failures and does not instruct pytest to skip.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND158_SAFE_HANDLING_DECISION_STATUS.json`.
