# EVE v3 Rounds 213-217 Handoff

## Rounds completed

- Round213 diagnosed the one-command operator suite git-safety failure after the guarded medium30k, EVE self-learning, and runtime-mapping measurements reported green.
- Round214 corrected git-safety classification for operator-local output.
- Round215 added focused tests for allowed and blocked git-safety behavior.
- Round216 documents the final one-command operator workflow.
- Round217 records the broader validation delta and next recommendation.

## Root cause

The suite reached the final git-safety layer after the operator-local measurement steps were green, then failed closed because latest main still contained a legacy tracked upload zip at `eve_v3_autonomous_handoff/packages/eve_v3_round96_code_only_no_medium_vectors.zip`. The previous guard also treated all `_operator_artifacts` paths as equally forbidden, without documenting the one allowed exact local output path for the suite-owned JSON report.

## Corrected git-safety behavior

Allowed only when all are true:

- `_operator_artifacts/operator_local_validation_latest.json` is the exact local report path.
- The report exists only as ignored local output.
- The report is not staged and not tracked.
- No other guarded artifact path appears in git status.

Blocked fail-closed cases:

- Any staged/tracked `_operator_artifacts` file.
- Any unignored `_operator_artifacts` leakage, including the local report path if `.gitignore` is not protecting it.
- Any staged/tracked or unignored guarded artifact leakage for `vectors.npy`, `vocab.txt`, `subset_manifest.json`, `seeds/subsets`, zip files, part files, or upload zips.

## Final one-command operator workflow

```bash
python scripts/operator_run_local_validation_suite.py
```

If the command exits `0`, copy the `OPERATOR_LOCAL_VALIDATION_SUMMARY` block into the next Codex prompt. If it exits nonzero, do not paste a green summary; inspect the compact JSON blockers first.

No manual Python snippet is required from the operator.

## Production persistence boundary

Production persistence remains **NO-GO**. This handoff does not enable production persistence, does not change the default runtime mapping flag to true, does not enable enforcement, does not bypass AGP, and does not load vector artifacts by default.

## Next recommendation

After merge, the operator should rerun the one-command suite from latest main and paste the green copy-paste summary only if the command exits `0`.
