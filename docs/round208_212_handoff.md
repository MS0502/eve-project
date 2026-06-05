# EVE v3 Rounds208-212 handoff

## Rounds completed

- **Round208** added the stable one-command operator-local suite: `scripts/operator_run_local_validation_suite.py`.
- **Round209** made the suite emit compact JSON, write `_operator_artifacts/operator_local_validation_latest.json`, and print a copy-paste summary block for the next Codex prompt.
- **Round210** added focused suite/report behavior tests that use fakes for guard behavior and do not fabricate vector contents.
- **Round211** documents the exact one-command operator workflow below.
- **Round212** records the broader validation delta and next recommendation.

## Operator one-command workflow

Run exactly:

```bash
python scripts/operator_run_local_validation_suite.py
```

The suite runs these checks in order and stops on the first failure:

1. `python scripts/operator_validate_medium30k.py --attempt-load`
2. `python scripts/operator_remeasure_eve_self_learning.py --artifact-dir _operator_artifacts/subset_medium_30k --target-word 민석 --context-word 한국어 --context-word 감정 --context-word 기억 --context-word 대화`
3. `python scripts/operator_measure_runtime_mapping_after_self_learning.py --artifact-dir _operator_artifacts/subset_medium_30k --target-word 민석 --context-word 한국어 --context-word 감정 --context-word 기억 --context-word 대화 --negative-token EVE`
4. Git status safety check for `_operator_artifacts`, vector/subset files, `seeds/subsets`, `*.zip`, and `*.part`.

## Output contract

The suite prints:

1. One compact JSON line.
2. An `OPERATOR_LOCAL_VALIDATION_SUMMARY:` block that can be pasted into the next Codex prompt.

It also writes the latest local report to:

```text
_operator_artifacts/operator_local_validation_latest.json
```

That file is operator-local and must not be staged or committed.

## Policy boundaries

Production persistence remains **NO-GO**. The suite must not enable production persistence, change the `runtime_mapping_enabled` default, enable enforcement, bypass AGP, create dummy vectors, download artifacts, stage files, or commit artifacts.

Do not stage or commit `_operator_artifacts`, `vectors.npy`, `vocab.txt`, `subset_manifest.json`, `seeds/subsets` changes, zip files, or part files.

Korean-first defaults remain `민석` with context words `한국어`, `감정`, `기억`, and `대화`; the negative token remains `EVE`.

## Next recommendation

After merge, the operator should run the one-command suite and paste the summary block into the next Codex task before any future runtime-mapping or persistence discussion. Production persistence remains out of scope.

## Cloud validation delta

- Focused suite tests are green in Codex Cloud: `python -m pytest -q tests/test_v3_round208_212_operator_local_validation_suite.py`.
- Compile check is green: `python -m compileall -q adapters tests main.py scripts`.
- Collection is green: `pytest --collect-only -q` collected 1350 tests.
- Full pytest remains red in this environment with 206 failures. The visible failure taxonomy is the existing missing vector-artifact baseline (for example, `seeds/subsets/cc.ko.300.subset.mini.1k/vectors.npy` is absent), not a production persistence issue and not a reason to fabricate vectors.
