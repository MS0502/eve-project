# EVE v3 Rounds187-191 Report — Operator Medium 30k Validation Script

## Rounds completed

- **Round187:** Added `scripts/operator_validate_medium30k.py` as the stable operator-local validation entrypoint for `_operator_artifacts/subset_medium_30k/`.
- **Round188:** The script emits compact single-line JSON and fails closed with exit code `1` only when validation, git-safety, preflight, or explicit-load validation fails.
- **Round189:** Added focused tests for script behavior without requiring real `vectors.npy`, `vocab.txt`, or `subset_manifest.json` artifacts.
- **Round190:** Documented the exact one-command operator workflow.
- **Round191:** Captured broader validation expectations and the next recommendation while keeping production persistence NO-GO.

## One-command operator workflow

Run from the repository root in the operator Codespace where the manually verified artifact exists:

```bash
python scripts/operator_validate_medium30k.py
```

The default command is read-only and no-load. It validates:

- `_operator_artifacts/subset_medium_30k/vocab.txt` exists.
- `_operator_artifacts/subset_medium_30k/vectors.npy` exists.
- `_operator_artifacts/subset_medium_30k/subset_manifest.json` exists.
- `vectors.npy` shape is `[30000, 300]`.
- `vectors.npy` dtype is `float32`.
- `vectors.npy` SHA256 is `f228cbca9816d539ce9532e63fbb1ea95e4c66a7c3df286c788f817e2055bd05`.
- Manifest consistency is green through the current canonical subset audit.
- Git status safety is green for `_operator_artifacts`, the three artifact file names, and `seeds/subsets`; additionally, `git ls-files` must not report operator artifact files as tracked.
- Current canonical seed/vector readiness and Round164 load-dependent preflight are green.

To explicitly test the guarded adapter load after all read-only gates are green, the operator may opt in:

```bash
python scripts/operator_validate_medium30k.py --attempt-load
```

## Boundaries preserved

- No production persistence enablement.
- No `runtime_mapping_enabled` default change.
- No enforcement enablement.
- No AGP bypass.
- No vector artifact creation, copying, download, fabrication, or commit.
- No committed `vectors.npy`, `vocab.txt`, `subset_manifest.json`, `_operator_artifacts`, zip files, part files, or seed-subset artifacts.

## Broader validation delta and next recommendation

The known broader baseline remains red until the real operator artifact is present in the local environment and the load-dependent cascades are repaired. The previously reported taxonomy remains the comparison point:

- seed/vector artifact cascade: 127
- EVE-specific vector/self-learning cascade: 40
- concept/runtime mapping cascade: 38

**Next recommendation:** run `python scripts/operator_validate_medium30k.py` in the operator Codespace with the real artifact. If green, run `python scripts/operator_validate_medium30k.py --attempt-load`, then remeasure the focused EVE-specific vector/self-learning cascade before considering broader repairs. Production persistence remains NO-GO.
