# EVE v3 Round205 Report — Guarded Local-Only Measurement/Repair Path

Round205 adds `scripts/operator_measure_runtime_mapping_after_self_learning.py` as the stable one-command operator-local workflow for the selected cluster.

## Stable operator-local command

```bash
python scripts/operator_measure_runtime_mapping_after_self_learning.py --artifact-dir _operator_artifacts/subset_medium_30k --target-word 민석 --context-word 한국어 --context-word 감정 --context-word 기억 --context-word 대화 --negative-token EVE --output eve_v3_autonomous_handoff/validation/operator_local_round203_207_runtime_mapping_after_self_learning.json
```

## Behavior

The command runs the existing medium30k guarded validation with `--attempt-load`, builds the engine only when that validation is green and explicitly operator-authorized, reruns the in-memory EVE-specific self-learning measurement for `민석`, and then runs the existing concept/runtime mapping chain through Round97 controlled smoke.

The controlled smoke must roll `runtime_mapping_enabled` back to false and must not persist runtime mapping. Enforcement remains disabled.

## Cloud behavior

In Codex Cloud, where `_operator_artifacts/subset_medium_30k` is absent, the command fails closed before engine build or measurement. The blocked Cloud JSON was written to `eve_v3_autonomous_handoff/validation/operator_local_round203_207_runtime_mapping_after_self_learning_BLOCKED_CLOUD.json`.

Validation JSON: `eve_v3_autonomous_handoff/validation/ROUND205_GUARDED_LOCAL_ONLY_RUNTIME_MAPPING_MEASUREMENT_CONTRACT_STATUS.json`.
