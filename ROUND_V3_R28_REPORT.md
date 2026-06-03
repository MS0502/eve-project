# EVE v3 Round28 Report — External Verification Result + Subset Planning

## Result

- Base: v3 round27 (`735 passed`)
- Current: v3 round28 (`740 passed`)
- Scope: external verification documentation + subset extraction planning
- `compileall` passed
- Production runtime changes: none

## External verification result

The registered Korean fastText seed was externally verified in Colab on 2026-05-11.

```yaml
seed: cc.ko.300.bin
status: verified
match: true
path: /content/cc.ko.300.bin
actual: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
expected: SHA256:a021ebbd5521ca4b3b33425fc25dacd60e4a795041d6f785997800d32a58acd7
method: external SHA256 verification, Python hashlib, no adapters import
context: Colab environment, cc.ko.300.bin.gz from Google Drive backup
```

This confirms that the external binary available in the operator environment matches the checksum registered in `seeds/MANIFEST.yaml`.

## Subset extraction plan

Round28 adds `docs/SUBSET_EXTRACTION_PLAN.md`.

The plan defines:

- deterministic vocabulary selection from `cc.ko.300.bin` frequency/model order;
- three extraction targets: mini `1k`, small `5k`, medium `30k`;
- recommended first output format: `vocab.txt` + `vectors.npy` + subset manifest;
- drift baseline policy for future EVE-specific update tracking;
- round29 non-goals and round30+ migration boundary.

## State after round28

```text
external_seed_state = registered
external verification = verified externally
runtime fastText load = False
subset extracted = False
seed used by self_embedding = False
self_embedding rewrite = False
```

## Non-goals preserved

- No actual subset extraction.
- No fastText runtime load.
- No seed binary commit.
- No self-embedding rewrite.
- No AGP runtime change.
- No threshold change.
- No fallback pool expansion.
- No semantic guard keyword addition.
- No memory/quarantine modification.

## Next recommendation

v3 round29:

- execute subset extraction in Colab or another operator environment;
- start with the mini `1k` target;
- write `vocab.txt`, `vectors.npy`, and a subset manifest;
- keep `self_embedding_adapter.py` unchanged until the extracted subset is verified.
