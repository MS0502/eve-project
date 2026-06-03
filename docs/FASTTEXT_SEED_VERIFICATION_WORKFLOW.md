# EVE v3 round27 — fastText seed verification workflow

This workflow verifies the external Korean fastText seed path without loading the
model and without touching `self_embedding_adapter.py`.

## Preconditions

- `seeds/MANIFEST.yaml` contains the registered `cc.ko.300.bin` provenance entry.
- The binary is stored externally. It is not committed to this repository.
- The file path is provided explicitly or through `EVE_FASTTEXT_SEED_PATH`.

## Run with an explicit path

```bash
python - <<'PY'
from adapters.fasttext_loader import seed_verification_runner

result = seed_verification_runner(explicit_path="/path/to/cc.ko.300.bin")
print(result)
PY
```

## Run with environment variable

```bash
export EVE_FASTTEXT_SEED_PATH=/path/to/cc.ko.300.bin
python - <<'PY'
from adapters.fasttext_loader import seed_verification_runner

result = seed_verification_runner()
print(result)
PY
```

## Status values

- `verified`: file exists and checksum matches the manifest.
- `mismatch`: file exists but checksum differs.
- `missing_file`: configured path does not exist.
- `skipped`: no explicit path and no environment variable were provided.
- `unregistered`: manifest does not contain the requested seed.

## Non-goals

Round27 does not load fastText, does not mark the seed as loaded, does not rewrite
self-embedding, and does not start subset extraction.
