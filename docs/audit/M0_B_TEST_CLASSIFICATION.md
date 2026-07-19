# M0-B Test Classification

Baseline: `main` at `78544d74af84afed450014d599b360c9b4af4f03`

M0-B classifies tests without deleting, skipping, xfail-marking, weakening, or rewriting any test.

Run the exact file-level inventory with:

```bash
python scripts/audit/m0_b_controlflow_concurrency_inventory.py --pretty
```

Generated JSON is an ephemeral audit artifact and must not be committed.

## KEEP

`KEEP` is the conservative default. Tests remain executable evidence unless exact AST evidence shows direct dependence on a real delay or an unseeded nondeterministic source.

This includes tests of gates, bypass rejection, outputs, fallback behavior, exception handling, queues, threads, task ordering, shutdown, cancellation, and concurrency safety. `KEEP` does not endorse the implementation; it preserves evidence for M0 and later migration.

## REWRITE

M0-B mechanically classifies a test file as `REWRITE` when either condition is present:

1. it directly calls `time.sleep` or `asyncio.sleep`;
2. it directly calls `uuid.uuid1`, `uuid.uuid4`, or `os.urandom`;
3. it uses `random`, `secrets`, `np.random`, or `numpy.random` without an explicit seed call in the same file.

The generated entry records the first exact evidence line and target. A future rewrite must replace the real delay or nondeterministic input with an injected deterministic clock/source while preserving the behavioral assertion.

This policy intentionally does not auto-classify all thread, queue, task, or executor tests as `REWRITE`. Concurrency tests may be valid deterministic evidence when synchronization is explicit.

## RETIRE

M0-B retires no test by default. Retirement requires exact file:line evidence that the test has no behavioral, migration, safety, historical, or regression value. That disposition remains deferred.

## Required fields

Every test classification records:

- repository-relative path;
- exact evidence line;
- `KEEP`, `REWRITE`, or `RETIRE`;
- mechanical evidence and detector;
- classification reason;
- confidence;
- unresolved state;
- manual-only state.

## Limits

The file-level seed check is conservative. A seed in one file may not govern another file, and a seeded generator can still be order-sensitive under concurrency. Conversely, an injected deterministic wrapper may internally call a normally nondeterministic API. These cases remain manual review candidates; the scanner does not silently guess.

M0-B changes no test behavior. Measured classification counts will be added only after independent validation against the unchanged branch head.
