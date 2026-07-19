# M0-B Test Classification

Baseline: `main` at `78544d74af84afed450014d599b360c9b4af4f03`

M0-B classifies tests without deleting, skipping, xfail-marking, weakening, or rewriting any test.

Run the exact file-level inventory with:

```bash
python scripts/audit/m0_b_controlflow_concurrency_inventory.py --pretty
```

Generated JSON is an ephemeral audit artifact and must not be committed.

## Validated result

```text
test files classified: 229
KEEP: 229
REWRITE: 0
RETIRE: 0
focused audit tests: 6 passed
full repository collection: 2,563 tests
full repository suite: 2,563 passed
```

No production test was modified by M0-B. The sixth focused audit test proves that bypass detection operates on symbol components: `force_alternative` remains evidence while `reinforce` and `enforcement` do not produce false `force` matches.

## KEEP

`KEEP` is the conservative default. Tests remain executable evidence unless exact AST evidence shows direct dependence on a real delay or an unseeded nondeterministic source.

This includes tests of gates, bypass rejection, outputs, fallback behavior, exception handling, queues, threads, task ordering, shutdown, cancellation, and concurrency safety. `KEEP` does not endorse the implementation; it preserves evidence for M0 and later migration.

All 229 mechanically classified test files were `KEEP` in the validated M0-B snapshot.

## REWRITE

M0-B mechanically classifies a test file as `REWRITE` when either condition is present:

1. it directly calls `time.sleep` or `asyncio.sleep`;
2. it directly calls `uuid.uuid1`, `uuid.uuid4`, or `os.urandom`;
3. it uses `random`, `secrets`, `np.random`, or `numpy.random` without an explicit seed call in the same file.

The generated entry records the first exact evidence line and target. A future rewrite must replace the real delay or nondeterministic input with an injected deterministic clock/source while preserving the behavioral assertion.

This policy intentionally does not auto-classify all thread, queue, task, or executor tests as `REWRITE`. Concurrency tests may be valid deterministic evidence when synchronization is explicit.

No test file met the mechanical `REWRITE` rule in the validated snapshot. This does not prove the absence of transitive real-clock or nondeterministic dependencies; those require call-graph review.

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

The classifier also follows the repository's tracked-path definition of a test file. Historical or root-level files whose names resemble tests but are not under a tracked test path may appear in the general runtime inventory rather than the test-classification count.

M0-B changes no test behavior. Later milestones may propose evidence-backed rewrites, but they must preserve the behavioral assertion and pass the full suite.
