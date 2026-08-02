# M3-C-R Resumable Phone Goal-Window Operator

## Scope

M3-C-R replaces the failed monolithic phone block with five independent,
non-interactive stages. It reuses the three completed stage-1 artifacts under
one exact private root and does not regenerate them.

```text
private root:
~/.local/share/eve-m3c-private-goal-window-4d22013a-20260801

immutable inputs:
canonical_goal_window_package.json
package_review_summary.json
forbidden_prior_path_digests.json
```

Accepted package review:

```text
package digest: bdc250fce7c746d527c378e240ec1fd3b307c3c1763306f43b8f4fafc3bd6c88
probe count: 4
operations: goal_set, tick, goal_set, tick
forbidden prior-path digests: 34
legacy goal authority transfer authorized: false
raw private text/path output: false
```

No stage changes the immutable input files. A mismatch stops before later
state is created.

## Seventh phone-habitat discovery

The first monolithic attempt completed package generation and then Android
terminated Termux near the interactive review prompt. The observed device state
was approximately 10 GiB total RAM with only 225 MiB available after engine
load. The completed package, summary, and forbidden-digest files remained
intact. No pin, receipt, store, or public review was completed.

This is the seventh phone-habitat discovery:

```text
failure class: Android low-memory process termination
observed available memory: approximately 225 MiB
vulnerability: full engine remained resident while shell waited for read -r
package corruption observed: false
pin completed: false
operator execution completed: false
legacy authority transfer: false
```

## Stage contract

### Stage 1 — review confirmation

`m3_c_r_stage1_record_review.py` verifies the exact accepted summary, package
digest, four-probe order, and 34 forbidden digests. `--reviewed` creates one
canonical 0600 `operator_review_confirmation.json`. Repetition returns the same
artifact. A conflicting/partial existing artifact is preserved and rejected.

Engine load: **zero**.

### Stage 2 — local pin capture

`m3_c_r_stage2_capture_pin.py` consumes only the immutable package and stage-1
confirmation. It reuses the merged M3-C-Q pin builder and creates:

```text
local_authorization_pin.json
authorization_capture_receipt.json
```

If both files already match the package binding, repetition returns the same
receipt. It never opens the operator seam and never imports or constructs the
full engine.

Engine load: **zero**.

### Stage 3 — execution preflight

`m3_c_r_stage3_preflight.py` proves the exact concrete path binding for:

```text
goal_dual_read_window.jsonl
legacy_goal_baseline.backup
legacy_goal_baseline.restore
```

It checks the real available-memory source and writes one canonical
`execution_preflight.json`. Android `/proc/meminfo` `MemAvailable` is primary;
`sysconf(SC_AVPHYS_PAGES, SC_PAGE_SIZE)` is the real-system fallback.

The fixed minimum is **3072 MiB available before engine construction**. The
threshold cannot be lowered through a command-line flag. Stage 4 repeats the
measurement immediately before construction, so an older successful stage-3
receipt cannot bypass a later low-memory state.

Engine load: **zero**.

### Stage 4 — one engine execution

`m3_c_r_stage4_execute_window.py` is the only new file containing a
`build_full_engine()` call, and it calls it exactly once. Before that call it:

1. verifies a clean exact checkout;
2. returns an already completed matching receipt without loading the engine;
3. validates stages 1–3 and the exact private package;
4. refuses any partial store/backup/restore state without a completed receipt;
5. repeats the 3072 MiB memory gate;
6. proves the exact package path binding;
7. loads the private pin only for one synchronous scoped session.

After successful execution it writes `operator_execution_receipt.json`, closes
all pin seams in `finally`, and renames the pin to
`local_authorization_pin.consumed.json`. A successful repeated invocation reads
the completed receipt and does not construct another engine.

Engine load: **exactly one on the first successful execution; zero on completed
resume**.

### Stage 5 — public review record

`m3_c_r_stage5_record_public_review.py --reviewed` records only digest-safe
review facts in:

```text
m3_c_private_device_goal_window_public_review.json
```

It does not transfer legacy goal authority, authorize migration, open M3-E, or
publish private paths/text.

Engine load: **zero**.

## Idempotency and interruption behavior

```text
matching completed stage output: return same result, no mutation
conflicting or partial stage output: preserve and fail closed
OOM before stage-4 output creation: stage 4 may be retried
any store/backup/restore path present without operator receipt:
    preserve all evidence and do not rerun stage 4
operator receipt present:
    return receipt and finish consumed-pin rename without engine load
single-use meaning: one successful completion, not one uninterrupted process
```

No stage deletes a canonical artifact. No stage repairs or overwrites a
conflicting file.

## Inspection and rollback commands

These commands inspect state only:

```bash
P="$HOME/.local/share/eve-m3c-private-goal-window-4d22013a-20260801"
find "$P" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
sha256sum \
  "$P/canonical_goal_window_package.json" \
  "$P/package_review_summary.json" \
  "$P/forbidden_prior_path_digests.json"
```

Safe rollback before stage 4 means stopping. The immutable three input files,
review confirmation, pin, and receipts remain preserved. No cleanup is needed.

After a stage-4 interruption, do **not** delete any of these if present:

```text
goal_dual_read_window.jsonl
legacy_goal_baseline.backup
legacy_goal_baseline.restore
operator_execution_receipt.json
local_authorization_pin.json
local_authorization_pin.consumed.json
```

Copy the complete private root to separate private storage before forensic
review. Only disposable editor files explicitly ending in `.tmp` may be removed
after confirming no process is active; M3-C-R itself does not create such files.

## Phone execution blocks

Do not run these blocks until this PR is reviewed, merged, and the exact merged
main SHA is substituted by `git rev-parse HEAD` after `git pull --ff-only`.
Keep the screen on, hold a wake lock, close other applications, and do not
switch away from Termux during stage 4.

### Common preflight

```bash
cd ~/EVE_Project
git switch main
git pull --ff-only
EXPECTED_HEAD="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)" || { echo "dirty worktree"; exit 1; }
P="$HOME/.local/share/eve-m3c-private-goal-window-4d22013a-20260801"
test -d "$P" || { echo "private root absent"; exit 1; }
termux-wake-lock
```

### Stage 1

```bash
python scripts/operator/m3_c_r_stage1_record_review.py \
  --expected-head "$EXPECTED_HEAD" \
  --private-root "$P" \
  --reviewed
```

### Stage 2

```bash
python scripts/operator/m3_c_r_stage2_capture_pin.py \
  --expected-head "$EXPECTED_HEAD" \
  --private-root "$P"
```

### Stage 3

```bash
python scripts/operator/m3_c_r_stage3_preflight.py \
  --expected-head "$EXPECTED_HEAD" \
  --private-root "$P"
```

### Stage 4

Run only after inspecting the stage-3 JSON and confirming
`memory_preflight.sufficient=true`:

```bash
python scripts/operator/m3_c_r_stage4_execute_window.py \
  --expected-head "$EXPECTED_HEAD" \
  --private-root "$P"
```

### Stage 5

First display the exact digest-only stage-4 receipt for human review:

```bash
cat "$P/operator_execution_receipt.json"
```

After review:

```bash
python scripts/operator/m3_c_r_stage5_record_public_review.py \
  --expected-head "$EXPECTED_HEAD" \
  --private-root "$P" \
  --reviewed
termux-wake-unlock
```

## Authority boundary

```text
legacy goal authority: authoritative
v4 goal result: shadow observation only
runtime default integration: false
legacy goal authority transferred: false
legacy migration authorized: false
action authorized: false
scheduler authorized: false
speech authorized: false
M3-E authority open: false
prior M3-C-J private evidence reused: false
raw private text/path public: false
```

The PR creates scripts and tests only. CI does not execute the phone workflow,
construct a production engine for this operator, or access the private root.
