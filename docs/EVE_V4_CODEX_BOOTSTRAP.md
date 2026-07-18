# EVE v4 Codex offline bootstrap

Read this file first, then read `.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md` from the same local checkout.

This file overrides the detailed task's baseline, current-branch, repository-coordinate, branch-creation, and final diff-base instructions. All other requirements remain unchanged.

Historical parent:

```text
65a9ec153f41233df1680ffb93bf79502aaa7fb7
```

## Isolated Codex runner rule

Codex may expose a task launched from GitHub `main` through a local ephemeral branch such as `work`, or through detached HEAD. The local branch name is not evidence that the task started from the wrong GitHub base.

Therefore, do **not** require `git branch --show-current` to return `main`.

Accept `main`, `work`, another isolated runner branch, or detached HEAD when all evidence checks below pass.

A missing local `origin` remote or blocked network access is also not a failure. Do not fetch GitHub and do not require `git remote -v` to identify the repository when the local evidence checks pass.

## Capture execution baseline

At task start, capture the current checked-out commit before making changes:

```bash
EXECUTION_BASELINE=$(git rev-parse HEAD)
RUNNER_BRANCH=$(git branch --show-current)
printf 'execution baseline: %s\n' "$EXECUTION_BASELINE"
printf 'runner branch: %s\n' "${RUNNER_BRANCH:-DETACHED_HEAD}"
```

Proceed only when:

- the worktree is clean
- both task files exist locally
- the historical parent is an ancestor of `EXECUTION_BASELINE`
- every repository change after the historical parent is limited to the two bootstrap/task files listed below
- `main.py` and the expected repository directories exist

Allowed bootstrap delta after the historical parent:

```text
.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md
docs/EVE_V4_CODEX_BOOTSTRAP.md
```

Verify with:

```bash
test -f .codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md
test -f docs/EVE_V4_CODEX_BOOTSTRAP.md
test -f main.py
test -d adapters
test -d core
test -d cognition
test -d tests
test -d scripts

git status --short
git merge-base --is-ancestor \
  65a9ec153f41233df1680ffb93bf79502aaa7fb7 \
  "$EXECUTION_BASELINE"

git diff --name-only \
  65a9ec153f41233df1680ffb93bf79502aaa7fb7..."$EXECUTION_BASELINE"
```

Fail closed if the diff contains any path other than the two allowed bootstrap files.

Do not fail merely because:

- the local branch is named `work`
- HEAD is detached
- no `origin` remote is configured
- GitHub network access is blocked

## Create implementation branch

Create the implementation branch from the captured execution baseline, not from the historical parent:

```bash
git switch -c codex/eve-v4-constitution-rebaseline "$EXECUTION_BASELINE"
```

If the isolated runner does not permit branch switching but the current ephemeral branch is writable, continue on that ephemeral branch and record this fact in the final report. Do not stop solely because the local branch is named `work`.

Do not modify either task file during the constitution implementation.

## Final diff base

Use the captured execution baseline for all final scope checks:

```bash
git diff --name-only "$EXECUTION_BASELINE"...HEAD
git diff --stat "$EXECUTION_BASELINE"...HEAD
```

The two pre-existing task/bootstrap files are not part of the constitution implementation diff and must not be counted among the eight allowed constitution/governance files.

After applying these overrides, execute every other instruction in `.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md` unchanged.