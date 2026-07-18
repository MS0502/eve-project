# EVE v4 Codex offline bootstrap

Read this file first, then read `.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md` from the same local checkout.

This file overrides the detailed task's baseline, current-branch, repository-coordinate, branch-creation, dependency-environment setup order, and final diff-base instructions. All other requirements remain unchanged.

Historical parent:

```text
65a9ec153f41233df1680ffb93bf79502aaa7fb7
```

## Isolated Codex runner rule

Codex may expose a task launched from GitHub `main` through a local ephemeral branch such as `work`, or through detached HEAD. The local branch name is not evidence that the task started from the wrong GitHub base.

Therefore, do **not** require `git branch --show-current` to return `main`.

Accept `main`, `work`, another isolated runner branch, or detached HEAD when all evidence checks below pass.

A missing local `origin` remote or blocked GitHub access is also not a failure. Do not fetch GitHub and do not require `git remote -v` to identify the repository when the local evidence checks pass.

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

## Development dependency bootstrap

The detailed task's pre-edit validation assumes the repository's declared development dependencies are installed. Missing packages declared in `requirements.txt` are an environment-preparation issue, not a repository compatibility failure.

Before the first `pytest --collect-only -q` or `python -m pytest -q`, run:

```bash
test -f requirements.txt
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
import numpy
import pytest
print(f"numpy={numpy.__version__}")
print(f"pytest={pytest.__version__}")
PY
```

Rules:

- Installing packages already declared in `requirements.txt` is allowed as isolated development-environment setup.
- Do not modify `requirements.txt`, add dependencies, change lock files, or commit environment artifacts.
- The no-network instruction applies to GitHub repository and remote fetching. It does not prohibit package installation through the Codex environment's configured Python package index.
- Do not classify `ModuleNotFoundError` for a declared dependency as a v4 test failure before attempting this bootstrap.
- After installation, run the exact compile and test commands required by the detailed task.
- If installation fails, stop and report the exact command, package, and installer error. Do not edit repository files and do not open the constitution PR.
- If collection or tests still fail after successful installation, follow the detailed task's fail-closed reporting rules.

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
