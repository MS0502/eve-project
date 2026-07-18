# EVE v4 Codex offline bootstrap

Read this file first, then read `.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md` from the same local checkout.

This file overrides only baseline, branch-creation, and diff-base instructions in the detailed task. All other requirements remain unchanged.

Historical parent: `65a9ec153f41233df1680ffb93bf79502aaa7fb7`.

At task start, capture the current `main` HEAD containing both task files:

```bash
EXECUTION_BASELINE=$(git rev-parse HEAD)
```

Proceed only when the worktree is clean, the repository is `MS0502/eve-project`, the branch is `main`, both task files exist locally, and every change after the historical parent is limited to:

```text
.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md
docs/EVE_V4_CODEX_BOOTSTRAP.md
```

Verify with:

```bash
test -f .codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md
test -f docs/EVE_V4_CODEX_BOOTSTRAP.md
git merge-base --is-ancestor 65a9ec153f41233df1680ffb93bf79502aaa7fb7 "$EXECUTION_BASELINE"
git diff --name-only 65a9ec153f41233df1680ffb93bf79502aaa7fb7..."$EXECUTION_BASELINE"
```

Create the implementation branch from the captured execution baseline, not from the historical parent:

```bash
git switch -c codex/eve-v4-constitution-rebaseline "$EXECUTION_BASELINE"
```

Do not modify either task file.

Use the captured execution baseline for final diff checks:

```bash
git diff --name-only "$EXECUTION_BASELINE"...HEAD
git diff --stat "$EXECUTION_BASELINE"...HEAD
```

After applying these overrides, execute every other instruction in `.codex/EVE_V4_CONSTITUTION_REBASELINE_TASK.md` unchanged.
