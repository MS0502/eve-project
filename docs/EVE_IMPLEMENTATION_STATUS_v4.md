# EVE v4 Implementation Status

Active constitution: EVE v4.0
Constitution status: provisional pending M0
Previous v3/v3.1 documents: historical reference only
Frozen work: all open implementation PRs
Next milestone: M0-A runtime entrypoint, mutation, import, and test audit
Required next baseline: exact main SHA after constitution merge
Planned revision: v4.1 after evidence from M0-A through M0-D

EVE v4.0 is provisional pending M0. It may be revised to v4.1 after evidence from M0. Evidence-based revision is part of the process and is not a project failure.

## Current state

This rebaseline is documentation and governance only. The v4 runtime is not claimed as implemented. No production persistence, enforcement, runtime mapping default, model activation, vector loading, database, checkpoint, or generated artifact is enabled here.

## Freeze

All existing implementation work remains frozen through M0, including open PRs #109, #97, #86, #84, #82, #11, #7, #4, and #1. Do not modify, rebase, merge, extend, or reuse those branches during M0.

## Next milestone

M0-A must audit runtime entrypoints, imports and dependency construction, mutation and direct-write sites, and test inventory with KEEP/RETIRE/REWRITE classification. The audit must separate mechanical evidence from manual classification, produce rerunnable evidence, and avoid runtime activation.
