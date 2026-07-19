#!/usr/bin/env python3
from pathlib import Path
import re

plan_path = Path("docs/audit/M0_C_AFFECT_MIGRATION_PLAN.md")
text = plan_path.read_text(encoding="utf-8")
lines = text.splitlines()
drop_axes = {"estrogen", "testosterone", "prolactin", "progesterone"}
output = []
for line in lines:
    if line.startswith("| `"):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 12:
            axis = cells[0].strip("`")
            family = cells[1].strip("`")
            if axis in drop_axes:
                cells[2] = "`PROPOSED-DROP`"
                cells[3] = "—"
                cells[4] = "—"
                cells[5] = "—"
                cells[6] = "—"
                cells[11] = "—"
            if family == "read_only_affect_registry":
                cells[10] = (
                    "Preserve the registry definition and any observed source value "
                    "with provenance; do not assume current persistence authority."
                )
            # The initial draft copied the connector display line, which includes two
            # metadata rows before source content. Convert those citations to actual
            # repository source lines; the AST checker independently verifies them.
            if family in {"legacy_mutable_hormone", "read_only_affect_registry"}:
                cells[9] = re.sub(
                    r"(?P<prefix>(?:hormone_system\.py|adapters/affect_hormone_neural_rhythm_registry\.py):)(?P<line>\d+)",
                    lambda match: f"{match.group('prefix')}{int(match.group('line')) - 2}",
                    cells[9],
                )
            line = "| " + " | ".join(cells) + " |"
    output.append(line)
text = "\n".join(output) + "\n"
text = text.replace(
    "Status: design-only migration contract and reviewer input.",
    "Status: reviewer-ruled design-only migration contract.",
)
text = text.replace(
    "identity projection; new energy = clamp(axis)",
    "direct projection; new energy = clamp(axis)",
)
text = text.replace("hormone_system.py:126-163", "hormone_system.py:124-161")
text = text.replace(
    "adapters/affect_hormone_neural_rhythm_registry.py:19-69",
    "adapters/affect_hormone_neural_rhythm_registry.py:17-67",
)
text = text.replace("adapters/hormone_adapter.py:31-39", "adapters/hormone_adapter.py:29-37")
marker = "## Reviewer questions\n"
assert marker in text
text = text.split(marker, 1)[0] + """## Reviewer rulings

The reviewer ruled all four initially unresolved endocrine-labelled axes as `PROPOSED-DROP`:

- `estrogen`
- `testosterone`
- `prolactin`
- `progesterone`

Reason: the current scalar labels do not provide sufficiently specific, EVE-grounded evidence for stable psychological or behavioral projection, and direct mappings would risk importing gendered, dominance, or caregiving stereotypes. Their original values, baselines, tier/phase metadata where present, and provenance remain preserved for historical replay. They receive no future drive, appraisal, emotion, speech, goal, identity, or agency authority.

Final mapping totals:

```text
MAPPED: 59
PROPOSED-DROP: 4
UNRESOLVED: 0
TOTAL: 63
```
"""
plan_path.write_text(text, encoding="utf-8")

status_path = Path("docs/EVE_IMPLEMENTATION_STATUS_v4.md")
status = status_path.read_text(encoding="utf-8")
status = status.replace(
    "Constitution status: provisional pending completion and reviewer ruling of the Affect Migration Plan, then human-reviewed v4.1 revision",
    "Constitution status: provisional pending merge of the reviewer-ruled Affect Migration Plan, then human-reviewed v4.1 revision",
)
status = status.replace(
    "Current milestone: M0-C Supplement — Affect Migration Plan",
    "Current milestone: reviewer-ruled M0-C Supplement — Affect Migration Plan",
)
status = status.replace(
    "Complete the deterministic checker and reviewer rulings for every found axis, independently validate the exact final head, merge the four-file Affect Migration Plan supplement, and then begin the human-reviewed v4.1 triangular revision using the seven conflict inputs.",
    "Independently validate the exact reviewer-ruled head, merge the four-file Affect Migration Plan supplement, and then begin the human-reviewed v4.1 triangular revision using the seven conflict inputs.",
)
status_path.write_text(status, encoding="utf-8")
