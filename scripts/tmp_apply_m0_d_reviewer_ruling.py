from pathlib import Path

script_path = Path('scripts/audit/m0_d_component_inventory.py')
text = script_path.read_text(encoding='utf-8')

anchor = 'BASELINE_SHA = "fe10cd954bdf445400ea6aa9708dd214ed761114"\n'
replacement = anchor + 'REVIEWER_ID = "reviewer"\nREVIEWER_DECISION = "ACCEPT_M0_D_RECOMMENDATION"\n'
if 'REVIEWER_DECISION = ' not in text:
    assert anchor in text
    text = text.replace(anchor, replacement, 1)

old = '        disposition, confidence, unresolved, reason = _module_disposition(path, path in reachable, parse_errors.get(path), components_by_path.get(path, []), refs)\n'
new = '        disposition, confidence, proposal_unresolved, reason = _module_disposition(path, path in reachable, parse_errors.get(path), components_by_path.get(path, []), refs)\n'
assert old in text
text = text.replace(old, new, 1)

old = '''            "detection": "AST import reachability plus M0-A/B/C and M0-D evidence",
            "evidence": f"reachable_from_active_root={path in reachable}; references={len(refs)}; component_evidence={len(components_by_path.get(path, []))}",
            "classification": disposition, "confidence": confidence, "unresolved": unresolved,
            "manual_only": path in MANUAL_DISPOSITION_OVERRIDES, "reason": reason,
            "reachable_from_active_root": path in reachable, "evidence_references": evidence_references,
'''
new = '''            "detection": "manual",
            "mechanical_detection": "AST import reachability plus M0-A/B/C and M0-D evidence",
            "evidence": f"reachable_from_active_root={path in reachable}; references={len(refs)}; component_evidence={len(components_by_path.get(path, []))}",
            "classification": disposition, "confidence": confidence, "unresolved": False,
            "manual_only": True, "reason": reason,
            "pre_ruling_unresolved": proposal_unresolved,
            "decided_by": REVIEWER_ID,
            "reviewer_ruling": REVIEWER_DECISION,
            "reachable_from_active_root": path in reachable, "evidence_references": evidence_references,
'''
assert old in text
text = text.replace(old, new, 1)

old_start = '    unresolved_items = [{"category": "governance_gap"'
start = text.index(old_start)
end = text.index('    unresolved_items.extend(parse_errors.values())', start)
replacement = '''    unresolved_items = [{
        "category": "governance_gap",
        "path": "docs/audit/M0_C_PERSISTENCE_AND_STATE_MAP.md",
        "line_start": 157,
        "line_end": 176,
        "symbol": "Hormone-to-drive migration inventory",
        "detection": "manual",
        "evidence": "M0-C inventories bridge candidates but provides no concrete migration plan required by EVE v4 section 10.",
        "classification": "M0_C_REQUIRED_MIGRATION_PLAN_ABSENT",
        "confidence": "high",
        "unresolved": False,
        "manual_only": True,
        "decided_by": REVIEWER_ID,
        "reviewer_ruling": "DEFER_TO_AFFECT_MIGRATION_PLAN_TASK",
    }]
    for entry in parse_errors.values():
        entry["detection"] = "manual"
        entry["mechanical_detection"] = "ast.parse"
        entry["unresolved"] = False
        entry["manual_only"] = True
        entry["decided_by"] = REVIEWER_ID
        entry["reviewer_ruling"] = "ACCEPT_DEPRECATE_RECOMMENDATION"
'''
text = text[:start] + replacement + text[end:]

conflict_start = text.index('def _build_conflicts()')
conflict_end = text.index('\n\ndef audit_repository', conflict_start)
conflict_block = text[conflict_start:conflict_end]
conflict_block = conflict_block.replace(
    '"unresolved": True}',
    '"unresolved": False, "decided_by": "reviewer", "reviewer_ruling": "ACCEPT_AS_V4_1_INPUT"}',
)
text = text[:conflict_start] + conflict_block + text[conflict_end:]

old = '            "unresolved_items": len(unresolved_items), "parse_errors": len(parse_errors),\n'
new = '            "unresolved_items": sum(bool(item["unresolved"]) for item in unresolved_items), "reviewer_resolved_items": sum(not bool(item["unresolved"]) for item in unresolved_items), "parse_errors": len(parse_errors),\n'
assert old in text
text = text.replace(old, new, 1)
script_path.write_text(text, encoding='utf-8')

test_path = Path('tests/audit/test_m0_d_component_inventory.py')
tests = test_path.read_text(encoding='utf-8')
start = tests.index('def test_m0_c_migration_plan_gap_is_prominent(report):')
end = tests.index('\n\ndef test_hormone_coupling_defaults_to_wrap_not_automatic_rewrite', start)
replacement = '''def test_m0_c_migration_plan_gap_is_prominent_and_reviewer_ruled(report):
    gaps = [entry for entry in report["unresolved_items"] if entry["classification"] == "M0_C_REQUIRED_MIGRATION_PLAN_ABSENT"]
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["confidence"] == "high"
    assert gap["unresolved"] is False
    assert gap["manual_only"] is True
    assert gap["detection"] == "manual"
    assert gap["decided_by"] == "reviewer"
    assert gap["reviewer_ruling"] == "DEFER_TO_AFFECT_MIGRATION_PLAN_TASK"
    conflicts = {entry["id"]: entry for entry in report["v4_runtime_conflicts"]}
    affect = conflicts["affect-migration-plan-missing"]
    assert affect["unresolved"] is False
    assert affect["decided_by"] == "reviewer"
    assert affect["reviewer_ruling"] == "ACCEPT_AS_V4_1_INPUT"
'''
tests = tests[:start] + replacement + tests[end:]

insert_at = tests.index('\n\ndef test_hormone_coupling_defaults_to_wrap_not_automatic_rewrite')
additional = '''

def test_reviewer_ruling_resolves_every_module_recommendation(report):
    entries = report["module_dispositions"]
    assert entries
    assert report["summary"]["unresolved_module_dispositions"] == 0
    assert report["summary"]["unresolved_items"] == 0
    assert report["summary"]["reviewer_resolved_items"] == 3
    for entry in entries:
        assert entry["unresolved"] is False
        assert entry["manual_only"] is True
        assert entry["detection"] == "manual"
        assert entry["decided_by"] == "reviewer"
        assert entry["reviewer_ruling"] == "ACCEPT_M0_D_RECOMMENDATION"
        assert "mechanical_detection" in entry
        assert "pre_ruling_unresolved" in entry


def test_rewrite_deprecate_and_remove_rulings_are_exact(report):
    by_category = {}
    for entry in report["module_dispositions"]:
        by_category.setdefault(entry["classification"], []).append(entry["path"])
    assert sorted(by_category["REWRITE"]) == [
        "adapters/hormone_adapter.py",
        "adapters/live_loop.py",
        "adapters/persistence_adapter.py",
        "core/autonomous.py",
        "language/streaming.py",
        "main.py",
    ]
    assert sorted(by_category["DEPRECATE"]) == [
        "eve_foundation_v10_2.py",
        "eve_foundation_v12_0.py",
    ]
    assert by_category.get("REMOVE", []) == []
'''
tests = tests[:insert_at] + additional + tests[insert_at:]
test_path.write_text(tests, encoding='utf-8')

disposition_path = Path('docs/audit/M0_D_MODULE_DISPOSITION.md')
doc = disposition_path.read_text(encoding='utf-8')
doc = doc.replace('Status: recommendations only. No module is deleted, deprecated in code, wrapped, rewritten, activated, or retired by M0-D.', 'Status: reviewer-ruled recommendations only. The rulings approve disposition labels as planning inputs; no module is deleted, deprecated in code, wrapped, rewritten, activated, or retired by M0-D.')
doc = doc.replace('resolved recommendations: 10', 'resolved recommendations: 288')
doc = doc.replace('unresolved recommendations requiring reviewer ruling: 278', 'unresolved recommendations requiring reviewer ruling: 0')
doc = doc.replace('Automatic hormone coupling now yields `WRAP/unresolved`; only six manually evidenced architecture conflicts remain `REWRITE`.', 'Automatic hormone coupling proposed `WRAP`; the reviewer accepted that conservative rule. Only six manually evidenced architecture conflicts are approved as `REWRITE`.')
doc = doc.replace(' | `YES` |\n', ' | `NO` |\n')
matrix_anchor = '## Complete evidence matrix\n'
ruling_section = '''## Reviewer bulk ruling

Detection for the final disposition decision is `manual` and every module entry records `decided_by: reviewer`. The reviewer accepted the evidence-backed proposal under these rules:

1. `KEEP`, `WRAP`, and `EXPERIMENTAL` preserve code and evidence; they do not activate, promote, or modify runtime behavior.
2. The six `REWRITE` labels approve future architectural replacement while preserving capability and tests; they authorize no rewrite in M0-D.
3. The two `DEPRECATE` labels preserve the parse-invalid versioned snapshots as historical/migration evidence and exclude future runtime authority; they authorize no deletion.
4. `REMOVE` remains empty. Lack of reachability is never deletion evidence.
5. The missing hormone-to-drive migration plan is accepted as a separate immediate Affect Migration Plan task and as direct v4.1 input; it is not silently filled by M0-D.

Confirmed `REWRITE` modules: `adapters/hormone_adapter.py`, `adapters/live_loop.py`, `adapters/persistence_adapter.py`, `core/autonomous.py`, `language/streaming.py`, `main.py`.

Confirmed `DEPRECATE` modules: `eve_foundation_v10_2.py`, `eve_foundation_v12_0.py`.

'''
if '## Reviewer bulk ruling' not in doc:
    doc = doc.replace(matrix_anchor, ruling_section + matrix_anchor, 1)
unresolved_heading = '## Prominent UNRESOLVED items\n'
if unresolved_heading in doc:
    doc = doc.split(unresolved_heading, 1)[0] + '''## Reviewer-ruling closure

All 288 module recommendations are reviewer-ruled in the canonical JSON with `detection: manual` and `decided_by: reviewer`. Open M0-D unresolved counts are zero.

The two parse failures are accepted under the `DEPRECATE` planning recommendation. The absent hormone-to-drive migration plan is assigned to the separate Affect Migration Plan task before v4.1 drafting. These rulings close M0-D review; they do not implement any disposition.
'''
disposition_path.write_text(doc, encoding='utf-8')

inventory_path = Path('docs/audit/M0_D_NEURAL_VECTOR_LIFELOOP_INVENTORY.md')
inv = inventory_path.read_text(encoding='utf-8')
inv = inv.replace('unresolved module dispositions: 278', 'unresolved module dispositions: 0')
inv = inv.replace('standalone unresolved items: 3', 'standalone unresolved items: 0')
inv = inv.replace('The conservative automatic result is `WRAP/unresolved`; only six manually evidenced architecture conflicts remain `REWRITE`.', 'The conservative automatic proposal is `WRAP`; the reviewer accepted that rule. Only six manually evidenced architecture conflicts are approved as `REWRITE`.')
inv = inv.replace('M0-D records `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT` as a high-confidence unresolved governance blocker. M0-D does not repair M0-C or invent constitutional migration policy outside its allowed scope.', 'M0-D records `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT` as a high-confidence governance defect. The reviewer assigns it to the separate Affect Migration Plan task before v4.1 drafting; M0-D does not repair M0-C or invent migration policy outside scope.')
inv = inv.replace('M0-D does not modify production code, existing tests outside its new audit test, data, models, vectors, configuration, persistence, defaults, or frozen PRs. It does not close any frozen PR and does not implement a disposition recommendation.', 'M0-D does not modify production code, existing tests outside its new audit test, data, models, vectors, configuration, persistence, or defaults. Reviewer rulings approve planning labels only and do not implement a disposition recommendation. Frozen-PR close actions remain separate post-merge operations.')
inventory_path.write_text(inv, encoding='utf-8')

status_path = Path('docs/EVE_IMPLEMENTATION_STATUS_v4.md')
status = status_path.read_text(encoding='utf-8')
status = status.replace('Constitution status: provisional pending completion and reviewer ruling of M0', 'Constitution status: provisional pending the Affect Migration Plan and human-reviewed v4.1 revision')
status = status.replace('Completed audit milestones: M0-A, M0-B, and M0-C merged', 'Completed audit milestones: M0-A, M0-B, and M0-C merged; M0-D reviewer rulings recorded in PR #125')
status = status.replace('M0-D records this as `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT`. It remains unresolved for reviewer ruling and a separate scope-compliant correction. M0-D does not silently fill the gap or change affect implementation.', 'M0-D records this as `M0_C_REQUIRED_MIGRATION_PLAN_ABSENT`. The reviewer assigns the correction to the separate Affect Migration Plan task before v4.1 drafting. M0-D does not silently fill the gap or change affect implementation.')
status = status.replace('Complete M0-D static analysis and independent exact-head validation. Review all unresolved rulings, the complete `REMOVE` and `DEPRECATE` recommendations, and the `v4.0 assumptions vs runtime reality` conflict list before any Ready or merge decision. v4.1 constitutional drafting remains a separate human-reviewed milestone.', 'Merge M0-D after independent exact-head validation, execute the separately approved frozen-PR closures, complete the Affect Migration Plan, and then draft v4.1 through human-reviewed triangular revision. `REMOVE` remains empty; the six `REWRITE` and two `DEPRECATE` planning labels are reviewer-confirmed.')
status_path.write_text(status, encoding='utf-8')
