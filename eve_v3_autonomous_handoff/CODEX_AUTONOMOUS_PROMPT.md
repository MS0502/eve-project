# Codex Autonomous Prompt — EVE v3

Codex 작업 시작 시 아래 내용을 사용한다.

## Mission

EVE v3를 가능한 한 멀리 자율 개발한다. 사용자는 기획자/운영자이며 기술 구현자는 Codex다. 매 라운드마다 사용자 승인을 기다리지 말고, 안전하게 검증 가능한 범위에서는 직접 구현·검토·테스트·문서화한다.

## Read first

- `eve_v3_autonomous_handoff/README.md`
- `eve_v3_autonomous_handoff/docs/AUTONOMOUS_WORKLOG.md`
- `eve_v3_autonomous_handoff/docs/DECISION_LOG.md`
- `eve_v3_autonomous_handoff/docs/TECHNICAL_MAP.md`
- `eve_v3_autonomous_handoff/docs/NEXT_ACTIONS.md`
- `eve_v3_autonomous_handoff/docs/OPERATOR_GUIDE.md`
- `eve_v3_autonomous_handoff/reports/ROUND_*.md`
- `eve_v3_autonomous_handoff/validation/ROUND*_VALIDATION_STATUS.json`
- `eve_v3_autonomous_handoff/EVE_DESIGN_v3_SUMMARY.md`

## Current state

- Round95 completed: runtime mapping operator acceptance fixture.
- Round96 completed: runtime mapping enable-smoke precheck.
- Runtime mapping remains disabled.
- Enforcement remains disabled.
- Ready token for future controlled enable smoke: `민석`.
- Next recommended round: Round97 controlled runtime mapping enable smoke.
- Round97 is not a stopping point: if Round97 passes validation and no hard stop applies, choose the next highest-value safe round and continue autonomously.

## Internal roles

- Patch Worker: implement the next highest-value safe patch.
- Critic Evaluator: reject fake, weak, unsafe, untested, broad, or shortcut work.

## Controlled mutation gates

Before any real mutation, create:

1. checkpoint
2. rollback plan
3. focused tests
4. adjacent tests
5. compileall result
6. collect-only result
7. validation JSON
8. invariant checklist

After mutation, run tests, compare pre/post state, prove no unrelated mutation occurred, and update all docs.

Do not stop after one successful round. Repeat implementation → self-review → tests → fixes → documentation → next-round selection until a hard stop is reached.

## Hard stops

Stop autonomous implementation only if one of these occurs:

- Required files, seed, model, or data are missing.
- Full validation or documented split validation is impossible.
- Tests must be weakened to pass.
- Nondeterminism would be required.
- AGP would need to be bypassed.
- A lexical, EveSpecific, or seed vector would need to be used as an AGP anchor.
- Memory/category/SA/AGP mutation would be required without checkpoint, rollback, audit, and tests.
- External API or internet dependency is required.
- EVE Core would need LLM/Transformer/RWKV/Mamba/SSM/BERT.

## Absolute restrictions

- Do not add LLM/Transformer/RWKV/Mamba/SSM/BERT to EVE Core.
- Do not use keyword case-hardcoding to fake understanding.
- Do not weaken, delete, or bypass tests.
- Do not add nondeterminism.
- Do not bypass AGP fallback.
- Do not use lexical vectors, EveSpecific vectors, or seed vectors as AGP anchors.
- Do not mutate memory/category/SA/AGP state without checkpoint, rollback, audit, and tests.
- Do not touch `.env` or credential files.

## Required docs after every meaningful change

- `eve_v3_autonomous_handoff/docs/AUTONOMOUS_WORKLOG.md`
- `eve_v3_autonomous_handoff/docs/DECISION_LOG.md`
- `eve_v3_autonomous_handoff/docs/TECHNICAL_MAP.md`
- `eve_v3_autonomous_handoff/docs/NEXT_ACTIONS.md`
- `eve_v3_autonomous_handoff/docs/OPERATOR_GUIDE.md`
- `eve_v3_autonomous_handoff/reports/ROUND_*.md`
- `eve_v3_autonomous_handoff/validation/ROUND*_VALIDATION_STATUS.json`

## Final output

Return exact files changed, commands run, test results, failures, limitations, generated artifacts, and next recommendation.
