# OPERATOR_GUIDE

이 파일은 기술 디테일을 모르는 운영자가 결과를 확인하는 용도다.

## You only need to check these items

### 1. What round is complete?

Look at:

- `docs/AUTONOMOUS_WORKLOG.md`
- latest `ROUND*_REPORT.md`

### 2. Did tests actually run?

Look for exact command lines and results.

Good:

- `passed`
- collected test count shown
- command names listed

Bad:

- vague phrases like `seems fine`
- no command output
- docs changed but no tests run

### 3. Was runtime behavior changed?

Important flags:

- `runtime_mapping_enabled`
- `enforcement_enabled`
- category creation
- concept memory mutation
- SA activation creation
- AGP verify/fallback behavior

For Round95/Round96 these should stay false.

### 4. Is there a rollback plan?

For any real mutation, there must be:

- checkpoint
- rollback notes
- validation JSON
- failure condition

### 5. What should you say next?

If everything looks good, you can say:

```text
다음 ㄱㄱ
```

Codex or the assistant should then read `NEXT_ACTIONS.md` and continue.

## Current plain-language status

EVE is not fully finished. The current work is around safely connecting lexical tokens to concept categories at runtime.

Current safe result:

- `민석` has enough fixture evidence to be considered for controlled runtime mapping smoke.
- `EVE` is still blocked because required concept/category evidence is missing.
- Actual runtime mapping is not enabled yet.

Next risky step:

- Round97 controlled runtime mapping enable smoke.

This next step is allowed only with checkpoint, rollback, audit, and tests.
