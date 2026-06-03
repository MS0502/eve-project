# EVE v3 Autonomous Handoff

이 디렉터리는 EVE v3를 Codex/에이전트가 계속 이어서 작업하기 위한 고정 기준점이다.

## 현재 상태

- 최신 작업 패키지: `eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip`는 ChatGPT 산출물로 생성됨.
- 이 GitHub repo에는 우선 Codex가 읽을 운영 문서와 작업 기준을 넣었다.
- 전체 소스 zip은 별도 업로드가 필요하다. 이 연결 도구는 UTF-8 텍스트 파일 생성/수정은 가능하지만, 40MB급 zip/`.npy` seed 바이너리를 repo에 직접 올리는 데 적합하지 않다.

## 현재 완료 라운드

- Round95: runtime mapping operator acceptance fixture
- Round96: runtime mapping enable-smoke precheck

## 다음 추천 작업

1. 최신 Round96 zip을 repo에 실제 소스 형태로 업로드한다.
2. Codex에게 `CODEX_AUTONOMOUS_PROMPT.md`를 읽고 `docs/NEXT_ACTIONS.md`부터 진행하게 한다.
3. 다음 코드 라운드는 `Round97 controlled runtime mapping enable smoke`다.

## 핵심 원칙

- EVE Core에 LLM/Transformer/RWKV/Mamba/SSM/BERT 금지.
- 케이스 하드코딩 금지.
- 비결정론 추가 금지.
- AGP 우회 금지.
- lexical/EveSpecific/seed vector를 AGP anchor로 쓰지 말 것.
- runtime/category/concept memory/SA/AGP mutation은 checkpoint/rollback/audit/test 없이는 금지.

## Round122-126 status update

Rounds122-126 completed a validation hygiene loop after the Round120/121 `NO-GO` package. The legacy root `spreading_activation` import blocker is diagnosed and recovered with a minimal compatibility shim, but collect-only remains partial due the next legacy root blocker (`working_memory`). Final recommendation remains `NO-GO`; production persistence, runtime mapping defaults, and enforcement defaults remain disabled.
