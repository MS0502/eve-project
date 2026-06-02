# Round96 source package upload and restore guide

이 디렉터리는 Round96 소스 패키지를 GitHub에 25MB 이하 분할 파일로 업로드하고, Codex가 다시 하나의 zip으로 복원하기 위한 위치다.

## Expected files

업로드 위치:

```text
eve_v3_autonomous_handoff/packages/
```

필수 파일:

```text
eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01
eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02
eve_v3_round96_split_manifest.json
```

예상 크기:

- `part01`: 약 24.0MB
- `part02`: 약 17.6MB

## Restore command

이 디렉터리에서 다음 명령을 실행한다.

```bash
cat eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01 \
    eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02 \
    > eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip
```

## Verify SHA-256

`eve_v3_round96_split_manifest.json`의 `source_sha256` 값과 복원된 zip의 SHA-256이 같아야 한다.

```bash
sha256sum eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip
python restore_round96_package.py --verify-only
```

## Extract source

검증이 성공하면 다음 명령으로 압축을 푼다.

```bash
unzip eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip -d round96_source
```

또는 복원/검증/압축 해제를 한 번에 수행한다.

```bash
python restore_round96_package.py
```

## After extraction

압축 해제가 끝나면 `eve_v3_autonomous_handoff/CODEX_AUTONOMOUS_PROMPT.md`와 `eve_v3_autonomous_handoff/docs/NEXT_ACTIONS.md`를 다시 읽고 Round97 controlled runtime mapping enable smoke를 진행한다.

Round97가 통과하고 hard stop이 없으면 Round97에서 멈추지 말고 다음 최고 가치 라운드를 스스로 선택해 계속 진행한다.
