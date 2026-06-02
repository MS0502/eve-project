# Round96 source package upload and restore guide

이 디렉터리는 Round96 소스 패키지를 GitHub에 25MB 이하 분할 파일로 업로드하고, Codex가 다시 하나의 zip으로 복원하기 위한 위치다.

## Expected files

업로드 위치:

```text
eve_v3_autonomous_handoff/packages/
```

### Preferred code-only package

현재 권장 입력은 25MB 이하 단일 code-only zip이다.

```text
eve_v3_round96_code_only_no_medium_vectors.zip
eve_v3_round96_code_only_manifest.json
```

이 package는 GitHub 업로드 제한을 피하기 위해 아래 medium vector 파일만 제외한다.

```text
seeds/subsets/cc.ko.300.subset.medium.30k/vectors.npy
```

Round95~Round96 focused/adjacent validation과 Round97 controlled runtime mapping enable smoke는 이 code-only source를 우선 사용한다. fastText medium vector가 필요한 full validation은 blocked/partial로 기록해야 한다.

### Legacy split package

이전 split package도 계속 인식된다. 긴 파일명 또는 짧은 업로드 파일명 중 하나를 사용할 수 있다.

긴 파일명:

```text
eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01
eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02
eve_v3_round96_split_manifest.json
```

짧은 업로드 파일명:

```text
part01
part02
manifest
```

예상 크기:

- `part01`: 약 24.0MB
- `part02`: 약 17.6MB

## Restore command

권장 방식은 helper를 사용하는 것이다. code-only zip이 있으면 code-only zip을 우선 사용하고, 없으면 legacy split package를 사용한다.

```bash
python restore_round96_package.py
```

legacy split package를 수동 복원해야 한다면 이 디렉터리에서 다음 명령을 실행한다.

```bash
cat eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part01 \
    eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip.part02 \
    > eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip

# 짧은 파일명으로 업로드한 경우:
cat part01 part02 > eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip
```

## Verify SHA-256

code-only manifest 또는 split manifest의 SHA-256 값과 zip의 SHA-256이 같아야 한다. helper는 `source_sha256`, `zip_sha256`, `sha256` 키를 허용한다.

```bash
sha256sum eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip
python restore_round96_package.py --verify-only
```

## Extract source

검증이 성공하면 helper가 자동으로 압축을 푼다. 수동으로 압축을 풀 경우 다음 명령을 사용한다.

```bash
unzip eve_v3_round96_runtime_mapping_enable_smoke_precheck.zip -d round96_source
```

## After extraction

압축 해제가 끝나면 `eve_v3_autonomous_handoff/CODEX_AUTONOMOUS_PROMPT.md`와 `eve_v3_autonomous_handoff/docs/NEXT_ACTIONS.md`를 다시 읽고 Round97 controlled runtime mapping enable smoke를 진행한다.

Round97가 통과하고 hard stop이 없으면 Round97에서 멈추지 말고 다음 최고 가치 라운드를 스스로 선택해 계속 진행한다.
