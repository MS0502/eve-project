# EVE v41 Round73 Patch 4 Report

## Result

- Full test suite: `541 passed`
- Runtime after patch 3 baseline: about `18.2s`
- Runtime after patch 4: about `4.5s`

## Changes

### VoiceLoopAdapter

- Changed VAD loading order to prefer lightweight local `webrtcvad`.
- Made heavy `silero-vad` / `torch.hub.load()` opt-in via `EVE_ENABLE_SILERO_VAD=1`.
- Prevents `status()` and graceful-degrade tests from blocking on model/network initialization.

### OpenAIServerAdapter

- Reduced HTTP server `serve_forever()` poll interval to make `stop()` responsive.
- Added `server_close()` in `stop()` for cleaner socket release.

## Validation

- `python -m compileall -q .`
- `python -m pytest -q`
- `python -m pytest -q --durations=20`

## Notes

- No test expectations were weakened.
- No semantic memory files were modified.
- No LLM calls or random behavior were added.
