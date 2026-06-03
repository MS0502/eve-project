"""
OpenAIServerAdapter — 라운드 26

EVE를 OpenAI 호환 HTTP API로 노출.
AIRI Stage Pocket (또는 다른 OpenAI 클라이언트) 가 그대로 붙음.

엔드포인트:
- POST /v1/chat/completions (streaming + non-streaming)
- GET  /v1/models           (모델 리스트)

원칙:
- Lazy load (aiohttp 또는 stdlib http.server)
- Background thread 0 (사용자가 명시 시작)
- 결정론 안 깸 (HTTP 노출만, 사고는 EVE)
- chat_stream → SSE (Server-Sent Events) 변환

OpenAI 호환 메시지 포맷:
  요청: {"model": "eve", "messages": [{"role": "user", "content": "..."}], "stream": true}
  응답 (stream=false): {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
  응답 (stream=true):  data: {"choices": [{"delta": {"content": "..."}}]}\n\n
                        data: [DONE]\n\n
"""

import json
import socket
import threading
import time
from typing import Optional


def _make_chat_response(text: str, model: str = "eve") -> dict:
    """non-streaming 응답."""
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0, "completion_tokens": len(text), "total_tokens": len(text),
        },
    }


def _make_chat_chunk(delta_text: str, model: str = "eve", done: bool = False) -> dict:
    """streaming chunk (SSE)."""
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {} if done else {"content": delta_text},
            "finish_reason": "stop" if done else None,
        }],
    }


class OpenAIServerAdapter:

    def __init__(self, engine, host: str = "127.0.0.1", port: int = 8080):
        self.engine = engine
        self.host = host
        self.port = port
        self.running = False
        self._server = None
        self._thread = None
        self._ready_event = threading.Event()

    # ===== 핵심 - 요청 처리 =====

    def _handle_chat_completions(self, body: dict) -> tuple:
        """
        OpenAI 호환 chat.completions 처리.
        Returns: (status_code, body_dict_or_generator, is_stream)
        """
        messages = body.get("messages", [])
        if not messages:
            return (400, {"error": {"message": "no messages"}}, False)
        # 마지막 user 메시지만 사용 (EVE는 단일-turn 처리)
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        if not user_text:
            return (400, {"error": {"message": "no user content"}}, False)

        stream = body.get("stream", False)

        if stream:
            # 입모양 ON (라운드 25 통합)
            if hasattr(self.engine, "character_adapter") and self.engine.character_adapter:
                self.engine.character_adapter.set_speaking(True)

            def gen():
                for chunk in self.engine.chat_stream(user_text):
                    yield "data: " + json.dumps(_make_chat_chunk(chunk), ensure_ascii=False) + "\n\n"
                # done
                yield "data: " + json.dumps(_make_chat_chunk("", done=True), ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
                # 입모양 OFF
                if hasattr(self.engine, "character_adapter") and self.engine.character_adapter:
                    self.engine.character_adapter.set_speaking(False)

            return (200, gen(), True)
        else:
            # non-streaming
            chunks = list(self.engine.chat_stream(user_text))
            full = " ".join(chunks)
            return (200, _make_chat_response(full), False)

    def _handle_models(self) -> tuple:
        return (200, {
            "object": "list",
            "data": [{
                "id": "eve",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "민석",
            }],
        }, False)

    # ===== HTTP 서버 (stdlib http.server 사용 — 의존성 0) =====

    def _build_handler(self):
        adapter = self
        from http.server import BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass     # 조용히

            def _send_json(self, status, body):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def _send_sse(self, generator):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    for chunk_str in generator:
                        self.wfile.write(chunk_str.encode("utf-8"))
                        self.wfile.flush()
                except Exception:
                    pass

            def do_OPTIONS(self):
                # CORS preflight
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()

            def do_GET(self):
                if self.path == "/v1/models":
                    status, body, _ = adapter._handle_models()
                    self._send_json(status, body)
                elif self.path == "/health" or self.path == "/":
                    self._send_json(200, {"status": "ok", "engine": "EVE v41"})
                else:
                    self._send_json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_len) if content_len > 0 else b""
                try:
                    body = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    self._send_json(400, {"error": {"message": "invalid JSON"}})
                    return

                if self.path == "/v1/chat/completions":
                    status, result, is_stream = adapter._handle_chat_completions(body)
                    if is_stream:
                        self._send_sse(result)
                    else:
                        self._send_json(status, result)
                else:
                    self._send_json(404, {"error": {"message": "not found"}})

        return Handler

    def start_background(self) -> bool:
        """별도 thread에서 HTTP 서버 시작."""
        if self.running:
            return True
        try:
            from http.server import ThreadingHTTPServer
        except ImportError:
            return False

        Handler = self._build_handler()
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            return False     # 포트 사용 중

        def runner():
            self._ready_event.set()
            try:
                # Small poll interval keeps stop() responsive in tests/mobile runs.
                self._server.serve_forever(poll_interval=0.01)
            except Exception:
                pass
            finally:
                self.running = False

        self._ready_event.clear()
        self.running = True
        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return True

    def wait_until_ready(self, timeout: float = 1.0) -> bool:
        """서버 thread가 요청을 받을 준비가 될 때까지 짧게 대기."""
        deadline = time.time() + timeout
        self._ready_event.wait(max(0.0, timeout))
        while time.time() < deadline:
            if not self.running:
                return False
            try:
                with socket.create_connection((self.host, self.port), timeout=0.05):
                    return True
            except OSError:
                time.sleep(0.01)
        return False

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            try:
                self._server.server_close()
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self.running = False
        self._ready_event.clear()

    def status(self) -> dict:
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "endpoints": [
                f"http://{self.host}:{self.port}/v1/models",
                f"http://{self.host}:{self.port}/v1/chat/completions",
            ],
        }
