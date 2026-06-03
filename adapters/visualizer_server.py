"""
VisualizerServer — 라운드 24

EVE 상태를 WebSocket으로 push, 브라우저가 2D 시각화.

원칙:
- Lazy load (websockets 없어도 EVE 안 죽음)
- background thread 0 (사용자가 명시적 시작)
- 결정론 안 깸 (관찰만, 사고는 EVE)
- 폰 Termux + Dell 7070 Micro 둘 다 OK

프로토콜 (양방향 WebSocket):
  EVE → 클라이언트:
    {"type": "state", "hormones": {...}, "location": "내방", ...}
    {"type": "chunk", "text": "..."}
    {"type": "done"}
  클라이언트 → EVE:
    {"type": "user_message", "text": "..."}
    {"type": "tick"}
    {"type": "save"}
"""

import asyncio
import json
import os
import threading
from typing import Optional


# 호르몬 → HSL 색상 매핑 (실제 26 호르몬 키 사용)
HORMONE_COLORS = {
    "dopamine":      "#FFD700",     # 금색 (보상)
    "oxytocin":      "#FFB6C1",     # 핑크 (사회 연결)
    "serotonin":     "#90EE90",     # 연녹 (안정)
    "cortisol":      "#4682B4",     # 파랑 (스트레스)
    "norepinephrine": "#FF6347",    # 토마토 (각성/주의)
    "endorphin":     "#FFA500",     # 주황 (쾌감)
    "melatonin":     "#191970",     # 어두운 청 (수면)
    "glutamate":     "#DC143C",     # 진홍 (인지 부하)
    "gaba":          "#9370DB",     # 보라 (억제, 평온)
    "glycine":       "#B0C4DE",     # 연파랑 (억제)
    "histamine":     "#FF69B4",     # 핫핑크 (각성/알레르기)
    "acetylcholine": "#00CED1",     # 청록 (집중/학습)
    "adenosine":     "#708090",     # 슬레이트 (피로)
    "vasopressin":   "#5F9EA0",     # 카뎃블루 (애착)
    "bdnf":          "#32CD32",     # 라임그린 (성장)
    "ngf":           "#3CB371",     # 미디엄시그린 (신경 성장)
    "estrogen":      "#DDA0DD",     # 자두 (호르몬 균형)
    "testosterone":  "#A0522D",     # 시에나 (활력)
    "insulin_brain": "#FFE4B5",     # 모카신 (대사)
    "thyroid":       "#F0E68C",     # 카키 (대사)
    "leptin":        "#98FB98",     # 연한녹 (포만)
    "ghrelin":       "#FF8C00",     # 진주황 (배고픔)
    "prolactin":     "#FFC0CB",     # 핑크 (애착/모성)
    "dhea":          "#DAA520",     # 골든로드 (활력)
    "progesterone":  "#D8BFD8",     # 시슬 (안정)
    "growth_hormone": "#7CFC00",    # 잔디녹 (성장)
}


def build_state_snapshot(engine) -> dict:
    """EVE 현재 상태를 JSON-friendly dict로."""
    state = {
        "type": "state",
        "hormones": {},
        "location": None,
        "objects": [],
        "npcs": [],
        "intimacy": 0.0,
        "feeling": "",
        "active_categories": [],
        "step_count": 0,
        "is_home": True,
    }
    # 호르몬
    try:
        for name, h in engine.hormone_adapter.hs.hormones.items():
            state["hormones"][name] = {
                "level": float(h.level),
                "baseline": float(h.baseline),
                "color": HORMONE_COLORS.get(name, "#888888"),
            }
    except Exception:
        pass
    # 위치
    try:
        state["location"] = engine.env_adapter.location_name()
        state["objects"] = engine.env_adapter.get_objects()
        state["is_home"] = engine.env_adapter.is_home()
        if not state["is_home"] and engine.env_adapter.social_env_adapter is not None:
            state["npcs"] = engine.env_adapter.social_env_adapter.get_npcs(state["location"])
    except Exception:
        pass
    # 친밀도
    try:
        state["intimacy"] = float(engine.user_presence_adapter.get_intimacy())
    except Exception:
        pass
    # 신체 감각
    try:
        state["feeling"] = engine.digital_somatic_adapter.feeling_phrase()
    except Exception:
        pass
    # 활성 카테고리 top 10
    try:
        sa = engine.activation_adapter.sa
        sorted_act = sorted(sa.activations.items(), key=lambda x: x[1], reverse=True)[:10]
        state["active_categories"] = [
            {"name": k, "level": float(v)} for k, v in sorted_act if v > 0.05
        ]
    except Exception:
        pass
    # 자율 step 카운트
    try:
        if engine.autonomous_loop is not None:
            state["step_count"] = engine.autonomous_loop.step_count
    except Exception:
        pass
    # 캐릭터 (라운드 25)
    try:
        if hasattr(engine, "character_adapter") and engine.character_adapter is not None:
            state["character"] = engine.character_adapter.snapshot()
    except Exception:
        state["character"] = {"expression": "neutral", "motion": "idle", "speaking": False}
    return state


class VisualizerServer:

    def __init__(self, engine, host: str = "127.0.0.1", port: int = 8765,
                 push_interval: float = 1.0):
        self.engine = engine
        self.host = host
        self.port = port
        self.push_interval = push_interval
        self.running = False
        self._loop = None
        self._thread = None
        self._connected_clients = set()
        self._websockets = None      # lazy

    def _load_websockets(self):
        if self._websockets is not None:
            return self._websockets
        try:
            import websockets
            self._websockets = websockets
            return self._websockets
        except ImportError:
            self._websockets = "unavailable"
            return None

    def available(self) -> bool:
        if self._websockets is None:
            self._load_websockets()
        return self._websockets is not None and self._websockets != "unavailable"

    # ===== 핸들러 =====

    async def handle_client(self, ws):
        """클라이언트 연결 핸들러."""
        self._connected_clients.add(ws)
        try:
            # 첫 연결 시 현재 상태 push
            await ws.send(json.dumps(build_state_snapshot(self.engine), ensure_ascii=False))
            # 메시지 수신
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle_message(ws, msg)
        except Exception:
            pass
        finally:
            self._connected_clients.discard(ws)

    async def _handle_message(self, ws, msg: dict):
        kind = msg.get("type")
        if kind == "user_message":
            text = msg.get("text", "")
            if text:
                # 입모양 ON
                if hasattr(self.engine, "character_adapter") and self.engine.character_adapter:
                    self.engine.character_adapter.set_speaking(True)
                # chat_stream 결과 chunk 별 push
                for chunk in self.engine.chat_stream(text):
                    await ws.send(json.dumps({
                        "type": "chunk", "text": chunk,
                    }, ensure_ascii=False))
                    # chunk마다 상태도 push (speaking 토글 자연스럽게)
                    await self._broadcast_state()
                # 입모양 OFF
                if hasattr(self.engine, "character_adapter") and self.engine.character_adapter:
                    self.engine.character_adapter.set_speaking(False)
                await ws.send(json.dumps({"type": "done"}, ensure_ascii=False))
                # 응답 후 상태 새로 push
                await self._broadcast_state()
        elif kind == "tick":
            n = msg.get("n", 1)
            for _ in range(n):
                if self.engine.autonomous_loop is None:
                    break
                r = self.engine.autonomous_loop.step(emit=False)
                await ws.send(json.dumps({
                    "type": "tick_result",
                    "step": r.get("step"),
                    "needs": r.get("needs", []),
                    "action": r.get("action"),
                }, ensure_ascii=False))
            await self._broadcast_state()
        elif kind == "save":
            try:
                from adapters.persistence_adapter import PersistenceAdapter
                p = PersistenceAdapter(self.engine)
                path = msg.get("path", "/tmp/eve.ckpt")
                p.save(path)
                await ws.send(json.dumps({"type": "saved", "path": path}, ensure_ascii=False))
            except Exception as e:
                await ws.send(json.dumps({"type": "error", "msg": str(e)}, ensure_ascii=False))
        elif kind == "request_state":
            await ws.send(json.dumps(build_state_snapshot(self.engine), ensure_ascii=False))

    async def _broadcast_state(self):
        if not self._connected_clients:
            return
        snap = json.dumps(build_state_snapshot(self.engine), ensure_ascii=False)
        # 끊긴 클라이언트 정리
        dead = []
        for ws in self._connected_clients:
            try:
                await ws.send(snap)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connected_clients.discard(ws)

    async def _periodic_push(self):
        """주기적으로 상태 push (push_interval 초마다)."""
        while self.running:
            await asyncio.sleep(self.push_interval)
            await self._broadcast_state()

    async def _serve(self):
        ws_lib = self._load_websockets()
        if ws_lib is None:
            return
        self.running = True
        async with ws_lib.serve(self.handle_client, self.host, self.port):
            await self._periodic_push()

    # ===== 공개 인터페이스 =====

    def start_background(self) -> bool:
        """별도 thread에서 서버 실행. 사용자가 명시적 호출."""
        if not self.available():
            return False
        if self.running:
            return True

        def runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except Exception:
                pass

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

    def status(self) -> dict:
        return {
            "available": self.available(),
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "clients": len(self._connected_clients),
        }
