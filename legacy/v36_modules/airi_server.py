"""
EVE v38 — AIRI WebSocket Server (Termux 친화)
==============================================

의존성: websockets (1개만, 순수 Python — Termux 빠른 설치)
설치: pip install websockets

실행:
  cd ~/eve_v36
  python3 v36_modules/airi_server.py
  
  또는 (env 변수로 포트):
  EVE_PORT=8765 python3 v36_modules/airi_server.py

연결:
  ws://localhost:8765   (같은 폰 안에서)
  ws://<폰IP>:8765      (같은 wifi 다른 기기에서)
  
Termux 설정 권장:
  termux-wake-lock     # 절전 방지 (백그라운드 실행)
  ifconfig             # 폰 IP 확인 (wifi)

프로토콜:
  Client → Server (JSON):
    {"type": "user_message", "text": "..."}
    {"type": "user_action", "action": "tick", "params": {"dt": 0.5}}
    {"type": "user_action", "action": "live_hours", "params": {"hours": 1.0}}
    {"type": "ping"}
  
  Server → Client (JSON):
    {"type": "state", ...}        # 주기적 상태 push
    {"type": "speech", ...}        # EVE 발화
    {"type": "proactive", ...}     # 자발 발화
    {"type": "action_result", ...} # 액션 결과
    {"type": "pong"}
"""

import sys
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Set, Optional

# Termux 친화 — 경로 자동 탐지
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name == 'v36_modules' else _HERE
_V35 = _ROOT / 'eve_v35_release'
sys.path.insert(0, str(_HERE))  # v36_modules 자체
if _V35.exists():
    sys.path.insert(0, str(_V35))
    sys.path.insert(0, str(_V35 / 'eve_modules'))

# websockets 의존성
try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
except ImportError:
    print("ERROR: websockets 라이브러리 없음")
    print("Termux에서 설치:")
    print("  pkg install python")
    print("  pip install websockets")
    sys.exit(1)

from eve_v36 import EVE_v36
from airi_adapter import AIRIAdapter


# ============================================================
# 로깅
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('eve_airi')


# ============================================================
# 서버 — 단일 EVE 인스턴스 + 다중 클라이언트
# ============================================================

class EVEServer:
    """
    EVE 백엔드를 WebSocket으로 노출.
    
    - 단일 EVE 인스턴스 (모든 클라이언트가 같은 EVE에 접속)
    - 다중 클라이언트 동시 연결 가능 (모든 클라이언트에 broadcast)
    - 백그라운드 자동 tick (호르몬 자연 변화 + DMN)
    - 백그라운드 주기적 state push
    - 자동 영속화 (옵션)
    """

    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 8765,
                 tick_interval: float = 0.5,
                 state_push_interval: float = 1.0,
                 sim_hour_per_tick: float = 0.0,
                 autosave_path: Optional[str] = None,
                 autosave_interval: int = 600):
        """
        Args:
            host: 바인드 주소. 0.0.0.0 = 모든 인터페이스 (wifi 다른 기기 접속 OK)
            port: 포트 (기본 8765)
            tick_interval: EVE.tick 주기 (초)
            state_push_interval: 상태 push 주기 (초)
            sim_hour_per_tick: 매 tick 시뮬 시간 진행 (시간). 0이면 자동 X
            autosave_path: 자동 저장 경로 (None = 비활성)
            autosave_interval: 자동 저장 주기 (초)
        """
        self.host = host
        self.port = port
        self.tick_interval = tick_interval
        self.state_push_interval = state_push_interval
        self.sim_hour_per_tick = sim_hour_per_tick
        self.autosave_path = autosave_path
        self.autosave_interval = autosave_interval

        # EVE 인스턴스 (백그라운드 단일)
        logger.info("EVE 인스턴스 생성 중...")
        self.eve = EVE_v36(seed_common_sense_on_init=True, enhance_say=True)
        self.eve.enter_room()  # 자기 방에서 시작
        self.adapter = AIRIAdapter(self.eve)
        logger.info(f"EVE 준비 완료: 호르몬 26, "
                   f"카테고리 {len(self.eve.sa.neighbors)}, "
                   f"위치 {self.eve._env_adapter.current_env.name}")

        # 클라이언트 set
        self.clients: Set = set()
        self.shutdown_event = asyncio.Event()

    # ============= 클라이언트 메시지 처리 =============
    async def handle_client_message(self, websocket, message: str):
        """클라이언트 → 서버 메시지 처리"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error', 'error': 'JSON 파싱 실패'
            }))
            return

        msg_type = data.get('type', '')

        if msg_type == 'ping':
            await websocket.send(json.dumps({'type': 'pong'}))
            return

        if msg_type == 'user_message':
            text = data.get('text', '')
            if not text:
                return
            logger.info(f"사용자: {text}")
            result = self.adapter.handle_user_message(text)
            if result.get('success'):
                # 응답 broadcast
                speech = self.adapter.build_speech_message(
                    text=result['response'],
                    is_response=True,
                    feeling=result.get('feeling', ''),
                    pattern=result.get('pattern', ''),
                )
                await self.broadcast(speech)
                logger.info(f"EVE: {result['response']}")
                # 응답 직후 새 상태 push
                await self.broadcast(self.adapter.build_state_message())
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'error': result.get('error', 'unknown')
                }))
            return

        if msg_type == 'user_action':
            action = data.get('action', '')
            params = data.get('params', {})
            result = self.adapter.handle_user_action(action, params)
            await websocket.send(json.dumps({
                'type': 'action_result',
                'action': action,
                **result,
            }))
            # 액션 후 상태 push
            await self.broadcast(self.adapter.build_state_message())
            return

        if msg_type == 'request_state':
            # 클라이언트가 즉시 상태 요청
            await websocket.send(json.dumps(self.adapter.build_state_message()))
            return

        await websocket.send(json.dumps({
            'type': 'error',
            'error': f'알 수 없는 메시지 type: {msg_type}',
        }))

    # ============= 클라이언트 연결 핸들러 =============
    async def handler(self, websocket):
        """새 WebSocket 연결 처리"""
        client_addr = websocket.remote_address
        logger.info(f"클라이언트 연결: {client_addr}")
        self.clients.add(websocket)

        try:
            # 환영: 즉시 현재 상태 보내기
            await websocket.send(json.dumps(self.adapter.build_state_message()))

            # 클라이언트 메시지 루프
            async for message in websocket:
                await self.handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"핸들러 에러: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"클라이언트 분리: {client_addr}")

    # ============= broadcast =============
    async def broadcast(self, message: dict):
        """모든 클라이언트에게 메시지 송신"""
        if not self.clients:
            return
        msg_str = json.dumps(message, ensure_ascii=False)
        # 분리된 클라이언트는 자동 제거
        dead = []
        for ws in self.clients:
            try:
                await ws.send(msg_str)
            except websockets.exceptions.ConnectionClosed:
                dead.append(ws)
            except Exception as e:
                logger.warning(f"broadcast 에러: {e}")
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # ============= 백그라운드 — 자동 tick =============
    async def tick_loop(self):
        """EVE 백그라운드 tick — 호르몬 변화 + DMN"""
        logger.info(f"tick loop 시작 (간격={self.tick_interval}s)")
        while not self.shutdown_event.is_set():
            try:
                # 시뮬 시간 진행 (옵션)
                if self.sim_hour_per_tick > 0:
                    self.eve.hs.sim_hour = (
                        (self.eve.hs.sim_hour + self.sim_hour_per_tick) % 24
                    )

                # tick
                self.eve.tick(dt=self.tick_interval)

                # proactive 발화 검사
                proactive = None
                try:
                    proactive = self.eve.proactive_message()
                except Exception:
                    pass
                if proactive and 'message' in proactive:
                    msg = self.adapter.build_proactive_message(
                        message=proactive['message'],
                        reason=proactive.get('reason', ''),
                    )
                    await self.broadcast(msg)
                    logger.info(f"EVE (자발): {proactive['message']}")

            except Exception as e:
                logger.error(f"tick 에러: {e}")

            await asyncio.sleep(self.tick_interval)

    # ============= 백그라운드 — state push =============
    async def state_push_loop(self):
        """주기적으로 클라이언트에게 상태 push"""
        logger.info(f"state push loop 시작 (간격={self.state_push_interval}s)")
        while not self.shutdown_event.is_set():
            try:
                if self.clients:
                    state = self.adapter.build_state_message()
                    await self.broadcast(state)
            except Exception as e:
                logger.error(f"state push 에러: {e}")
            await asyncio.sleep(self.state_push_interval)

    # ============= 백그라운드 — autosave =============
    async def autosave_loop(self):
        """자동 영속화"""
        if not self.autosave_path:
            return
        logger.info(f"autosave loop 시작 (간격={self.autosave_interval}s, 경로={self.autosave_path})")
        while not self.shutdown_event.is_set():
            await asyncio.sleep(self.autosave_interval)
            try:
                info = self.eve.save(self.autosave_path)
                logger.info(f"자동 저장: {info.get('path', self.autosave_path)} "
                           f"({info.get('size_bytes', 0)}B)")
            except Exception as e:
                logger.error(f"autosave 에러: {e}")

    # ============= 메인 =============
    async def run(self):
        """서버 실행"""
        # 자동 로드 — autosave 경로가 있고 파일 존재하면
        if self.autosave_path:
            for candidate in [self.autosave_path, self.autosave_path + '.gz']:
                if os.path.exists(candidate):
                    try:
                        report = self.eve.load(candidate)
                        logger.info(f"자동 로드: {candidate} (복원 {len(report.get('restored', []))}개)")
                        break
                    except Exception as e:
                        logger.warning(f"자동 로드 실패: {e}")

        # 백그라운드 태스크
        background_tasks = [
            asyncio.create_task(self.tick_loop()),
            asyncio.create_task(self.state_push_loop()),
        ]
        if self.autosave_path:
            background_tasks.append(asyncio.create_task(self.autosave_loop()))

        # WebSocket 서버 시작
        logger.info(f"WebSocket 서버 시작: ws://{self.host}:{self.port}")
        logger.info(f"  같은 폰 안: ws://localhost:{self.port}")
        logger.info(f"  같은 wifi: ws://<폰IP>:{self.port} (ifconfig로 IP 확인)")

        async with serve(self.handler, self.host, self.port):
            try:
                await self.shutdown_event.wait()
            except KeyboardInterrupt:
                logger.info("종료 신호 수신")
                self.shutdown_event.set()

        # 정리
        for task in background_tasks:
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)
        logger.info("서버 종료")


# ============================================================
# 진입점
# ============================================================
def main():
    """CLI 진입점 — 환경 변수 + 인자 모두 지원"""
    import argparse
    parser = argparse.ArgumentParser(description='EVE AIRI WebSocket 서버')
    parser.add_argument('--host', default=os.environ.get('EVE_HOST', '0.0.0.0'),
                       help='바인드 호스트 (기본 0.0.0.0)')
    parser.add_argument('--port', type=int,
                       default=int(os.environ.get('EVE_PORT', 8765)),
                       help='포트 (기본 8765)')
    parser.add_argument('--tick', type=float,
                       default=float(os.environ.get('EVE_TICK', 0.5)),
                       help='tick 간격 (초)')
    parser.add_argument('--state-push', type=float,
                       default=float(os.environ.get('EVE_STATE_PUSH', 1.0)),
                       help='상태 push 간격 (초)')
    parser.add_argument('--sim-hour-per-tick', type=float,
                       default=float(os.environ.get('EVE_SIM_HOUR_PER_TICK', 0.0)),
                       help='매 tick당 시뮬 시간 진행 (시간). 0=자동 X')
    parser.add_argument('--autosave',
                       default=os.environ.get('EVE_AUTOSAVE'),
                       help='자동 저장 경로 (예: ~/eve.ckpt)')
    parser.add_argument('--autosave-interval', type=int,
                       default=int(os.environ.get('EVE_AUTOSAVE_INTERVAL', 600)),
                       help='자동 저장 주기 (초)')
    args = parser.parse_args()

    server = EVEServer(
        host=args.host,
        port=args.port,
        tick_interval=args.tick,
        state_push_interval=args.state_push,
        sim_hour_per_tick=args.sim_hour_per_tick,
        autosave_path=args.autosave,
        autosave_interval=args.autosave_interval,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
