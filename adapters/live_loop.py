"""
LiveLoop — 라운드 27

EVE를 *연속적으로 살아있게* 한다.
인공 룰/임계/limit 0. 호르몬 + agency만이 결정.

원칙:
- 매 1초 (인간 신경 발화 주기 정도)
- 호르몬 자연 변화 (시간 흐름 = 호르몬 흐름)
- DMN (Default Mode Network) 기본 활성
- 자율 행동/발화 = autonomous_loop + agency
- 사용자 입력 = 인터럽트 (queue로 push)
- 자동 persistence (default 5분)

학계:
- Raichle 2001 — DMN 기본 모드
- Bar 2007 — 뇌는 항상 예측 모드, 입력 없을 때도 활성
- Friston Active Inference — 호르몬 변화 = free energy 변화
"""

import os
import threading
import time
import queue
from typing import Optional, Callable


class LiveLoop:

    def __init__(self,
                 engine,
                 interval: float = 1.0,
                 autosave_path: Optional[str] = None,
                 autosave_interval: float = 300.0,
                 emit_callback: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.interval = interval
        self.autosave_path = autosave_path
        self.autosave_interval = autosave_interval
        self.emit_callback = emit_callback or (lambda text: print(f"\n[EVE] {text}"))
        # 상태
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._user_input_queue: queue.Queue = queue.Queue()
        self._last_autosave = 0.0
        self._last_emit_time = 0.0
        self._wake_event = threading.Event()
        self._tick_event = threading.Event()
        self._input_event = threading.Event()
        # trace
        self.tick_count = 0
        self.emit_count = 0
        self.error_count = 0
        self.processed_input_count = 0

    # ===== 사용자 입력 처리 =====

    def push_user_input(self, text: str):
        """사용자 입력을 LiveLoop에 인터럽트로 전달.

        대기 중인 루프를 즉시 깨워서 interval 전체를 기다리지 않게 한다.
        """
        self._user_input_queue.put(text)
        self._wake_event.set()

    def _drain_user_inputs(self):
        """큐에 쌓인 입력 처리."""
        while True:
            try:
                text = self._user_input_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_user_input(text)
            self.processed_input_count += 1
            self._input_event.set()

    def _handle_user_input(self, text: str):
        """사용자 입력 처리 — 가르침 평가 + 일반 chat."""
        # 라운드 61: 입력 받자마자 mark_input — 이후 30초 자율발화 차단
        autonomy = getattr(self.engine, "autonomy_adapter", None)
        if autonomy is not None:
            try:
                autonomy.mark_input()
            except Exception:
                pass
        
        # 1. 가르침 인식 + 평가 (라운드 28)
        if hasattr(self.engine, "teaching_adapter") and self.engine.teaching_adapter:
            tr = self.engine.teaching_adapter.process_teaching(text)
            if tr is not None:
                self.emit_callback(tr["response"])
                if tr["decision"] == "accept":
                    self.engine.teaching_adapter.weaken_last()
                self._last_emit_time = time.time()
                return
            # 수정 아니면 강화
            self.engine.teaching_adapter.reinforce_if_no_correction()

        # 2. 일반 chat
        chunks = list(self.engine.chat_stream(text))
        full = " ".join(chunks)
        self.emit_callback(full)
        self._last_emit_time = time.time()
        # 라운드 62: 응답 후에도 mark_input — 대화 *유지* 상태
        if autonomy is not None:
            try:
                autonomy.mark_input()
            except Exception:
                pass

    # ===== 메인 루프 =====

    def _run(self):
        try:
            while self.running:
                start = time.time()
                self.tick_count += 1
                self._tick_event.set()

                # 1. 사용자 입력 처리 (인터럽트)
                self._drain_user_inputs()

                # 2. 호르몬 자연 변화 (시간 흐름)
                try:
                    self.engine.hormone_adapter.hs.update(dt=self.interval)
                except Exception as e:
                    self.error_count += 1

                # 2.5. Salience tick (라운드 37)
                try:
                    sal = getattr(self.engine, "salience_adapter", None)
                    if sal is not None:
                        sal.tick(dt=self.interval)
                except Exception:
                    self.error_count += 1

                # 3. autonomous step — 라운드 61: is_due 검사 (입력 직후 차단)
                emitted = False
                autonomy = getattr(self.engine, "autonomy_adapter", None)
                # 자율발화 가능한 상태인지 — 입력 후 idle_threshold 지나야
                can_autonomy = True
                if autonomy is not None:
                    try:
                        can_autonomy = autonomy.is_due()
                    except Exception:
                        can_autonomy = True
                
                if can_autonomy:
                    try:
                        result = self.engine.autonomous_loop.step(emit=True)
                        response = result.get("response")
                        
                        # 자율 발화 있으면 emit
                        if result.get("emitted") and response and isinstance(response, str):
                            self.emit_callback(response)
                            self._last_emit_time = time.time()
                            self.emit_count += 1
                            emitted = True
                    except Exception as e:
                        self.error_count += 1

                # 3.5. autonomous가 발화 안 했으면 — urge 체크 후 proactive_stream
                # 라운드 61: can_autonomy=False면 proactive도 차단
                if not emitted and can_autonomy:
                    try:
                        urge = getattr(self.engine, "urge_adapter", None)
                        # urge가 임계 넘었을 때만 발화 시도
                        if urge is None or urge.should_emit(time.time()):
                            chunks = list(self.engine.proactive_stream(force=True))
                            if chunks:
                                full = " ".join(c for c in chunks if c).strip()
                                if full:
                                    # Exploration 체크 — 반복 응답 skip
                                    safety = getattr(self.engine, "safety_adapter", None)
                                    if safety is not None and safety.is_repetitive(full):
                                        pass    # 반복 — 침묵
                                    else:
                                        self.emit_callback(full)
                                        if safety is not None:
                                            safety.record_response(full)
                                        if urge is not None:
                                            urge.mark_emitted(time.time())
                                        self._last_emit_time = time.time()
                                        self.emit_count += 1
                                        if self.engine.autonomy_adapter is not None:
                                            self.engine.autonomy_adapter.mark_proactive()
                    except Exception as e:
                        self.error_count += 1

                # 4. 자동 저장
                if self.autosave_path and (time.time() - self._last_autosave) > self.autosave_interval:
                    self._do_autosave()

                # 5. 1초 대기 (인터벌 보정)
                elapsed = time.time() - start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    self._wake_event.wait(sleep_time)
                    self._wake_event.clear()
        except Exception:
            self.running = False

    def _do_autosave(self):
        try:
            from adapters.persistence_adapter import PersistenceAdapter
            p = PersistenceAdapter(self.engine)
            p.save(self.autosave_path)
            self._last_autosave = time.time()
        except Exception:
            self.error_count += 1

    # ===== 공개 인터페이스 =====

    def start(self):
        if self.running:
            return False
        self.running = True
        self._last_autosave = time.time()
        # 시작 시 _last_emit_time을 cooldown 만큼 *과거*로 — 즉시 첫 발화 가능
        if self.engine.autonomy_adapter is not None:
            self._last_emit_time = time.time() - (self.engine.autonomy_adapter.cooldown + 1)
        else:
            self._last_emit_time = time.time() - 100
        self._wake_event.clear()
        self._tick_event.clear()
        self._input_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def wait_for_tick(self, min_count: int = 1, timeout: float = 1.0) -> bool:
        """테스트/제어용: tick_count가 목표치에 도달할 때까지 대기."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.tick_count >= min_count:
                return True
            remaining = max(0.0, deadline - time.time())
            self._tick_event.wait(min(0.05, remaining))
            self._tick_event.clear()
        return self.tick_count >= min_count

    def wait_for_processed_inputs(self, min_count: int = 1, timeout: float = 1.0) -> bool:
        """테스트/제어용: 사용자 입력 처리 완료를 카운트 기반으로 대기."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.processed_input_count >= min_count:
                return True
            remaining = max(0.0, deadline - time.time())
            self._input_event.wait(min(0.05, remaining))
            self._input_event.clear()
        return self.processed_input_count >= min_count

    def status(self) -> dict:
        return {
            "running": self.running,
            "interval": self.interval,
            "tick_count": self.tick_count,
            "emit_count": self.emit_count,
            "error_count": self.error_count,
            "processed_input_count": self.processed_input_count,
            "autosave_path": self.autosave_path,
            "autosave_interval": self.autosave_interval,
            "queue_size": self._user_input_queue.qsize(),
        }
