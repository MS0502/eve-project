"""
AutonomyAdapter

idle 감지 + 자동 proactive 트리거.

원칙:
- background thread 안 씀 (결정론 유지)
- main 루프가 명시적으로 check 호출
- 시간은 외부에서 주입 가능 (테스트 결정론 위해)

흐름:
1. 매 chat_stream — 마지막 입력 시간 갱신
2. 외부에서 idle_speak_if_due(now=...) 호출
3. now - last_input_time > threshold면 proactive_stream 트리거 OK 신호

REPL 통합 예시:
    while True:
        line = read_with_timeout(5초)
        if line is None:                            # timeout
            if engine.autonomy.is_due(now=...):
                for c in engine.proactive_stream(force=True):
                    print(c)
        else:
            engine.chat_stream(line)
"""

import time as _time


class AutonomyAdapter:

    def __init__(self,
                 idle_threshold_sec: float = 30.0,
                 cooldown_sec: float = 60.0,
                 time_fn=_time.time):
        """
        idle_threshold_sec: 마지막 입력 후 N초 지나면 자발 발화 가능
        cooldown_sec: 자발 발화 후 다음 발화까지 최소 대기
        """
        self.idle_threshold = idle_threshold_sec
        self.cooldown = cooldown_sec
        self._time = time_fn
        self.last_input_time = self._time()
        self.last_proactive_time = 0.0
        # 라운드 62: 대화 상태 기반 차단
        # 사용자 입력 받으면 True. 새 입력 안 오고 idle_threshold 지나면 False.
        self.in_conversation = False
        self.conversation_timeout = 30.0   # 30초 입력 없으면 대화 끝난 걸로 (자율발화 가능)

    def mark_input(self, now: float | None = None):
        """사용자 입력 받았을 때 호출 (StreamingEngine 자동)."""
        self.last_input_time = now if now is not None else self._time()
        # 라운드 62: 대화 시작/지속
        self.in_conversation = True

    def mark_proactive(self, now: float | None = None):
        """자발 발화 했을 때 호출."""
        self.last_proactive_time = now if now is not None else self._time()

    def is_due(self, now: float | None = None) -> bool:
        """지금 자발 발화 해도 되는지.
        
        라운드 62: in_conversation 우선 검사 — 대화 중이면 무조건 False.
        """
        now = now if now is not None else self._time()
        
        # 라운드 62: 대화 중이면 자율발화 절대 X
        # conversation_timeout 지나면 대화 끝난 걸로 간주
        if self.in_conversation:
            since_input = now - self.last_input_time
            if since_input < self.conversation_timeout:
                return False
            # timeout 지났으면 대화 종료
            self.in_conversation = False
        
        idle = now - self.last_input_time
        cooldown_passed = (now - self.last_proactive_time) > self.cooldown
        return idle > self.idle_threshold and cooldown_passed

    def get_idle_seconds(self, now: float | None = None) -> float:
        now = now if now is not None else self._time()
        return now - self.last_input_time
