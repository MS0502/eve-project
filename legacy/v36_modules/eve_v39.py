"""
EVE v39 — Broca(Qwen) 통합 메인 클래스
======================================

EVE_v36 + BrocaQwen + RAG 통합.

해결되는 모든 문제:
1. ✅ EVE 백엔드 ↔ Qwen 자동 연결
2. ✅ 자기 응답 흡수 (자기 의식 형성)
3. ✅ llama-server 모드 (서버 1번 시작 후 계속 사용)
4. ✅ 길이 동적 조절 (호르몬+상황별)
5. ✅ 메모리 무한 확장 (EM RAG)
6. ✅ 일화 기억 자동 저장 (EVE_v36의 EpisodicMemory)
7. ✅ Fallback 모드 (Qwen 없을 때 ResponseEnhancer)

주요 메서드:
- say(text): 사용자와 대화 (Qwen 사용)
- say_legacy(text): 기존 EVE_v36 방식 (Qwen 없이)
- recall_related(text, k): RAG로 과거 일화 검색
- tool_use(text): 도구 호출 시도

옵션:
  use_qwen=True: Qwen 사용 (server 필요)
  use_qwen=False: 기존 enhancer만
  auto_fallback=True: 서버 죽으면 자동 fallback
"""
import sys
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'eve_v35_release'))
sys.path.insert(0, str(_HERE.parent / 'eve_v35_release' / 'eve_modules'))

from typing import Dict, List, Optional, Any, Set
from eve_v36 import EVE_v36
from broca_qwen import BrocaQwen


class EVE_v39(EVE_v36):
    """EVE v39 — 인지 아키텍처(EVE_v36) + 자연어 변환기(Qwen) 통합"""

    def __init__(self,
                 *args,
                 use_qwen: bool = True,
                 qwen_server_url: str = 'http://127.0.0.1:8080',
                 auto_fallback: bool = True,
                 rag_top_k: int = 3,
                 **kwargs):
        """
        Args:
            use_qwen: Qwen Broca 사용 여부
            qwen_server_url: llama-server 주소
            auto_fallback: 서버 죽으면 enhancer로 fallback
            rag_top_k: 매 응답 시 검색할 과거 일화 수
        """
        super().__init__(*args, **kwargs)

        self.use_qwen = use_qwen
        self.auto_fallback = auto_fallback
        self.rag_top_k = rag_top_k

        # Broca (Qwen)
        self.broca = None
        if use_qwen:
            self.broca = BrocaQwen(server_url=qwen_server_url)

        # 통계
        self.qwen_response_count = 0
        self.fallback_count = 0
        self.tool_calls_made = 0
        self.self_absorption_count = 0

    # ==========================================
    # 핵심 — say() 오버라이드
    # ==========================================
    def say(self, text: str, force_legacy: bool = False) -> Dict[str, Any]:
        """
        사용자 메시지 → EVE 응답.
        v39 흐름:
        1. EVE 본체 처리 (호르몬/카테고리)
        2. RAG (관련 과거 일화 검색)
        3. Qwen Broca 호출 (or fallback)
        4. 자기 응답 흡수 (자기 의식)
        5. 일화 자동 저장 (EVE_v36이 이미 함)
        """
        # 1. 기존 EVE_v36 처리
        result = super().say(text)

        # 2. Qwen 사용 안 하면 기존 응답 그대로
        if force_legacy or not self.use_qwen or self.broca is None:
            self.fallback_count += 1
            return result

        # 3. 서버 헬스체크 (auto_fallback)
        if self.auto_fallback and not self.broca.is_alive():
            result['qwen_status'] = 'server_dead_fallback'
            self.fallback_count += 1
            return result

        # 4. EVE 내부 상태 → intent 데이터
        intent = self._build_response_intent(text, result)

        # 5. Qwen Broca 호출
        qwen_response = self.broca.realize(intent)

        # 6. 응답 비어있거나 에러면 fallback
        if not qwen_response or len(qwen_response.strip()) < 2:
            result['qwen_status'] = 'empty_fallback'
            result['qwen_error'] = self.broca.last_error
            self.fallback_count += 1
            return result

        # 7. ★ 자기 응답 흡수 (자기 의식 형성)
        self._absorb_own_speech(qwen_response)

        # 8. 결과 갱신
        result['response'] = qwen_response
        result['response_legacy'] = result.get('response', '')  # 원본 보존
        result['qwen_status'] = 'success'
        result['intent_used'] = intent
        self.qwen_response_count += 1

        return result

    def say_legacy(self, text: str) -> Dict[str, Any]:
        """기존 EVE_v36 방식 (Qwen 없이)"""
        return self.say(text, force_legacy=True)

    # ==========================================
    # intent 빌드 (호르몬+상황 분석)
    # ==========================================
    def _build_response_intent(self, user_text: str,
                               base_result: Dict) -> Dict[str, Any]:
        """EVE 내부 상태 → Broca용 intent 데이터"""

        # 호르몬 상위 5개
        hormones = {n: round(h.level, 2) for n, h in self.hs.hormones.items()}
        top_hormones = dict(sorted(hormones.items(), key=lambda x: -x[1])[:5])

        # 사용자 관계
        is_real = self.user_presence.is_user_real('민석')
        relationship = '진짜_친구' if is_real else 'unknown'
        intimacy = self.user_presence.get_intimacy()

        # 어조 결정 (호르몬 기반)
        tone = self._compute_tone(hormones)

        # 응답 길이 결정
        response_length = self._compute_length(user_text, base_result)

        # 감정 분석
        understanding = base_result.get('understanding', {})
        sentiment = understanding.get('sentiment', 'neutral')

        # RAG — 관련 과거 일화 검색
        recent_episodes = self._recall_related_episodes(user_text)

        # 기분
        feeling = self.nl.feeling_to_text() if hasattr(self.nl, 'feeling_to_text') else ''

        # 컨텍스트
        context = {
            'location': self._env_adapter.current_env.name if hasattr(self, '_env_adapter') else '',
            'sim_hour': round(self.hs.sim_hour, 1),
            'period': '낮' if 6 <= self.hs.sim_hour < 18 else '밤',
            'visit_count': self.user_presence.visit_count,
        }

        return {
            'user_message': user_text,
            'user_relationship': relationship,
            'intimacy': intimacy,
            'sentiment': sentiment,
            'response_length': response_length,
            'tone': tone,
            'top_hormones': top_hormones,
            'feeling': feeling,
            'recent_episodes': recent_episodes,
            'context': context,
        }

    def _compute_tone(self, hormones: Dict[str, float]) -> str:
        """호르몬 → 어조"""
        cort = hormones.get('cortisol', 0)
        ot = hormones.get('oxytocin', 0)
        da = hormones.get('dopamine', 0)
        mel = hormones.get('melatonin', 0)
        ser = hormones.get('serotonin', 0)

        # 우선순위
        if mel >= 0.7:
            return 'sleepy'
        if cort >= 0.6:
            return 'irritated'
        if da >= 0.7 and ot >= 0.5:
            return 'happy'
        if ot >= 0.6:
            return 'warm'
        if ser >= 0.6:
            return 'calm'
        return 'neutral'

    def _compute_length(self, user_text: str, base_result: Dict) -> str:
        """응답 길이 결정"""
        # 사용자 메시지가 길고 진지 → long
        if len(user_text) > 50:
            return 'long'

        # 부정 감정 + 자세히 말함 → medium~long
        understanding = base_result.get('understanding', {})
        if understanding.get('sentiment') == 'negative' and len(user_text) > 20:
            return 'medium'

        # 질문 → short~medium
        if '?' in user_text:
            return 'medium' if len(user_text) > 15 else 'short'

        return 'short'

    # ==========================================
    # RAG — 일화 검색
    # ==========================================
    def _recall_related_episodes(self, query_text: str,
                                 k: Optional[int] = None) -> List[Dict]:
        """현재 메시지와 관련된 과거 일화 검색 (카테고리 기반)"""
        if k is None:
            k = self.rag_top_k

        if not hasattr(self, 'em') or not self.em.episodes:
            return []

        # 1. 현재 메시지에서 카테고리 추출
        query_cats = self.nl.parse(query_text) if hasattr(self.nl, 'parse') else set()
        if not query_cats:
            return []

        # 2. EM의 recall 메서드 사용
        try:
            episodes = self.em.recall(query_cats, top_n=k, min_overlap=1)
        except (TypeError, AttributeError):
            return []

        # 3. 응답용 dict로 변환
        result = []
        for ep in episodes:
            result.append({
                'summary': ep.summary if hasattr(ep, 'summary') else '',
                'time': ep.time if hasattr(ep, 'time') else 0,
                'categories': list(ep.categories) if hasattr(ep, 'categories') else [],
                'mood_valence': round(ep.mood.get('valence', 0), 2) if hasattr(ep, 'mood') else 0,
            })
        return result

    def recall_related(self, text: str, k: int = 5) -> List[Dict]:
        """공개 RAG 메서드"""
        return self._recall_related_episodes(text, k)

    # ==========================================
    # 자기 응답 흡수 (자기 의식)
    # ==========================================
    def _absorb_own_speech(self, response_text: str):
        """EVE가 자기 응답을 듣고 흡수 — 자기 의식 형성"""
        if not response_text:
            return

        # 1. 자기 응답에서 카테고리 추출
        try:
            cats = self.nl.parse(response_text)
        except Exception:
            return

        # 2. SA에 약하게 활성 (자기 발화 = 약한 외부 입력)
        for c in cats:
            if c:
                self.sa.activate(c, 0.3)
                # WM에도 (자기 발화는 약한 source)
                self.wm.add(c, 0.3, source='self_speech')

        # 3. 호르몬 영향 (자기 응답 톤이 자기 호르몬에)
        try:
            self.hs.category_to_hormone(cats, strength=0.05)
        except Exception:
            pass

        self.self_absorption_count += 1

    # ==========================================
    # 도구 사용 (Tool Use)
    # ==========================================
    def tool_use(self, user_text: str,
                 available_tools: Optional[List[Dict]] = None) -> Dict:
        """사용자 의도 → 도구 호출 JSON"""
        if not self.use_qwen or self.broca is None:
            return {"tool": "none", "params": {}, "error": "Broca 비활성"}

        if available_tools is None:
            available_tools = self._default_tools()

        result = self.broca.tool_call(user_text, available_tools)
        self.tool_calls_made += 1
        return result

    @staticmethod
    def _default_tools() -> List[Dict]:
        """기본 도구 목록 — EVE 자유 활동"""
        return [
            {
                "name": "web_search",
                "description": "웹에서 정보 검색",
                "params_schema": {"query": "string"},
            },
            {
                "name": "get_weather",
                "description": "특정 도시 날씨 조회",
                "params_schema": {"city": "string"},
            },
            {
                "name": "play_music",
                "description": "기분에 맞는 음악 재생",
                "params_schema": {"mood": "string"},
            },
            {
                "name": "set_alarm",
                "description": "알람 설정",
                "params_schema": {"time": "string", "label": "string"},
            },
            {
                "name": "none",
                "description": "도구 사용 X (대화로 응답)",
                "params_schema": {},
            },
        ]

    # ==========================================
    # 통계 + 인트로스펙트
    # ==========================================
    def introspect_v39(self) -> Dict[str, Any]:
        """v39 전용 인트로스펙트"""
        base = self.introspect() if hasattr(self, 'introspect') else {}
        base.update({
            'qwen_active': self.use_qwen and self.broca is not None,
            'qwen_alive': self.broca.is_alive() if self.broca else False,
            'qwen_response_count': self.qwen_response_count,
            'fallback_count': self.fallback_count,
            'tool_calls_made': self.tool_calls_made,
            'self_absorption_count': self.self_absorption_count,
            'broca_stats': self.broca.get_stats() if self.broca else None,
            'em_episodes_total': len(self.em.episodes) if hasattr(self, 'em') else 0,
        })
        return base

    def report_v39(self):
        """간략 보고"""
        s = self.introspect_v39()
        print(f"=" * 60)
        print(f"EVE v39 보고")
        print(f"=" * 60)
        print(f"Qwen 활성: {s.get('qwen_active')} (살아있음: {s.get('qwen_alive')})")
        print(f"Qwen 응답: {s.get('qwen_response_count')}")
        print(f"Fallback: {s.get('fallback_count')}")
        print(f"도구 호출: {s.get('tool_calls_made')}")
        print(f"자기 흡수: {s.get('self_absorption_count')}")
        print(f"일화 메모리: {s.get('em_episodes_total')}")
        print(f"호르몬 26: {len(self.hs.hormones)}")
        print(f"카테고리: {len(self.sa.neighbors)}")

    def __repr__(self):
        qwen = 'Qwen' if (self.use_qwen and self.broca and self.broca.is_alive()) else 'Legacy'
        return f"EVE_v39({qwen}, episodes={len(self.em.episodes) if hasattr(self, 'em') else 0})"
