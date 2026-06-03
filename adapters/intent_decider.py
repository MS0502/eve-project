"""
IntentDecider — 라운드 67

GPT 짚은 판단 레이어.
- 입력 → 판단 → 표현 구조
- 3개 intent: ack / empathy / interest

EVE 원칙 준수:
- random.random() 사용 X (결정론)
- 대신 호르몬 상태로 다양성 (Damasio Somatic Marker)
- 같은 입력 + 같은 호르몬 = 같은 응답 (결정론)
- 호르몬 변하면 자연스레 다양

학계: Damasio Somatic Marker Hypothesis (1994)
- 신체 상태가 결정의 가중치 변경
- "이성 + 감정"의 통합 (감정 X 결정 X)
"""

from typing import Optional


# 감정 단어 — 최소만 (확장 X)
EMOTION_WORDS = ["힘들", "피곤", "슬프", "짜증", "우울", "지쳐", "외롭", "괴로"]
POSITIVE_WORDS = ["좋아", "행복", "기뻐", "재밌", "신나", "뿌듯"]


class IntentDecider:
    """
    입력 + 호르몬 → intent (ack / empathy / interest)
    
    호르몬 상태가 의도 *기본 성향* 결정:
    - 외로움 (oxytocin↓) → empathy 우선
    - 호기심 (dopamine↑) → interest 우선
    - 피곤 (melatonin↑) → ack 우선
    """

    def __init__(self, engine=None):
        self.engine = engine
        # 통계
        self.decisions = {"ack": 0, "empathy": 0, "interest": 0}
        # 라운드 71: 맥락 유지 1턴만 (GPT 짚은 핵심)
        # last_intent: 직전 응답 intent
        # last_text: 직전 사용자 입력 (반복 감지)
        self.last_intent: Optional[str] = None
        self.last_text: Optional[str] = None

    def decide(self, text: str, ctx: Optional[dict] = None) -> str:
        """
        text + ctx → intent string.
        결정론 (같은 text + 같은 호르몬 = 같은 결과).
        라운드 71: last_intent 활용 (맥락 유지).
        """
        text = (text or "").strip()
        if not text:
            return "ack"
        
        # 라운드 71: 질문 감지 — "?" 끝나면 *상대*에 대한 질문
        # 감정 단어 있어도 → 내 감정 공감 X (질문 답해야)
        is_question = text.endswith("?") or text.endswith("？")
        # "너", "이브" 들어간 질문 = EVE한테 묻는 거
        asks_eve = is_question and any(w in text for w in ["너", "이브", "EVE", "당신"])
        
        # 라운드 71: 같은 입력 반복 감지
        is_repeat = (text == self.last_text) and self.last_intent is not None
        
        # 점수 초기화
        scores = {"ack": 0.3, "empathy": 0.3, "interest": 0.3}
        
        # 라운드 71: 질문은 *interest* (대화 이어가기)
        # EVE한테 묻는 질문 → interest 큰 가중치
        if asks_eve:
            scores["interest"] += 0.7    # empathy 점수보다 크게
        elif is_question:
            scores["interest"] += 0.4
        
        # 1. 감정 단어 → empathy (단, 질문이면 감정 공감 X)
        if not is_question:
            if any(w in text for w in EMOTION_WORDS):
                scores["empathy"] += 0.5
            if any(w in text for w in POSITIVE_WORDS):
                scores["empathy"] += 0.3
        
        # 2. 행동/활동 공유 → interest (대화 이어가기)
        from utils.korean_endings import is_action_word
        entities = (ctx or {}).get("entities", []) if ctx else []
        for ent in entities:
            if is_action_word(ent):
                scores["interest"] += 0.4
                break
        
        # 3. 짧은 문장 → ack (한 단어 답)
        if len(text) < 6 and not is_question:
            scores["ack"] += 0.3
        
        # 4. 호르몬 상태 — 결정론 다양성 (random 대신)
        if self.engine is not None:
            try:
                h = self.engine.hormone_adapter.hs.hormones
                # dopamine 높음 → 호기심 (interest)
                da = h.get("dopamine")
                if da and da.level > 0.6:
                    scores["interest"] += 0.2
                # oxytocin 낮음 (외로움) → empathy
                ot = h.get("oxytocin")
                if ot and ot.level < 0.35:
                    scores["empathy"] += 0.2
                # melatonin 높음 (피곤) → ack
                mel = h.get("melatonin")
                if mel and mel.level > 0.55:
                    scores["ack"] += 0.2
            except Exception:
                pass
        
        # 5. 라운드 68: ContextSeed — 결정론 다양성 (turn + history 포함)
        # 이전: seed_offset = (text % 7) * 0.02 (text만 의존)
        # 지금: ContextSeed (text + hormone + turn + history)
        if self.engine is not None and hasattr(self.engine, "context_seed"):
            cs = self.engine.context_seed.compute(text, ctx)
            offset = (cs % 7) * 0.02
            scores["ack"] += offset
            scores["empathy"] += offset * 0.5
        else:
            seed_offset = (sum(ord(c) for c in text) % 7) * 0.02
            scores["ack"] += seed_offset
            scores["empathy"] += seed_offset * 0.5
        
        # 최고 점수 선택
        best = max(scores, key=scores.get)
        self.decisions[best] += 1
        
        # 라운드 71: last_intent / last_text 기록 (맥락 유지)
        # 반복 정보는 ctx에 전달 (generate_response에서 사용)
        if ctx is not None:
            ctx["is_repeat"] = is_repeat
            ctx["asks_eve"] = asks_eve
        self.last_intent = best
        self.last_text = text
        
        return best

    def stats(self) -> dict:
        return dict(self.decisions)
    
    def generate_response(self, intent: str, text: str, 
                          ctx: Optional[dict] = None) -> Optional[str]:
        """
        intent → 응답.
        - ack: minimal 인정
        - empathy: reaction_for (이미 호르몬별 다양화 됨)
        - interest: 동적 질문 (entities 기반, 키워드 분기 X)
        """
        if intent == "ack":
            # 라운드 68: ContextSeed
            if self.engine and hasattr(self.engine, "context_seed"):
                seed = self.engine.context_seed.compute(text, ctx)
            else:
                seed = sum(ord(c) for c in text) if text else 0
            ack_pool = ["응", "음", "그래", "아 그렇구나", "그렇네"]
            return ack_pool[seed % len(ack_pool)]
        
        if intent == "empathy":
            # 호르몬별 공감 — SpeechHub 사용
            if self.engine and hasattr(self.engine, "speech_hub"):
                # 부정 vs 긍정 분기
                is_pos = any(w in text for w in POSITIVE_WORDS)
                kind = "empathy_positive" if is_pos else "empathy_negative"
                # 라운드 68: ContextSeed
                if hasattr(self.engine, "context_seed"):
                    seed = self.engine.context_seed.compute(text, ctx)
                else:
                    seed = sum(ord(c) for c in text) if text else 0
                result = self.engine.speech_hub.generate(kind, ctx or {}, seed=seed)
                return result["core"]
            return "그랬구나"
        
        if intent == "interest":
            ctx = ctx or {}
            is_repeat = ctx.get("is_repeat", False)
            asks_eve = ctx.get("asks_eve", False)
            
            # 라운드 71: 같은 질문 반복 → "왜 궁금해?" 같은 다른 응답
            if is_repeat:
                seed = sum(ord(c) for c in text)
                repeat_pool = [
                    "왜 궁금해?",
                    "음, 그게 왜?",
                    "또 물어보네. 무슨 일이야?",
                    "응? 같은 거 또?",
                ]
                return repeat_pool[seed % len(repeat_pool)]
            
            # 라운드 71: 원본 텍스트에서 *직접* 동사 어절 찾기
            from utils.korean_endings import is_action_word, reaction_for, _is_safe_input
            
            words = (text or "").split()
            for w in words:
                if not _is_safe_input(w):
                    continue
                if is_action_word(w):
                    base = reaction_for(w)
                    if base is None:
                        continue
                    return f"{base}. 어땠어?"
            
            # 라운드 71: 질문인데 행동 entity 없음 — 이어가기 질문
            if asks_eve:
                # "너는 X?" → 다양한 답
                seed = sum(ord(c) for c in text)
                eve_q_pool = [
                    "음, 글쎄.",
                    "어, 모르겠어.",
                    "음... 너는?",
                ]
                return eve_q_pool[seed % len(eve_q_pool)]
            
            return "더 얘기해줘"
        
        return None
    
    @staticmethod
    def _extract_stem(word: str) -> Optional[str]:
        """동사 과거형 → 어간 ('운동했어' → '운동')."""
        from utils.korean_endings import PAST_VERB_ENDINGS
        for ending in sorted(PAST_VERB_ENDINGS.keys(), key=len, reverse=True):
            if word.endswith(ending):
                stem = word[:-len(ending)]
                # 너무 짧으면 X
                if len(stem) >= 1:
                    return stem
        return None
