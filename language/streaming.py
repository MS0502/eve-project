"""
Streaming Engine — 전체 파이프라인 통합 진입점.

eve2.md 4. 실행 파이프라인:
    text -> understand -> graph -> workspace
         -> System1 후보 -> System2 필터
         -> Plan -> 점진적 generate
         -> learn

이 파일이 v41의 '척추'다. LLM 호출 0회, random 0회.
"""

import time
from typing import Iterator

from utils.types import Experience
from language.understanding import LanguageUnderstanding
from language.planner import ResponsePlanner
from language.generator import LanguageGenerator
from cognition.meaning_graph import build_graph
from core.workspace import GlobalWorkspace
from core.system1 import System1
from core.system2 import System2


# 청크 사이 기본 sleep — 외부에서 0으로 끄면 테스트 빨라짐
DEFAULT_DELAYS = {
    "stage1": 0.6,
    "core":   1.0,
    "expand": 1.0,
    "final":  0.8,
}


# v3 round39: language/streaming fastText migration is observation-only.
STREAMING_FASTTEXT_TRACE_CAP = 1000
STREAMING_FASTTEXT_OBSERVATION_LAYER = "streaming"
STREAMING_GENERATION_EMBEDDING_SOURCE = "wrapper"
STREAMING_OPERATION_STREAM_CHUNK = "stream_chunk"


class StreamingEngine:
    """모듈 인스턴스 보관 + 학습용 경험 누적."""

    def __init__(self, delays: dict | None = None, sleep_fn=time.sleep,
                 hormone_adapter=None,
                 activation_adapter=None,
                 memory_adapter=None,
                 nl_adapter=None,
                 sd_adapter=None,
                 dmn_adapter=None,
                 vsa_adapter=None,
                 ai_adapter=None,
                 goal_adapter=None,
                 norm_adapter=None,
                 continual_adapter=None,
                 allostatic_adapter=None,
                 env_adapter=None,
                 autonomy_adapter=None,
                 counterfactual_adapter=None,
                 analogy_adapter=None,
                 temporal_adapter=None,
                 humor_adapter=None,
                 suffering_adapter=None,
                 narrative_adapter=None,
                 creative_adapter=None,
                 metacognition_adapter=None,
                 emotion_regulation_adapter=None,
                 multi_stream_adapter=None,
                 deep_reasoning_adapter=None,
                 reasoning_loop=None,
                 world_model_adapter=None,
                 simulation_engine=None,
                 agency_adapter=None,
                 autonomous_loop=None,
                 apc_adapter=None,
                 corpus_adapter=None,
                 frame_adapter=None,
                 hypergraph_adapter=None,
                 semantic_distance_adapter=None,
                 digital_somatic_adapter=None,
                 user_presence_adapter=None,
                 integrated_self_adapter=None,
                 enhancer_adapter=None,
                 proactive_adapter=None,
                 sensory_adapter=None,
                 curiosity_adapter=None,
                 task_solver=None):
        self.lu = LanguageUnderstanding(nl_adapter=nl_adapter)
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.memory_adapter = memory_adapter
        self.nl_adapter = nl_adapter
        self.sd_adapter = sd_adapter
        self.dmn_adapter = dmn_adapter
        self.vsa_adapter = vsa_adapter
        self.ai_adapter = ai_adapter
        self.goal_adapter = goal_adapter
        self.norm_adapter = norm_adapter
        self.continual_adapter = continual_adapter
        self.allostatic_adapter = allostatic_adapter
        self.env_adapter = env_adapter
        self.autonomy_adapter = autonomy_adapter
        self.counterfactual_adapter = counterfactual_adapter
        self.analogy_adapter = analogy_adapter
        self.temporal_adapter = temporal_adapter
        self.humor_adapter = humor_adapter
        self.suffering_adapter = suffering_adapter
        self.narrative_adapter = narrative_adapter
        self.creative_adapter = creative_adapter
        self.metacognition_adapter = metacognition_adapter
        self.emotion_regulation_adapter = emotion_regulation_adapter
        self.multi_stream_adapter = multi_stream_adapter
        self.deep_reasoning_adapter = deep_reasoning_adapter
        self.reasoning_loop = reasoning_loop
        self.world_model_adapter = world_model_adapter
        self.simulation_engine = simulation_engine
        self.agency_adapter = agency_adapter
        self.autonomous_loop = autonomous_loop
        self.apc_adapter = apc_adapter
        self.corpus_adapter = corpus_adapter
        self.frame_adapter = frame_adapter
        self.hypergraph_adapter = hypergraph_adapter
        self.semantic_distance_adapter = semantic_distance_adapter
        self.digital_somatic_adapter = digital_somatic_adapter
        self.user_presence_adapter = user_presence_adapter
        self.integrated_self_adapter = integrated_self_adapter
        self.enhancer_adapter = enhancer_adapter
        self.proactive_adapter = proactive_adapter
        self.sensory_adapter = sensory_adapter
        self.curiosity_adapter = curiosity_adapter
        # task_solver는 default로 만들어서 산술 같은 task 자동 처리
        if task_solver is not None:
            self.task_solver = task_solver
        else:
            from learning.code_synthesis import TaskSolver
            self.task_solver = TaskSolver()

        self.workspace = GlobalWorkspace(hormone_adapter=hormone_adapter)
        self.s1 = System1(activation_adapter=activation_adapter, vsa_adapter=vsa_adapter)
        self.s2 = System2(sd_adapter=sd_adapter)
        self.planner = ResponsePlanner(
            hormone_adapter=hormone_adapter,
            memory_adapter=memory_adapter,
            goal_adapter=goal_adapter,
            task_solver=self.task_solver,
            env_adapter=env_adapter,
            counterfactual_adapter=counterfactual_adapter,
            temporal_adapter=temporal_adapter,
            humor_adapter=humor_adapter,
            suffering_adapter=suffering_adapter,
            reasoning_loop=reasoning_loop,
            world_model_adapter=world_model_adapter,
            simulation_engine=simulation_engine,
            digital_somatic_adapter=digital_somatic_adapter,
            enhancer_adapter=enhancer_adapter,
            nl_adapter=nl_adapter,
        )
        self.generator = LanguageGenerator()

        self.delays = delays if delays is not None else DEFAULT_DELAYS
        self._sleep = sleep_fn
        self.history: list[Experience] = []

        # v3 round39: fastText parallel observation only.
        # This must never affect chunk text, order, timing shape, or
        # engine.self_embedding ownership.
        self.fasttext_trace: list[dict] = []
        self.fasttext_observation_count = 0
        self.fasttext_trace_cap = STREAMING_FASTTEXT_TRACE_CAP
        self.fasttext_operation_counts = {
            STREAMING_OPERATION_STREAM_CHUNK: 0,
        }

    def chat_stream(self, input_text: str) -> Iterator[str]:
        # 0) 입력 시간 기록 (autonomy)
        if self.autonomy_adapter is not None:
            self.autonomy_adapter.mark_input()

        # 0.05) 라운드 39a: emotion_cause에 사용자 입력 기록
        engine_ref = getattr(self.planner, "_engine_ref", None)
        if engine_ref is not None and hasattr(engine_ref, "emotion_cause"):
            engine_ref.emotion_cause.record_user_input(input_text)
        
        # 라운드 68: turn_count 증가 (변동성 위해)
        if engine_ref is not None and hasattr(engine_ref, "context_seed"):
            engine_ref.context_seed.increment_turn()
        
        # 라운드 71: 같은 입력 반복 감지 (GPT 짚은 맥락 유지)
        # 직전 입력과 정확히 같으면 → variation 응답 (시스템 분기 우회)
        normalized = (input_text or "").strip()
        last_input = getattr(self, "_last_input", None)
        if normalized and last_input == normalized:
            seed = sum(ord(c) for c in normalized)
            repeat_pool = [
                "또 물어보네. 왜 궁금해?",
                "음, 같은 거 또?",
                "응? 무슨 일이야?",
                "어, 또 그거?",
            ]
            yield repeat_pool[seed % len(repeat_pool)]
            self._last_input = normalized
            return
        self._last_input = normalized

        # 0.06) 라운드 52: 사용자 메타 지시 파싱
        # "응 듣고있어 빼고 말해", "짧게 답해", "그냥 안녕이라고만 해봐"
        if engine_ref is not None and hasattr(engine_ref, "user_instruction"):
            ui = engine_ref.user_instruction
            ui.tick()  # duration 감소
            instr = ui.parse_instruction(input_text)
            if instr is not None:
                ui.add_constraint(instr)
                # 라운드 54: 지시 시 자율 발화 cooldown (잠시 조용)
                if self.autonomy_adapter is not None:
                    self.autonomy_adapter.mark_proactive()
                # 인정 응답 (지시 받았다는 표시)
                yield ui.acknowledge_response(instr)
                return  # 다른 처리 X — 지시는 단독 처리

        # 0.07) 라운드 54: EVE 직전 질문 → 사용자 답 인식
        # 예: EVE "인사 처음 들어봐. 설명해주면 기억할게." (질문 패턴)
        #     사용자 "처음 만났을때 하는거야" (답)
        #     → ConceptMemory.learn(인사, 처음 만났을때 하는거야)
        if engine_ref is not None and hasattr(engine_ref, "dialogue_context"):
            dc = engine_ref.dialogue_context
            answer = dc.receive_answer(input_text)
            if answer is not None:
                target, definition = answer
                # ConceptMemory에 학습
                if hasattr(engine_ref, "concept_memory") and engine_ref.concept_memory is not None:
                    try:
                        engine_ref.concept_memory.learn(target, definition, source="dialogue_answer")
                    except Exception:
                        pass
                # 인정 응답
                yield f"아, 그게 {target}구나. 알았어, 기억할게."
                return

        # 0.1) 사용자 방문 인식 (UserPresence — 김민석 = 진짜 친구)
        if self.user_presence_adapter is not None:
            self.user_presence_adapter.on_user_arrive()

        meaning = self.lu.parse(input_text)

        # 라운드 48: 어텐션 통합 (Layer 1) — 분산된 spike를 한 곳에서
        # 분석 결과를 meaning.context에 저장 (planner/ThoughtChain이 활용)
        engine_ref = getattr(self.planner, "_engine_ref", None)
        attention_result = None
        if engine_ref is not None and hasattr(engine_ref, "attention"):
            try:
                emotions = meaning.emotions or {}
                attention_result = engine_ref.attention.analyze(
                    text=input_text,
                    entities=meaning.entities or [],
                    emotions=emotions,
                )
                # meaning.context에 저장 (다른 모듈이 활용)
                if not hasattr(meaning, "context") or meaning.context is None:
                    meaning.context = {}
                meaning.context["attention"] = {
                    "scores": attention_result.scores,
                    "top": [e for e, _ in attention_result.top_entities[:3]],
                    "primary": attention_result.primary_node,
                    "emotion": attention_result.dominant_emotion,
                    "urgency": attention_result.urgency,
                }
                
                # 라운드 49: 어텐션 top 2 → Compositor 합성 의미
                if hasattr(engine_ref, "compositor"):
                    top_ents = [e for e, _ in attention_result.top_entities[:2]]
                    if len(top_ents) >= 2:
                        try:
                            comp = engine_ref.compositor.compose(top_ents[0], top_ents[1])
                            if comp is not None:
                                meaning.context["composition"] = {
                                    "inputs": list(comp.inputs),
                                    "concepts": comp.concepts,
                                    "method": comp.method,
                                    "confidence": comp.confidence,
                                    "phrase": engine_ref.compositor.composition_phrase(comp),
                                }
                        except Exception:
                            pass
            except Exception:
                pass

        # 라운드 47: SelfEmbedding observe — EVE 자체 의미 학습
        if engine_ref is not None and hasattr(engine_ref, "self_embedding"):
            try:
                engine_ref.self_embedding.observe(input_text)
            except Exception:
                pass

        # Round58: EVE-specific continuous lexical observation. Observation is
        # allowed during runtime, but vector promotion is still explicit and
        # handled by EveSelfLearningAdapter.commit_eve_specific_vectors().
        if engine_ref is not None and hasattr(engine_ref, "eve_self_learning"):
            try:
                engine_ref.eve_self_learning.observe_text(
                    input_text,
                    source="chat_stream_user_input",
                    context={
                        "intent": getattr(meaning, "intent", None),
                        "entities": list(getattr(meaning, "entities", []) or []),
                    },
                )
            except Exception:
                pass

        # 라운드 40 + 48: Salience spike — 어텐션 점수 기반 (통합)
        if engine_ref is not None and hasattr(engine_ref, "salience_adapter"):
            sal = engine_ref.salience_adapter
            if sal is not None and attention_result is not None:
                # 어텐션 점수 → salience boost (점수 강할수록 더 spike)
                for ent, score in attention_result.scores.items():
                    if score > 0.05:
                        # 0~1 점수 → spike 강도
                        is_emotional = (score > 0.4)  # 강한 점수면 emotional spike
                        sal.spike(ent, emotional=is_emotional)

        # Orchestrator 분류 + 컨텍스트 (라운드 29 B-1)
        # teaching 처리 전에 분류하여 모든 입력이 통계에 잡히도록.
        # B-1은 기록만 — 응답 흐름은 안 바꿈. B-2에서 응답에 활용.
        orchestrator = getattr(self, "orchestrator_adapter", None)
        if orchestrator is not None:
            try:
                orch_result = orchestrator.process(meaning)
                if not hasattr(meaning, "context") or meaning.context is None:
                    meaning.context = {}
                meaning.context["orchestrator"] = orch_result
            except Exception:
                pass

        # 0.15) 라운드 43: 정의 가르침 → ConceptMemory 저장
        # teaching_adapter보다 먼저 — 정의 패턴 ("X는 Y야") 우선 처리
        # teaching_adapter는 "X라고 말해" 같은 *반응* 학습 담당
        if engine_ref is not None and hasattr(engine_ref, "concept_memory"):
            from adapters.concept_memory_adapter import ConceptMemoryAdapter
            pair = ConceptMemoryAdapter.extract_definition(input_text)
            if pair is not None:
                key, definition = pair
                # 일반 대명사/짧은 거 제외 (이미 extract에서 일부 했지만 중복 안전)
                if key not in ("나", "내", "너", "니", "그", "이", "저") and len(key) >= 1:
                    learned = engine_ref.concept_memory.learn(key, definition, source="user")
                    if learned is not None:
                        from utils.korean_particles import topic_particle, quote_ending
                        from utils.speech_style import get_emotional_state, pick_learn_template
                        
                        # 동사 활용형 처리
                        verb_endings = ('는', '한', '온', '간', '쓴', '된')
                        if definition.endswith(verb_endings) or definition.endswith('하'):
                            def_for_speech = f"{definition} 거"
                        else:
                            def_for_speech = definition
                        
                        # 호르몬 상태 + 결정론
                        state = get_emotional_state(engine_ref)
                        seed = sum(ord(c) for c in input_text)
                        s1, core_t, fin = pick_learn_template(state, seed)
                        
                        # 변수 치환
                        key_t = topic_particle(key)
                        last_char = def_for_speech[-1] if def_for_speech else ""
                        is_ja_eum = last_char and 0xAC00 <= ord(last_char) <= 0xD7A3 and (ord(last_char) - 0xAC00) % 28 != 0
                        quote_end = "이라는" if is_ja_eum else "라는"
                        
                        yield s1
                        yield core_t.format(key_t=key_t, def_speech=def_for_speech, quote_end=quote_end)
                        yield fin
                        return

        # 0.2) 가르침 인식 + 평가 (라운드 27 + 28)
        teaching_adapter = getattr(self, "teaching_adapter", None)
        if teaching_adapter is not None:
            tr = teaching_adapter.process_teaching(input_text)
            if tr is not None:
                # 가르침 → 평가에 따라 다른 반응
                yield tr["response"]
                # accept면 이전 응답 약화 (수정으로 간주)
                if tr["decision"] == "accept":
                    teaching_adapter.weaken_last()
                return
            # 가르침 아니면 강화
            teaching_adapter.reinforce_if_no_correction()

        # 0.3) 시간 간격 인사 (라운드 27) — 인사 의도일 때만
        if self.user_presence_adapter is not None:
            try:
                gap_phrase = self.user_presence_adapter.gap_phrase()
                # 인사 키워드 있고 간격 있으면 인사에 반영
                greeting_keywords = ("안녕", "안뇽", "하이", "왔어", "오랜만")
                if gap_phrase and any(k in input_text for k in greeting_keywords):
                    yield f"{gap_phrase}이네."
            except Exception:
                pass

        # 1) 의미 (위에서 이미 parse됨, 재사용)

        # 1.2) 환경 컨텍스트 박음
        if self.env_adapter is not None:
            self.env_adapter.enrich_meaning(meaning)

        # 1.3) counterfactual / temporal 패턴 감지
        if self.counterfactual_adapter is not None:
            self.counterfactual_adapter.enrich_meaning(meaning)
        if self.temporal_adapter is not None:
            self.temporal_adapter.enrich_meaning(meaning)
        if self.world_model_adapter is not None:
            self.world_model_adapter.enrich_meaning(meaning)
        if self.frame_adapter is not None:
            self.frame_adapter.enrich_meaning(meaning)

        # 1.5) 활성/작업기억으로 흘려보내기 (어댑터 있을 때)
        if self.activation_adapter is not None:
            self.activation_adapter.ingest(meaning)

        # 1.6) 목표/규범 어댑터에 의미 흘림 (학습)
        if self.goal_adapter is not None:
            self.goal_adapter.observe_meaning(meaning)
        if self.norm_adapter is not None:
            self.norm_adapter.observe_meaning(meaning)

        # 2) 그래프 + workspace 갱신
        graph = build_graph(meaning)
        self.workspace.update(graph, meaning)
        state = self.workspace.get_state()

        # 3) System1 후보 → System2 검증
        candidates = self.s1.generate_candidates(state)
        candidates = self.s2.filter(candidates)

        # 4) 응답 계획
        plan = self.planner.build(state, candidates)

        # 5+6+7) 생성 + 학습 (공통)
        yield from self._stream_from_plan(plan, meaning, input_text)

    def proactive_stream(self, force: bool = True) -> Iterator[str]:
        """
        DMN 자발 활성으로 응답 생성. 외부 입력 없을 때 호출.
        force=True면 무조건 한 번 발화. False면 조건 충족할 때만 (DMN.tick 결과).

        v41 라운드 19: ProactiveAdapter (NEED→fragment 자율 선택) 우선 시도,
        실패 시 DMN 단순 mind_wandering fallback.
        """
        # ProactiveAdapter 우선 (NEED 기반 자연스러운 발화)
        if self.proactive_adapter is not None:
            msg = None
            if force:
                msg = self.proactive_adapter.message()
            else:
                speak, _ = self.proactive_adapter.should_speak()
                if speak:
                    msg = self.proactive_adapter.message()
            if msg and isinstance(msg, dict):
                text = msg.get("message") or msg.get("text") or ""
                if text:
                    if self.autonomy_adapter is not None:
                        self.autonomy_adapter.mark_proactive()
                    # 짧게 한 줄 + final
                    yield self._stage1_word("warm")
                    self._sleep(self.delays.get("stage1", 0))
                    yield text + ("." if not text.endswith(("?", "!", ".")) else "")
                    self._sleep(self.delays.get("core", 0))
                    return

        # Fallback: DMN
        if self.dmn_adapter is None:
            return
        meaning = self.dmn_adapter.try_spontaneous_meaning(force=force)
        if meaning is None:
            return

        # autonomy 기록
        if self.autonomy_adapter is not None:
            self.autonomy_adapter.mark_proactive()

        # workspace 갱신 (focus 노출)
        graph = build_graph(meaning)
        self.workspace.update(graph, meaning)
        state = self.workspace.get_state()

        # proactive intent니 후보는 만들지 않고 plan만 (planner가 분기)
        plan = self.planner.build(state, candidates=[])
        # 자율 발화 = 다양한 finalize (혼잣말 느낌 — 같은 거 매번 X)
        plan.is_proactive = True

        yield from self._stream_from_plan(plan, meaning, meaning.raw_text)

    def _stage1_word(self, tone: str) -> str:
        from language.generator import STAGE1_BY_TONE
        base = STAGE1_BY_TONE.get(tone, STAGE1_BY_TONE["neutral"])
        if self.hormone_adapter is not None:
            return self.hormone_adapter.stage1_word(base)
        return base

    def _apply_constraints(self, text: str) -> str:
        """라운드 52+59: yield 시 사용자 constraint + PostProcessor 적용."""
        engine_ref = getattr(self.planner, "_engine_ref", None)
        if engine_ref is None:
            return text
        # 1. 사용자 constraint
        if hasattr(engine_ref, "user_instruction"):
            try:
                text = engine_ref.user_instruction.apply_constraints(text)
            except Exception:
                pass
        # 2. 라운드 59: SpeechHub PostProcessor (forbidden + 중복 정리)
        if hasattr(engine_ref, "speech_hub"):
            try:
                text = engine_ref.speech_hub.post_process(text)
            except Exception:
                pass
        return text

    def _stream_from_plan(self, plan, meaning, input_text: str) -> Iterator[str]:
        """plan → 점진적 yield + EM 인코딩 + 학습 기록 (공통 백엔드)."""
        # AI predict (응답 전)
        if self.ai_adapter is not None:
            self.ai_adapter.predict_before_response()

        chunks: list[str] = []

        # stage1 — 자율 발화면 다양한 풀에서 (의미 기반 hash로 결정론)
        if getattr(plan, "is_proactive", False):
            from language.generator import PROACTIVE_STAGE1_POOL
            pool = PROACTIVE_STAGE1_POOL.get(
                plan.emotional_tone, PROACTIVE_STAGE1_POOL["neutral"]
            )
            # core_message hash → 같은 메시지 = 같은 stage1 (결정론)
            seed_str = plan.core_message or plan.emotional_tone
            idx = sum(ord(c) for c in seed_str) % len(pool)
            s1_text = pool[idx]
        else:
            s1_text = self.generator.generate_stage1(plan)
        
        if s1_text:
            s1_text = self._apply_constraints(s1_text)
            if s1_text:
                chunks.append(s1_text)
                self._observe_fasttext_parallel("stage1", meaning, input_text, s1_text, len(chunks) - 1)
                yield s1_text
                self._sleep(self.delays.get("stage1", 0))

        core_text = self.generator.generate_core(plan)
        core_text = self._apply_constraints(core_text)
        if core_text:
            chunks.append(core_text)
            self._observe_fasttext_parallel("core", meaning, input_text, core_text, len(chunks) - 1)
            yield core_text
            self._sleep(self.delays.get("core", 0))

        # 학습된 응답 (라운드 27) — 가르침 받은 거 매칭되면 출력
        teaching_adapter = getattr(self, "teaching_adapter", None)
        if teaching_adapter is not None and self.activation_adapter is not None:
            try:
                active = {
                    cat for cat, lvl in self.activation_adapter.sa.activations.items()
                    if lvl > 0.1
                }
                match = teaching_adapter.find_match(active, min_overlap=1)
                if match is not None:
                    response = self._apply_constraints(match.response)
                    if response:
                        chunks.append(response)
                        self._observe_fasttext_parallel("teaching", meaning, input_text, response, len(chunks) - 1)
                        yield response
                        teaching_adapter.mark_emitted(match)
                        self._sleep(self.delays.get("expand", 0))
            except Exception:
                pass

        for sub in plan.sub_points:
            sub_text = self.generator.expand(sub)
            sub_text = self._apply_constraints(sub_text)
            if sub_text:
                chunks.append(sub_text)
                self._observe_fasttext_parallel("expand", meaning, input_text, sub_text, len(chunks) - 1)
                yield sub_text
                self._sleep(self.delays.get("expand", 0))

        if not getattr(plan, "skip_finalize", False):
            # 자율 발화면 다양한 finalize 풀에서 (의미 기반 hash로 결정론)
            if getattr(plan, "is_proactive", False):
                from language.generator import PROACTIVE_FINALIZE_POOL
                pool = PROACTIVE_FINALIZE_POOL.get(
                    plan.emotional_tone, PROACTIVE_FINALIZE_POOL["neutral"]
                )
                # core_message hash + 1 (stage1과 다른 시드)
                seed_str = (plan.core_message or "") + "_final"
                idx = sum(ord(c) for c in seed_str) % len(pool)
                final_text = pool[idx]
            else:
                final_text = self.generator.finalize(plan)
            
            if final_text:
                final_text = self._apply_constraints(final_text)
                if final_text:
                    chunks.append(final_text)
                    self._observe_fasttext_parallel("final", meaning, input_text, final_text, len(chunks) - 1)
                    yield final_text
                self._sleep(self.delays.get("final", 0))

        full_response = " ".join(chunks)

        # 라운드 54: EVE 응답 기록 (질문 패턴 감지 → PendingQuestion 저장)
        engine_ref_for_dc = getattr(self.planner, "_engine_ref", None)
        if engine_ref_for_dc is not None and hasattr(engine_ref_for_dc, "dialogue_context"):
            try:
                engine_ref_for_dc.dialogue_context.record_eve_response(
                    response=full_response,
                    recent_user_input=input_text,
                )
            except Exception:
                pass

        # 라운드 52: constraint는 *post-process로* 적용 (이미 yield된 거라 *기록*만)
        # 다음 turn에 같은 응답 만들면 그땐 적용됨
        engine_ref_for_ui = getattr(self.planner, "_engine_ref", None)
        if engine_ref_for_ui is not None and hasattr(engine_ref_for_ui, "user_instruction"):
            try:
                ui = engine_ref_for_ui.user_instruction
                for c in ui.constraints:
                    if c.type == "remove_phrase" and c.target in full_response:
                        c.remaining = max(c.remaining, 5)
            except Exception:
                pass

        # 라운드 45: safety_adapter에 응답 기록 (반복 감지용)
        engine_for_safety = getattr(self.planner, "_engine_ref", None)
        if engine_for_safety is not None and hasattr(engine_for_safety, "safety_adapter"):
            try:
                engine_for_safety.safety_adapter.record_response(full_response)
            except Exception:
                pass

        # AI observe (응답 후) — outcome은 단순 추정
        if self.ai_adapter is not None:
            outcome = self._infer_outcome(meaning, full_response)
            self.ai_adapter.observe_after_response(outcome=outcome)

        # Allostatic 학습 (호르몬 패턴)
        if self.allostatic_adapter is not None:
            self.allostatic_adapter.observe(meaning, full_response)

        # APC 자율 학습 (활성 카테고리 예측-관찰)
        if self.apc_adapter is not None:
            self.apc_adapter.observe(meaning)

        # Corpus 자율 학습 (한국어 텍스트 → SA pair)
        if self.corpus_adapter is not None:
            self.corpus_adapter.auto_learn_from_input(meaning)

        # Frame → HyperGraph 저장 (장기 기억)
        if self.frame_adapter is not None:
            self.frame_adapter.store_to_hg(meaning)

        # Curiosity — 모름 카테고리 자율 검색 (응답 끝난 후)
        if self.curiosity_adapter is not None:
            try:
                self.curiosity_adapter.explore_meaning(meaning)
            except Exception:
                pass

        # 사용자 떠남 (UserPresence)
        if self.user_presence_adapter is not None:
            self.user_presence_adapter.on_user_leave(
                message=meaning.raw_text or "", response=full_response,
            )

        # Continual rehearsal (정체성 보존 - 매 turn 약하게)
        if self.continual_adapter is not None:
            self.continual_adapter.tick(dt=1.0)

        # Creative incubation (조건 충족 시)
        if self.creative_adapter is not None:
            self.creative_adapter.maybe_incubate(meaning)

        # Emotion regulation (필요 시 자동)
        if self.emotion_regulation_adapter is not None:
            self.emotion_regulation_adapter.auto_regulate()

        # Multi-stream tick (백그라운드 사고)
        if self.multi_stream_adapter is not None:
            self.multi_stream_adapter.tick(dt=0.5)

        # EM 인코딩
        if self.memory_adapter is not None:
            self.memory_adapter.encode_turn(meaning, full_response)

        # 라운드 31: mood_adapter에 turn 기록 (대화 흐름 추적)
        mood_adapter = getattr(self, "mood_adapter", None)
        if mood_adapter is not None:
            try:
                situation = None
                orch = (meaning.context or {}).get("orchestrator")
                if orch:
                    situation = orch.get("situation")
                mood_adapter.record_turn(input_text, full_response, situation)
            except Exception:
                pass

        self.history.append(Experience(
            input_text=input_text,
            meaning=meaning,
            response=full_response,
            success_score=0.0,
            timestamp=time.time(),
        ))

    # ===== v3 round39 fastText streaming observation helpers =====

    def _fasttext_adapter_loaded(self):
        engine_ref = getattr(self.planner, "_engine_ref", None)
        if engine_ref is None:
            return None
        adapter = getattr(engine_ref, "fasttext_embedding", None)
        if adapter is None or not hasattr(adapter, "is_loaded"):
            return None
        try:
            if not adapter.is_loaded():
                return None
        except Exception:
            return None
        return adapter

    def _observe_fasttext_parallel(self, stage: str, meaning, input_text: str, chunk: str, chunk_index: int) -> None:
        """Record fastText streaming diagnostics without changing chunks.

        Streaming is user-visible and order-sensitive. Round39 keeps the
        existing PMI+SVD/self_embedding path as the only active generation path
        and records fastText signals only when the adapter has been explicitly
        loaded by tests/operators. Trace data is never fed back into chunk
        selection, chunk text, chunk order, sleep/timing, or engine.self_embedding.
        """
        adapter = self._fasttext_adapter_loaded()
        if adapter is None:
            return
        operation = STREAMING_OPERATION_STREAM_CHUNK
        try:
            raw_text = str(getattr(meaning, "raw_text", "") or input_text or "")
            chunk_text = str(chunk or "")
            entities = list(getattr(meaning, "entities", []) or [])
            lookup_text = chunk_text or raw_text or " ".join(str(e) for e in entities)
            text_norm = 0.0
            text_known = False
            text_self_similarity = 0.0
            vector = None
            if lookup_text and hasattr(adapter, "get_vector"):
                vector = adapter.get_vector(lookup_text)
            if vector is not None:
                text_known = True
                try:
                    import numpy as np
                    text_norm = float(np.linalg.norm(vector))
                except Exception:
                    text_norm = 0.0
            if lookup_text and raw_text and hasattr(adapter, "text_similarity"):
                text_self_similarity = float(adapter.text_similarity(lookup_text, raw_text))
            trace = {
                "meaning_layer": STREAMING_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "operation": operation,
                "stage": str(stage or ""),
                "chunk_index": int(chunk_index),
                "in_use_by_generation": STREAMING_GENERATION_EMBEDDING_SOURCE,
                "self_embedding_result": {
                    "active": True,
                    "chunk": chunk_text,
                    "raw_text": raw_text,
                },
                "fasttext_result": {
                    "subset_name": getattr(adapter, "subset_name", None),
                    "loaded": True,
                    "text": lookup_text,
                    "text_known": bool(text_known),
                    "text_vector_norm": text_norm,
                    "text_self_similarity": text_self_similarity,
                },
            }
            self._record_fasttext_trace(trace)
        except Exception as exc:  # fail-open for observation-only diagnostics
            self._record_fasttext_trace({
                "meaning_layer": STREAMING_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "operation": operation,
                "stage": str(stage or ""),
                "chunk_index": int(chunk_index),
                "error": str(exc),
                "in_use_by_generation": STREAMING_GENERATION_EMBEDDING_SOURCE,
            })

    def _record_fasttext_trace(self, trace: dict) -> None:
        self.fasttext_observation_count += 1
        operation = str(trace.get("operation") or "unknown")
        if operation not in self.fasttext_operation_counts:
            self.fasttext_operation_counts[operation] = 0
        self.fasttext_operation_counts[operation] += 1
        self.fasttext_trace.append(dict(trace))
        if len(self.fasttext_trace) > self.fasttext_trace_cap:
            del self.fasttext_trace[: len(self.fasttext_trace) - self.fasttext_trace_cap]

    def stats(self) -> dict:
        return {
            "fasttext_parallel_observation": "fasttext" if self._fasttext_adapter_loaded() is not None else None,
            "fasttext_trace_size": len(self.fasttext_trace),
            "fasttext_observation_count": self.fasttext_observation_count,
            "fasttext_trace_cap": self.fasttext_trace_cap,
            "operations_observed": dict(self.fasttext_operation_counts),
            "in_use_by_generation": STREAMING_GENERATION_EMBEDDING_SOURCE,
        }

    def _infer_outcome(self, meaning, response: str) -> str:
        """응답 결과 단순 추정 (AI 학습용 라벨)."""
        # negative emotion 감지하면 'negative' (위로해야 할 상황)
        for emo in (meaning.emotions or {}):
            if emo in ("fatigue", "sadness", "anger", "anxiety"):
                return "negative"
        if "joy" in (meaning.emotions or {}) or "warmth" in (meaning.emotions or {}):
            return "positive"
        return "neutral"


# 모듈-레벨 편의 함수 (main.py에서 호출용)
def generate_response_stream(input_text: str, engine: StreamingEngine | None = None) -> Iterator[str]:
    if engine is None:
        engine = StreamingEngine()
    yield from engine.chat_stream(input_text)
