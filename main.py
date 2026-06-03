"""
EVE v41 entry point.

기본 REPL은 풀스택(12 어댑터)으로 실행.
순수 v41 만 실험하려면 build_minimal_engine() 사용.

REPL 명령:
  /p              자발 발화 (proactive)
  /save <path>    상태 저장
  /load <path>    상태 복원
  /q              종료
"""

import os
import sys

from language.streaming import StreamingEngine


def build_full_engine(delays: dict | None = None) -> StreamingEngine:
    """v40 24모듈 어댑터 다 끼운 풀스택.

    기본값은 테스트/모바일 실행용 무지연이다. 사람이 보는 REPL만
    별도로 StreamingEngine 기본 delay를 넘겨서 체감 발화 속도를 만든다.
    """
    if delays is None:
        delays = {"stage1": 0, "core": 0, "expand": 0, "final": 0}
    from adapters.hormone_adapter import HormoneAdapter
    from adapters.activation_adapter import ActivationAdapter
    from adapters.memory_adapter import MemoryAdapter
    from adapters.nl_adapter import NLAdapter
    from adapters.sd_adapter import SDAdapter
    from adapters.dmn_adapter import DMNAdapter
    from adapters.vsa_adapter import VSAAdapter
    from adapters.ai_adapter import AIAdapter
    from adapters.goal_adapter import GoalAdapter
    from adapters.norm_adapter import NormAdapter
    from adapters.continual_adapter import ContinualAdapter
    from adapters.allostatic_adapter import AllostaticAdapter
    from adapters.env_adapter import EnvironmentAdapter
    from adapters.autonomy_adapter import AutonomyAdapter
    from adapters.counterfactual_adapter import CounterfactualAdapter
    from adapters.analogy_adapter import AnalogyAdapter
    from adapters.temporal_adapter import TemporalAdapter
    from adapters.humor_adapter import HumorAdapter
    from adapters.suffering_adapter import SufferingAdapter
    from adapters.narrative_adapter import NarrativeAdapter
    from adapters.creative_adapter import CreativeAdapter
    from adapters.metacognition_adapter import MetacognitionAdapter
    from adapters.emotion_regulation_adapter import EmotionRegulationAdapter
    from adapters.multi_stream_adapter import MultiStreamAdapter

    from adapters.deep_reasoning_adapter import DeepReasoningAdapter
    from adapters.world_model_adapter import WorldModelAdapter
    from adapters.agency_adapter import AgencyAdapter
    from adapters.apc_adapter import APCAdapter
    from adapters.corpus_adapter import CorpusAdapter
    from adapters.frame_adapter import FrameAdapter
    from adapters.hypergraph_adapter import HypergraphAdapter
    from adapters.semantic_distance_adapter import SemanticDistanceAdapter
    from adapters.digital_somatic_adapter import DigitalSomaticAdapter
    from adapters.user_presence_adapter import UserPresenceAdapter
    from adapters.integrated_self_adapter import IntegratedSelfAdapter
    from adapters.social_env_adapter import SocialEnvAdapter
    from adapters.enhancer_adapter import EnhancerAdapter
    from adapters.proactive_adapter import ProactiveAdapter
    from core.reasoning import ReasoningLoop
    from core.simulation import SimulationEngine
    from core.autonomous import AutonomousLoop

    h = HormoneAdapter()
    a = ActivationAdapter(hormone_adapter=h)
    m = MemoryAdapter(activation_adapter=a, hormone_adapter=h)
    n = NLAdapter(hormone_adapter=h, activation_adapter=a)
    s = SDAdapter(hormone_adapter=h, activation_adapter=a)
    d = DMNAdapter(hormone_adapter=h, activation_adapter=a)
    v = VSAAdapter(hormone_adapter=h, activation_adapter=a)
    ai = AIAdapter(hormone_adapter=h, activation_adapter=a, sd_adapter=s)
    g = GoalAdapter(hormone_adapter=h, activation_adapter=a)
    nr = NormAdapter(hormone_adapter=h, activation_adapter=a, nl_adapter=n, dmn_adapter=d)
    cr = ContinualAdapter(hormone_adapter=h, activation_adapter=a)
    al = AllostaticAdapter(hormone_adapter=h, activation_adapter=a)
    env_social = SocialEnvAdapter()
    env = EnvironmentAdapter(
        hormone_adapter=h, activation_adapter=a,
        social_env_adapter=env_social,
    )
    auto = AutonomyAdapter(idle_threshold_sec=30.0, cooldown_sec=60.0)
    cf = CounterfactualAdapter(hormone_adapter=h, activation_adapter=a)
    an = AnalogyAdapter(hormone_adapter=h, activation_adapter=a)
    tm = TemporalAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m)
    hu = HumorAdapter(hormone_adapter=h, activation_adapter=a)
    sf = SufferingAdapter(hormone_adapter=h, activation_adapter=a)
    nv = NarrativeAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m)
    cra = CreativeAdapter(hormone_adapter=h, activation_adapter=a, memory_adapter=m, dmn_adapter=d)
    mc = MetacognitionAdapter(hormone_adapter=h, activation_adapter=a,
                              sd_adapter=s, ai_adapter=ai, dmn_adapter=d)
    era = EmotionRegulationAdapter(hormone_adapter=h, activation_adapter=a,
                                   memory_adapter=m, cf_adapter=cf, tm_adapter=tm)
    msa = MultiStreamAdapter(hormone_adapter=h, activation_adapter=a,
                             dmn_adapter=d, memory_adapter=m, tm_adapter=tm,
                             goal_adapter=g, nl_adapter=n)
    dr = DeepReasoningAdapter(hormone_adapter=h, activation_adapter=a, nl_adapter=n)
    wm_adapter = WorldModelAdapter(
        hormone_adapter=h, activation_adapter=a,
        memory_adapter=m, tm_adapter=tm, an_adapter=an,
    )
    sim = SimulationEngine(
        activation_adapter=a, hormone_adapter=h,
        cf_adapter=cf, ai_adapter=ai, dr_adapter=dr,
    )
    reasoning = ReasoningLoop(
        hormone_adapter=h,
        deep_reasoning_adapter=dr,
        temporal_adapter=tm,
        counterfactual_adapter=cf,
        metacognition_adapter=mc,
    )
    agency = AgencyAdapter(
        hormone_adapter=h, activation_adapter=a,
        sd_adapter=s, dmn_adapter=d, ai_adapter=ai,
    )
    apc = APCAdapter(hormone_adapter=h, activation_adapter=a)
    corpus = CorpusAdapter(hormone_adapter=h, activation_adapter=a, nl_adapter=n)
    hg_adapter = HypergraphAdapter(hormone_adapter=h, activation_adapter=a)
    frame = FrameAdapter(
        hormone_adapter=h, activation_adapter=a, nl_adapter=n,
        hypergraph_adapter=hg_adapter,
    )
    sd_adapter = SemanticDistanceAdapter(
        hormone_adapter=h, activation_adapter=a,
        vsa_adapter=v, hypergraph_adapter=hg_adapter,
    )
    ds_adapter = DigitalSomaticAdapter(hormone_adapter=h, activation_adapter=a)
    engine = StreamingEngine(
        delays=delays,
        hormone_adapter=h, activation_adapter=a, memory_adapter=m,
        nl_adapter=n, sd_adapter=s, dmn_adapter=d,
        vsa_adapter=v, ai_adapter=ai,
        goal_adapter=g, norm_adapter=nr,
        continual_adapter=cr, allostatic_adapter=al,
        env_adapter=env, autonomy_adapter=auto,
        counterfactual_adapter=cf, analogy_adapter=an, temporal_adapter=tm,
        humor_adapter=hu, suffering_adapter=sf, narrative_adapter=nv,
        creative_adapter=cra, metacognition_adapter=mc,
        emotion_regulation_adapter=era, multi_stream_adapter=msa,
        deep_reasoning_adapter=dr, reasoning_loop=reasoning,
        world_model_adapter=wm_adapter, simulation_engine=sim,
        agency_adapter=agency,
        apc_adapter=apc, corpus_adapter=corpus,
        frame_adapter=frame, hypergraph_adapter=hg_adapter,
        semantic_distance_adapter=sd_adapter,
        digital_somatic_adapter=ds_adapter,
    )
    # autonomous + UP + IS는 engine 만든 후 (engine 의존)
    engine.autonomous_loop = AutonomousLoop(engine)
    engine.user_presence_adapter = UserPresenceAdapter(engine, user_name="민석")
    engine.integrated_self_adapter = IntegratedSelfAdapter(engine)
    engine.enhancer_adapter = EnhancerAdapter(engine)
    engine.proactive_adapter = ProactiveAdapter(engine)
    # Enhancer는 planner에 늦게 주입 — planner도 engine 만든 후에 다시 연결
    engine.planner.enhancer_adapter = engine.enhancer_adapter

    # 라운드 20: 시각/청각 (lazy, 라이브러리 없어도 EVE 시작 OK)
    from adapters.sensory_adapter import SensoryAdapter
    engine.sensory_adapter = SensoryAdapter(engine)

    # 라운드 21: 자동 음성 대화 루프 (lazy)
    from adapters.voice_loop_adapter import VoiceLoopAdapter
    engine.voice_loop_adapter = VoiceLoopAdapter(engine)

    # 라운드 22: 인터넷 학습 + 비전 (lazy)
    from adapters.web_learning_adapter import WebLearningAdapter
    from adapters.vision_adapter import VisionAdapter
    engine.web_learning_adapter = WebLearningAdapter(engine)
    engine.vision_adapter = VisionAdapter(engine)

    # 라운드 23: 자율 검색 (모르는 카테고리 자동 학습)
    from adapters.curiosity_adapter import CuriosityAdapter
    engine.curiosity_adapter = CuriosityAdapter(engine)

    # 라운드 24: 2D 시각화 WebSocket 서버 (lazy, 명시 시작 필요)
    from adapters.visualizer_server import VisualizerServer
    engine.visualizer_server = VisualizerServer(engine)

    # 라운드 25: 캐릭터 (호르몬→표정, 행동→모션, 입모양)
    from adapters.character_adapter import CharacterAdapter
    engine.character_adapter = CharacterAdapter(engine)

    # 라운드 26: OpenAI 호환 HTTP API (AIRI 등 외부 클라이언트 연결용)
    from adapters.openai_server_adapter import OpenAIServerAdapter
    engine.openai_server_adapter = OpenAIServerAdapter(engine)

    # 라운드 27: 가르침 학습 + 살아있는 루프
    from adapters.teaching_adapter import TeachingAdapter
    from adapters.live_loop import LiveLoop
    engine.teaching_adapter = TeachingAdapter(engine)
    engine.live_loop = LiveLoop(engine)

    # 라운드 29 (B-1): Orchestrator 분류기
    from adapters.orchestrator_adapter import OrchestratorAdapter
    engine.orchestrator_adapter = OrchestratorAdapter(engine)

    # 라운드 30 (B-2): Situation 기반 응답 합성
    from adapters.situation_responder import SituationResponder
    engine.situation_responder = SituationResponder(engine)
    engine.planner.situation_responder = engine.situation_responder

    # 라운드 31 (B-3): 분위기 + 맥락 추론
    from adapters.mood_adapter import MoodAdapter
    engine.mood_adapter = MoodAdapter(engine)

    # 라운드 32 (B-4): 모듈 선택 학습
    from adapters.module_learning_adapter import ModuleLearningAdapter
    engine.module_learning_adapter = ModuleLearningAdapter(engine)

    # 라운드 33-34 (C-1, C-2): 한국어 형태소 + 구조 분석
    from adapters.korean_language_adapter import KoreanLanguageAdapter
    engine.korean_adapter = KoreanLanguageAdapter(engine)

    # 라운드 35 (C-3): 4개 안전장치
    from adapters.safety_adapter import SafetyAdapter
    engine.safety_adapter = SafetyAdapter(engine)

    # 라운드 36: 발화 충동
    from adapters.urge_adapter import UrgeAdapter
    engine.urge_adapter = UrgeAdapter(engine)

    # 라운드 37: SalienceAdapter
    from adapters.salience_adapter import SalienceAdapter
    engine.salience_adapter = SalienceAdapter(engine)

    # 라운드 39a: EmotionCauseTracker
    from adapters.emotion_cause_adapter import EmotionCauseTracker
    engine.emotion_cause = EmotionCauseTracker(engine)
    
    # planner가 emotion_cause 접근 가능하게 engine 백참조
    if hasattr(engine, "planner"):
        engine.planner._engine_ref = engine

    # 라운드 39b: ThoughtChainAdapter
    from adapters.thought_chain_adapter import ThoughtChainAdapter
    engine.thought_chain = ThoughtChainAdapter(engine)

    # 라운드 43: ConceptMemory
    from adapters.concept_memory_adapter import ConceptMemoryAdapter
    engine.concept_memory = ConceptMemoryAdapter(engine)

    # 라운드 47: SelfEmbedding
    from adapters.self_embedding_adapter import SelfEmbeddingAdapter
    engine.self_embedding = SelfEmbeddingAdapter(engine)
    # Bootstrap is explicit/lazy; running SVD on every engine build is too slow for tests/mobile.

    # v3 round41: final embedding swap.
    # Keep the original PMI+SVD adapter as rollback/fallback, load the extracted
    # fastText subset explicitly, then route engine.self_embedding through a
    # fail-open wrapper. self_embedding_adapter.py remains preserved.
    from adapters.fasttext_embedding_adapter import FasttextEmbeddingAdapter
    from adapters.embedding_wrapper import EmbeddingWrapper
    engine.self_embedding_backup = engine.self_embedding
    engine.fasttext_embedding = FasttextEmbeddingAdapter(engine=engine)
    try:
        engine.fasttext_embedding.load()
    except ValueError as exc:
        # Code-only handoff packages intentionally omit the medium 30k
        # vectors.npy artifact. If a restored small 5k subset is available in
        # the local checkout, focused runtime-mapping validation may use it;
        # otherwise the wrapper remains fail-open to PMI+SVD and medium/full
        # validation is reported as blocked rather than overstated.
        blocked_markers = (
            "missing_vectors_file",
            "missing_subset_dir",
            "subset_not_registered",
            "subset_audit_failed",
        )
        if not any(marker in str(exc) for marker in blocked_markers):
            raise
        from pathlib import Path
        from adapters.external_seed_manifest import (
            FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
            FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
        )
        small_vectors = Path(FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH) / "vectors.npy"
        if small_vectors.exists():
            engine.fasttext_embedding = FasttextEmbeddingAdapter(
                engine=engine,
                subset_name=FASTTEXT_KOREAN_SUBSET_SMALL_5K_NAME,
                subset_dir=FASTTEXT_KOREAN_SUBSET_SMALL_5K_PATH,
            )
            engine.fasttext_embedding.load()
        else:
            engine.fasttext_embedding_code_only_blocked = {
                "reason": str(exc),
                "medium_vectors_present": False,
                "small_vectors_present": False,
                "focused_validation_uses_fail_open_fallback": True,
            }
    # Round56: Initialize the Eve-specific vector store and insert it between
    # the fastText primary and the PMI+SVD fallback in the embedding wrapper.
    from adapters.eve_vocab_tracker import EveVocabTracker
    from adapters.eve_vector_store import EveSpecificVectorStore
    from adapters.eve_self_learning_adapter import EveSelfLearningAdapter
    engine.eve_vocab_tracker = EveVocabTracker(engine=engine)
    engine.eve_specific_vector_store = EveSpecificVectorStore(engine=engine)
    engine.self_embedding = EmbeddingWrapper(
        primary=engine.fasttext_embedding,
        fallback=engine.self_embedding_backup,
        eve_specific=engine.eve_specific_vector_store,
    )
    # Round58: continuous observation is enabled, but vector promotion remains
    # explicit. Round64 enforces a stricter commit gate: an EVE-specific
    # candidate must be observed at least twice before vector mutation. Round65
    # exposes a read-only threshold config/manual rollback snapshot. Round69
    # also requires context-diverse observation evidence before commit. Round70
    # surfaces blocked-candidate reporting and a read-only manual rollback drill.
    engine.eve_self_learning = EveSelfLearningAdapter(
        engine=engine,
        tracker=engine.eve_vocab_tracker,
        vector_store=engine.eve_specific_vector_store,
        auto_observe_enabled=True,
        auto_promotion_enabled=False,
        min_observations_for_commit=2,
        context_diversity_gate_enabled=True,
    )

    # v3 round77: lexical → concept mapping planning boundary.
    # Planning only: lexical vectors remain evidence, not AGP anchors.
    from adapters.lex_concept_mapping_adapter import LexConceptMappingAdapter
    engine.lex_concept_mapping = LexConceptMappingAdapter(engine)

    # 라운드 48: AttentionAnalyzer
    from adapters.attention_analyzer import AttentionAnalyzer
    engine.attention = AttentionAnalyzer(engine)

    # v3 round8: AGP observation adapter (no veto/fallback activation yet)
    from adapters.agp_adapter import AGPAdapter
    engine.agp_adapter = AGPAdapter(
        sa_adapter=engine.activation_adapter,
        hormone_adapter=engine.hormone_adapter,
        mode="observation",
    )

    # 라운드 49: CompositorAdapter
    from adapters.compositor_adapter import CompositorAdapter
    engine.compositor = CompositorAdapter(engine)

    # 라운드 52: UserInstructionAdapter — 사용자 메타 지시 처리
    from adapters.user_instruction_adapter import UserInstructionAdapter
    engine.user_instruction = UserInstructionAdapter(engine)

    # 라운드 54: DialogueContextAdapter
    from adapters.dialogue_context_adapter import DialogueContextAdapter
    engine.dialogue_context = DialogueContextAdapter(engine)

    # 라운드 56: SpeechHub — 단일 응답 생성 hub (하드코드 통합 시작)
    from adapters.speech_hub import SpeechHub
    engine.speech_hub = SpeechHub(engine)

    # 라운드 67: IntentDecider — GPT 판단 레이어
    from adapters.intent_decider import IntentDecider
    engine.intent_decider = IntentDecider(engine)

    # 라운드 68: ContextSeed — 결정론적 변동성
    from adapters.context_seed import ContextSeed
    engine.context_seed = ContextSeed(engine)

    # 라운드 72: SelfState — GPT 짚은 자기 일관성
    from adapters.self_state import SelfState
    engine.self_state = SelfState(engine)

    # round73 patch7: StateDebug — fallback/routing/state dump
    from adapters.state_debug_adapter import StateDebugAdapter
    engine.state_debug = StateDebugAdapter(engine)

    return engine


def build_minimal_engine() -> StreamingEngine:
    """순수 v41 (어댑터 0개) — eve1.md 설계 그대로."""
    return StreamingEngine()


def chat(engine: StreamingEngine, input_text: str):
    for chunk in engine.chat_stream(input_text):
        print(chunk, flush=True)


def proactive(engine: StreamingEngine):
    for chunk in engine.proactive_stream(force=True):
        print(chunk, flush=True)


def repl():
    print("EVE v41 (LLM 없음, 결정론). 명령:")
    print("  /p              자발 발화")
    print("  /me             자기 이야기 (NarrativeSelf)")
    print("  /tick [N]       자율 사이클 N번 (기본 1)")
    print("  /learn <text>   텍스트에서 카테고리 페어 학습")
    print("  /say <text>     텍스트 → 음성 (Piper 필요)")
    print("  /listen <path>  오디오 → STT → EVE 응답 (Whisper 필요)")
    print("  /read <path>    이미지 → OCR → EVE 학습 (Tesseract 필요)")
    print("  /voice          자동 음성 한 번 (마이크→EVE→스피커)")
    print("  /voiceloop [N]  자동 음성 N회 반복 (기본 5, 0=무한)")
    print("  /web <url>      URL → 텍스트 학습")
    print("  /wiki <topic>   위키피디아 한국어 → 학습")
    print("  /yt <url|id>    유튜브 자막 → 학습")
    print("  /see <path>     이미지 → 카테고리 학습 (CLIP)")
    print("  /curiosity      자율 검색 상태 (오늘 검색 횟수)")
    print("  /viz [start|stop|status]  2D 시각화 WebSocket 서버")
    print("  /api [start|stop|status]  OpenAI 호환 HTTP API (AIRI 연결용)")
    print("  /live [start|stop|status] 자율 살아있는 모드 (background)")
    print("  /quiet          자율 발화 잠시 끔 (조용히)")
    print("  /teach          학습된 응답 패턴 보기")
    print("  /orch           Orchestrator 분류 통계 (라운드 29)")
    print("  /mlearn         모듈 선택 학습 통계 (라운드 32)")
    print("  /safety         4개 안전장치 통계 (라운드 35: Confidence/Decay/Conflict/Exploration)")
    print("  /dump <text>    입력 라우팅/상태 진단 덤프")
    print("  /sensory        시각/청각 모듈 상태")
    print("  /save <path>    저장")
    print("  /load <path>    복원")
    print("  /q              종료")
    print()
    from language.streaming import DEFAULT_DELAYS
    engine = build_full_engine(delays=DEFAULT_DELAYS)
    persistence = None  # 첫 요청에 lazy 생성

    # 라운드 35: 자율 발화 default ON — 진짜 끊임없이
    # 인간 = 깨어있는 동안 사고 + 발화 멈추지 않음 (혼잣말, 흥얼거림)
    # cooldown 0 = 매 tick 발화 가능, Exploration이 반복 막음
    if engine.autonomy_adapter is not None:
        engine.autonomy_adapter.idle_threshold = 0.0   # 즉시 발화 가능
        engine.autonomy_adapter.cooldown = 0.0          # 끊임없이
    
    # interval 3초 — 너무 빠르면 콘솔 깨짐, 너무 느리면 안 살아있는 느낌
    engine.live_loop.interval = 3.0
    
    try:
        engine.live_loop.autosave_path = os.path.expanduser("~/eve.ckpt")
        if engine.live_loop.start():
            print("✨ EVE 살아있는 모드 자동 시작 (자율 발화 ON)")
            print("   /quiet 으로 끄거나 /live stop")
            print()
    except Exception as e:
        print(f"⚠️  자율 모드 시작 실패: {e} (수동: /live start)")
        print()

    while True:
        try:
            # 라운드 62: input prompt 보이는 시점 = 사용자 활성. 자율발화 차단
            if engine.live_loop.running and engine.autonomy_adapter is not None:
                engine.autonomy_adapter.mark_input()
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/q":
            break
        if text == "/p":
            proactive(engine)
        elif text == "/me":
            if engine.integrated_self_adapter is not None:
                print(f"  {engine.integrated_self_adapter.snapshot_phrase()}")
            if engine.narrative_adapter is not None:
                phrase = engine.narrative_adapter.narrate_phrase()
                print(f"  {phrase}")
                coh = engine.narrative_adapter.coherence()
                print(f"  [coherence: {coh:.3f}]")
            if engine.user_presence_adapter is not None:
                print(f"  [intimacy: {engine.user_presence_adapter.get_intimacy():.3f}]")
        elif text.startswith("/tick"):
            parts = text.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            for _ in range(n):
                r = engine.autonomous_loop.step(emit=True)
                print(f"  step {r['step']}: needs={r['needs']}, "
                      f"action={r['action']}, decision={r['decision']}")
                if r["emitted"]:
                    print(f"  → {r['response']}")
        elif text.startswith("/learn "):
            content = text[7:].strip()
            if engine.corpus_adapter is not None and content:
                r = engine.corpus_adapter.learn_text(content)
                pairs = r.get("pairs_made", 0) if isinstance(r, dict) else 0
                cats = r.get("categories", set()) if isinstance(r, dict) else set()
                print(f"  학습: 카테고리 {len(cats)}개, 페어 {pairs}개")
            else:
                print("  corpus_adapter 없음 또는 빈 입력")
        elif text == "/sensory":
            s = engine.sensory_adapter.status()
            print(f"  STT (Whisper): {'✅ 가능' if s['stt'] else '❌ 라이브러리 없음 (pip install faster-whisper)'}")
            print(f"  TTS (Piper):   {'✅ 가능' if s['tts'] else '❌ 라이브러리 없음 (pip install piper-tts + 한국어 voice)'}")
            print(f"  OCR (Tesseract): {'✅ 가능' if s['ocr'] else '❌ 라이브러리 없음 (pip install pytesseract + tesseract 바이너리)'}")
            v = engine.voice_loop_adapter.status()
            print(f"  Audio I/O:    {'✅ 가능' if v['audio_io'] else '❌ (pip install sounddevice numpy)'}")
            print(f"  VAD:          {'✅ ' + (v['vad_kind'] or '') if v['vad'] else '❌ (pip install silero-vad torch 또는 webrtcvad)'}")
            w = engine.web_learning_adapter.status()
            print(f"  HTTP:         {'✅' if w['requests'] else '❌ (pip install requests)'}")
            print(f"  Wiki:         {'✅' if w['wikipedia'] else '❌ (pip install wikipedia)'}")
            print(f"  YouTube:      {'✅' if w['youtube_transcript'] else '❌ (pip install youtube-transcript-api)'}")
            vis = engine.vision_adapter.status()
            print(f"  CLIP:         {'✅' if vis['clip'] else '❌ (pip install transformers torch pillow)'}")
        elif text == "/curiosity":
            s = engine.curiosity_adapter.status()
            print(f"  오늘 검색: {s['searches_today']}/{s['daily_limit']}")
            print(f"  남은 검색: {s['remaining']}")
            print(f"  쿨다운: {s['cooldown_seconds']}초")
            print(f"  시도한 카테고리: {s['tried_categories']}개")
            history = engine.curiosity_adapter.history(5)
            if history:
                print(f"  최근:")
                for h in history:
                    print(f"    - {h['category']}: 학습 {'✅' if h['learned'] else '❌'} ({h['text_length']}자)")
        elif text.startswith("/api"):
            parts = text.split()
            cmd = parts[1] if len(parts) > 1 else "status"
            a = engine.openai_server_adapter
            if cmd == "start":
                if a.running:
                    print(f"  이미 실행 중 (http://{a.host}:{a.port})")
                else:
                    ok = a.start_background()
                    if ok:
                        print(f"  ✅ 시작됨: http://{a.host}:{a.port}")
                        print(f"  AIRI Settings → OpenAI Custom URL: http://<폰IP>:{a.port}/v1")
                        print(f"  API key: 아무거나 (검사 X)")
                    else:
                        print(f"  시작 실패 (포트 사용 중일 수 있음)")
            elif cmd == "stop":
                a.stop()
                print(f"  중지됨")
            else:
                s = a.status()
                print(f"  running: {'✅' if s['running'] else '❌'}")
                print(f"  endpoints:")
                for ep in s['endpoints']:
                    print(f"    {ep}")
        elif text == "/safety":
            s = engine.safety_adapter.stats()
            print(f"  Confidence 검사: {s['confidence_checks']} (낮음 {s['confidence_low']})")
            print(f"  Decay 실행: {s['decay_runs']}")
            print(f"  Beliefs: {s['beliefs_count']}, 모순감지: {s['conflicts_detected']}, 해소: {s['conflicts_resolved']}")
            print(f"  Exploration trigger: {s['exploration_triggers']}")
        elif text.startswith("/dump"):
            target = text[5:].strip()
            if not target:
                if getattr(engine, "state_debug", None) is not None and engine.state_debug.last_dump:
                    print(engine.state_debug.render())
                else:
                    print("  사용법: /dump <진단할 입력>")
            else:
                dump = engine.state_debug.diagnose_text(target)
                print(engine.state_debug.render(dump))
        elif text == "/quiet":
            ll = engine.live_loop
            ll.stop()
            print("  🤫 조용히. 자율 발화 멈춤.")
            print("  다시 켜려면: /live start")
        elif text.startswith("/live"):
            parts = text.split()
            cmd = parts[1] if len(parts) > 1 else "status"
            ll = engine.live_loop
            if cmd == "start":
                # 사용자가 path 줄 수도, 기본값은 ~/eve.ckpt
                ll.autosave_path = os.path.expanduser(parts[2]) if len(parts) > 2 else os.path.expanduser("~/eve.ckpt")
                if ll.start():
                    print(f"  ✅ EVE 살아있는 모드 시작 (interval={ll.interval}s)")
                    print(f"  자동 저장: {ll.autosave_path} ({int(ll.autosave_interval)}초마다)")
                    print(f"  EVE가 자율로 발화할 수 있음. 너 입력은 인터럽트로.")
                else:
                    print(f"  이미 실행 중")
            elif cmd == "stop":
                ll.stop()
                print(f"  ✅ 살아있는 모드 중지")
            else:
                s = ll.status()
                print(f"  running: {'✅' if s['running'] else '❌'}")
                print(f"  ticks: {s['tick_count']}, emits: {s['emit_count']}, errors: {s['error_count']}")
                print(f"  autosave: {s['autosave_path']}")
        elif text == "/teach":
            t = engine.teaching_adapter
            s = t.stats()
            print(f"  학습된 응답 {s['learned_count']}개")
            print(f"  사용 횟수 {s['total_uses']}, 강화 {s['total_reinforced']}")
            for lr in t.learned[:10]:
                print(f"    [{lr.strength:.2f}] {lr.response!r} ← {lr.triggers} (사용 {lr.times_used}회)")
        elif text == "/orch":
            o = engine.orchestrator_adapter
            s = o.stats()
            print(f"  분류 횟수: {s['total_classifications']}")
            print(f"  상황별 분포:")
            for sit, count in sorted(s['situations'].items(), key=lambda x: -x[1]):
                print(f"    {sit}: {count}")
            recent = o.history(5)
            if recent:
                print(f"  최근 분류:")
                for h in recent:
                    print(f"    [{h['situation']}] {h['text'][:50]!r}")
        elif text == "/mlearn":
            ml = engine.module_learning_adapter
            s = ml.stats()
            print(f"  조정 {s['adjustments']}회 (긍정 {s['positive']}, 부정 {s['negative']})")
            print(f"  학습된 (situation, module) 쌍: {s['learned_pairs']}개")
            if s['top_5']:
                print(f"  상위 5 (효과 좋은):")
                for (sit, mod), util in s['top_5']:
                    print(f"    [{sit}] {mod}: {util:.2f}")
            if s['bottom_5']:
                print(f"  하위 5 (효과 약한):")
                for (sit, mod), util in s['bottom_5']:
                    print(f"    [{sit}] {mod}: {util:.2f}")
        elif text.startswith("/viz"):
            parts = text.split()
            cmd = parts[1] if len(parts) > 1 else "status"
            v = engine.visualizer_server
            if cmd == "start":
                if not v.available():
                    print(f"  ❌ websockets 라이브러리 없음 (pip install websockets)")
                elif v.running:
                    print(f"  이미 실행 중 ({v.host}:{v.port})")
                else:
                    ok = v.start_background()
                    if ok:
                        print(f"  ✅ 시작됨: ws://{v.host}:{v.port}")
                        print(f"  브라우저에서 visualizer/index.html 열기")
                    else:
                        print(f"  시작 실패")
            elif cmd == "stop":
                v.stop()
                print(f"  중지 요청")
            else:
                s = v.status()
                print(f"  available: {'✅' if s['available'] else '❌'}")
                print(f"  running: {'✅' if s['running'] else '❌'}")
                print(f"  host: {s['host']}:{s['port']}")
                print(f"  연결된 클라이언트: {s['clients']}")
        elif text.startswith("/web "):
            url = text[5:].strip()
            r = engine.web_learning_adapter.fetch_and_learn(url)
            print(f"  URL: {r['url']}")
            print(f"  텍스트 길이: {r['text_length']}")
            print(f"  학습: {'✅' if r['learned'] else '❌'}")
        elif text.startswith("/wiki "):
            topic = text[6:].strip()
            r = engine.web_learning_adapter.wiki_and_learn(topic)
            print(f"  주제: {r['topic']}")
            print(f"  텍스트 길이: {r['text_length']}")
            print(f"  학습: {'✅' if r['learned'] else '❌'}")
        elif text.startswith("/yt "):
            v = text[4:].strip()
            r = engine.web_learning_adapter.youtube_and_learn(v)
            print(f"  영상: {r['video']}")
            print(f"  자막 길이: {r['text_length']}")
            print(f"  학습: {'✅' if r['learned'] else '❌'}")
        elif text.startswith("/see "):
            path = text[5:].strip()
            r = engine.vision_adapter.see_and_learn(path)
            if not r["labels"]:
                print(f"  비전 사용 불가 또는 인식 실패")
            else:
                desc = engine.vision_adapter.describe_image(path)
                print(f"  봤어: {desc}")
                print(f"  카테고리: {[(n, f'{s:.2f}') for n, s in r['labels']]}")
                print(f"  학습: {'✅' if r['learned'] else '❌'}")
        elif text == "/voice":
            v = engine.voice_loop_adapter
            if not v.audio_available():
                print("  마이크 사용 불가 (sounddevice + numpy 필요)")
            else:
                print("  말하세요...")
                r = v.chat_once()
                if r["transcribed"]:
                    print(f"  들었어: {r['transcribed']}")
                    print(f"  답: {r['response']}")
                    if r["played"]:
                        print(f"  ♪ 재생됨")
                    elif r["synthesized"]:
                        print(f"  (합성됐지만 재생 실패)")
                else:
                    print(f"  녹음/인식 실패: {r}")
        elif text.startswith("/voiceloop"):
            parts = text.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            v = engine.voice_loop_adapter
            if not v.audio_available():
                print("  마이크 사용 불가")
            else:
                count = 0
                while n == 0 or count < n:
                    print(f"\n  [{count + 1}{'/' + str(n) if n else ''}] 말하세요...")
                    try:
                        r = v.chat_once()
                    except KeyboardInterrupt:
                        print("\n  중단")
                        break
                    if r["transcribed"]:
                        print(f"  들었어: {r['transcribed']}")
                        print(f"  답: {r['response']}")
                    else:
                        print(f"  (조용 또는 인식 실패 - 계속)")
                    count += 1
        elif text.startswith("/say "):
            content = text[5:].strip()
            if not engine.sensory_adapter.tts_available():
                print("  TTS 사용 불가 (라이브러리 또는 voice 모델 필요)")
            else:
                wav = engine.sensory_adapter.synthesize(content)
                print(f"  합성됨: {wav}")
        elif text.startswith("/listen "):
            path = text[8:].strip()
            if not engine.sensory_adapter.stt_available():
                print("  STT 사용 불가 (faster-whisper 필요)")
            else:
                user_text = engine.sensory_adapter.transcribe(path)
                if not user_text:
                    print("  음성 인식 실패")
                else:
                    print(f"  들었어: {user_text}")
                    for c in engine.chat_stream(user_text):
                        print(f"  {c}")
        elif text.startswith("/read "):
            path = text[6:].strip()
            if not engine.sensory_adapter.ocr_available():
                print("  OCR 사용 불가 (pytesseract + tesseract 바이너리 필요)")
            else:
                img_text = engine.sensory_adapter.read_and_learn(path)
                if not img_text:
                    print("  텍스트 인식 실패")
                else:
                    print(f"  읽었어: {img_text[:200]}{'...' if len(img_text) > 200 else ''}")
                    print(f"  → corpus 학습 완료")
        elif text.startswith("/save "):
            from adapters.persistence_adapter import PersistenceAdapter
            persistence = persistence or PersistenceAdapter(engine)
            path = text[6:].strip()
            r = persistence.save(path)
            print(f"  저장: {r.get('path', path)}")
        elif text.startswith("/load "):
            from adapters.persistence_adapter import PersistenceAdapter
            persistence = persistence or PersistenceAdapter(engine)
            path = text[6:].strip()
            r = persistence.load(path)
            print(f"  복원: v40={r['v40_ok']}, v41={r['v41_loaded']}")
        else:
            # LiveLoop 활성이면 인터럽트로 전달
            if engine.live_loop.running:
                engine.live_loop.push_user_input(text)
            else:
                chat(engine, text)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--minimal":
        engine = build_minimal_engine()
        for line in sys.stdin:
            chat(engine, line.strip())
    else:
        repl()

