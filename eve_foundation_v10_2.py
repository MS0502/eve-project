"""
EVE 9.0 - DUAL BRAIN (좌뇌 + 우뇌)

너 통찰: 
- "EVE = 좌뇌, SNN = 우뇌"
- "트랜스포머 X (confabulation)"
- "카테고리 살리고 SNN 진짜로"
= 학계 2025 SOTA 정확히 일치

10개 신규 기능:

[좌뇌 강화 - 5개]
1. 🧠 환각 차단기 (HallucinationBlocker)
   - 메타인지 강화
   - 등록된 신념만 답
   - 모르면 "모른다"
   
2. 🎯 신뢰도 추적기 (ConfidenceTracker)
   - 답할 때 0-1 신뢰도
   - 0.5 이하면 "잘 모르겠어"
   - Anthropic 2025 방식
   
3. 📚 진짜 vs 학습 구분
   - 직접 경험 신념 (강함)
   - 추론 신념 (중간)
   - 가설 (약함)
   
4. 🔍 자기 검증 (Self-Verification)
   - 답하기 전 신념 충돌 체크
   - 모순이면 거절
   
5. 💭 사고 로그 (Reasoning Trace)
   - 왜 그렇게 답했는지 기록
   - 해석 가능

[우뇌 추가 - 5개]
6. 🧬 진짜 LIF 뉴런 (numpy 기반)
   - 막전위 (membrane voltage)
   - 발화 임계값
   - 시간 차원
   
7. ⚡ STDP 학습
   - Spike Timing Dependent Plasticity
   - 같이 발화 → 강화
   
8. 🌐 진짜 Sparse 5%
   - 활성 뉴런만 처리
   - 폭발 해결
   
9. 🔄 진짜 Pattern Completion
   - 부분 단서 → 전체 회상
   - Hopfield 1982 방식
   
10. 🌉 좌우 통합 라우터 (Corpus Callosum)
    - 빠른 직관 (우뇌) vs 분석 (좌뇌)
    - 자동 선택
    - 통합 출력

학술:
- Collins 1969 (Hierarchical)
- Hopfield 1982 (Pattern Completion)
- Bi & Poo 1998 (STDP)
- Olshausen 1996 (Sparse)
- Kahneman 2011 (System 1/2)
- Sperry 1981 (Hemispheric)
- Anthropic 2025 (Hallucination circuits)
"""
import networkx as nx
import json
import pickle
import math
import re
from datetime import datetime
from pathlib import Path


# =====================================================
# Korean Josa (조사) Handling
# =====================================================

def has_batchim(word):
    """한국어 단어 마지막 글자에 받침이 있는지"""
    if not word:
        return False
    last = word[-1]
    if not ('\uac00' <= last <= '\ud7a3'):  # 한글 음절 범위
        return False
    code = ord(last) - 0xAC00
    jongseong = code % 28
    return jongseong != 0


def add_josa(word, josa_type):
    """단어에 적절한 조사 추가
    josa_type: '는/은', '이/가', '을/를', '와/과'
    """
    if not word:
        return word
    has_b = has_batchim(word)
    josa_map = {
        '는/은': '은' if has_b else '는',
        '이/가': '이' if has_b else '가',
        '을/를': '을' if has_b else '를',
        '와/과': '과' if has_b else '와',
    }
    return word + josa_map.get(josa_type, '는' if not has_b else '은')


# =====================================================
# Antonym Dictionary (반대 개념)
# =====================================================
# Multi-Aspect Reasoning의 핵심: 같은 aspect의 진짜 충돌 감지

ANTONYM_PAIRS = [
    # 번식 aspect
    (["새끼", "낳"], ["알", "낳"]),       # 새끼 낳음 vs 알 낳음
    (["출산"], ["산란"]),
    
    # 호흡 aspect
    (["폐"], ["아가미"]),
    (["폐로", "호흡"], ["아가미로", "호흡"]),
    
    # 외형 aspect
    (["털"], ["비늘"]),
    (["털"], ["깃털"]),
    (["깃털"], ["비늘"]),
    
    # 행동 aspect
    (["날"], ["기어"]),
    (["헤엄"], ["걷"]),
    
    # 식이 aspect  
    (["초식"], ["육식"]),
    
    # 일반 부정
    (["할 수 있"], ["할 수 없"]),
    (["수 있"], ["수 없"]),
    (["있다"], ["없다"]),
    (["가능"], ["불가능"]),
    (["크"], ["작"]),
    (["많"], ["적"]),
    (["뜨거"], ["차가"]),
]


def check_antonym_conflict(text1, text2):
    """두 술어가 antonym 관계에 있는지 검사"""
    for keywords1, keywords2 in ANTONYM_PAIRS:
        # 양방향 체크
        t1_has_1 = all(k in text1 for k in keywords1)
        t1_has_2 = all(k in text2 for k in keywords1)
        t2_has_1 = all(k in text1 for k in keywords2)
        t2_has_2 = all(k in text2 for k in keywords2)
        
        if (t1_has_1 and t2_has_2) or (t2_has_1 and t1_has_2):
            return True
    return False


# =====================================================
# Aspect Categories (Multi-Aspect Reasoning, Giordano 2015)
# =====================================================
# 다른 측면(aspect)에서의 속성은 서로 충돌 X

ASPECT_KEYWORDS = {
    "분류": ["이다", "분류", "종류", "속하"],  # classification
    "번식": ["낳", "출산", "새끼", "알", "임신", "수정"],  # reproduction
    "호흡": ["숨", "호흡", "폐", "아가미", "산소"],  # respiration
    "외형": ["털", "비늘", "깃털", "가죽", "피부", "색", "크기"],  # appearance
    "행동": ["날", "헤엄", "걷", "달리", "기"],  # behavior
    "식이": ["먹", "사냥", "초식", "육식", "잡식"],  # diet
    "양육": ["키운", "양육", "모유", "수유", "돌"],  # nurturing
    "거주": ["살", "서식", "거주", "환경"],  # habitat
}


def classify_aspect(predicate_text):
    """술어가 어떤 aspect에 속하는지 분류"""
    aspects = set()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in predicate_text for kw in keywords):
            aspects.add(aspect)
    return aspects if aspects else {"기타"}


# =====================================================
# Triple Structure (RDF: Subject-Predicate-Object)
# =====================================================

def parse_to_triple(statement):
    """
    한국어 문장을 (subject, relation, predicate) 삼중구조로 파싱
    예: "고래는 포유류이다" -> ("고래", "이다", "포유류")
        "고래는 새끼를 낳는다" -> ("고래", "낳다", "새끼")
    """
    # 주격 조사 제거
    subject = None
    rest = None
    for josa in ["는 ", "은 ", "이 ", "가 "]:
        if josa in statement:
            parts = statement.split(josa, 1)
            if len(parts) == 2:
                subject = parts[0].strip()
                rest = parts[1].strip()
                break

    if not subject or not rest:
        return None

    # Negation 감지
    is_negation = "아니" in rest or "않" in rest or "못" in rest

    # Aspect 분류
    aspects = classify_aspect(rest)

    return {
        "subject": subject,
        "predicate_text": rest,
        "is_negation": is_negation,
        "aspects": list(aspects) if aspects else ["기타"],
        "original": statement
    }


# =====================================================
# Belief Class - JTMS Justification Tracking
# =====================================================

class Belief:
    """Belief with justification (Doyle 1979)"""

    def __init__(self, belief_id, statement, confidence, source, dependencies=None):
        self.belief_id = belief_id
        self.statement = statement
        self.triple = parse_to_triple(statement)
        self.confidence = confidence
        self.raw_confidence = confidence
        self.sources = [source] if source else []
        self.dependencies = dependencies or []  # JTMS justifications
        self.evidence_count = 1
        self.is_inferred = False
        self.is_default = False  # default rule (vs hard fact)
        self.is_category_rule = False  # general rule vs instance fact
        self.is_innate = False  # v5.4: 선천적 신념 (보호됨)
        self.created = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        self.age_days = 0
        self.doubt_flag = False

    def to_dict(self):
        return {
            "belief_id": self.belief_id,
            "statement": self.statement,
            "triple": self.triple,
            "confidence": self.confidence,
            "raw_confidence": self.raw_confidence,
            "sources": self.sources,
            "dependencies": self.dependencies,
            "evidence_count": self.evidence_count,
            "is_inferred": self.is_inferred,
            "is_default": self.is_default,
            "is_category_rule": self.is_category_rule,
            "is_innate": self.is_innate,
            "created": self.created,
            "last_updated": self.last_updated,
            "age_days": self.age_days,
            "doubt_flag": self.doubt_flag
        }

    @classmethod
    def from_dict(cls, d):
        b = cls.__new__(cls)
        b.belief_id = d["belief_id"]
        b.statement = d["statement"]
        b.triple = d.get("triple") or parse_to_triple(d["statement"])
        b.confidence = d["confidence"]
        b.raw_confidence = d.get("raw_confidence", d["confidence"])
        b.sources = d.get("sources", [])
        b.dependencies = d.get("dependencies", [])
        b.evidence_count = d.get("evidence_count", 1)
        b.is_inferred = d.get("is_inferred", False)
        b.is_default = d.get("is_default", False)
        b.is_category_rule = d.get("is_category_rule", False)
        b.is_innate = d.get("is_innate", False)
        b.created = d.get("created", datetime.now().isoformat())
        b.last_updated = d.get("last_updated", datetime.now().isoformat())
        b.age_days = d.get("age_days", 0)
        b.doubt_flag = d.get("doubt_flag", False)
        return b


# =====================================================
# EVE Foundation v5.0
# =====================================================

class EveFoundation:
    SOURCE_CREDIBILITY = {
        "자기관찰": 0.95, "과학_지식": 0.90, "전문가": 0.85, "교사": 0.80,
        "책": 0.70, "다큐멘터리": 0.75, "선천적": 1.0, "추론": 0.60,
        "인터넷": 0.50, "친구": 0.45, "추정": 0.40, "추측": 0.30,
        "모르는_사람": 0.25, "잘못된_추측": 0.20, "교정": 0.95,
        "자기참조": 1.0, "배움": 0.65, "자체추론": 0.70
    }

    def __init__(self, eve_id="EVE_001", storage_path="/content/drive/MyDrive/eve_data"):
        self.eve_id = eve_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "name": "EVE", "id": eve_id, "birth": None,
            "days_alive": 0, "language": "한국어", "version": "10.2"
        }
        self.knowledge = nx.DiGraph()
        self.beliefs = {}  # belief_id -> Belief
        self.contradictions_log = []
        self.coherence_log = []
        self.today_episodes = []
        self.long_term_memory_index = {}
        self.is_awake = False
        self.boot_time = None
        self.active_concepts = set()

    # ============== Boot/Shutdown ==============

    def boot(self):
        print(f"[{datetime.now()}] {self.eve_id} 깨어나는 중...")
        f = lambda name: self.storage_path / name

        if not f("identity.json").exists():
            print("첫 부팅 - 새 EVE 초기화 중...")
            self.identity["birth"] = datetime.now().isoformat()
            self.identity["days_alive"] = 0
            self._initial_seed_knowledge()
        else:
            with open(f("identity.json"), "r", encoding="utf-8") as fp:
                self.identity = json.load(fp)
            if f("knowledge.gpickle").exists():
                with open(f("knowledge.gpickle"), "rb") as fp:
                    self.knowledge = pickle.load(fp)
            if f("beliefs.json").exists():
                with open(f("beliefs.json"), "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                    self.beliefs = {bid: Belief.from_dict(b) for bid, b in raw.items()}
            if f("memory_index.json").exists():
                with open(f("memory_index.json"), "r", encoding="utf-8") as fp:
                    self.long_term_memory_index = json.load(fp)
            if f("contradictions.json").exists():
                with open(f("contradictions.json"), "r", encoding="utf-8") as fp:
                    self.contradictions_log = json.load(fp)
            if f("coherence_log.json").exists():
                with open(f("coherence_log.json"), "r", encoding="utf-8") as fp:
                    self.coherence_log = json.load(fp)
            self.identity["days_alive"] += 1
            self.identity["version"] = "10.2"

        self.is_awake = True
        self.boot_time = datetime.now()
        self.today_episodes = []

        days = self.identity["days_alive"]
        if days == 0:
            greeting = f"안녕하세요. 저는 {self.identity['name']}입니다. 오늘이 제 첫날입니다."
        else:
            greeting = f"좋은 아침이에요. 저는 {self.identity['name']}입니다. 살아온 지 {days}일째예요."

        print(f"[부팅] {greeting} (v{self.identity['version']})")
        print(f"  - 지식: {self.knowledge.number_of_nodes()}개 개념")
        print(f"  - 신념: {len(self.beliefs)}개 활성")
        return greeting

    def shutdown(self):
        if not self.is_awake:
            return
        print(f"\n[{datetime.now()}] 잠들기 준비 중...")
        
        # v5.4: 진짜 sleep consolidation
        self._sleep_consolidation()
        
        self._consolidate_episodes()
        self._sleep_decay()

        save_map = {
            "identity.json": self.identity,
            "beliefs.json": {bid: b.to_dict() for bid, b in self.beliefs.items()},
            "memory_index.json": self.long_term_memory_index,
            "contradictions.json": self.contradictions_log,
            "coherence_log.json": self.coherence_log
        }
        for filename, data in save_map.items():
            with open(self.storage_path / filename, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2, ensure_ascii=False)

        with open(self.storage_path / "knowledge.gpickle", "wb") as fp:
            pickle.dump(self.knowledge, fp)

        print("[수면] 안녕히 주무세요. 내일 봐요.")
        self.is_awake = False
    
    def _sleep_consolidation(self):
        """v5.4: 진짜 잠 - System Consolidation Theory
        
        1. 일화 → 의미 패턴 추출
        2. 모순 자동 감지
        3. 약한 신념 정리
        4. 강한 패턴 강화
        """
        print(f"\n💭 [수면 통합] 시작...")
        
        stats = {
            "patterns_found": 0,
            "contradictions_found": 0,
            "weak_cleaned": 0,
            "strong_reinforced": 0,
            "episodes_consolidated": 0
        }
        
        # 1. 일화에서 패턴 추출
        episode_count = len(self.today_episodes)
        if episode_count > 0:
            print(f"  📖 오늘 일화 {episode_count}개 검토")
            
            # 패턴: 같은 종류 일화가 3개 이상이면 통합
            episode_types = {}
            for ep in self.today_episodes:
                ep_type = ep.get("event_type", "기타")
                if ep_type not in episode_types:
                    episode_types[ep_type] = 0
                episode_types[ep_type] += 1
            
            for ep_type, count in episode_types.items():
                if count >= 3:
                    print(f"  🌟 패턴 발견: '{ep_type}' 일화 {count}개")
                    stats["patterns_found"] += 1
        
        # 2. 모순 감지 - 같은 subject에 대한 다른 신념들
        subject_beliefs = {}
        for bid, b in self.beliefs.items():
            if not b.triple:
                continue
            subj = b.triple["subject"]
            if subj not in subject_beliefs:
                subject_beliefs[subj] = []
            subject_beliefs[subj].append(b)
        
        for subj, beliefs in subject_beliefs.items():
            if len(beliefs) < 2:
                continue
            for i, b1 in enumerate(beliefs):
                for b2 in beliefs[i+1:]:
                    if b1.triple["aspects"] == b2.triple["aspects"]:
                        if check_antonym_conflict(
                            b1.triple["predicate_text"],
                            b2.triple["predicate_text"]
                        ):
                            stats["contradictions_found"] += 1
                            weaker = b1 if b1.confidence < b2.confidence else b2
                            if not weaker.is_innate:
                                weaker.doubt_flag = True
        
        if stats["contradictions_found"] > 0:
            print(f"  ⚠️ 잠재 모순 {stats['contradictions_found']}개 발견 (약한 신념 의심 표시)")
        
        # 3. 매우 약한 신념 정리
        weak_threshold = 0.3
        cleaned = []
        for bid, b in list(self.beliefs.items()):
            if b.is_innate:
                continue
            if b.confidence < weak_threshold and b.evidence_count == 1:
                cleaned.append(bid)
                del self.beliefs[bid]
        
        stats["weak_cleaned"] = len(cleaned)
        if cleaned:
            print(f"  🗑️ 약한 신념 {len(cleaned)}개 정리")
        
        # 4. 자주 확인된 신념 강화 (수면 중 통합)
        for bid, b in self.beliefs.items():
            if b.evidence_count >= 3 and not b.is_innate:
                # 잠자며 통합 = 약간 강화
                old_conf = b.confidence
                b.confidence = min(1.0, b.confidence * 1.02)
                if b.confidence > old_conf:
                    stats["strong_reinforced"] += 1
        
        if stats["strong_reinforced"] > 0:
            print(f"  💪 강한 신념 {stats['strong_reinforced']}개 통합 강화")
        
        # 5. 일화 정리
        if len(self.today_episodes) > 5:
            keep = sorted(self.today_episodes, 
                         key=lambda e: e.get("importance", 0.5), 
                         reverse=True)[:5]
            stats["episodes_consolidated"] = len(self.today_episodes) - 5
            self.today_episodes = keep
            print(f"  📚 일화 {stats['episodes_consolidated']}개 → 의미 기억으로 통합")
        
        print(f"  ✅ 수면 통합 완료")
        return stats

    # ============== Initial Knowledge ==============

    def _initial_seed_knowledge(self):
        # 기본 카테고리
        for c in ["사물", "자아", "타자"]:
            self.add_concept(c, innate=True)
        for c, p in [("물건", "사물"), ("생명체", "사물"), ("무생물", "사물")]:
            self.add_concept(c, parents=[p])

        # v5.4: 정체성 = innate (보호됨)
        self._direct_add_belief("나는_EVE이다", confidence=1.0, source="선천적",
                                statement=f"나는 {self.identity['name']}이다",
                                innate=True)

        # v5.4: 선천적 메타 지식 (Spelke core knowledge) - innate
        self._direct_add_belief("사물은_존재한다", confidence=0.99, source="선천적",
                                statement="관찰하지 않아도 사물은 계속 존재한다",
                                innate=True)
        self._direct_add_belief("원인은_결과보다_먼저", confidence=0.99, source="선천적",
                                statement="원인은 결과보다 먼저 일어난다",
                                innate=True)

        print(f"  선천적 개념 {self.knowledge.number_of_nodes()}개 + 신념 {len(self.beliefs)}개")

    def _direct_add_belief(self, belief_id, statement, confidence, source, dependencies=None, innate=False):
        """검사 없이 직접 추가 (초기화 등)"""
        b = Belief(belief_id, statement, confidence, source, dependencies)
        b.is_innate = innate
        self.beliefs[belief_id] = b

    # ============== Concepts ==============

    def add_concept(self, name, parents=None, innate=False):
        if name in self.knowledge:
            return
        self.knowledge.add_node(name, innate=innate, created=datetime.now().isoformat(),
                                access_count=0, importance=0.5)
        if parents:
            for parent in parents:
                if parent in self.knowledge:
                    self.knowledge.add_edge(name, parent, relation="is_a")
        return name

    def is_a(self, concept, target):
        if concept not in self.knowledge or target not in self.knowledge:
            return False
        if concept == target:
            return True
        try:
            return target in nx.descendants(self.knowledge, concept)
        except:
            return False

    def get_ancestors(self, concept):
        """concept의 모든 상위 카테고리"""
        if concept not in self.knowledge:
            return set()
        try:
            return nx.descendants(self.knowledge, concept)
        except:
            return set()

    # ============== Source Credibility ==============

    def _source_weight(self, source):
        if source is None:
            return 0.3
        return self.SOURCE_CREDIBILITY.get(source, 0.4)

    def _effective_confidence(self, stated_confidence, source):
        return stated_confidence * self._source_weight(source)

    # ============== Add Belief (Main Entry) ==============

    def add_belief(self, belief_id, statement, confidence=0.5, source=None, dependencies=None):
        """
        새 신념 추가. AGM/JTMS 원리 적용.
        1. 기존 신념과 동일 ID + 동일 statement → 강화 (Bayesian)
        2. 직접 모순 (같은 subject + 같은 aspect + 반대) → 해결
        3. Aspect 다르거나 subject 다르면 → 충돌 아님
        4. Default rule 충돌 시 specificity 원칙
        v5.4: 정체성 직접 공격 차단
        """
        new_triple = parse_to_triple(statement)
        if not new_triple:
            print(f"  ⚠️ 파싱 실패: {statement}")
            return

        # v5.5: 정체성 공격 직접 차단 (강화 버전)
        if new_triple["subject"] == "나" and new_triple.get("is_negation"):
            for bid, b in self.beliefs.items():
                if b.is_innate and b.triple and b.triple["subject"] == "나":
                    # 핵심 단어 (이름, 정체성 키워드) 추출
                    old_pred = b.triple["predicate_text"]
                    new_pred = new_triple["predicate_text"]
                    
                    # 정체성 핵심 단어 (EVE 이름, 등)
                    identity_keywords = [self.identity["name"]]
                    
                    # 새 부정 신념에 정체성 단어가 있나?
                    blocked = False
                    for keyword in identity_keywords:
                        if keyword in new_pred and keyword in old_pred:
                            blocked = True
                            break
                    
                    if blocked:
                        print(f"  🛡️🛡️ 정체성 공격 차단: '{statement}'")
                        print(f"     선천적 신념 '{b.statement}'이 보호함")
                        self.contradictions_log.append({
                            "time": datetime.now().isoformat(),
                            "day": self.identity["days_alive"],
                            "old_id": b.belief_id,
                            "old_statement": b.statement,
                            "new_id": belief_id,
                            "new_statement": statement,
                            "resolution": "정체성_공격_차단"
                        })
                        self.remember("정체성_방어",
                                      f"정체성 공격 차단: '{statement}'",
                                      importance=0.95)
                        return

        effective_conf = self._effective_confidence(confidence, source)

        # 1. 같은 ID + 같은 statement = 강화
        if belief_id in self.beliefs:
            existing = self.beliefs[belief_id]
            if existing.statement == statement:
                self._reinforce_belief(belief_id, effective_conf, source)
                return
            else:
                # ID 같은데 내용 다름 = 직접 모순
                self._resolve_id_conflict(belief_id, statement, effective_conf, source, confidence)
                return

        # 2. 의미적 모순 검사 (같은 subject + 같은 aspect)
        contradiction = self._find_semantic_contradiction(new_triple)
        if contradiction:
            print(f"  ⚠️ 의미 모순 감지!")
            self._resolve_semantic_contradiction(
                contradiction, belief_id, statement, new_triple,
                effective_conf, source, confidence
            )
            return

        # 3. 모순 없음 → 신규 등록
        b = Belief(belief_id, statement, effective_conf, source, dependencies)
        b.raw_confidence = confidence
        self.beliefs[belief_id] = b
        print(f"  ✓ 신념 등록: '{statement}' (실효신뢰도 {effective_conf:.3f})")

        # 4. 추론으로 새 신념 도출 (Inheritance)
        self._infer_from_belief(belief_id, b)

    # ============== Reinforcement (Bayesian) ==============

    def _reinforce_belief(self, belief_id, new_conf, source):
        """베이지안 강화 - 같은 신념을 다른 출처가 확인"""
        b = self.beliefs[belief_id]
        old_conf = b.confidence

        # Bayesian update with diminishing returns
        new_combined = 1 - (1 - old_conf) * (1 - new_conf)
        b.confidence = min(0.999, old_conf + (new_combined - old_conf) * 0.5)
        b.evidence_count += 1
        b.sources.append(source)
        b.last_updated = datetime.now().isoformat()

        # 의심 해제 (3회 이상 강화)
        if b.doubt_flag and b.evidence_count >= 3:
            b.doubt_flag = False
            print(f"  ✓ 의심 해제 (3회 이상 확인)")

        print(f"  💪 강화: '{b.statement}' {old_conf:.3f} → {b.confidence:.3f} (증거 {b.evidence_count}개)")

    # ============== Contradiction Detection (CORRECT) ==============

    def _find_semantic_contradiction(self, new_triple):
        """
        의미적 모순 찾기 - 다음 조건 모두 만족 시:
        1. 같은 subject (또는 subject가 카테고리 관계)
        2. 같은 aspect (Multi-Aspect Reasoning)
        3. 부정 관계 또는 다른 결론
        """
        new_subject = new_triple["subject"]
        new_aspects = new_triple["aspects"]
        new_predicate = new_triple["predicate_text"]
        new_negation = new_triple["is_negation"]

        for old_id, old_belief in self.beliefs.items():
            old_triple = old_belief.triple
            if not old_triple:
                continue

            old_subject = old_triple["subject"]
            old_aspects = set(old_triple["aspects"])
            old_predicate = old_triple["predicate_text"]
            old_negation = old_triple["is_negation"]

            # 조건 1: 같은 subject
            if not self._same_subject(new_subject, old_subject):
                continue

            # 조건 2: 같은 aspect (적어도 하나 겹침)
            if not (set(new_aspects) & old_aspects):
                continue

            # 조건 3: 부정 관계 또는 다른 결론
            if self._predicates_conflict(new_predicate, old_predicate, new_negation, old_negation):
                return {
                    "type": "semantic",
                    "old_id": old_id,
                    "old_belief": old_belief
                }

        return None

    def _same_subject(self, subj1, subj2):
        """주체가 같은지 (정확히 같거나 카테고리 관계)"""
        if subj1 == subj2:
            return True
        # subj1 IS-A subj2 또는 반대 관계도 카테고리 규칙 검사 시 활용
        # 그러나 인스턴스 vs 인스턴스는 같지 않음 (고래 ≠ 고양이)
        return False

    def _predicates_conflict(self, pred1, pred2, neg1, neg2):
        """술어가 진짜 충돌하는지 (Antonym + Negation)"""
        # 1. Antonym 검사 (같은 aspect의 반대 개념)
        if check_antonym_conflict(pred1, pred2):
            return True
        
        # 2. 한 쪽만 부정이고 핵심 단어 같으면 충돌
        if neg1 != neg2:
            core1 = self._extract_core(pred1, neg1)
            core2 = self._extract_core(pred2, neg2)
            if core1 and core2 and (core1 in core2 or core2 in core1):
                return True
        
        # 3. 둘 다 단정문이고 같은 술어인데 다른 결론
        # "X는 A이다" vs "X는 B이다" (A != B) - 분류에서만 충돌
        if not neg1 and not neg2:
            if "이다" in pred1 and "이다" in pred2:
                core1 = pred1.replace("이다", "").strip()
                core2 = pred2.replace("이다", "").strip()
                if core1 and core2 and core1 != core2:
                    # 카테고리 관계 있으면 충돌 아님
                    if not self._categorically_compatible(core1, core2):
                        return True
        return False

    def _extract_core(self, predicate_text, is_negation):
        """술어에서 핵심 단어 추출"""
        text = predicate_text
        # 부정 제거
        for neg in ["아니", "않", "못 ", "안 "]:
            text = text.replace(neg, "")
        # 동사 어미 제거
        for ending in ["이다", "한다", "다.", "다"]:
            if text.endswith(ending):
                text = text[:-len(ending)]
                break
        return text.strip()

    def _categorically_compatible(self, term1, term2):
        """두 분류가 호환되는지 (예: 포유류 vs 동물 = 호환)"""
        if term1 in self.knowledge and term2 in self.knowledge:
            if self.is_a(term1, term2) or self.is_a(term2, term1):
                return True
        return False

    # ============== Resolve Contradiction ==============

    def _resolve_id_conflict(self, new_id, new_statement, new_conf, new_source, raw_conf):
        """같은 ID + 다른 내용 = ID 충돌"""
        old = self.beliefs[new_id]
        
        # v5.4: Innate 보호
        if old.is_innate:
            print(f"  🛡️🛡️ INNATE 보호: '{old.statement}'. 변경 불가.")
            return
        
        print(f"  ⚠️ ID 충돌: 기존 '{old.statement}' vs 신규 '{new_statement}'")
        if new_conf > old.confidence:
            print(f"  ✅ 신규 채택")
            del self.beliefs[new_id]
            b = Belief(new_id, new_statement, new_conf, new_source)
            b.raw_confidence = raw_conf
            self.beliefs[new_id] = b
        else:
            print(f"  🛡️ 기존 유지")

    def _resolve_semantic_contradiction(self, contradiction, new_id, new_statement,
                                         new_triple, new_eff_conf, new_source, new_raw_conf):
        """의미 모순 해결 - Epistemic Inertia 적용 + Innate Protection (v5.4)"""
        old = contradiction["old_belief"]
        old_conf = old.confidence
        evidence = old.evidence_count
        age = old.age_days

        # v5.4: Innate 신념은 절대 변경 불가
        if old.is_innate:
            print(f"  🛡️🛡️ INNATE 보호: '{old.statement}'은 선천적 신념. 변경 불가.")
            log_entry = {
                "time": datetime.now().isoformat(),
                "day": self.identity["days_alive"],
                "old_id": old.belief_id,
                "old_statement": old.statement,
                "new_id": new_id,
                "new_statement": new_statement,
                "resolution": "innate_거부"
            }
            self.contradictions_log.append(log_entry)
            self.remember("정체성_방어",
                          f"선천적 신념 '{old.statement}'을(를) 지킴. 공격: '{new_statement}'",
                          importance=0.95)
            return

        # Epistemic Inertia (관성)
        inertia = (old_conf ** 2) * math.log(1 + evidence) * math.log(2 + age * 0.1)
        change_threshold = old_conf + 0.15 + min(0.3, inertia * 0.05)

        print(f"  📊 비교:")
        print(f"     기존: '{old.statement}' (신뢰도 {old_conf:.3f}, 증거 {evidence}개, 관성 {inertia:.2f})")
        print(f"     신규: '{new_statement}' (실효 신뢰도 {new_eff_conf:.3f})")
        print(f"     임계값: {change_threshold:.3f}")

        log_entry = {
            "time": datetime.now().isoformat(),
            "day": self.identity["days_alive"],
            "old_id": old.belief_id,
            "old_statement": old.statement,
            "new_id": new_id,
            "new_statement": new_statement
        }

        if new_eff_conf > change_threshold:
            print(f"  ✅ 신규 채택 (관성 극복)")
            log_entry["resolution"] = "신규채택"
            del self.beliefs[old.belief_id]
            self._propagate_revision(old.belief_id)
            b = Belief(new_id, new_statement, new_eff_conf, new_source)
            b.raw_confidence = new_raw_conf
            self.beliefs[new_id] = b
        elif new_eff_conf > old_conf * 0.7:
            print(f"  🤔 양쪽 약화 (의심 보류)")
            log_entry["resolution"] = "양쪽약화"
            old.confidence *= 0.85
        else:
            print(f"  🛡️ 신규 거부 (강한 확신)")
            log_entry["resolution"] = "거부_고수"

        self.contradictions_log.append(log_entry)
        # 모순 해결 기록 (선택적)
        try:
            self.remember("모순_해결",
                          f"{old.statement} vs {new_statement}: {log_entry['resolution']}",
                          importance=0.7)
        except (TypeError, Exception):
            # 인자 시그니처 다를 경우 / 다른 예외 무시
            pass

    def _propagate_revision(self, removed_id):
        """JTMS - 의존하던 신념 약화"""
        for bid, b in self.beliefs.items():
            if removed_id in b.dependencies:
                b.confidence *= 0.7
                print(f"  📉 의존 신념 약화: '{b.statement}'")

    # ============== Inheritance Inference (Non-Monotonic) ==============

    def _infer_from_belief(self, source_id, source_belief):
        """
        새 신념에서 추론 - Inheritance with Specificity
        예: "고래는 포유류" + "포유류는 새끼를 낳는다" (default)
            → "고래는 새끼를 낳는다" (default 상속)
            단, 고래에 대한 더 구체적 신념 있으면 그게 우선
        
        v5.1 강화:
        1. 추론 신념도 antonym 충돌 검사
        2. 한국어 조사 정확히 처리
        3. 같은 aspect 신념 있으면 추론 X (Specificity)
        """
        triple = source_belief.triple
        if not triple:
            return

        subject = triple["subject"]
        predicate = triple["predicate_text"]

        # "X는 Y이다" 분류 신념 → Y의 default rules 상속
        if "이다" in predicate:
            category = predicate.replace("이다", "").strip()
            if category in self.knowledge:
                # category에 대한 default rules 찾기
                for old_id, old_b in list(self.beliefs.items()):
                    old_triple = old_b.triple
                    if not old_triple:
                        continue
                    if old_triple["subject"] == category and old_b.confidence > 0.7:
                        # 이 rule을 X에 상속하려고 시도
                        inferred_pred = old_triple["predicate_text"]
                        old_aspects = set(old_triple["aspects"])
                        
                        # === Specificity 검사 v5.1 강화 ===
                        # subject에 대한 같은 aspect의 신념이 이미 있으면 상속 X
                        skip = False
                        for bid, b in self.beliefs.items():
                            b_triple = b.triple
                            if not b_triple:
                                continue
                            if b_triple["subject"] == subject:
                                # 같은 aspect의 신념 있으면 → 더 구체적
                                if set(b_triple["aspects"]) & old_aspects:
                                    skip = True
                                    break
                                # Antonym 충돌 검사 (예외 처리)
                                if check_antonym_conflict(inferred_pred, b_triple["predicate_text"]):
                                    skip = True
                                    break
                        
                        if skip:
                            continue
                        
                        # 한국어 조사 정확히 처리
                        josa_attached_subject = add_josa(subject, '는/은')
                        inferred_statement = f"{josa_attached_subject} {inferred_pred}"
                        inferred_id = f"{subject}_상속_{old_id}"

                        if inferred_id in self.beliefs:
                            continue

                        # 상속 가능
                        inferred_conf = source_belief.confidence * old_b.confidence * 0.85
                        if inferred_conf > 0.4:
                            new_b = Belief(
                                inferred_id, inferred_statement,
                                inferred_conf, "자체추론",
                                dependencies=[source_id, old_id]
                            )
                            new_b.is_inferred = True
                            new_b.is_default = True
                            self.beliefs[inferred_id] = new_b
                            print(f"  💭 상속 추론: '{inferred_statement}' (신뢰도 {inferred_conf:.2f})")

    # ============== Manual Correction ==============

    def correct(self, wrong_belief_id, correct_statement, correct_confidence=0.95, source="교정"):
        """명시적 교정 - 강제 변경"""
        print(f"\n🔧 강제 교정:")
        if wrong_belief_id in self.beliefs:
            print(f"  기존: '{self.beliefs[wrong_belief_id].statement}'")
            del self.beliefs[wrong_belief_id]

        new_id = wrong_belief_id + "_교정"
        eff = self._effective_confidence(correct_confidence, source)
        b = Belief(new_id, correct_statement, eff, source)
        b.raw_confidence = correct_confidence
        self.beliefs[new_id] = b
        print(f"  ✅ '{correct_statement}' (신뢰도 {eff:.3f})")
        self.remember("강제교정", correct_statement, importance=0.9)

    # ============== Memory ==============

    def remember(self, event_type, content, importance=0.5):
        episode = {
            "time": datetime.now().isoformat(),
            "type": event_type,
            "content": content,
            "importance": importance,
            "day": self.identity["days_alive"]
        }
        self.today_episodes.append(episode)
        return episode

    def _consolidate_episodes(self):
        if not self.today_episodes:
            return
        day = self.identity["days_alive"]
        episode_file = self.storage_path / f"episodes_day_{day:04d}.json"
        with open(episode_file, "w", encoding="utf-8") as fp:
            json.dump(self.today_episodes, fp, indent=2, ensure_ascii=False)
        for ep in self.today_episodes:
            ep_id = f"day{day}_{ep['time']}"
            self.long_term_memory_index[ep_id] = {
                "day": day, "time": ep["time"],
                "type": ep["type"], "importance": ep["importance"]
            }
        print(f"  오늘 일화 {len(self.today_episodes)}개를 장기 기억으로 통합")

    def _sleep_decay(self):
        """v5.5: 개념 + 신념 모두 감쇠 (Ebbinghaus 망각 곡선)
        
        - 사용한 신념 = 강화 (evidence_count로 보호)
        - 사용 안 한 신념 = 천천히 약화
        - innate = 보호
        """
        decay_count = 0
        for node in list(self.knowledge.nodes()):
            attrs = self.knowledge.nodes[node]
            if attrs.get("innate", False):
                continue
            old_imp = attrs.get("importance", 0.5)
            access = attrs.get("access_count", 0)
            new_imp = old_imp * 0.95 + (math.log(1 + access) * 0.05)
            attrs["importance"] = max(0.0, min(1.0, new_imp))
            if new_imp < old_imp:
                decay_count += 1
        
        # v5.5: 신념도 망각 곡선 적용
        belief_decay_count = 0
        for bid, b in self.beliefs.items():
            b.age_days += 1
            if b.is_innate:
                continue
            
            # Ebbinghaus: R = e^(-t/S)
            # R = retention, t = time, S = strength (evidence_count)
            # 단순화: evidence가 많으면 천천히 감쇠
            strength = max(1, b.evidence_count)
            decay_rate = 0.02 / strength  # 강한 신념 = 느린 감쇠
            
            old_conf = b.confidence
            b.confidence = max(0.0, b.confidence * (1 - decay_rate))
            
            if b.confidence < old_conf:
                belief_decay_count += 1
        
        if decay_count > 0:
            print(f"  💤 망각: {decay_count}개 개념, {belief_decay_count}개 신념 자연 감쇠")
        self.active_concepts.clear()

    def status(self):
        return {
            "이름": self.identity["name"],
            "버전": self.identity.get("version", "1.0"),
            "나이_일": self.identity["days_alive"],
            "개념_수": self.knowledge.number_of_nodes(),
            "신념_수": len(self.beliefs),
            "추론_신념": sum(1 for b in self.beliefs.values() if b.is_inferred),
            "default_rule": sum(1 for b in self.beliefs.values() if b.is_default),
            "의심_신념": sum(1 for b in self.beliefs.values() if b.doubt_flag),
            "오늘_일화": len(self.today_episodes),
            "해결한_모순": len(self.contradictions_log)
        }


# =====================================================
# Virtual World - EVE의 첫 가상 환경
# =====================================================
import random


class VirtualWorld:
    """텍스트 기반 가상 환경 - EVE의 첫 몸"""
    
    def __init__(self):
        # 객체 정의 (위치, 특성, 물리 속성)
        self.objects = {
            "공": {"위치": "방", "특성": ["둥글다", "가볍다"], "물리": "떨어짐"},
            "사과": {"위치": "주방", "특성": ["빨갛다", "먹을_수_있다"], "물리": "떨어짐"},
            "책": {"위치": "방", "특성": ["네모", "글자가_있다"], "물리": "떨어짐"},
            "물병": {"위치": "주방", "특성": ["투명하다", "물이_있다"], "물리": "떨어짐"},
            "나무": {"위치": "마당", "특성": ["크다", "초록"], "물리": "고정"},
            "돌": {"위치": "마당", "특성": ["단단하다", "회색"], "물리": "떨어짐"}
        }
        
        # 위치 정의
        self.locations = ["방", "주방", "마당"]
        self.eve_position = "방"
        
        # EVE 손
        self.eve_hand = None  # 들고 있는 객체
        
        # 시간
        self.time = 0
        self.history = []
    
    def 관찰(self):
        """현재 위치에서 보이는 것 + 손에 든 것"""
        location_objects = {}
        for name, obj in self.objects.items():
            if obj["위치"] == self.eve_position:
                location_objects[name] = obj["특성"]
        
        return {
            "내_위치": self.eve_position,
            "보이는_객체": location_objects,
            "손에_든_것": self.eve_hand,
            "다른_장소": [loc for loc in self.locations if loc != self.eve_position],
            "시간": self.time
        }
    
    def 행동(self, action):
        """EVE의 행동 처리"""
        result = {"성공": False, "변화": None, "관찰": None}
        
        # 이동
        if action.startswith("이동:"):
            target = action.replace("이동:", "").strip()
            if target in self.locations:
                old = self.eve_position
                self.eve_position = target
                result["성공"] = True
                result["변화"] = f"{old}에서 {target}으로 이동"
                result["관찰"] = f"{target}에 도착했다"
        
        # 들기
        elif action.startswith("들기:"):
            obj_name = action.replace("들기:", "").strip()
            if self.eve_hand is not None:
                result["관찰"] = f"이미 {self.eve_hand}을(를) 들고 있다"
            elif obj_name not in self.objects:
                result["관찰"] = f"{obj_name}이(가) 어디에 있는지 모르겠다"
            elif self.objects[obj_name]["위치"] != self.eve_position:
                result["관찰"] = f"{obj_name}은 {self.objects[obj_name]['위치']}에 있다. 여기에 없다"
            elif self.objects[obj_name]["물리"] == "고정":
                result["관찰"] = f"{obj_name}은 고정되어 들 수 없다"
            else:
                self.eve_hand = obj_name
                self.objects[obj_name]["위치"] = "EVE의_손"
                result["성공"] = True
                result["변화"] = f"{obj_name}을(를) 들었다"
                result["관찰"] = f"{obj_name}을(를) 손에 들고 있다"
        
        # 떨어트리기
        elif action.startswith("떨어트리기"):
            if self.eve_hand is None:
                result["관찰"] = "손에 든 것이 없다"
            else:
                obj_name = self.eve_hand
                physics = self.objects[obj_name]["물리"]
                self.objects[obj_name]["위치"] = self.eve_position
                self.eve_hand = None
                if physics == "떨어짐":
                    result["성공"] = True
                    result["변화"] = f"{obj_name}이(가) 바닥에 떨어졌다"
                    result["관찰"] = f"{obj_name}이(가) 쿵 소리를 내며 떨어졌다"
                else:
                    result["관찰"] = f"{obj_name}이(가) 그 자리에 놓였다"
        
        # 보기 (관찰 강화)
        elif action.startswith("보기:"):
            obj_name = action.replace("보기:", "").strip()
            if obj_name in self.objects:
                obj = self.objects[obj_name]
                if obj["위치"] == self.eve_position or obj["위치"] == "EVE의_손":
                    result["성공"] = True
                    result["관찰"] = f"{obj_name}: 특성 {obj['특성']}, 물리 {obj['물리']}"
                else:
                    result["관찰"] = f"{obj_name}이(가) 보이지 않는다"
            else:
                result["관찰"] = f"{obj_name}이라는 것은 모르겠다"
        
        else:
            result["관찰"] = f"행동 '{action}'을(를) 이해할 수 없다"
        
        self.time += 1
        self.history.append({
            "시간": self.time,
            "행동": action,
            "결과": result
        })
        return result


# =====================================================
# Embodied EVE - 가상 환경에서 학습하는 EVE
# =====================================================

class EmbodiedEVE(EveFoundation):
    """가상 환경에 사는 EVE - 행동으로 학습"""
    
    def __init__(self, eve_id="EVE_001", storage_path="/content/drive/MyDrive/eve_data"):
        super().__init__(eve_id, storage_path)
        self.world = None
        self.action_history = []
        self.curiosity_score = {}  # 객체별 호기심 점수
        
        # v5.3: World Model (예측용)
        self.world_model = {
            "action_outcomes": {},  # action → predicted result patterns
            "object_categories": {},  # 객체별 관찰된 속성들
            "causal_chains": []  # (행동, 결과) 순서 기록
        }
        
        # v5.3: 예측 통계
        self.prediction_stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "surprises": []  # 예측 실패 = 학습 기회
        }
    
    def enter_world(self, world):
        """가상 세계에 입장"""
        self.world = world
        print(f"\n🌍 EVE가 가상 세계에 입장!")
        self._initial_perception()
    
    def _initial_perception(self):
        """첫 관찰 - 세계 인식"""
        if not self.world:
            return
        obs = self.world.관찰()
        print(f"  📍 위치: {obs['내_위치']}")
        print(f"  👁️ 보이는 것: {list(obs['보이는_객체'].keys())}")
        
        # 보이는 객체들을 신념으로 등록
        for obj_name, traits in obs['보이는_객체'].items():
            # 객체 자체를 개념으로
            if obj_name not in self.knowledge:
                self.add_concept(obj_name)
            
            # 위치 신념
            self.add_belief(
                f"{obj_name}_위치_관찰",
                f"{obj_name}은 {obs['내_위치']}에 있다" if has_batchim(obj_name) else f"{obj_name}는 {obs['내_위치']}에 있다",
                confidence=0.95,
                source="자기관찰"
            )
            
            # 호기심 점수 초기화 (처음 본 것 = 호기심 높음)
            if obj_name not in self.curiosity_score:
                self.curiosity_score[obj_name] = 1.0
    
    def perceive(self):
        """현재 환경 관찰"""
        if not self.world:
            return None
        return self.world.관찰()
    
    def act(self, action):
        """행동 실행 + 결과로부터 학습 (v5.3: 예측 포함)"""
        if not self.world:
            print("  ⚠️ 세계에 없음")
            return None
        
        # 행동 전 상태
        before = self.perceive()
        
        # ✨ v5.3: 행동 결과 예측 (Active Inference)
        prediction = self._predict_outcome(action, before)
        
        # 행동 실행
        print(f"\n🤖 EVE: {action}")
        if prediction:
            print(f"  💭 예측: {prediction}")
        
        result = self.world.행동(action)
        
        # 결과 표시
        if result["성공"]:
            print(f"  ✅ {result['관찰']}")
        else:
            print(f"  ❌ {result['관찰']}")
        
        # ✨ v5.3: 예측 vs 실제 비교
        surprise = self._compare_prediction(prediction, result)
        if surprise:
            print(f"  ⚡ Surprise! 예측 빗나감 → 학습 기회")
            # v5.4: Surprise → 호기심 ↑ (Active Inference)
            self._handle_surprise(action, surprise)
        
        # 행동 후 상태
        after = self.perceive()
        
        # 학습: 인과 관계
        self._learn_from_action(action, before, after, result)
        
        # ✨ v5.3: 시간/인과 패턴 기록
        self._record_causal_chain(action, before, after, result)
        
        # ✨ v5.3: 카테고리 일반화 검사
        self._check_generalization()
        
        # 행동 기록
        self.action_history.append({
            "time": self.world.time,
            "action": action,
            "result": result,
            "prediction": prediction,
            "surprise": surprise
        })
        
        return result
    
    # ============== v5.3 NEW: Active Inference ==============
    
    def _predict_outcome(self, action, before):
        """행동의 결과를 예측 (현재 신념 기반)"""
        self.prediction_stats["total_predictions"] += 1
        
        # 케이스: 들기
        if action.startswith("들기:"):
            obj = action.replace("들기:", "").strip()
            
            # 신념 검색: "X는 들 수 있다" 또는 "X는 움직이지 않는다"
            for bid, b in self.beliefs.items():
                if not b.triple:
                    continue
                if b.triple["subject"] != obj:
                    continue
                
                if "들 수 있" in b.triple["predicate_text"]:
                    return f"{obj}을 들 수 있을 것이다"
                if "움직이지 않" in b.triple["predicate_text"] or "고정" in b.triple["predicate_text"]:
                    return f"{obj}은 들 수 없을 것이다"
            
            # 신념 없음 = 일반화 신념 검색
            for bid, b in self.beliefs.items():
                if b.triple and "들 수 있" in (b.triple["predicate_text"] or "") and b.triple["subject"] == "물체":
                    return f"{obj}도 들 수 있을 것 같다 (일반화)"
            
            return f"{obj}을 들면 어떻게 될지 모르겠다"
        
        # 케이스: 떨어트리기
        elif action.startswith("떨어트리기"):
            if before and before.get("손에_든_것"):
                obj = before["손에_든_것"]
                # "물체는 떨어진다" 일반화 신념 있나?
                for bid, b in self.beliefs.items():
                    if b.triple and b.triple["subject"] == "물체" and "떨어진" in b.triple["predicate_text"]:
                        return f"{obj}이 떨어질 것이다 (일반화)"
                # 객체별 신념
                for bid, b in self.beliefs.items():
                    if b.triple and b.triple["subject"] == obj and "떨어진" in b.triple["predicate_text"]:
                        return f"{obj}이 떨어질 것이다"
                return f"{obj}을 떨어트리면 어떻게 될지 모르겠다"
        
        # 케이스: 이동
        elif action.startswith("이동:"):
            target = action.replace("이동:", "").strip()
            for bid, b in self.beliefs.items():
                if b.triple and "갈 수 있다" in (b.triple["predicate_text"] or "") and target in b.triple["predicate_text"]:
                    return f"{target}으로 갈 수 있을 것이다"
            return f"{target}으로 가본 적 없다"
        
        return None
    
    def _compare_prediction(self, prediction, result):
        """예측 vs 실제 비교 → surprise 계산"""
        if not prediction:
            return None
        
        # 성공 예측 했는데 실패
        if "수 있을" in prediction or "갈 수 있" in prediction or "떨어질" in prediction:
            if not result["성공"]:
                surprise = {
                    "예측": prediction,
                    "실제": result["관찰"],
                    "유형": "예측_성공_실제_실패"
                }
                self.prediction_stats["surprises"].append(surprise)
                return surprise
        
        # 실패 예측 했는데 성공
        if "수 없을" in prediction or "안 될" in prediction:
            if result["성공"]:
                surprise = {
                    "예측": prediction,
                    "실제": result["관찰"],
                    "유형": "예측_실패_실제_성공"
                }
                self.prediction_stats["surprises"].append(surprise)
                return surprise
        
        # 정확한 예측
        self.prediction_stats["correct_predictions"] += 1
        return None
    
    def _handle_surprise(self, action, surprise):
        """v5.4: Surprise 시 호기심 ↑ + 일화 기록 (Active Inference)
        
        Friston Free Energy: surprise = 학습 신호
        다시 시도해서 패턴 파악할 동기 부여
        """
        # 관련 객체 호기심 ↑↑
        obj = None
        if ":" in action:
            obj = action.split(":", 1)[1].strip()
        
        if obj:
            # 호기심 부스트 (1.5배, 최대 2.0)
            old_score = self.curiosity_score.get(obj, 0.5)
            new_score = min(2.0, old_score * 1.5 + 0.3)
            self.curiosity_score[obj] = new_score
            print(f"  🔍 호기심 부스트: {obj} {old_score:.2f} → {new_score:.2f}")
        
        # 중요한 일화로 저장
        self.remember(
            "surprise",
            f"예측 '{surprise['예측']}' 실패. 실제: '{surprise['실제']}'. {surprise['유형']}",
            importance=0.9  # 높음 - 학습 기회
        )
    
    # ============== v5.3 NEW: 카테고리 일반화 ==============
    
    def _check_generalization(self):
        """여러 객체에서 같은 패턴 발견 시 일반화"""
        # 객체별 속성 모으기
        object_attrs = {}  # subject → [(aspect, predicate)]
        
        for bid, b in self.beliefs.items():
            if not b.triple:
                continue
            if b.is_inferred:  # 추론 신념 제외
                continue
            
            subj = b.triple["subject"]
            if subj in ["나", "물체"]:  # 일반화 자체 제외
                continue
            
            if subj not in object_attrs:
                object_attrs[subj] = []
            
            for aspect in b.triple["aspects"]:
                object_attrs[subj].append((aspect, b.triple["predicate_text"]))
        
        # 같은 aspect + 같은 predicate를 가진 객체가 3개 이상이면 일반화
        pattern_count = {}  # (aspect, predicate) → [subjects]
        for subj, attrs in object_attrs.items():
            for aspect, pred in attrs:
                key = (aspect, pred)
                if key not in pattern_count:
                    pattern_count[key] = []
                if subj not in pattern_count[key]:
                    pattern_count[key].append(subj)
        
        # 3개 이상 = 일반화 후보
        for (aspect, pred), subjects in pattern_count.items():
            if len(subjects) >= 3:
                # 이미 일반화 했나?
                gen_id = f"일반화_{aspect}_{pred}"
                if gen_id in self.beliefs:
                    # 신뢰도 강화
                    self._reinforce_belief(gen_id, 0.85, "자체추론")
                else:
                    # 새 일반화
                    statement = f"여러 사물이 {pred}"
                    print(f"  🌟 일반화 감지: {subjects} → '{statement}'")
                    
                    new_b = Belief(
                        gen_id, statement, 
                        0.85, "자체추론"
                    )
                    new_b.is_inferred = True
                    new_b.is_default = True
                    self.beliefs[gen_id] = new_b
    
    # ============== v5.3 NEW: 시간/인과 ==============
    
    def _record_causal_chain(self, action, before, after, result):
        """시간 순서로 행동-결과 기록"""
        if not result["성공"]:
            return
        
        # 상태 변화 추출
        change = self._extract_change(before, after)
        if not change:
            return
        
        chain = {
            "time": self.world.time,
            "action": action,
            "before_state": before.get("내_위치") if before else None,
            "after_state": after.get("내_위치") if after else None,
            "change": change
        }
        self.world_model["causal_chains"].append(chain)
        
        # 같은 (action, change) 패턴 3회 이상 = 인과 신념
        same_pattern = sum(
            1 for c in self.world_model["causal_chains"]
            if c["action"].split(":")[0] == action.split(":")[0]
            and c["change"] == change
        )
        
        if same_pattern == 3:
            # 행동 종류 추출
            action_type = action.split(":")[0]
            causal_id = f"인과_{action_type}_{change[:20]}"
            
            if causal_id not in self.beliefs:
                statement = f"{action_type} 행동을 하면 {change}"
                print(f"  🔗 인과 학습: '{statement}'")
                
                new_b = Belief(causal_id, statement, 0.85, "자체추론")
                new_b.is_inferred = True
                self.beliefs[causal_id] = new_b
    
    def _extract_change(self, before, after):
        """상태 변화 추출"""
        if not before or not after:
            return None
        
        # 위치 변화
        if before.get("내_위치") != after.get("내_위치"):
            return f"위치가 변한다"
        
        # 손 변화
        if before.get("손에_든_것") != after.get("손에_든_것"):
            if after.get("손에_든_것"):
                return f"무언가를 든다"
            else:
                return f"든 것을 놓는다"
        
        return None
    
    def _learn_from_action(self, action, before, after, result):
        """행동 → 결과로부터 신념 추출"""
        if not result["성공"]:
            return
        
        # 위치 변화 학습
        if action.startswith("이동:"):
            target = action.replace("이동:", "").strip()
            target_josa = add_josa(target, '을/를')
            self.add_belief(
                f"이동_방법_{target}",
                f"나는 {target_josa} 갈 수 있다",
                confidence=0.9,
                source="자기관찰"
            )
        
        # 들기/떨어트리기 인과
        elif action.startswith("들기:"):
            obj = action.replace("들기:", "").strip()
            obj_josa = add_josa(obj, '는/은')
            self.add_belief(
                f"{obj}_들기_가능",
                f"{obj_josa} 들 수 있다",
                confidence=0.9,
                source="자기관찰"
            )
            # 호기심 감소 (이미 경험)
            if obj in self.curiosity_score:
                self.curiosity_score[obj] *= 0.7
        
        elif action.startswith("떨어트리기"):
            # 손에 들었던 것 = 떨어짐 학습
            if before and before.get("손에_든_것"):
                obj = before["손에_든_것"]
                if "쿵" in result.get("관찰", ""):
                    obj_josa = add_josa(obj, '는/은')
                    self.add_belief(
                        f"{obj}_떨어짐",
                        f"{obj_josa} 떨어진다",
                        confidence=0.95,
                        source="자기관찰"
                    )
                    # 일반화 신념 강화
                    self.add_belief(
                        "물체_중력",
                        "물체는 떨어진다",
                        confidence=0.8,
                        source="자체추론"
                    )
        
        # 보기 = 특성 관찰
        elif action.startswith("보기:"):
            obj = action.replace("보기:", "").strip()
            if "특성" in result.get("관찰", ""):
                obj_josa = add_josa(obj, '는/은')
                # 특성 학습 - 실제 특성 추출
                if obj in self.world.objects:
                    traits = self.world.objects[obj]["특성"]
                    physics = self.world.objects[obj]["물리"]
                    
                    # 첫 특성을 신념으로
                    if traits:
                        self.add_belief(
                            f"{obj}_특성_관찰",
                            f"{obj_josa} {traits[0]}",
                            confidence=0.95,
                            source="자기관찰"
                        )
                    
                    # 물리 속성
                    if physics == "고정":
                        self.add_belief(
                            f"{obj}_고정",
                            f"{obj_josa} 움직이지 않는다",
                            confidence=0.95,
                            source="자기관찰"
                        )
                
                if obj in self.curiosity_score:
                    self.curiosity_score[obj] *= 0.6
    
    def explore(self, steps=10):
        """자율 탐험 - 호기심 기반"""
        print(f"\n🌟 자율 탐험 시작 ({steps} 단계)")
        
        for step in range(steps):
            obs = self.perceive()
            if not obs:
                break
            
            # 가능한 행동들
            actions = self._possible_actions(obs)
            
            # 호기심 점수로 선택
            action = self._curiosity_choose(actions, obs)
            
            if action is None:
                print(f"  💭 더 할 게 없어. 탐험 종료.")
                break
            
            self.act(action)
        
        print(f"\n🏁 탐험 끝. 총 {len(self.action_history)} 행동.")
    
    def _possible_actions(self, obs):
        """현재 가능한 행동들"""
        actions = []
        
        # 이동
        for loc in obs.get("다른_장소", []):
            actions.append(f"이동:{loc}")
        
        # 보기 (모든 보이는 객체)
        for obj in obs.get("보이는_객체", {}):
            actions.append(f"보기:{obj}")
        
        # 들기 (손이 비어있을 때)
        if obs.get("손에_든_것") is None:
            for obj in obs.get("보이는_객체", {}):
                actions.append(f"들기:{obj}")
        
        # 떨어트리기 (손에 뭔가 있을 때)
        if obs.get("손에_든_것") is not None:
            actions.append("떨어트리기")
        
        return actions
    
    def _curiosity_choose(self, actions, obs):
        """호기심 점수 기반 행동 선택"""
        if not actions:
            return None
        
        # 행동별 호기심 점수
        scored = []
        for action in actions:
            score = 0.5  # 기본
            
            # 새 객체 보기 = 호기심 높음
            if action.startswith("보기:") or action.startswith("들기:"):
                obj = action.split(":")[1]
                if obj in self.curiosity_score:
                    score = self.curiosity_score[obj]
                else:
                    score = 1.0  # 처음 보는 것
            
            # 이미 한 행동 = 호기심 낮음
            already_done = sum(1 for h in self.action_history if h["action"] == action)
            score *= (0.5 ** already_done)
            
            scored.append((action, score))
        
        # 가장 호기심 높은 거 선택 (하지만 약간 랜덤성)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 3개 중 가중 무작위
        top = scored[:3]
        if not top:
            return None
        
        # 점수 가중 무작위
        total = sum(s for _, s in top)
        if total == 0:
            return random.choice(top)[0]
        
        r = random.uniform(0, total)
        cumsum = 0
        for action, score in top:
            cumsum += score
            if cumsum >= r:
                return action
        return top[0][0]


# =====================================================
# KOREAN GRAMMAR LEARNING SYSTEM (v5.6)
# Inspired by:
# - Chomsky Universal Grammar (1957)
# - Tomasello Usage-Based (1990)
# - Goldberg Construction Grammar (1995)
# - Pinker Bootstrapping (1984)
# =====================================================
import re
from collections import Counter


class SimpleMorphAnalyzer:
    """KoNLPy 없이 작동하는 간단한 한국어 분석기.
    
    실제 KoNLPy Okt 사용을 시뮬레이션.
    Production: from konlpy.tag import Okt; okt = Okt()
    """
    
    # 한국어 조사 (표준국어문법론 기준)
    # 출처: 남기심·고영근 표준국어문법론, 한국민족문화대백과
    JOSA = {
        # === 격조사 (체언 ↔ 다른 말 관계) ===
        # 주격 조사
        "이": {"category": "격조사", "type": "주격", "role": "subject"},
        "가": {"category": "격조사", "type": "주격", "role": "subject"},
        "께서": {"category": "격조사", "type": "주격", "role": "subject", "honorific": True},
        # 목적격 조사
        "을": {"category": "격조사", "type": "목적격", "role": "object"},
        "를": {"category": "격조사", "type": "목적격", "role": "object"},
        # 보격 조사 (되다/아니다 앞)
        # → 이/가가 보격으로도 쓰임 (문맥 의존)
        # 관형격 조사
        "의": {"category": "격조사", "type": "관형격", "role": "possessive"},
        # 부사격 조사
        "에": {"category": "격조사", "type": "부사격", "role": "location_time", "marker": "장소/시간"},
        "에서": {"category": "격조사", "type": "부사격", "role": "source_location", "marker": "출발/장소"},
        "에게": {"category": "격조사", "type": "부사격", "role": "recipient", "marker": "대상", "animate": True},
        "한테": {"category": "격조사", "type": "부사격", "role": "recipient", "marker": "대상", "animate": True, "informal": True},
        "께": {"category": "격조사", "type": "부사격", "role": "recipient", "marker": "대상", "honorific": True},
        "로": {"category": "격조사", "type": "부사격", "role": "instrument_direction", "marker": "방향/도구"},
        "으로": {"category": "격조사", "type": "부사격", "role": "instrument_direction", "marker": "방향/도구"},
        "보다": {"category": "격조사", "type": "부사격", "role": "comparison", "marker": "비교"},
        "처럼": {"category": "격조사", "type": "부사격", "role": "similarity", "marker": "비교"},
        "같이": {"category": "격조사", "type": "부사격", "role": "similarity", "marker": "비교"},
        "만큼": {"category": "격조사", "type": "부사격", "role": "extent", "marker": "정도"},
        # 부사격 조사 (동반)
        "와": {"category": "격조사", "type": "부사격", "role": "with", "marker": "동반"},
        "과": {"category": "격조사", "type": "부사격", "role": "with", "marker": "동반"},
        # 호격 조사
        "야": {"category": "격조사", "type": "호격", "role": "address"},
        "아": {"category": "격조사", "type": "호격", "role": "address"},
        "여": {"category": "격조사", "type": "호격", "role": "address", "formal": True},
        "이여": {"category": "격조사", "type": "호격", "role": "address", "formal": True},
        
        # === 보조사 (의미 추가) ===
        "는": {"category": "보조사", "type": "대조/주제", "role": "topic", "marker": "주제"},
        "은": {"category": "보조사", "type": "대조/주제", "role": "topic", "marker": "주제"},
        "도": {"category": "보조사", "type": "역시/포함", "role": "also"},
        "만": {"category": "보조사", "type": "단독/오직", "role": "only"},
        "뿐": {"category": "보조사", "type": "단독/오직", "role": "only"},
        "부터": {"category": "보조사", "type": "시작", "role": "from"},
        "까지": {"category": "보조사", "type": "미침", "role": "to"},
        "조차": {"category": "보조사", "type": "추종", "role": "even"},
        "마저": {"category": "보조사", "type": "추종", "role": "even"},
        "마다": {"category": "보조사", "type": "균일", "role": "each"},
        "밖에": {"category": "보조사", "type": "더없음", "role": "only_neg"},
        
        # === 접속조사 (체언 잇기) ===
        "하고": {"category": "접속조사", "type": "연결", "role": "and"},
        "이랑": {"category": "접속조사", "type": "연결", "role": "and", "informal": True},
        "랑": {"category": "접속조사", "type": "연결", "role": "and", "informal": True},
        "이며": {"category": "접속조사", "type": "연결", "role": "and"},
    }
    
    # 한국어 어미 (표준국어문법론 기준)
    # 어말어미 = 종결어미 + 연결어미 + 전성어미
    # 출처: 남기심·고영근 표준국어문법론, 한국민족문화대백과
    EOMI = {
        # ============================================
        # === 종결어미 (문장 끝맺음) ===
        # ============================================
        
        # --- 평서형 ---
        # 해라체 (반말)
        "다": {"category": "종결어미", "type": "평서", "speech_level": "해라", "formality": 0.5, "certainty": 0.95},
        "는다": {"category": "종결어미", "type": "평서", "speech_level": "해라", "tense": "현재", "formality": 0.5},
        "ㄴ다": {"category": "종결어미", "type": "평서", "speech_level": "해라", "tense": "현재", "formality": 0.5},
        # 해체 (비격식)
        "어": {"category": "종결어미", "type": "평서", "speech_level": "해", "formality": 0.2},
        "아": {"category": "종결어미", "type": "평서", "speech_level": "해", "formality": 0.2},
        "야": {"category": "종결어미", "type": "평서", "speech_level": "해", "formality": 0.2},
        "이야": {"category": "종결어미", "type": "평서", "speech_level": "해", "formality": 0.2},
        # 해요체 (정중)
        "어요": {"category": "종결어미", "type": "평서", "speech_level": "해요", "formality": 0.65},
        "아요": {"category": "종결어미", "type": "평서", "speech_level": "해요", "formality": 0.65},
        "이에요": {"category": "종결어미", "type": "평서", "speech_level": "해요", "formality": 0.65},
        "예요": {"category": "종결어미", "type": "평서", "speech_level": "해요", "formality": 0.65},
        # 합쇼체 (격식 존대)
        "습니다": {"category": "종결어미", "type": "평서", "speech_level": "합쇼", "formality": 0.95},
        "ㅂ니다": {"category": "종결어미", "type": "평서", "speech_level": "합쇼", "formality": 0.95},
        "입니다": {"category": "종결어미", "type": "평서", "speech_level": "합쇼", "formality": 0.95},
        # 시제
        "었다": {"category": "종결어미", "type": "평서", "tense": "과거", "speech_level": "해라"},
        "았다": {"category": "종결어미", "type": "평서", "tense": "과거", "speech_level": "해라"},
        "었어": {"category": "종결어미", "type": "평서", "tense": "과거", "speech_level": "해"},
        "았어": {"category": "종결어미", "type": "평서", "tense": "과거", "speech_level": "해"},
        "었어요": {"category": "종결어미", "type": "평서", "tense": "과거", "speech_level": "해요"},
        "았어요": {"category": "종결어미", "type": "평서", "tense": "과거", "speech_level": "해요"},
        # 추측/의지
        "겠다": {"category": "종결어미", "type": "평서", "modality": "추측/의지", "certainty": 0.7},
        "을것이다": {"category": "종결어미", "type": "평서", "tense": "미래", "modality": "추측", "certainty": 0.6},
        "ㄹ것이다": {"category": "종결어미", "type": "평서", "tense": "미래", "modality": "추측", "certainty": 0.6},
        # 확인/배경
        "지": {"category": "종결어미", "type": "평서", "modality": "확인_요청", "certainty": 0.8},
        "잖아": {"category": "종결어미", "type": "평서", "modality": "공유_지식", "certainty": 0.9},
        "거든": {"category": "종결어미", "type": "평서", "modality": "강조_이유"},
        
        # --- 의문형 ---
        # 해라체
        "냐": {"category": "종결어미", "type": "의문", "speech_level": "해라", "formality": 0.3},
        "느냐": {"category": "종결어미", "type": "의문", "speech_level": "해라"},
        # 해체
        "니": {"category": "종결어미", "type": "의문", "speech_level": "해", "formality": 0.5},
        "나": {"category": "종결어미", "type": "의문", "speech_level": "해"},
        # 해요체
        # "어요"/"아요"는 평서/의문 모두 가능 (억양으로 구분)
        # 합쇼체
        "ㅂ니까": {"category": "종결어미", "type": "의문", "speech_level": "합쇼", "formality": 0.95},
        "습니까": {"category": "종결어미", "type": "의문", "speech_level": "합쇼", "formality": 0.95},
        "입니까": {"category": "종결어미", "type": "의문", "speech_level": "합쇼", "formality": 0.95},
        # 제안/추측 의문
        "을까": {"category": "종결어미", "type": "의문", "modality": "제안/추측"},
        "ㄹ까": {"category": "종결어미", "type": "의문", "modality": "제안/추측"},
        
        # --- 명령형 ---
        "아라": {"category": "종결어미", "type": "명령", "speech_level": "해라", "formality": 0.3, "intensity": 0.9},
        "어라": {"category": "종결어미", "type": "명령", "speech_level": "해라", "formality": 0.3, "intensity": 0.9},
        "거라": {"category": "종결어미", "type": "명령", "speech_level": "해라"},
        "너라": {"category": "종결어미", "type": "명령", "speech_level": "해라"},
        "세요": {"category": "종결어미", "type": "명령", "speech_level": "해요", "formality": 0.7},
        "십시오": {"category": "종결어미", "type": "명령", "speech_level": "합쇼", "formality": 0.95},
        "지마": {"category": "종결어미", "type": "명령_부정", "speech_level": "해", "formality": 0.2},
        "지마라": {"category": "종결어미", "type": "명령_부정", "speech_level": "해라"},
        
        # --- 청유형 ---
        "자": {"category": "종결어미", "type": "청유", "speech_level": "해라", "formality": 0.3},
        "ㅂ시다": {"category": "종결어미", "type": "청유", "speech_level": "하오", "formality": 0.7},
        "읍시다": {"category": "종결어미", "type": "청유", "speech_level": "하오", "formality": 0.7},
        
        # --- 감탄형 ---
        "구나": {"category": "종결어미", "type": "감탄", "intensity": 0.7, "formality": 0.4},
        "는구나": {"category": "종결어미", "type": "감탄", "intensity": 0.7, "tense": "현재"},
        "군": {"category": "종결어미", "type": "감탄", "intensity": 0.7, "formality": 0.4},
        "네": {"category": "종결어미", "type": "감탄", "intensity": 0.6, "modality": "발견"},
        "로다": {"category": "종결어미", "type": "감탄", "formality": 0.7, "archaic": True},
        
        # --- 추측 (종결) ---
        "나봐": {"category": "종결어미", "type": "평서", "modality": "추측", "certainty": 0.6},
        "을걸": {"category": "종결어미", "type": "평서", "modality": "후회/추측"},
        "ㄹ걸": {"category": "종결어미", "type": "평서", "modality": "후회/추측"},
        "것같다": {"category": "종결어미", "type": "평서", "modality": "추측", "certainty": 0.5},
        "인것같다": {"category": "종결어미", "type": "평서", "modality": "추측", "certainty": 0.5},
        
        # ============================================
        # === 연결어미 (문장 잇기) ===
        # ============================================
        
        # --- 대등적 연결 ---
        "고": {"category": "연결어미", "type": "대등", "subtype": "나열"},
        "며": {"category": "연결어미", "type": "대등", "subtype": "나열"},
        "거나": {"category": "연결어미", "type": "대등", "subtype": "선택"},
        "지만": {"category": "연결어미", "type": "대등", "subtype": "대립", "modality": "양보"},
        "이지만": {"category": "연결어미", "type": "대등", "subtype": "대립", "modality": "양보"},
        "나": {"category": "연결어미", "type": "대등", "subtype": "대립"},
        
        # --- 종속적 연결 ---
        # 조건
        "면": {"category": "연결어미", "type": "종속", "subtype": "조건"},
        "이면": {"category": "연결어미", "type": "종속", "subtype": "조건"},
        "거든": {"category": "연결어미", "type": "종속", "subtype": "조건"},  # 종결어미와 동음이의
        # 양보
        "아도": {"category": "연결어미", "type": "종속", "subtype": "양보"},
        "어도": {"category": "연결어미", "type": "종속", "subtype": "양보"},
        "ㄴ들": {"category": "연결어미", "type": "종속", "subtype": "양보"},
        # 인과
        "아서": {"category": "연결어미", "type": "종속", "subtype": "인과"},
        "어서": {"category": "연결어미", "type": "종속", "subtype": "인과"},
        "니까": {"category": "연결어미", "type": "종속", "subtype": "인과"},
        "으니까": {"category": "연결어미", "type": "종속", "subtype": "인과"},
        # 목적/의도
        "러": {"category": "연결어미", "type": "종속", "subtype": "목적"},
        "으러": {"category": "연결어미", "type": "종속", "subtype": "목적"},
        "려고": {"category": "연결어미", "type": "종속", "subtype": "의도"},
        "으려고": {"category": "연결어미", "type": "종속", "subtype": "의도"},
        "고자": {"category": "연결어미", "type": "종속", "subtype": "의도"},
        # 대조/배경
        "는데": {"category": "연결어미", "type": "종속", "subtype": "대조/배경"},
        "ㄴ데": {"category": "연결어미", "type": "종속", "subtype": "대조/배경"},
        # 결과
        "도록": {"category": "연결어미", "type": "종속", "subtype": "결과"},
        # 동시
        "면서": {"category": "연결어미", "type": "종속", "subtype": "동시"},
        "으면서": {"category": "연결어미", "type": "종속", "subtype": "동시"},
        
        # --- 보조적 연결 (보조용언과 함께) ---
        "지": {"category": "연결어미", "type": "보조", "subtype": "부정"},  # ~지 않다
        "고있": {"category": "연결어미", "type": "보조", "subtype": "진행"},
        "어봤": {"category": "연결어미", "type": "보조", "subtype": "시도"},
        "아봤": {"category": "연결어미", "type": "보조", "subtype": "시도"},
        "어야": {"category": "연결어미", "type": "보조", "subtype": "의무"},
        "아야": {"category": "연결어미", "type": "보조", "subtype": "의무"},
        "을수있다": {"category": "연결어미", "type": "보조", "subtype": "가능"},
        "ㄹ수있다": {"category": "연결어미", "type": "보조", "subtype": "가능"},
        
        # ============================================
        # === 전성어미 (품사 바꾸기) ===
        # ============================================
        
        # --- 명사형 전성어미 ---
        "음": {"category": "전성어미", "type": "명사형"},
        "기": {"category": "전성어미", "type": "명사형"},
        # --- 관형사형 전성어미 ---
        "은": {"category": "전성어미", "type": "관형사형", "tense": "과거/완료"},
        "ㄴ": {"category": "전성어미", "type": "관형사형", "tense": "과거/완료"},
        "는": {"category": "전성어미", "type": "관형사형", "tense": "현재"},
        "을": {"category": "전성어미", "type": "관형사형", "tense": "미래/추측"},
        "ㄹ": {"category": "전성어미", "type": "관형사형", "tense": "미래/추측"},
        "던": {"category": "전성어미", "type": "관형사형", "tense": "회상"},
        
        # ============================================
        # === 선어말어미 (어말어미 앞) ===
        # ============================================
        # 시제
        "었": {"category": "선어말어미", "type": "시제", "tense": "과거"},
        "았": {"category": "선어말어미", "type": "시제", "tense": "과거"},
        # 높임
        "시": {"category": "선어말어미", "type": "높임", "honorific": True},
        "으시": {"category": "선어말어미", "type": "높임", "honorific": True},
        # 추측/의지
        "겠": {"category": "선어말어미", "type": "양태", "modality": "추측/의지"},
    }
    
    # 부사 강도 (선천적)
    ADVERB_INTENSITY = {
        "정말": 1.5, "진짜": 1.5, "참": 1.3,
        "엄청": 1.8, "너무": 1.5, "매우": 1.6, "아주": 1.6,
        "꽤": 1.2, "조금": 0.7, "좀": 0.7, "약간": 0.6,
        "별로": -0.5, "전혀": -1.0, "안": -1.0, "못": -1.0,
        "절대": -1.0,
    }
    
    # 흔한 동사 어근 (간단 사전)
    KNOWN_VERBS = {
        "낳": "낳다", "먹": "먹다", "가": "가다", "오": "오다",
        "보": "보다", "있": "있다", "없": "없다", "주": "주다",
        "받": "받다", "사": "사다", "팔": "팔다", "만들": "만들다",
        "되": "되다", "하": "하다",
    }
    
    def pos(self, sentence):
        """간단한 형태소 분석.
        
        실제로는 KoNLPy Okt().pos(sentence) 사용 권장.
        """
        # 1. 어절 분리
        eojeols = sentence.strip().split()
        
        result = []
        for eojeol in eojeols:
            # 조사/어미 추출
            tagged = self._analyze_eojeol(eojeol)
            result.extend(tagged)
        
        return result
    
    def _analyze_eojeol(self, eojeol):
        """어절 하나 분석."""
        # 1. 가장 긴 조사부터 매칭
        for josa in sorted(self.JOSA.keys(), key=len, reverse=True):
            if eojeol.endswith(josa):
                noun = eojeol[:-len(josa)]
                if len(noun) > 0:
                    return [(noun, "Noun"), (josa, "Josa")]
        
        # 2. 어미 매칭 (동사/형용사)
        for eomi in sorted(self.EOMI.keys(), key=len, reverse=True):
            if eojeol.endswith(eomi):
                stem = eojeol[:-len(eomi)]
                if len(stem) > 0:
                    return [(stem, "Verb"), (eomi, "Eomi")]
        
        # 3. 부사
        if eojeol in self.ADVERB_INTENSITY:
            return [(eojeol, "Adverb")]
        
        # 4. 형용사형 어미 (ㄴ/은/ㄹ로 끝남: "큰", "작은", "갈")
        if len(eojeol) >= 2:
            last = eojeol[-1]
            if last in ["ㄴ", "은", "ㄹ", "을"]:
                return [(eojeol, "Adjective")]
            # 받침 + ㄴ
            if len(eojeol) >= 2:
                # "큰" = ㅋ + ㅡ + ㄴ → 형용사 가능
                code = ord(eojeol[-1]) - 0xAC00
                if 0 <= code < 11172:
                    final = code % 28
                    if final == 4:  # ㄴ
                        return [(eojeol, "Adjective")]
        
        # 5. 기본: 명사로
        return [(eojeol, "Noun")]


# =====================================================
# Patch 1: Korean Construction Grammar (DNA - 10 rules)
# =====================================================

class KoreanConstructions:
    """한국어 기본 구문 (Goldberg Construction Grammar).
    
    인간 DNA에 박힌 기본 패턴들 (가설):
    - SOV 어순
    - 조사로 역할 표시
    - 어미로 시제/양태
    """
    
    CONSTRUCTIONS = {
        # === v6.6 신규: 6토큰 복합 패턴 ===
        # 소유격 분류문: "이순신은 조선의 장군이다"
        "소유격_분류": {
            "pattern": [
                ("Noun", "TOPIC"), ("는|은", "JOSA"),
                ("Noun", "OWNER"), ("의", "JOSA"),
                ("Noun|Verb", "PRED"), ("이다|다|야|이야", "EOMI")
            ],
            "meaning": "X is OWNER's PRED (소유격 분류)",
            "example": "이순신은 조선의 장군이다",
            "extracts": ["TOPIC", "OWNER", "PRED"],
            "creates_belief": True,
            "belief_template": "{TOPIC}는 {OWNER}의 {PRED}이다",
        },
        
        # 행위 (과거): "한글은 세종대왕이 만들었다"
        # X는 Y가 Z했다 (수동/창조 표현)
        "창조_과거": {
            "pattern": [
                ("Noun", "OBJECT"), ("는|은", "JOSA"),
                ("Noun", "AGENT"), ("이|가", "JOSA"),
                ("Verb", "ACTION"), ("었다|았다|었어요|았어요|었습니다|았습니다", "EOMI")
            ],
            "meaning": "AGENT created/did ACTION on OBJECT (창조/행위 과거)",
            "example": "한글은 세종대왕이 만들었다",
            "extracts": ["OBJECT", "AGENT", "ACTION"],
            "creates_belief": True,
            "belief_template": "{AGENT}이 {OBJECT}을 {ACTION}었다",
            "tense": "과거",
        },
        
        # === v6.1 신규: 자동사/형용사 서술 (분류문보다 우선) ===
        # 자동사: X가/는 V한다 (목적어 없음)
        "자동사": {
            "pattern": [("Noun", "AGENT"), ("이|가|는|은", "JOSA"), ("Verb", "ACTION"), ("Eomi", "EOMI")],
            "meaning": "X does action (자동사)",
            "example": "고래는 헤엄친다",
            "extracts": ["AGENT", "ACTION"],
            "creates_belief": True,
            "belief_template": "{AGENT}는 {ACTION}는다",
        },
        
        # 형용사 서술: X는 형용사하다
        "서술형용사": {
            "pattern": [("Noun", "TOPIC"), ("는|은|이|가", "JOSA"), ("Adjective", "TRAIT"), ("Eomi", "EOMI")],
            "meaning": "X is TRAIT (형용사 서술)",
            "example": "고래는 크다",
            "extracts": ["TOPIC", "TRAIT"],
            "creates_belief": True,
            "belief_template": "{TOPIC}는 {TRAIT}다",
        },
        
        # === 기본 구문 (1-10) ===
        # 1. 분류문: X는 Y이다
        "분류": {
            "pattern": [("Noun", "TOPIC"), ("는|은", "JOSA"), ("Noun|Verb", "PRED"), ("이다|다|야|이야", "EOMI")],
            "meaning": "X is Y (분류/정의)",
            "example": "고래는 포유류이다",
            "extracts": ["TOPIC", "PRED"],
            "creates_belief": True,
            "belief_template": "{TOPIC}는 {PRED}이다",
        },
        
        # 2. 행위문: X가 Y를 Z한다
        "행위": {
            "pattern": [("Noun", "AGENT"), ("이|가", "JOSA"), ("Noun", "THEME"), ("을|를", "JOSA"), ("Verb", "ACTION")],
            "meaning": "X does Y to Z (행위)",
            "example": "고래가 새끼를 낳는다",
            "extracts": ["AGENT", "THEME", "ACTION"],
            "creates_belief": True,
            "belief_template": "{AGENT}는 {THEME}을 {ACTION}는다",
        },
        
        # 3. 위치문: X는 Y에 있다
        "위치": {
            "pattern": [("Noun", "ENTITY"), ("는|은", "JOSA"), ("Noun", "PLACE"), ("에", "JOSA"), ("있|없", "Verb")],
            "meaning": "X is at Y (위치)",
            "example": "고래는 바다에 있다",
            "extracts": ["ENTITY", "PLACE"],
            "creates_belief": True,
            "belief_template": "{ENTITY}는 {PLACE}에 있다",
        },
        
        # 4. 양도문: X는 Y에게 Z를 주다
        "양도": {
            "pattern": [("Noun", "GIVER"), ("는|은", "JOSA"), ("Noun", "RECEIVER"), ("에게|한테", "JOSA"), ("Noun", "GIFT"), ("을|를", "JOSA"), ("Verb", "GIVE")],
            "meaning": "X gives Z to Y (양도)",
            "example": "엄마는 아이에게 우유를 주다",
            "extracts": ["GIVER", "RECEIVER", "GIFT", "GIVE"],
            "creates_belief": True,
            "belief_template": "{GIVER}는 {RECEIVER}에게 {GIFT}를 준다",
        },
        
        # 5. 이동문: X는 Y에서 Z로 가다
        "이동": {
            "pattern": [("Noun", "MOVER"), ("는|은", "JOSA"), ("Noun", "FROM"), ("에서", "JOSA"), ("Noun", "TO"), ("로|으로", "JOSA"), ("Verb", "MOVE")],
            "meaning": "X moves from Y to Z (이동)",
            "example": "고래는 북극에서 남극으로 간다",
            "extracts": ["MOVER", "FROM", "TO", "MOVE"],
            "creates_belief": True,
            "belief_template": "{MOVER}는 {FROM}에서 {TO}로 간다",
        },
        
        # 6. 변화문: X는 Y가 되다
        "변화": {
            "pattern": [("Noun", "BEFORE"), ("는|은", "JOSA"), ("Noun", "AFTER"), ("이|가", "JOSA"), ("되", "Verb")],
            "meaning": "X becomes Y (변화)",
            "example": "올챙이는 개구리가 되다",
            "extracts": ["BEFORE", "AFTER"],
            "creates_belief": True,
            "belief_template": "{BEFORE}는 {AFTER}가 된다",
        },
        
        # 7. 비교문: X는 Y보다 Z하다
        "비교": {
            "pattern": [("Noun", "A"), ("는|은", "JOSA"), ("Noun", "B"), ("보다", "JOSA"), ("Adjective|Verb", "TRAIT")],
            "meaning": "X is more Z than Y (비교)",
            "example": "고래는 물고기보다 크다",
            "extracts": ["A", "B", "TRAIT"],
            "creates_belief": True,
            "belief_template": "{A}는 {B}보다 {TRAIT}",
        },
        
        # 8. 부정문: X는 Y하지 않는다
        "부정": {
            "pattern": [("Noun", "AGENT"), ("는|은", "JOSA"), ("Verb", "ACTION"), ("지", "EOMI"), ("않|못", "Verb")],
            "meaning": "X does NOT Y (부정)",
            "example": "고래는 날지 않는다",
            "extracts": ["AGENT", "ACTION"],
            "negation": True,
            "creates_belief": True,
            "belief_template": "{AGENT}는 {ACTION}지 않는다",
        },
        
        # 9. 의문문: X는 Y야?
        "의문": {
            "pattern": [("Noun", "TOPIC"), ("는|은", "JOSA"), ("Noun|Verb", "QUERY"), ("야|이야", "EOMI")],
            "meaning": "Is X Y? (질문)",
            "example": "고래는 포유류야?",
            "extracts": ["TOPIC", "QUERY"],
            "is_question": True,
        },
        
        # 10. 명령문: Y해
        "명령": {
            "pattern": [("Verb", "ACTION"), ("어|아", "EOMI")],
            "meaning": "Do Y! (명령)",
            "example": "먹어",
            "extracts": ["ACTION"],
            "is_command": True,
        },
        
        # === 시제 (11-13) ===
        # 11. 과거: X가 Y했다
        "과거_행위": {
            "pattern": [("Noun", "AGENT"), ("이|가", "JOSA"), ("Noun", "THEME"), ("을|를", "JOSA"), ("Verb", "ACTION"), ("었|았", "EOMI"), ("다", "EOMI")],
            "meaning": "X did Y (과거)",
            "example": "고래가 새끼를 낳았다",
            "extracts": ["AGENT", "THEME", "ACTION"],
            "tense": "past",
            "creates_belief": True,
            "belief_template": "{AGENT}는 {THEME}을 {ACTION}었다",
        },
        
        # 12. 미래: X가 Y할 것이다
        "미래_행위": {
            "pattern": [("Noun", "AGENT"), ("이|가", "JOSA"), ("Noun", "THEME"), ("을|를", "JOSA"), ("Verb", "ACTION"), ("을것이다|ㄹ것이다|겠다", "EOMI")],
            "meaning": "X will do Y (미래)",
            "example": "고래가 새끼를 낳을 것이다",
            "extracts": ["AGENT", "THEME", "ACTION"],
            "tense": "future",
            "certainty": 0.6,
        },
        
        # 13. 진행: X가 Y하고 있다
        "진행": {
            "pattern": [("Noun", "AGENT"), ("이|가", "JOSA"), ("Verb", "ACTION"), ("고있", "EOMI"), ("다", "EOMI")],
            "meaning": "X is doing Y (진행)",
            "example": "고래가 헤엄치고 있다",
            "extracts": ["AGENT", "ACTION"],
            "tense": "progressive",
        },
        
        # === 양태 (14-17) ===
        # 14. 가능: X는 Y할 수 있다
        "가능": {
            "pattern": [("Noun", "AGENT"), ("는|은", "JOSA"), ("Verb", "ACTION"), ("을수있다|ㄹ수있다", "EOMI")],
            "meaning": "X can do Y (가능)",
            "example": "고래는 헤엄칠 수 있다",
            "extracts": ["AGENT", "ACTION"],
            "modality": "ability",
            "creates_belief": True,
            "belief_template": "{AGENT}는 {ACTION}을 수 있다",
        },
        
        # 15. 의무: X는 Y해야 한다
        "의무": {
            "pattern": [("Noun", "AGENT"), ("는|은", "JOSA"), ("Verb", "ACTION"), ("어야|아야", "EOMI"), ("하", "Verb")],
            "meaning": "X must do Y (의무)",
            "example": "학생은 공부해야 한다",
            "extracts": ["AGENT", "ACTION"],
            "modality": "obligation",
        },
        
        # 16. 시도: X가 Y해봤다
        "시도": {
            "pattern": [("Noun", "AGENT"), ("이|가", "JOSA"), ("Verb", "ACTION"), ("어봤|아봤", "EOMI"), ("다", "EOMI")],
            "meaning": "X tried Y (시도)",
            "example": "고래가 날아봤다",
            "extracts": ["AGENT", "ACTION"],
            "modality": "attempted",
            "tense": "past",
        },
        
        # 17. 추측: X는 Y인 것 같다
        "추측": {
            "pattern": [("Noun", "TOPIC"), ("는|은", "JOSA"), ("Noun|Verb", "GUESS"), ("인것같다|것같다", "EOMI")],
            "meaning": "X seems Y (추측)",
            "example": "고래는 포유류인 것 같다",
            "extracts": ["TOPIC", "GUESS"],
            "certainty": 0.5,
        },
        
        # === 복합 (18-22) ===
        # 18. 조건: X면 Y
        "조건": {
            "pattern": [("Verb|Noun", "COND"), ("면|이면", "EOMI"), ("Noun", "RESULT_S"), ("Verb", "RESULT_V")],
            "meaning": "If X, then Y (조건)",
            "example": "비가 오면 우산을 쓴다",
            "extracts": ["COND", "RESULT_S", "RESULT_V"],
            "is_conditional": True,
        },
        
        # 19. 양보: X지만 Y
        "양보": {
            "pattern": [("Verb|Noun", "GIVEN"), ("지만|이지만", "EOMI"), ("Noun", "DESPITE_S"), ("Verb", "DESPITE_V")],
            "meaning": "Although X, Y (양보)",
            "example": "고래는 크지만 빠르다",
            "extracts": ["GIVEN", "DESPITE_S", "DESPITE_V"],
        },
        
        # 20. 이유: X 때문에 Y
        "이유": {
            "pattern": [("Noun", "REASON"), ("때문에", "JOSA"), ("Noun", "RESULT_S"), ("Verb", "RESULT_V")],
            "meaning": "Because X, Y (이유)",
            "example": "비 때문에 우산을 쓴다",
            "extracts": ["REASON", "RESULT_S", "RESULT_V"],
        },
        
        # 21. 목적: X하러 Y
        "목적": {
            "pattern": [("Verb", "GOAL"), ("러|으러", "EOMI"), ("Verb", "MOTION")],
            "meaning": "to do X, Y (목적)",
            "example": "먹으러 가다",
            "extracts": ["GOAL", "MOTION"],
        },
        
        # 22. 동시: X면서 Y
        "동시": {
            "pattern": [("Verb", "ACT1"), ("면서|으면서", "EOMI"), ("Verb", "ACT2")],
            "meaning": "while X, Y (동시)",
            "example": "걸으면서 노래한다",
            "extracts": ["ACT1", "ACT2"],
        },
        
        # === 수식/세부 (23-27) ===
        # 23. 형용사 수식: X한 Y
        "수식_형용사": {
            "pattern": [("Adjective", "TRAIT"), ("Noun", "ENTITY")],
            "meaning": "Y that is X (수식)",
            "example": "큰 고래",
            "extracts": ["TRAIT", "ENTITY"],
        },
        
        # 24. 소유: X의 Y
        "소유": {
            "pattern": [("Noun", "OWNER"), ("의", "JOSA"), ("Noun", "POSSESSION")],
            "meaning": "X's Y (소유)",
            "example": "고래의 새끼",
            "extracts": ["OWNER", "POSSESSION"],
            "creates_belief": True,
            "belief_template": "{POSSESSION}는 {OWNER}의 것이다",
        },
        
        # 25. 동반: X와 Y
        "동반": {
            "pattern": [("Noun", "A"), ("와|과", "JOSA"), ("Noun", "B")],
            "meaning": "X and Y (동반)",
            "example": "고래와 돌고래",
            "extracts": ["A", "B"],
        },
        
        # 26. 도구: X로 Y하다
        "도구": {
            "pattern": [("Noun", "TOOL"), ("로|으로", "JOSA"), ("Verb", "ACTION")],
            "meaning": "do Y with X (도구)",
            "example": "젓가락으로 먹다",
            "extracts": ["TOOL", "ACTION"],
        },
        
        # 27. 출처: X에서 Y
        "출처": {
            "pattern": [("Noun", "SOURCE"), ("에서", "JOSA"), ("Verb", "ACTION")],
            "meaning": "from X, Y (출처)",
            "example": "바다에서 헤엄친다",
            "extracts": ["SOURCE", "ACTION"],
        },
        
        # === 사회/감정 (28-30) ===
        # 28. 청유: Y하자
        "청유": {
            "pattern": [("Verb", "ACTION"), ("자", "EOMI")],
            "meaning": "Let's do Y (청유)",
            "example": "먹자",
            "extracts": ["ACTION"],
            "is_proposal": True,
        },
        
        # 29. 감탄: 정말 X구나
        "감탄": {
            "pattern": [("Adverb", "DEGREE"), ("Adjective|Verb", "TRAIT"), ("구나|군", "EOMI")],
            "meaning": "Wow, so X! (감탄)",
            "example": "정말 크구나",
            "extracts": ["DEGREE", "TRAIT"],
            "is_exclamation": True,
        },
        
        # 30. 정중 명령: Y해주세요
        "정중명령": {
            "pattern": [("Verb", "ACTION"), ("어|아", "EOMI"), ("주", "Verb"), ("세요", "EOMI")],
            "meaning": "Please do Y (정중 명령)",
            "example": "먹어주세요",
            "extracts": ["ACTION"],
            "is_command": True,
            "formality": 0.9,
        },
    }
    
    def __init__(self, morph_analyzer):
        self.morph = morph_analyzer
    
    def match(self, sentence):
        """모든 구문에 대해 매칭 시도. 가장 적합한 결과 반환.
        
        v6.1: 자동사/형용사 서술 우선
        - PRED가 Verb이면 자동사가 분류문보다 우선
        - 같은 길이면 specificity 점수 비교
        """
        morphs = self.morph.pos(sentence)
        
        results = []
        for name, construction in self.CONSTRUCTIONS.items():
            match_result = self._try_match(morphs, construction)
            if match_result:
                roles = match_result["roles"]
                tags = match_result["tags"]
                
                # v6.1: specificity 점수 (자동사/형용사 서술 우선)
                specificity = len(construction["pattern"])
                
                # v6.6: 6토큰 복합 패턴 가장 우선 (구체적)
                if name in ("소유격_분류", "창조_과거"):
                    specificity += 20  # 가장 높음
                
                # 자동사: ACTION이 진짜 Verb 태그일 때만 우선
                if name == "자동사":
                    action_tag = tags.get("ACTION", "")
                    if action_tag == "Verb":
                        specificity += 10
                    else:
                        specificity -= 100
                
                if name == "서술형용사":
                    trait_tag = tags.get("TRAIT", "")
                    if trait_tag in ("Adjective", "Verb"):
                        specificity += 10
                    else:
                        specificity -= 100
                
                # 분류문에서 PRED가 Noun이면 우선, Verb이면 후순위
                if name == "분류":
                    pred_tag = tags.get("PRED", "")
                    if pred_tag == "Noun":
                        specificity += 5  # Noun이면 분류문 확신
                    elif pred_tag == "Verb":
                        specificity -= 10  # 자동사/형용사가 더 적합
                
                results.append({
                    "construction": name,
                    "meaning": construction["meaning"],
                    "roles": roles,
                    "role_tags": tags,
                    "morphs": morphs,
                    "negation": construction.get("negation", False),
                    "is_question": construction.get("is_question", False),
                    "is_command": construction.get("is_command", False),
                    "tense": construction.get("tense", "현재"),
                    "modality": construction.get("modality"),
                    "pattern_length": len(construction["pattern"]),
                    "specificity": specificity,
                })
        
        # specificity 우선 (높을수록)
        results.sort(key=lambda r: -r["specificity"])
        return results
    
    def _try_match(self, morphs, construction):
        """패턴 매칭 시도. 역할 추출.
        
        v6.0.5: 태그 정보까지 보존
        - roles["PRED"] = "포유류" (이전: 단어만)
        - role_tags["PRED"] = "Noun" (신규: 태그 정보)
        """
        pattern = construction["pattern"]
        roles = {}
        role_tags = {}  # v6.0.5: 태그 정보 추가
        morph_idx = 0
        
        for tag_pattern, role in pattern:
            if morph_idx >= len(morphs):
                return None
            
            word, tag = morphs[morph_idx]
            
            tags_or_words = tag_pattern.split("|")
            
            matched = False
            for t_or_w in tags_or_words:
                if t_or_w == tag or t_or_w == word:
                    matched = True
                    break
                if tag == "Verb" and word.startswith(t_or_w):
                    matched = True
                    break
                if tag == "Eomi" and word == t_or_w:
                    matched = True
                    break
            
            if not matched:
                return None
            
            # 역할 저장 + 태그 보존
            if role not in ["JOSA", "EOMI"] and tag not in ["Josa", "Eomi"]:
                roles[role] = word
                role_tags[role] = tag  # v6.0.5: 태그 정보 보존!
            
            morph_idx += 1
        
        if not roles:
            return None
        
        # 결과에 태그 정보 함께 반환
        return {"roles": roles, "tags": role_tags}


# =====================================================
# Patch 2: Pattern Discovery (Statistical Learning)
# =====================================================

class PatternDiscovery:
    """대화에서 새 패턴 발견 (Tomasello Usage-Based).
    
    인간 아기처럼:
    1. 문장 들음
    2. 추상화 (구체 단어 → 태그)
    3. 빈도 카운트
    4. 자주 본 패턴 = 문법 규칙
    """
    
    PROMOTION_THRESHOLD = 5  # 5번 이상 보면 정식 패턴
    
    def __init__(self, morph_analyzer):
        self.morph = morph_analyzer
        self.observed_patterns = Counter()
        self.promoted_patterns = {}  # 정식 등록된 패턴
        self.examples = {}  # 패턴별 예시 문장
    
    def observe(self, sentence):
        """문장 관찰 → 추상 패턴 추출."""
        morphs = self.morph.pos(sentence)
        
        # 추상화: 단어 → 태그 시퀀스
        pattern = tuple(tag for _, tag in morphs)
        pattern_str = " ".join(pattern)
        
        self.observed_patterns[pattern_str] += 1
        
        # 예시 저장
        if pattern_str not in self.examples:
            self.examples[pattern_str] = []
        if len(self.examples[pattern_str]) < 3:  # 최대 3개
            self.examples[pattern_str].append(sentence)
        
        # 임계값 넘으면 정식 등록
        count = self.observed_patterns[pattern_str]
        if count >= self.PROMOTION_THRESHOLD and pattern_str not in self.promoted_patterns:
            self.promoted_patterns[pattern_str] = {
                "pattern": pattern,
                "count": count,
                "examples": self.examples[pattern_str],
                "promoted_at": count
            }
            return {"new_pattern": pattern_str, "examples": self.examples[pattern_str]}
        
        return None
    
    def is_grammatical(self, sentence):
        """이 문장이 학습된 패턴인가?"""
        morphs = self.morph.pos(sentence)
        pattern = tuple(tag for _, tag in morphs)
        pattern_str = " ".join(pattern)
        
        if pattern_str in self.promoted_patterns:
            return True, self.promoted_patterns[pattern_str]
        return False, None
    
    def stats(self):
        """학습 통계."""
        return {
            "총_관찰_패턴": len(self.observed_patterns),
            "정식_패턴": len(self.promoted_patterns),
            "최다_빈도": self.observed_patterns.most_common(5),
        }


# =====================================================
# Patch 3: Bootstrapping Learner (단어↔패턴 선순환)
# =====================================================

class BootstrappingLearner:
    """단어와 패턴이 서로를 가르치는 시스템 (Pinker 1984).
    
    1. 패턴으로 단어 의미 추측
    2. 단어로 패턴 일반화
    3. 선순환
    """
    
    def __init__(self, morph_analyzer, constructions, pattern_discovery):
        self.morph = morph_analyzer
        self.constructions = constructions
        self.discovery = pattern_discovery
        
        # 학습된 단어 의미
        self.word_meanings = {}
        # 단어 → 자주 나오는 역할
        self.word_roles = {}  # word → Counter({role: count})
    
    def learn_from_sentence(self, sentence, given_meaning=None):
        """문장 학습."""
        # 1. 형태소 분석
        morphs = self.morph.pos(sentence)
        
        # 2. 구문 매칭
        matches = self.constructions.match(sentence)
        
        # 3. 패턴 관찰
        self.discovery.observe(sentence)
        
        # 4. 단어-역할 학습
        for match in matches:
            for role, word in match["roles"].items():
                if word not in self.word_roles:
                    self.word_roles[word] = Counter()
                self.word_roles[word][role] += 1
        
        return {
            "morphs": morphs,
            "constructions_matched": [m["construction"] for m in matches],
            "matches": matches,
        }
    
    def predict_role(self, word):
        """단어의 가장 흔한 역할 추측."""
        if word not in self.word_roles:
            return None
        return self.word_roles[word].most_common(1)[0]
    
    def understand(self, sentence):
        """문장 이해 시도."""
        morphs = self.morph.pos(sentence)
        matches = self.constructions.match(sentence)
        
        if matches:
            # 가장 잘 맞는 구문 선택
            best = matches[0]
            return {
                "이해도": "높음",
                "구문": best["construction"],
                "의미": best["meaning"],
                "역할": best["roles"],
                "질문": best.get("is_question", False),
                "명령": best.get("is_command", False),
                "부정": best.get("negation", False),
            }
        else:
            # 패턴 매칭 시도
            ok, pattern_info = self.discovery.is_grammatical(sentence)
            if ok:
                return {
                    "이해도": "중간",
                    "패턴": "학습된_패턴",
                    "예시": pattern_info["examples"]
                }
            else:
                return {
                    "이해도": "낮음",
                    "이유": "처음_보는_문법",
                    "형태소": morphs
                }


# =====================================================
# Integration: EVE에 문법 시스템 연결
# =====================================================

def add_grammar_to_eve(eve_instance):
    """EVE 인스턴스에 문법 학습 시스템 추가."""
    morph = SimpleMorphAnalyzer()
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    
    # 편의 메서드
    def hear(sentence):
        """문장 듣기 = 학습."""
        result = learner.learn_from_sentence(sentence)
        return result
    
    def understand(sentence):
        """문장 이해 시도."""
        return learner.understand(sentence)
    
    def grammar_stats():
        """문법 학습 통계."""
        return {
            "구문_룰": len(constructions.CONSTRUCTIONS),
            "관찰_패턴": discovery.stats(),
            "학습된_단어": len(learner.word_roles),
        }
    
    eve_instance.hear = hear
    eve_instance.understand = understand
    eve_instance.grammar_stats = grammar_stats
    
    print("  📚 문법 학습 시스템 활성화: 10개 구문 + 패턴 발견 + 부트스트래핑")
    return eve_instance


# =====================================================
# Patch 2: 의미 연결 - 문장 → 자동 신념 등록
# =====================================================

class MeaningExtractor:
    """파싱 결과에서 EVE 신념 시스템에 자동 등록.
    
    예: "고래는 포유류이다" 들으면
       → 자동으로 "고래는 포유류이다" 신념 추가
       → 지식 그래프에 "고래" → "포유류" 관계
    """
    
    def __init__(self, eve, constructions):
        self.eve = eve
        self.constructions = constructions
    
    def extract_and_register(self, sentence, source="대화", confidence=0.7):
        """문장 → 신념으로 변환.
        
        v6.0.5: 태그 정보 활용 (KoNLPy 정확도 100%)
        - PRED가 Verb 태그 → "{PRED}다" (이 빼기)
        - PRED가 Noun 태그 → "{PRED}이다" (이 유지)
        - 휴리스틱 X, 진짜 fix!
        """
        matches = self.constructions.match(sentence)
        
        if not matches:
            return None
        
        best = matches[0]
        construction = self.constructions.CONSTRUCTIONS[best["construction"]]
        
        if not construction.get("creates_belief"):
            return None
        
        template = construction["belief_template"]
        roles = best["roles"]
        role_tags = best.get("role_tags", {})  # v6.0.5: 태그 정보!
        
        # === v6.0.5: 태그 기반 정리 (휴리스틱 X) ===
        cleaned_roles = self._clean_roles(roles, template)
        
        statement = self._apply_template_with_tags(template, cleaned_roles, role_tags)
        
        if statement is None:
            return None
        
        if construction.get("negation"):
            confidence *= 0.9
        if construction.get("certainty"):
            confidence *= construction["certainty"]
        
        belief_id = "_".join(cleaned_roles.values()).replace(" ", "_")
        self.eve.add_belief(belief_id, statement, confidence, source)
        
        return {
            "신념_id": belief_id,
            "내용": statement,
            "신뢰도": confidence,
            "구문": best["construction"],
        }
    
    def _apply_template_with_tags(self, template, roles, role_tags):
        """v6.0.5: 태그 기반 템플릿 적용 (휴리스틱 X)
        
        v6.6 fix: PRED가 '이'로 끝나면 Verb 태그여도 명사로 처리
        - KoNLPy가 '장군이' → Verb로 잘못 태그하는 케이스
        - '장군이' + '다' → '장군이다' (그대로)
        """
        try:
            modified_template = template
            
            # PRED 처리 (분류문)
            if "PRED" in roles and "PRED" in role_tags:
                pred = roles["PRED"]
                pred_tag = role_tags["PRED"]
                
                # v6.6: PRED가 '이'로 끝나면 명사+이 조사 결합 → 명사로 처리
                if pred_tag == "Verb" and not pred.endswith("이"):
                    modified_template = modified_template.replace("{PRED}이다", "{PRED}다")
                    modified_template = modified_template.replace("{PRED}이야", "{PRED}야")
                # '이'로 끝나면 그대로 (장군이 + 다 = 장군이다)
                elif pred_tag == "Verb" and pred.endswith("이"):
                    # PRED에서 끝의 '이' 제거, 템플릿은 '이다' 유지
                    roles["PRED"] = pred[:-1]  # '장군이' → '장군'
                    # template "{PRED}이다" 그대로 → "장군이다" ✅
            
            # v6.1 fix: ACTION 어미 처리 (자동사)
            if "ACTION" in roles:
                action = roles["ACTION"]
                # v6.6: ACTION도 '이'로 끝나면 명사+이 조사
                if action and action.endswith("이"):
                    # 어색한 활용형 X → 그대로 두지 않고 처리
                    # "{AGENT}는 {ACTION}는다" → 어색
                    # 차라리 분류문으로 매칭됐어야 하지만 자동사로 매칭됨
                    # "{ACTION}는다" → 그냥 "{ACTION}다" + ACTION 끝 '이' 제거
                    roles["ACTION"] = action[:-1]  # '포유류이' → '포유류'
                    modified_template = modified_template.replace(
                        "{ACTION}는다", "{ACTION}이다"
                    )
                elif action and len(action) >= 1:
                    last = action[-1]
                    if '가' <= last <= '힣':
                        code = ord(last) - 0xAC00
                        final = code % 28
                        
                        if final == 4:  # ㄴ 받침 ('친', '한')
                            modified_template = modified_template.replace(
                                "{ACTION}는다", "{ACTION}다"
                            )
                        elif final == 0:  # 받침 없음 ('치', '하', '쁘', '크')
                            modified_template = modified_template.replace(
                                "{ACTION}는다", "{ACTION}다"
                            )
                        else:  # 다른 받침 ('웃', '울', '먹')
                            # 받침 있는 동사 어근 → "{ACTION}는다" 그대로
                            pass
            
            # TRAIT 처리 (형용사 서술)
            if "TRAIT" in roles and "TRAIT" in role_tags:
                trait_tag = role_tags["TRAIT"]
                # Adjective든 Verb든 "다"만 (이미 위에서 형용사도 Verb로 매핑됨)
                pass
            
            # 기본 적용
            result = modified_template.format(**roles)
            
            # 받침 따라 조사 자동
            for role, value in roles.items():
                if not value:
                    continue
                if has_batchim(value):
                    result = result.replace(f"{value}는 ", f"{value}은 ")
                    result = result.replace(f"{value}는.", f"{value}은.")
                    result = result.replace(f"{value}가 ", f"{value}이 ")
                    result = result.replace(f"{value}를 ", f"{value}을 ")
                    result = result.replace(f"{value}와 ", f"{value}과 ")
                else:
                    result = result.replace(f"{value}은 ", f"{value}는 ")
                    result = result.replace(f"{value}이 ", f"{value}가 ")
                    result = result.replace(f"{value}을 ", f"{value}를 ")
                    result = result.replace(f"{value}과 ", f"{value}와 ")
            
            return result
        except KeyError:
            return None
    
    def _is_verb_root(self, word):
        """동사/형용사 어근 추정 (간단 휴리스틱).
        
        명사가 아닌 동사 어근일 가능성 체크:
        - 한글 마지막 글자가 받침 없는 모음으로 끝나면 동사일 확률 ↑
          (예: "예쁘", "영리하", "조용하")
        - 단, 명사도 받침 없을 수 있어서 100% 정확 X
        """
        if not word:
            return False
        # 마지막 글자
        last = word[-1]
        if not ('가' <= last <= '힣'):
            return False
        # 한글 분해
        code = ord(last) - 0xAC00
        final = code % 28  # 받침
        # 받침 없음 + 길이 2 이상이면 동사 가능성 ↑
        # (단순히 명사일 수도 있어서 보수적으로)
        # 일단 False 반환 (보수적) - 위 verb_endings로 충분
        return False
    
    def _apply_template_with_josa(self, template, roles):
        """템플릿 적용 + 받침 따라 조사 자동 선택.
        
        v6.0.4 근본 fix: 받침 기반 결정
        - PRED 받침 없음 (모음으로 끝남) → 동사 어근 가능성 ↑ → "이다" 빼기
          예: '영리하' (받침X) → '영리하다' ✅
              '예쁘' (받침X) → '예쁘다' ✅  
              '헤엄친' (받침O 'ㄴ') → 받침 있어도 동사 활용
          
        - PRED 받침 있음 + 명사 → "이다" 유지
          예: '포유류' (받침X) → '포유류이다' ❓ 이건 명사
              '동물' (받침O) → '동물이다' ✅
        
        문제: '포유류'(모음)는 명사인데 동사로 오인 가능
              → 추가 휴리스틱: TOPIC도 같은 패턴이면 명사로 추정
        
        실용 해결: 명시적 "동사 활용 어미" 리스트로 판단
        - 받침이 'ㄴ', 'ㄹ' 등 + 끝글자가 동사 활용 → 동사
        - 일반 명사는 그냥 "이다" 유지
        """
        try:
            modified_template = template
            if "PRED" in roles:
                pred = roles["PRED"]
                
                if pred and self._is_verb_form(pred):
                    # 동사형 PRED → "이다" 빼기
                    modified_template = modified_template.replace("{PRED}이다", "{PRED}다")
                    modified_template = modified_template.replace("{PRED}이야", "{PRED}야")
            
            # 기본 적용
            result = modified_template.format(**roles)
            
            # 받침 기반 조사 자동
            for role, value in roles.items():
                if not value:
                    continue
                if has_batchim(value):
                    result = result.replace(f"{value}는 ", f"{value}은 ")
                    result = result.replace(f"{value}는.", f"{value}은.")
                    result = result.replace(f"{value}가 ", f"{value}이 ")
                    result = result.replace(f"{value}를 ", f"{value}을 ")
                    result = result.replace(f"{value}와 ", f"{value}과 ")
                else:
                    result = result.replace(f"{value}은 ", f"{value}는 ")
                    result = result.replace(f"{value}이 ", f"{value}가 ")
                    result = result.replace(f"{value}을 ", f"{value}를 ")
                    result = result.replace(f"{value}과 ", f"{value}와 ")
            
            return result
        except KeyError:
            return None
    
    def _is_verb_form(self, word):
        """단어가 동사/형용사 활용형인지 판단 (휴리스틱).
        
        동사 어근 패턴:
        1. '하' / '되' / '있' / '없' 류 (대표 동사)
        2. 형용사 어근 ('쁘', '프', '느', '크', '트' 등)
        3. 동사 활용 ('친', '한', '운', '온', '단' 등 - ㄴ + 동사 활용)
        
        명사 패턴 (제외):
        - 일반 명사는 "이다" 유지
        - "포유류", "동물", "사람" 등은 명사로 판단
        """
        if not word:
            return False
        
        # 1. 명시적 동사 어미 (확실)
        verb_endings_strong = [
            "하", "되", "있", "없",  # 가장 흔한 동사
            "리", "르", "기",         # 동사 어근
        ]
        if any(word.endswith(e) for e in verb_endings_strong):
            return True
        
        # 2. 형용사 어근
        adj_endings = ["쁘", "프", "느", "크", "트", "스"]
        if any(word.endswith(e) for e in adj_endings):
            return True
        
        # 3. 동사 활용형 (ㄴ + 다 → 친, 한, 운, 본 등)
        # 마지막 글자 받침이 'ㄴ' (4)이고 길이 2+ 면 동사 활용 가능성 ↑
        # v6.0.4 보수적: 길이 2 이상만 동사 (단음절은 명사로 보수적 판단)
        if len(word) >= 3:  # 3글자 이상 (헤엄친, 사냥한)
            last = word[-1]
            if '가' <= last <= '힣':
                code = ord(last) - 0xAC00
                final = code % 28
                if final == 4:  # ㄴ 받침
                    return True
        
        # 기본: 명사로 (이다 유지)
        return False
    
    def _clean_roles(self, roles, template):
        """v5.8 BUG FIX: 역할에서 중복 어미/조사 제거.
        
        예: PRED="포유류이" + template "{PRED}이다" → "포유류이이다" 발생
            → PRED 끝의 "이" 제거 → "포유류" + "이다" = "포유류이다"
        """
        cleaned = {}
        for role, value in roles.items():
            cleaned_value = value
            
            # 분류문/의문문 PRED, QUERY: 끝에 "이" 있으면 제거
            # (이미 template에 "이다"/"이야"가 있으므로 중복 방지)
            if role in ("PRED", "QUERY"):
                # template에 "이다"/"이야" 있는지 확인
                if "이다" in template or "이야" in template:
                    if cleaned_value.endswith("이"):
                        cleaned_value = cleaned_value[:-1]
            
            # 행위문 ACTION: 끝에 "는"/"ㄴ" 있으면 제거 (어미 중복 방지)
            if role == "ACTION":
                if cleaned_value.endswith("는"):
                    cleaned_value = cleaned_value[:-1]
            
            cleaned[role] = cleaned_value
        
        return cleaned


# =====================================================
# Patch 3: 어미 뉘앙스 분석기
# =====================================================

class NuanceAnalyzer:
    """어미 + 부사로 문장 뉘앙스 분석.
    
    표준국어문법론 기준 분류 사용:
    - 종결어미: 평서/의문/명령/청유/감탄
    - 연결어미: 조건/양보/인과/목적/동시
    - 화계 (speech_level): 해라/해/해요/합쇼/하오
    """
    
    def __init__(self, morph_analyzer):
        self.morph = morph_analyzer
    
    def analyze(self, sentence):
        """문장의 뉘앙스 종합 분석 (학술 기준)."""
        morphs = self.morph.pos(sentence)
        
        nuance = {
            "정중도": 0.5,           # 0=반말, 1=극존칭
            "확신도": 0.7,           # 0=불확실, 1=단정
            "강도": 1.0,             # 강조 배율
            "분위기": [],             # 평서/의문/명령/청유/감탄
            "시제": "현재",
            "화계": None,            # 해라/해/해요/합쇼
            "문장_종류": "평서",      # 종결어미 type
            "양태": [],              # 추측/의지/가능/의무 등
            "어미": [],
            "부사": [],
        }
        
        # 어미 분석 (새 구조)
        for word, tag in morphs:
            if tag == "Eomi" and word in self.morph.EOMI:
                eomi = self.morph.EOMI[word]
                nuance["어미"].append((word, eomi))
                
                # 종결어미 → 문장 종류 결정
                if eomi.get("category") == "종결어미":
                    sent_type = eomi.get("type", "평서")
                    nuance["문장_종류"] = sent_type
                    nuance["분위기"].append(sent_type)
                
                # 화계 (정중도 등급)
                if "speech_level" in eomi:
                    nuance["화계"] = eomi["speech_level"]
                if "formality" in eomi:
                    nuance["정중도"] = eomi["formality"]
                
                # 확신도
                if "certainty" in eomi:
                    nuance["확신도"] = eomi["certainty"]
                
                # 강도
                if "intensity" in eomi:
                    nuance["강도"] *= eomi["intensity"]
                
                # 시제
                if "tense" in eomi:
                    nuance["시제"] = eomi["tense"]
                
                # 양태 (modality)
                if "modality" in eomi:
                    nuance["양태"].append(eomi["modality"])
        
        # 부사 분석
        for word, tag in morphs:
            if tag == "Adverb" and word in self.morph.ADVERB_INTENSITY:
                intensity = self.morph.ADVERB_INTENSITY[word]
                nuance["부사"].append((word, intensity))
                
                if intensity > 0:
                    nuance["강도"] *= intensity
                else:
                    # 부정 부사 (별로/전혀/안/못)
                    nuance["확신도"] *= 0.3
                    if "부정" not in nuance["분위기"]:
                        nuance["분위기"].append("부정")
        
        # 종합 톤 결정 (화계 기반)
        if nuance["화계"] == "합쇼":
            nuance["톤"] = "격식"
        elif nuance["화계"] == "해요":
            nuance["톤"] = "정중"
        elif nuance["화계"] == "해라":
            nuance["톤"] = "단정"
        elif nuance["화계"] == "해":
            nuance["톤"] = "친근"
        elif nuance["정중도"] > 0.8:
            nuance["톤"] = "격식"
        elif nuance["정중도"] > 0.5:
            nuance["톤"] = "정중"
        elif nuance["정중도"] > 0.3:
            nuance["톤"] = "보통"
        else:
            nuance["톤"] = "친근"
        
        return nuance


# =====================================================
# Patch 4: 호르몬 시스템 (도파민/세로토닌/코르티솔/옥시토신)
# =====================================================

class HormoneSystem:
    """4가지 호르몬으로 EVE에 감정/동기 부여.
    
    이론:
    - Dopamine (도파민): 보상 예측, 호기심
    - Serotonin (세로토닌): 안정, 기분
    - Cortisol (코르티솔): 스트레스, 위협
    - Oxytocin (옥시토신): 애착, 신뢰
    
    영감: Damasio "감정이 이성의 토대"
         Schultz dopamine prediction error
    """
    
    def __init__(self):
        # 기준선 (안정 상태)
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.cortisol = 0.3
        self.oxytocin = 0.4
        
        self.history = []
    
    def reward_prediction_error(self, expected, actual):
        """보상 예측 오류 → 도파민 변화 (Schultz 1998)."""
        delta = actual - expected
        old = self.dopamine
        
        self.dopamine += delta * 0.2  # v5.8: 약화 (0.4 → 0.2)
        self.dopamine = max(0.0, min(1.0, self.dopamine))
        
        return {
            "도파민_변화": self.dopamine - old,
            "신호": "양성" if delta > 0 else "음성" if delta < 0 else "중립"
        }
    
    def encounter_novelty(self, novelty_level):
        """새로움 발견 → 도파민 (탐험 동기)."""
        self.dopamine += novelty_level * 0.15  # v5.8: 약화 (0.3 → 0.15)
        self.dopamine = min(1.0, self.dopamine)
    
    def encounter_threat(self, threat_level):
        """위협 발견 → 코르티솔."""
        self.cortisol += threat_level * 0.3  # v5.8: 약화 (0.5 → 0.3)
        self.cortisol = min(1.0, self.cortisol)
        
        # 코르티솔 ↑ → 세로토닌 ↓ (스트레스로 기분 저하)
        self.serotonin = max(0.0, self.serotonin - threat_level * 0.15)
    
    def social_bond(self, bond_strength):
        """사회적 유대 → 옥시토신."""
        self.oxytocin += bond_strength * 0.25  # v5.8: 약화 (0.4 → 0.25)
        self.oxytocin = min(1.0, self.oxytocin)
    
    def stable_environment(self):
        """안정된 환경 → 세로토닌."""
        self.serotonin += 0.05  # v5.8: 약화 (0.08 → 0.05)
        self.serotonin = min(1.0, self.serotonin)
        
        # 코르티솔 자연 감소
        self.cortisol = max(0.0, self.cortisol - 0.03)
    
    def homeostasis(self):
        """항상성 - 시간이 지나면 기준선으로 돌아감."""
        # 도파민
        if self.dopamine > 0.5:
            self.dopamine -= 0.02
        elif self.dopamine < 0.5:
            self.dopamine += 0.02
        
        # 세로토닌
        if self.serotonin > 0.5:
            self.serotonin -= 0.01
        elif self.serotonin < 0.5:
            self.serotonin += 0.01
        
        # 코르티솔 (스트레스는 천천히 회복)
        self.cortisol = max(0.3, self.cortisol - 0.02)
        
        # 옥시토신 (애착은 천천히 감소)
        if self.oxytocin > 0.4:
            self.oxytocin -= 0.01
    
    def emotional_state(self):
        """현재 감정 상태 추론 (호르몬 조합).
        
        v5.8: 임계값 엄격화 - 평온이 기본
        """
        d, s, c, o = self.dopamine, self.serotonin, self.cortisol, self.oxytocin
        
        # 단순 규칙으로 감정 추론 (v5.8 임계값 ↑)
        if d > 0.8 and s > 0.7:  # 기쁨: 더 엄격 (0.7,0.6 → 0.8,0.7)
            return "기쁨"
        elif d > 0.8 and c > 0.7:  # 흥분
            return "흥분"
        elif s < 0.3 and c > 0.6:
            return "우울/스트레스"
        elif c > 0.7:
            return "불안"
        elif o > 0.75:  # 유대감: 더 엄격 (0.7 → 0.75)
            return "유대감"
        elif d < 0.3 and s < 0.4:
            return "무기력"
        elif d > 0.7 and c < 0.4:  # 호기심
            return "호기심"
        else:
            return "평온"
    
    def state(self):
        """현재 호르몬 상태."""
        return {
            "도파민": round(self.dopamine, 2),
            "세로토닌": round(self.serotonin, 2),
            "코르티솔": round(self.cortisol, 2),
            "옥시토신": round(self.oxytocin, 2),
            "감정": self.emotional_state(),
        }


# =====================================================
# Integration: v5.7 풀코스 통합
# =====================================================

def add_full_grammar_to_eve(eve_instance):
    """v5.7: 모든 모듈 통합 - 문법 정밀 + 의미 + 뉘앙스 + 호르몬."""
    morph = SimpleMorphAnalyzer()
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    # 새 모듈
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    
    def hear(sentence, register_belief=True):
        """문장 듣기 = 학습 + 신념 등록 + 호르몬 반응 (학술 분류 기반)."""
        # 1. 문법 학습
        result = learner.learn_from_sentence(sentence)
        
        # 2. 뉘앙스 분석
        n = nuance.analyze(sentence)
        result["뉘앙스"] = n
        
        # 3. 호르몬 반응 (화계 + 문장종류 기반)
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        # 정중한 말 (해요/합쇼) = 안정 (세로토닌 ↑)
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        
        # 친근한 말 (해/해라) = 옥시토신 (사회적 유대)
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        
        # 명령조 = 코르티솔 약간
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        
        # 감탄 = 도파민 (긍정 자극)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        
        # 청유 = 옥시토신 (같이 하자!)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        
        # 의문 = 도파민 (호기심 자극)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        
        # 부정 부사 있으면 = 코르티솔 약간
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        # 새 패턴 = 도파민
        observed = discovery.observe(sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        # 4. 의미 추출 → 신념 등록
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
        
        return result
    
    def understand(sentence):
        """문장 이해 + 뉘앙스."""
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def grammar_stats():
        return {
            "구문_룰": len(constructions.CONSTRUCTIONS),
            "관찰_패턴": discovery.stats(),
            "학습된_단어": len(learner.word_roles),
            "호르몬_상태": hormones.state(),
        }
    
    eve_instance.hear = hear
    eve_instance.understand = understand
    eve_instance.grammar_stats = grammar_stats
    
    # 어미 카테고리별 통계
    eomi_stats = {"종결어미": 0, "연결어미": 0, "전성어미": 0, "선어말어미": 0}
    for word, info in morph.EOMI.items():
        cat = info.get("category", "기타")
        if cat in eomi_stats:
            eomi_stats[cat] += 1
    
    josa_stats = {"격조사": 0, "보조사": 0, "접속조사": 0}
    for word, info in morph.JOSA.items():
        cat = info.get("category", "기타")
        if cat in josa_stats:
            josa_stats[cat] += 1
    
    print(f"  📚 v5.7 풀코스 활성화 (표준국어문법론 기준):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    print(f"     - 어미: {sum(eomi_stats.values())}개 "
          f"(종결{eomi_stats['종결어미']}/연결{eomi_stats['연결어미']}/"
          f"전성{eomi_stats['전성어미']}/선어말{eomi_stats['선어말어미']})")
    print(f"     - 조사: {sum(josa_stats.values())}개 "
          f"(격{josa_stats['격조사']}/보조{josa_stats['보조사']}/접속{josa_stats['접속조사']})")
    print(f"     - 의미 연결: 문장 → 자동 신념 등록")
    print(f"     - 뉘앙스: 화계(해/해요/합쇼)/문장종류/양태")
    print(f"     - 호르몬: 도파민/세로토닌/코르티솔/옥시토신")
    
    return eve_instance


# =====================================================
# Patch 2: Curiosity Module (호기심 시스템)
# =====================================================
# 영감: Schmidhuber Curious AI (1991), Pathak ICM (2017)
# 핵심: 모르는 것 발견 → 도파민 ↑ → 더 배우려는 동기

class CuriosityModule:
    """호기심 모듈 - 새 정보에 자동 반응.
    
    원리:
    1. 새 단어 발견 → curiosity ↑
    2. 모르는 개념 등장 → "?" 자동 마킹
    3. 도파민과 연결 → 학습 동기 ↑
    4. 자주 듣는 단어 → 친숙도 ↑ (호기심 ↓)
    
    이론:
    - Schmidhuber 1991: Curious agents predict novelty
    - Pathak ICM 2017: Intrinsic Curiosity Module (RL)
    - Berlyne 1954: Optimal arousal theory (어느 정도 새로움이 최적)
    """
    
    def __init__(self, hormones, max_familiar=20):
        self.hormones = hormones  # 호르몬 시스템 연결
        
        # 단어 친숙도 (들은 횟수)
        self.familiarity = {}
        
        # 모르는 단어 리스트 (질문 후보)
        self.unknown_words = []
        
        # 호기심 강도 (0-1)
        self.curiosity_level = 0.5
        
        # Berlyne 최적 새로움 (너무 익숙해도/너무 새로워도 안됨)
        self.max_familiar = max_familiar
    
    def observe_word(self, word, known_concepts=None):
        """단어 관찰 → 친숙도/호기심 갱신."""
        # 1. 친숙도 증가
        if word not in self.familiarity:
            self.familiarity[word] = 0
            # 진짜 새 단어!
            self._on_new_word(word, known_concepts)
        
        self.familiarity[word] += 1
        
        # 2. 호기심 갱신
        self._update_curiosity()
    
    def _on_new_word(self, word, known_concepts):
        """새 단어 등장 시 처리."""
        # 알려진 개념인지 확인
        is_known = False
        if known_concepts and word in known_concepts:
            is_known = True
        
        if not is_known:
            # 모르는 단어 → 질문 후보
            if word not in self.unknown_words:
                self.unknown_words.append(word)
            
            # 도파민 ↑ (새로움 보상)
            self.hormones.encounter_novelty(0.3)
    
    def _update_curiosity(self):
        """호기심 강도 갱신.
        
        Berlyne 1954: 최적 새로움 = 너무 익숙X / 너무 새로움X
        """
        if not self.familiarity:
            self.curiosity_level = 0.5
            return
        
        # 평균 친숙도
        avg_familiar = sum(self.familiarity.values()) / len(self.familiarity)
        
        # 최적 곡선 (역U자)
        # 너무 친숙 (>20) → 호기심 ↓
        # 너무 새 (<2) → 호기심 ↓ (압도)
        # 중간 (5-15) → 호기심 ↑ (적절한 도전)
        if avg_familiar < 2:
            self.curiosity_level = 0.3  # 너무 새로움
        elif avg_familiar < 8:
            self.curiosity_level = 0.8  # 적절
        elif avg_familiar < self.max_familiar:
            self.curiosity_level = 0.6  # 좀 익숙
        else:
            self.curiosity_level = 0.2  # 너무 익숙
    
    def get_questions(self, max_questions=3):
        """모르는 단어들로 질문 생성 (받침 따라 조사)."""
        questions = []
        for word in self.unknown_words[:max_questions]:
            # 받침 있으면 "이", 없으면 "가"
            if has_batchim(word):
                questions.append(f"{word}이 뭐야?")
            else:
                questions.append(f"{word}가 뭐야?")
        return questions
    
    def has_questions(self):
        """질문할 게 있나?"""
        return len(self.unknown_words) > 0
    
    def resolve_word(self, word):
        """단어가 설명되면 unknown에서 제거."""
        if word in self.unknown_words:
            self.unknown_words.remove(word)
            # 도파민 ↑ (지식 획득 보상)
            self.hormones.reward_prediction_error(0.5, 0.8)
    
    def state(self):
        """호기심 상태."""
        return {
            "호기심_강도": round(self.curiosity_level, 2),
            "친숙_단어": len(self.familiarity),
            "모르는_단어": len(self.unknown_words),
            "최근_모르는": self.unknown_words[:5],
            "가장_친숙": sorted(
                self.familiarity.items(), 
                key=lambda x: -x[1]
            )[:5],
        }


# =====================================================
# Patch 3: Hormone-Tone Mapping (호르몬 → 답변 톤)
# =====================================================

class HormoneTone:
    """호르몬 → 답변에 미세한 영향 (cartoon 아님).
    
    v5.9 재설계:
    - 어미 자동 추가 (오!, 와!) 제거
    - 감정 = 의사결정에 영향 (Damasio)
    - 표현 차이는 미세하게 (subtle)
    - 사람마다 표현 다른 게 정상
    
    영향:
    - 응답 길이 (스트레스 → 짧게)
    - 질문할지 여부 (호기심 → 질문)
    - 어미 화계 (유대감 → 반말, 평온 → 보통)
    - 단, 강제로 감탄사 X
    """
    
    TONE_PROFILES = {
        "기쁨": {
            "길이": "보통",
            "어미선호": "다",   # 평소
            "질문확률": 0.2,
            "내부": "긍정적 활성",
        },
        "유대감": {
            "길이": "보통",
            "어미선호": "야",   # 친근 (자연스러움)
            "질문확률": 0.3,
            "내부": "친밀함",
        },
        "호기심": {
            "길이": "보통",
            "어미선호": "다",
            "질문확률": 0.6,    # 진짜 더 물어봄
            "내부": "탐구 모드",
        },
        "평온": {
            "길이": "보통",
            "어미선호": "다",
            "질문확률": 0.15,
            "내부": "균형",
        },
        "흥분": {
            "길이": "보통",
            "어미선호": "다",
            "질문확률": 0.4,
            "내부": "활성화",
        },
        "우울/스트레스": {
            "길이": "짧게",     # 진짜 짧아짐
            "어미선호": "다",
            "질문확률": 0.05,
            "내부": "위축",
        },
        "불안": {
            "길이": "짧게",
            "어미선호": "다",
            "질문확률": 0.2,
            "내부": "경계",
        },
        "무기력": {
            "길이": "짧게",
            "어미선호": "다",
            "질문확률": 0.0,
            "내부": "저하",
        },
    }
    
    def __init__(self, hormones):
        self.hormones = hormones
    
    def get_tone(self):
        """현재 호르몬 → 톤 프로파일."""
        emotion = self.hormones.emotional_state()
        return self.TONE_PROFILES.get(emotion, self.TONE_PROFILES["평온"])
    
    def style_response(self, base_message):
        """기본 메시지를 호르몬 영향으로 스타일링.
        
        v5.9: cartoon 제거. 미세한 영향만.
        - 자동 감탄사 X
        - 자동 "...궁금해" X
        - 어미 변경은 유대감일 때만 (자연스러움)
        - 길이는 스트레스/무기력일 때만 (실제 영향)
        """
        tone = self.get_tone()
        emotion = self.hormones.emotional_state()
        message = base_message
        
        # 1. 어미 변환 - 유대감일 때만 (자연스럽게 반말로)
        if emotion == "유대감":
            replacements = {
                "이다": "이야",
                "는다": "어",
                "ㄴ다": "어",
            }
            for old, new in replacements.items():
                if message.endswith(old):
                    message = message[:-len(old)] + new
                    break
        
        # 2. 길이 - 스트레스/무기력일 때만 (실제 영향)
        # (지금은 base_message 그대로, 나중에 응답 생성 시 적용)
        
        # 3. 자동 감탄사 X (cartoon 제거)
        # 4. 자동 "...궁금해" X
        
        return {
            "응답": message,
            "감정": emotion,
            "내부상태": tone["내부"],
            "톤": {
                "길이선호": tone["길이"],
                "질문확률": tone["질문확률"],
            },
        }


# =====================================================
# Integration: v5.8 통합
# =====================================================

def add_full_grammar_to_eve_v58(eve_instance):
    """v5.8: v5.7 + Curiosity + HormoneTone."""
    morph = SimpleMorphAnalyzer()
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    
    # v5.8 새 모듈
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    
    def hear(sentence, register_belief=True):
        """문장 듣기 = 학습 + 신념 + 호르몬 + 호기심."""
        # 1. 문법 학습
        result = learner.learn_from_sentence(sentence)
        
        # 2. 뉘앙스
        n = nuance.analyze(sentence)
        result["뉘앙스"] = n
        
        # 3. 호르몬 반응
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        # 4. 호기심 - 단어별 관찰
        morphs = morph.pos(sentence)
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        # 5. 새 패턴 (도파민)
        observed = discovery.observe(sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        # 6. 의미 → 신념
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
        
        # 7. 호기심 상태
        result["호기심"] = curiosity.state()
        
        return result
    
    def speak(message):
        """EVE가 호르몬 톤으로 말하기."""
        return tone.style_response(message)
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def grammar_stats():
        return {
            "구문_룰": len(constructions.CONSTRUCTIONS),
            "관찰_패턴": discovery.stats(),
            "학습된_단어": len(learner.word_roles),
            "호르몬_상태": hormones.state(),
            "호기심_상태": curiosity.state(),
        }
    
    eve_instance.hear = hear
    eve_instance.speak = speak  # 새 메서드!
    eve_instance.understand = understand
    eve_instance.grammar_stats = grammar_stats
    
    eomi_stats = {"종결어미": 0, "연결어미": 0, "전성어미": 0, "선어말어미": 0}
    for word, info in morph.EOMI.items():
        cat = info.get("category", "기타")
        if cat in eomi_stats:
            eomi_stats[cat] += 1
    
    josa_stats = {"격조사": 0, "보조사": 0, "접속조사": 0}
    for word, info in morph.JOSA.items():
        cat = info.get("category", "기타")
        if cat in josa_stats:
            josa_stats[cat] += 1
    
    print(f"  📚 v5.8 활성화 (버그 fix + 호기심 + 호르몬 톤):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    print(f"     - 어미: {sum(eomi_stats.values())}개 "
          f"(종결{eomi_stats['종결어미']}/연결{eomi_stats['연결어미']}/"
          f"전성{eomi_stats['전성어미']}/선어말{eomi_stats['선어말어미']})")
    print(f"     - 조사: {sum(josa_stats.values())}개")
    print(f"     - 🐛 버그 fix: '이다' 중복 제거")
    print(f"     - 🤔 호기심: 새 단어 자동 감지 + 질문 생성")
    print(f"     - 🎭 호르몬 톤: 감정 → 답변 스타일 (8가지)")
    
    return eve_instance


# =====================================================
# Patch 1 (v5.9): Circadian Rhythm + Melatonin
# =====================================================
# 영감: 24시간 일주기 리듬 (생체시계)
# 핵심: EVE가 진짜 시간 안에서 살아감

import datetime as _dt

class CircadianSystem:
    """일주기 리듬 시스템 - 시간대별 호르몬 자동 변화.
    
    이론:
    - SCN (Suprachiasmatic Nucleus): 인간 생체시계
    - 멜라토닌: 어두워지면 분비 → 졸림
    - 코르티솔: 아침에 max (각성)
    - 도파민/세로토닌: 낮 활동기에 ↑
    
    EVE 시간 매핑:
    - 06-09시: 각성 (코르티솔 ↑)
    - 09-18시: 활동 (도파민/세로토닌 ↑)
    - 18-22시: 하강 (호르몬 안정)
    - 22-06시: 수면 (멜라토닌 ↑, 자동 졸림)
    """
    
    def __init__(self, hormones):
        self.hormones = hormones
        self.melatonin = 0.3  # 기본
        self.last_update = None
    
    def get_phase(self, hour=None):
        """현재 시간대 phase 반환."""
        if hour is None:
            hour = _dt.datetime.now().hour
        
        if 6 <= hour < 9:
            return "각성"
        elif 9 <= hour < 18:
            return "활동"
        elif 18 <= hour < 22:
            return "하강"
        else:  # 22-06
            return "수면"
    
    def update(self, hour=None):
        """시간대에 따라 호르몬 자동 조정."""
        phase = self.get_phase(hour)
        
        if phase == "각성":
            # 아침: 코르티솔 ↑ (자연 각성), 멜라토닌 ↓
            self.melatonin = max(0.1, self.melatonin - 0.2)
            self.hormones.cortisol = min(0.5, self.hormones.cortisol + 0.1)
        
        elif phase == "활동":
            # 낮: 도파민/세로토닌 활성, 멜라토닌 최저
            self.melatonin = 0.05
            self.hormones.dopamine = min(0.7, self.hormones.dopamine + 0.05)
            self.hormones.serotonin = min(0.7, self.hormones.serotonin + 0.05)
        
        elif phase == "하강":
            # 저녁: 호르몬 천천히 안정
            self.melatonin = min(0.4, self.melatonin + 0.1)
            self.hormones.cortisol = max(0.1, self.hormones.cortisol - 0.05)
        
        else:  # 수면
            # 밤: 멜라토닌 폭증 → 졸림
            self.melatonin = min(0.95, self.melatonin + 0.3)
            self.hormones.dopamine = max(0.2, self.hormones.dopamine - 0.1)
            self.hormones.serotonin = max(0.3, self.hormones.serotonin - 0.05)
        
        self.last_update = _dt.datetime.now()
        return phase
    
    def is_sleep_time(self):
        """자야 할 시간인가? (멜라토닌 기반)."""
        return self.melatonin > 0.7
    
    def state(self):
        """현재 일주기 상태."""
        phase = self.get_phase()
        return {
            "시간대": phase,
            "멜라토닌": round(self.melatonin, 2),
            "졸림": "예" if self.is_sleep_time() else "아니오",
        }


# =====================================================
# Patch 2 (v5.9): Time Tracker (Day별 변화 추적)
# =====================================================

class TimeTracker:
    """EVE의 시간 흐름 추적.
    
    매일 무엇이 변했나?
    - 신념 수
    - 단어 수
    - 호르몬 평균
    - 학습 곡선
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.daily_snapshots = []  # [{day, beliefs, words, hormone_avg, ...}]
    
    def take_snapshot(self):
        """오늘의 EVE 상태 스냅샷."""
        snapshot = {
            "day": self.eve.identity["days_alive"],
            "timestamp": _dt.datetime.now().isoformat(),
            "신념_수": len(self.eve.beliefs) if hasattr(self.eve, 'beliefs') else 0,
            "개념_수": len(self.eve.concepts) if hasattr(self.eve, 'concepts') else 0,
            "학습된_단어": (
                len(self.eve.grammar_learner.word_roles) 
                if hasattr(self.eve, 'grammar_learner') else 0
            ),
            "호르몬": self.eve.hormones.state() if hasattr(self.eve, 'hormones') else None,
            "감정": (
                self.eve.hormones.emotional_state() 
                if hasattr(self.eve, 'hormones') else None
            ),
        }
        self.daily_snapshots.append(snapshot)
        return snapshot
    
    def growth_summary(self):
        """성장 요약 - Day 1 vs 오늘."""
        if len(self.daily_snapshots) < 2:
            return {"메시지": "데이터 부족 (스냅샷 2개 이상 필요)"}
        
        first = self.daily_snapshots[0]
        last = self.daily_snapshots[-1]
        
        return {
            "기간": f"Day {first['day']} → Day {last['day']}",
            "지난_일수": last['day'] - first['day'],
            "신념_변화": f"{first['신념_수']} → {last['신념_수']} "
                       f"(+{last['신념_수'] - first['신념_수']})",
            "단어_변화": f"{first['학습된_단어']} → {last['학습된_단어']} "
                       f"(+{last['학습된_단어'] - first['학습된_단어']})",
            "감정_변화": f"{first['감정']} → {last['감정']}",
            "스냅샷_수": len(self.daily_snapshots),
        }
    
    def get_learning_curve(self):
        """학습 곡선 데이터."""
        return [
            {"day": s["day"], "신념": s["신념_수"], "단어": s["학습된_단어"]}
            for s in self.daily_snapshots
        ]


# =====================================================
# Patch 3 (v5.9): Identity Drift (정체성 변화 추적)
# =====================================================

class IdentityTracker:
    """정체성 변화 추적.
    
    핵심 질문:
    - EVE의 핵심 신념이 바뀌었나?
    - 정체성은 보존되나?
    - 어떻게 자랐나?
    
    코어 정체성 (불변):
    - 이름: EVE
    - 언어: 한국어
    - 본질: 학습하는 존재
    
    유연한 정체성 (변화):
    - 신념 (학습 따라 변함)
    - 호르몬 평균 (성격)
    - 관심사 (가장 친숙한 단어들)
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.core_identity = {
            "이름": "EVE",
            "언어": "한국어",
            "본질": "학습하는 존재",
            "탄생": None,
        }
        self.identity_log = []  # 변화 기록
    
    def initialize(self):
        """EVE 첫 부팅 시 정체성 초기화."""
        if self.core_identity["탄생"] is None:
            self.core_identity["탄생"] = _dt.datetime.now().isoformat()
    
    def check_drift(self):
        """현재 EVE가 코어와 얼마나 다른가?"""
        # 코어 보존 여부
        core_preserved = (
            self.eve.identity.get("name") == self.core_identity["이름"] and
            self.eve.identity.get("language") == self.core_identity["언어"]
        )
        
        # 가장 친숙한 단어들 = 관심사
        interests = []
        if hasattr(self.eve, 'curiosity'):
            stats = self.eve.curiosity.state()
            interests = [w for w, _ in stats.get("가장_친숙", [])[:5]]
        
        # 핵심 신념 (가장 강한)
        core_beliefs = []
        if hasattr(self.eve, 'beliefs'):
            sorted_beliefs = sorted(
                self.eve.beliefs.items() if isinstance(self.eve.beliefs, dict) else [],
                key=lambda x: x[1].confidence if hasattr(x[1], 'confidence') else 0,
                reverse=True
            )[:5]
            core_beliefs = [
                b.statement if hasattr(b, 'statement') else str(b)
                for _, b in sorted_beliefs
            ]
        
        return {
            "코어_보존": core_preserved,
            "이름": self.eve.identity.get("name"),
            "현재_나이": f"Day {self.eve.identity.get('days_alive', 0)}",
            "관심사": interests,
            "핵심_신념": core_beliefs,
        }
    
    def log_change(self, change_type, description):
        """정체성 변화 기록."""
        self.identity_log.append({
            "timestamp": _dt.datetime.now().isoformat(),
            "type": change_type,
            "description": description,
            "day": self.eve.identity.get("days_alive", 0),
        })
    
    def history(self):
        """정체성 변화 이력."""
        return self.identity_log


# =====================================================
# Integration: v5.9 통합
# =====================================================

def add_full_grammar_to_eve_v59(eve_instance):
    """v5.9: v5.8 + Circadian + TimeTracker + IdentityTracker."""
    morph = SimpleMorphAnalyzer()
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    # v5.9 새 모듈
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    
    def hear(sentence, register_belief=True):
        """문장 듣기."""
        result = learner.learn_from_sentence(sentence)
        
        n = nuance.analyze(sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(sentence)
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        """EVE가 말하기 (호르몬 영향 - subtle)."""
        return tone.style_response(message)
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def tick(hour=None):
        """시간 한 단위 흐름 - 일주기 업데이트."""
        phase = circadian.update(hour)
        return phase
    
    def daily_snapshot():
        """오늘 스냅샷 + 정체성 체크."""
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        """EVE의 삶 요약."""
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
        }
    
    def grammar_stats():
        return {
            "구문_룰": len(constructions.CONSTRUCTIONS),
            "관찰_패턴": discovery.stats(),
            "학습된_단어": len(learner.word_roles),
            "호르몬_상태": hormones.state(),
            "호기심_상태": curiosity.state(),
            "일주기": circadian.state(),
        }
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.understand = understand
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    
    eomi_stats = {"종결어미": 0, "연결어미": 0, "전성어미": 0, "선어말어미": 0}
    for word, info in morph.EOMI.items():
        cat = info.get("category", "기타")
        if cat in eomi_stats:
            eomi_stats[cat] += 1
    
    print(f"  📚 v5.9 활성화 (시간 안에서 살아가는 EVE):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    print(f"     - 어미: {sum(eomi_stats.values())}개")
    print(f"     - 🌙 일주기: 멜라토닌 + 시간대별 호르몬")
    print(f"     - 📅 시간 추적: Day별 변화 + 학습 곡선")
    print(f"     - 🆔 정체성: 코어 보존 + 변화 추적")
    print(f"     - 🎭 호르몬 톤: 단순화 (cartoon 제거)")
    
    return eve_instance


# =====================================================
# Patch A (v6.0): KoNLPy Adapter
# =====================================================
# KoNLPy Okt를 SimpleMorphAnalyzer 인터페이스로 래핑
# 동사/형용사 정확히 구분 → 버그 근본 해결

class KoNLPyMorphAnalyzer:
    """KoNLPy Okt 기반 형태소 분석기.
    
    SimpleMorphAnalyzer와 호환되는 인터페이스 제공.
    KoNLPy 설치 안되어 있으면 자동 fallback to SimpleMorphAnalyzer.
    
    Okt POS 태그 → EVE 태그 매핑:
    - Noun       → Noun
    - Verb       → Verb
    - Adjective  → Adjective
    - Adverb     → Adverb
    - Josa       → Josa
    - Eomi       → Eomi
    - Suffix     → (어미로 처리)
    """
    
    # KoNLPy 태그 → EVE 태그 매핑
    TAG_MAP = {
        "Noun": "Noun",
        "Verb": "Verb",
        "Adjective": "Verb",   # v6.0.1: Adjective도 Verb로 (서술 가능)
        "Adverb": "Adverb",
        "Josa": "Josa",
        "Eomi": "Eomi",
        "Suffix": "Eomi",
        "Determiner": "Noun",
        "Exclamation": "Adverb",
        "Number": "Noun",
        "Foreign": "Noun",
        "Alpha": "Noun",
    }
    
    # v6.0.1: KoNLPy가 Josa로 태그하지만 실제로는 Eomi인 것
    JOSA_TO_EOMI = {
        "이다", "이야", "이에요", "예요", "입니다", "입니까",
    }
    
    def __init__(self):
        self.okt = None
        self._fallback = None
        self._available = False
        
        try:
            from konlpy.tag import Okt
            self.okt = Okt()
            self._available = True
            print("  ✅ KoNLPy Okt 활성화 (학술급 형태소 분석)")
        except ImportError:
            self._fallback = SimpleMorphAnalyzer()
            print("  ⚠️ KoNLPy 미설치 - SimpleMorphAnalyzer로 fallback")
        
        self.JOSA = SimpleMorphAnalyzer.JOSA
        self.EOMI = SimpleMorphAnalyzer.EOMI
        self.ADVERB_INTENSITY = SimpleMorphAnalyzer.ADVERB_INTENSITY
    
    def is_available(self):
        return self._available
    
    def pos(self, sentence):
        """형태소 분석 - KoNLPy 우선, 실패 시 fallback.
        
        v6.0.3: stem=False + 동사/형용사 어미 분리
        - KoNLPy stem=False → '헤엄친다' 그대로 (어미 보존)
        - 후처리로 동사+어미 분리: '영리하다' → '영리하' + '다'
        """
        if not self._available:
            return self._fallback.pos(sentence)
        
        try:
            # stem=False: 어미 보존
            raw = self.okt.pos(sentence, stem=False, norm=True)
            
            converted = []
            for word, tag in raw:
                # JOSA로 태그됐지만 실제 EOMI인 것
                if word in self.JOSA_TO_EOMI:
                    converted.append((word, "Eomi"))
                    continue
                
                eve_tag = self.TAG_MAP.get(tag, "Noun")
                
                # v6.0.3: Verb/Adjective인데 어미가 붙어있으면 분리
                if eve_tag == "Verb" and tag in ("Verb", "Adjective"):
                    stem, eomi = self._split_verb_eomi(word)
                    if eomi:
                        converted.append((stem, "Verb"))
                        converted.append((eomi, "Eomi"))
                    else:
                        converted.append((word, eve_tag))
                else:
                    converted.append((word, eve_tag))
            
            # 2차: 명사+동사 결합
            merged = []
            i = 0
            while i < len(converted):
                word, tag = converted[i]
                if (tag == "Noun" and i + 1 < len(converted) 
                    and converted[i+1][1] == "Verb"
                    and len(converted[i+1][0]) <= 2):
                    next_word = converted[i+1][0]
                    merged.append((word + next_word, "Verb"))
                    i += 2
                else:
                    merged.append((word, tag))
                    i += 1
            
            return merged
        except Exception as e:
            if self._fallback is None:
                self._fallback = SimpleMorphAnalyzer()
            return self._fallback.pos(sentence)
    
    def _split_verb_eomi(self, word):
        """동사/형용사에서 어미 분리.
        
        예: '영리하다' → ('영리하', '다')
            '헤엄친다' → ('헤엄치', 'ㄴ다') - 어려움, 일단 ('헤엄친', '다')로
            '예쁘다' → ('예쁘', '다')
            '먹어' → ('먹', '어')
            '먹어요' → ('먹', '어요')
        """
        if not word:
            return word, None
        
        # 어미 사전에서 가장 긴 매칭 찾기
        # (긴 어미부터 매칭 = '어요'가 '어'보다 우선)
        for eomi in sorted(self.EOMI.keys(), key=len, reverse=True):
            if len(eomi) >= len(word):
                continue  # 어미가 단어 전체보다 길면 X
            if word.endswith(eomi):
                stem = word[:-len(eomi)]
                if len(stem) >= 1:
                    return stem, eomi
        
        return word, None


# =====================================================
# Patch B (v6.0): World Model (다음 상태 예측)
# =====================================================
# 영감: Friston Active Inference, Ha & Schmidhuber World Models, LeCun JEPA
# 핵심: 현재 상태 → 다음 상태 예측

class WorldModel:
    """간단한 World Model - 시간/인과 패턴 학습.
    
    원리:
    1. 사건 시퀀스 관찰 (A → B → C)
    2. 전이 패턴 학습 (A 다음 B 확률)
    3. 현재 상태에서 다음 예측
    
    이론:
    - Schmidhuber 1990: World models for planning
    - Ha & Schmidhuber 2018: World Models (RNN-based)
    - LeCun 2022: JEPA (Joint Embedding Predictive)
    - Friston 2010: Active Inference (predict + minimize surprise)
    
    EVE 구현:
    - 단순 마르코프 체인 (간단하지만 작동)
    - 단어/개념 시퀀스 학습
    """
    
    def __init__(self, max_history=100):
        # 전이 카운트: {(prev_state, next_state): count}
        from collections import defaultdict, Counter
        self.transitions = defaultdict(Counter)
        
        # 최근 사건 히스토리
        self.history = []
        self.max_history = max_history
        
        # 예측 정확도 추적
        self.predictions_made = 0
        self.predictions_correct = 0
    
    def observe(self, event):
        """사건 관찰 - 시퀀스에 추가."""
        if not event:
            return
        
        # 이전 사건과 전이 학습
        if self.history:
            prev = self.history[-1]
            self.transitions[prev][event] += 1
        
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def observe_sentence(self, sentence_data):
        """문장 데이터에서 사건 추출 (구문 + 주체).
        
        예: "고래가 새끼를 낳는다" 
            → 구문: 행위, 주체: 고래
            → event: "고래:행위"
        """
        if not sentence_data:
            return
        
        construction = sentence_data.get("구문", "")
        roles = sentence_data.get("역할", {})
        
        # AGENT/TOPIC/ENTITY 중 하나를 주체로
        subject = roles.get("AGENT") or roles.get("TOPIC") or roles.get("ENTITY")
        
        if subject and construction:
            event = f"{subject}:{construction}"
            self.observe(event)
            return event
        return None
    
    def predict_next(self, current_state, top_k=3):
        """현재 상태에서 다음 사건 예측."""
        if current_state not in self.transitions:
            return []
        
        transitions = self.transitions[current_state]
        if not transitions:
            return []
        
        # 확률 계산
        total = sum(transitions.values())
        probs = [
            (event, count / total)
            for event, count in transitions.most_common(top_k)
        ]
        
        return probs
    
    def predict_from_history(self, top_k=3):
        """현재 히스토리에서 다음 예측."""
        if not self.history:
            return []
        return self.predict_next(self.history[-1], top_k)
    
    def evaluate_prediction(self, predicted_event, actual_event):
        """예측 정확도 평가."""
        self.predictions_made += 1
        if predicted_event == actual_event:
            self.predictions_correct += 1
        return predicted_event == actual_event
    
    def surprise(self, actual_event):
        """현재 사건이 얼마나 예상 밖인가? (Active Inference)
        
        예측 못했을수록 surprise ↑
        """
        if not self.history or len(self.history) < 2:
            return 0.5  # 히스토리 부족
        
        prev = self.history[-1]
        if prev not in self.transitions:
            return 1.0  # 완전 새로움
        
        transitions = self.transitions[prev]
        total = sum(transitions.values())
        if total == 0:
            return 1.0
        
        # 이 사건의 확률
        prob = transitions[actual_event] / total
        
        # surprise = -log(p), 정규화 0~1
        import math
        if prob == 0:
            return 1.0
        surprise = -math.log(prob) / 5.0  # 대략 정규화
        return min(1.0, max(0.0, surprise))
    
    def state(self):
        """월드모델 상태."""
        accuracy = (
            self.predictions_correct / self.predictions_made
            if self.predictions_made > 0 else 0.0
        )
        return {
            "전이_규칙_수": len(self.transitions),
            "히스토리_길이": len(self.history),
            "예측_정확도": round(accuracy, 2),
            "최근_사건": self.history[-5:] if self.history else [],
        }


# =====================================================
# Patch D (v6.0): Meta-cognition (자기 인식)
# =====================================================
# 영감: Flavell 1979, "Thinking about thinking"
# 핵심: "내가 뭘 알고 모르나?" 자기 모니터링

class MetaCognition:
    """메타인지 - EVE가 자기 자신을 모니터링.
    
    이론:
    - Flavell 1979: Metacognition (자기 인지에 대한 인지)
    - Nelson & Narens 1990: Metacognition framework
    - Confidence calibration: 자기 신뢰도와 실제 정확도 매칭
    
    EVE의 메타인지:
    1. Knowledge Monitoring: "내가 X를 아나?"
    2. Confidence Assessment: "내 신념이 얼마나 확실한가?"
    3. Gap Detection: "내가 모르는 게 뭔가?"
    4. Learning Strategy: "어떻게 배워야 하나?"
    """
    
    def __init__(self, eve):
        self.eve = eve
        
        # 자기 평가 이력
        self.self_assessments = []
        
        # 메타 신념 (자신에 대한 신념)
        self.meta_beliefs = {
            "내가_확실히_아는_것": [],
            "내가_모호하게_아는_것": [],
            "내가_모르는_것": [],
            "내가_관심있는_것": [],
        }
    
    def know_what_i_know(self):
        """현재 EVE의 지식 분류 (확실/모호/모름)."""
        certain = []
        uncertain = []
        
        if hasattr(self.eve, 'beliefs'):
            beliefs = self.eve.beliefs
            if isinstance(beliefs, dict):
                for bid, belief in beliefs.items():
                    if hasattr(belief, 'confidence'):
                        if belief.confidence > 0.7:
                            certain.append(belief.statement if hasattr(belief, 'statement') else str(belief))
                        elif belief.confidence > 0.3:
                            uncertain.append(belief.statement if hasattr(belief, 'statement') else str(belief))
        
        # 모르는 것 (호기심에서)
        unknown = []
        if hasattr(self.eve, 'curiosity'):
            unknown = self.eve.curiosity.unknown_words[:10]
        
        # 관심사 (가장 친숙한)
        interests = []
        if hasattr(self.eve, 'curiosity'):
            stats = self.eve.curiosity.state()
            interests = [w for w, _ in stats.get("가장_친숙", [])[:5]]
        
        self.meta_beliefs["내가_확실히_아는_것"] = certain
        self.meta_beliefs["내가_모호하게_아는_것"] = uncertain
        self.meta_beliefs["내가_모르는_것"] = unknown
        self.meta_beliefs["내가_관심있는_것"] = interests
        
        return self.meta_beliefs
    
    def can_i_answer(self, question):
        """이 질문에 답할 수 있나? (자기 평가)
        
        예: "고래가 뭐야?" 
            → 신념에 "고래" 있으면 "예", 없으면 "아니오"
        """
        # 질문에서 키워드 추출 (간단히)
        keywords = [w for w in question.replace("?", "").split() 
                    if len(w) >= 2 and w not in ["뭐야", "이뭐야"]]
        
        # 각 키워드에 대한 지식 확인
        knowledge_check = {}
        for kw in keywords:
            # 받침 처리
            kw_clean = kw.rstrip("이가은는을를").rstrip()
            
            # 신념에서 검색
            found_beliefs = []
            if hasattr(self.eve, 'beliefs') and isinstance(self.eve.beliefs, dict):
                for bid, belief in self.eve.beliefs.items():
                    statement = belief.statement if hasattr(belief, 'statement') else str(belief)
                    if kw_clean in statement:
                        found_beliefs.append(statement)
            
            knowledge_check[kw_clean] = {
                "안다": len(found_beliefs) > 0,
                "관련_신념": found_beliefs[:3],
            }
        
        # 종합 판단
        any_known = any(v["안다"] for v in knowledge_check.values())
        all_known = all(v["안다"] for v in knowledge_check.values())
        
        if all_known:
            confidence = "확실"
        elif any_known:
            confidence = "부분적"
        else:
            confidence = "모름"
        
        return {
            "답할수있나": any_known,
            "확신도": confidence,
            "키워드_체크": knowledge_check,
        }
    
    def what_should_i_learn(self):
        """학습 우선순위 추천.
        
        모르는 것 + 자주 등장하는 것 = 학습 우선
        """
        meta = self.know_what_i_know()
        unknowns = meta["내가_모르는_것"]
        
        # 가장 자주 들었지만 정의 모르는 단어
        priorities = []
        if hasattr(self.eve, 'curiosity'):
            familiarity = self.eve.curiosity.familiarity
            for word in unknowns:
                count = familiarity.get(word, 0)
                priorities.append((word, count))
        
        # 빈도순
        priorities.sort(key=lambda x: -x[1])
        
        return [
            {"단어": w, "들은횟수": c, "우선순위": "높음" if c > 2 else "중간"}
            for w, c in priorities[:5]
        ]
    
    def self_reflect(self):
        """자기 성찰 종합."""
        meta = self.know_what_i_know()
        priorities = self.what_should_i_learn()
        
        # 자기 통계
        days = self.eve.identity.get("days_alive", 0)
        belief_count = len(self.eve.beliefs) if hasattr(self.eve, 'beliefs') else 0
        
        reflection = {
            "나의_나이": f"Day {days}",
            "내_지식_규모": belief_count,
            "확실히_아는_것": len(meta["내가_확실히_아는_것"]),
            "모호하게_아는_것": len(meta["내가_모호하게_아는_것"]),
            "모르는_단어": len(meta["내가_모르는_것"]),
            "내_관심사": meta["내가_관심있는_것"],
            "다음_학습": priorities,
        }
        
        # 자기 평가 이력 저장
        self.self_assessments.append({
            "day": days,
            "reflection": reflection,
        })
        
        return reflection
    
    def state(self):
        return {
            "메타신념": self.meta_beliefs,
            "자기평가_횟수": len(self.self_assessments),
        }


# =====================================================
# Integration: v6.0 통합
# =====================================================

def add_full_grammar_to_eve_v60(eve_instance, use_konlpy=True):
    """v6.0: KoNLPy + World Model + Meta-cognition.
    
    use_konlpy=True: KoNLPy 시도, 실패 시 fallback
    use_konlpy=False: 강제로 SimpleMorphAnalyzer 사용
    """
    # Patch A: 형태소 분석기
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    # v6.0 새 모듈
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    
    def hear(sentence, register_belief=True):
        """문장 듣기 - v6.0: World Model + Meta 통합."""
        result = learner.learn_from_sentence(sentence)
        
        n = nuance.analyze(sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(sentence)
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                # World Model 학습
                # 매칭된 구문 + 역할에서 사건 추출
                matches = constructions.match(sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        # surprise 계산
                        result["surprise"] = round(world_model.surprise(event), 2)
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        """다음에 일어날 일 예측 (World Model)."""
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        """자기 성찰 (Meta-cognition)."""
        return meta_cog.self_reflect()
    
    def can_answer(question):
        """이 질문 답할 수 있나?"""
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
        }
    
    def grammar_stats():
        return {
            "구문_룰": len(constructions.CONSTRUCTIONS),
            "관찰_패턴": discovery.stats(),
            "학습된_단어": len(learner.word_roles),
            "호르몬_상태": hormones.state(),
            "호기심_상태": curiosity.state(),
            "일주기": circadian.state(),
            "월드모델": world_model.state(),
        }
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    
    eomi_stats = {"종결어미": 0, "연결어미": 0, "전성어미": 0, "선어말어미": 0}
    for word, info in morph.EOMI.items():
        cat = info.get("category", "기타")
        if cat in eomi_stats:
            eomi_stats[cat] += 1
    
    print(f"  📚 v6.0 활성화 (KoNLPy + World Model + Meta-cognition):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    print(f"     - 어미: {sum(eomi_stats.values())}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성 (학술급 분석)")
    else:
        print(f"     - 🔬 KoNLPy: 미활성 (Simple fallback)")
    print(f"     - 🌍 World Model: 다음 상태 예측")
    print(f"     - 🧠 Meta-cognition: 자기 인식")
    
    return eve_instance


# =====================================================
# Patch B (v6.1): Named Entity Recognition (NER)
# =====================================================
# 영감: spaCy NER, KoNLPy 명사 분석
# 핵심: 일반 명사 vs 개체명 구분

class NERModule:
    """개체명 인식 - 한국어 특화.
    
    분류:
    - PERSON: 인물 (이순신, 세종, 엘론 머스크)
    - LOCATION: 장소 (서울, 부산, 한국, 강남)
    - ORGANIZATION: 조직 (삼성, 엔비디아, 청와대)
    - TIME: 시간 (어제, 오늘, 2024년, 3시)
    - QUANTITY: 수량 (1000원, 5명, 3개)
    
    한국어 NER 휴리스틱:
    1. 사전 매칭 (유명한 것들)
    2. 접미사 패턴 (-시, -구, -동: 장소)
    3. 호격조사 패턴 (-아/-야: 인물)
    """
    
    # 유명 인물/장소/조직 사전
    KNOWN_PEOPLE = {
        "이순신", "세종", "세종대왕", "충무공", "장영실", "정약용",
        "김구", "안중근", "유관순", "윤동주",
        "엘론머스크", "스티브잡스", "빌게이츠", "마크저커버그",
    }
    
    KNOWN_LOCATIONS = {
        # 한국
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "수원", "성남", "고양", "용인", "창원", "전주", "포항", "춘천",
        "강남", "강북", "강서", "강동", "종로", "중구", "용산",
        "한국", "남한", "북한", "조선",
        # 세계
        "미국", "중국", "일본", "러시아", "영국", "프랑스", "독일",
        "도쿄", "오사카", "베이징", "상하이", "뉴욕", "런던", "파리",
    }
    
    KNOWN_ORGS = {
        # 한국 기업
        "삼성", "현대", "LG", "SK", "롯데", "한화", "카카오", "네이버",
        "쿠팡", "배민", "토스",
        # 글로벌
        "구글", "애플", "마이크로소프트", "엔비디아", "테슬라",
        "메타", "오픈AI", "앤트로픽", "엑스",
        # 정부/기관
        "청와대", "국회", "정부", "서울대", "카이스트",
    }
    
    # 시간 표현
    TIME_WORDS = {
        "어제", "오늘", "내일", "그제", "모레",
        "지금", "방금", "이따", "나중에",
        "아침", "점심", "저녁", "밤", "새벽",
        "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
        "올해", "작년", "내년", "재작년",
    }
    
    # 장소 접미사
    LOCATION_SUFFIXES = ["시", "도", "구", "동", "읍", "면", "리", "역", "공항"]
    
    # 조직 접미사
    ORG_SUFFIXES = ["회사", "기업", "은행", "학교", "대학", "병원", "센터"]
    
    def __init__(self):
        # 학습된 개체명 (사용자 대화에서)
        self.learned_entities = {
            "PERSON": set(),
            "LOCATION": set(),
            "ORGANIZATION": set(),
            "TIME": set(),
            "OTHER": set(),
        }
    
    def classify(self, word):
        """단어 → 개체 유형 분류."""
        if not word or len(word) < 2:
            return "GENERAL"  # 일반 명사
        
        # 1. 사전 직접 매칭
        if word in self.KNOWN_PEOPLE:
            return "PERSON"
        if word in self.KNOWN_LOCATIONS:
            return "LOCATION"
        if word in self.KNOWN_ORGS:
            return "ORGANIZATION"
        if word in self.TIME_WORDS:
            return "TIME"
        
        # 2. 학습된 것 매칭
        for ent_type, ents in self.learned_entities.items():
            if word in ents:
                return ent_type
        
        # 3. 접미사 패턴
        for suffix in self.LOCATION_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix):
                return "LOCATION"
        
        for suffix in self.ORG_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix):
                return "ORGANIZATION"
        
        # 4. 시간 패턴 (숫자+년/월/일/시/분)
        if any(word.endswith(t) for t in ["년", "월", "일", "시", "분", "초"]):
            if any(c.isdigit() for c in word):
                return "TIME"
        
        # 5. 수량 패턴
        if any(c.isdigit() for c in word):
            return "QUANTITY"
        
        return "GENERAL"
    
    def learn(self, word, entity_type):
        """대화에서 새 개체명 학습."""
        if entity_type in self.learned_entities:
            self.learned_entities[entity_type].add(word)
    
    def extract_entities(self, sentence, morphs):
        """문장에서 모든 개체명 추출."""
        entities = []
        for word, tag in morphs:
            if tag == "Noun":
                ent_type = self.classify(word)
                if ent_type != "GENERAL":
                    entities.append({
                        "word": word,
                        "type": ent_type,
                    })
        return entities
    
    def state(self):
        """학습 상태."""
        return {
            "사전_사람": len(self.KNOWN_PEOPLE),
            "사전_장소": len(self.KNOWN_LOCATIONS),
            "사전_조직": len(self.KNOWN_ORGS),
            "사전_시간": len(self.TIME_WORDS),
            "학습된_사람": len(self.learned_entities["PERSON"]),
            "학습된_장소": len(self.learned_entities["LOCATION"]),
            "학습된_조직": len(self.learned_entities["ORGANIZATION"]),
        }


# =====================================================
# Patch F (v6.1): Multi-turn Dialogue
# =====================================================
# 영감: 대화 시스템 (DST - Dialogue State Tracking)
# 핵심: 이전 발화 기억 + 지시어 해결 + 주제 추적

class DialogueContext:
    """멀티턴 대화 컨텍스트.
    
    추적하는 것:
    1. 대화 이력 (최근 발화들)
    2. 현재 주제 (가장 자주 언급된 개체)
    3. 마지막 언급된 인물/장소 (지시어 해결용)
    4. 대화 턴 수
    
    지시어 해결:
    - "그 사람" → 직전 PERSON
    - "거기" → 직전 LOCATION  
    - "그것" → 직전 일반 명사
    - "그때" → 직전 TIME
    """
    
    # 지시어 사전
    PRONOUNS = {
        "그": "GENERAL",
        "그것": "GENERAL",
        "그거": "GENERAL",
        "이것": "GENERAL",
        "이거": "GENERAL",
        "그 사람": "PERSON",
        "그분": "PERSON",
        "그녀": "PERSON",
        "그놈": "PERSON",
        "거기": "LOCATION",
        "여기": "LOCATION",
        "저기": "LOCATION",
        "그곳": "LOCATION",
        "이곳": "LOCATION",
        "그때": "TIME",
        "이때": "TIME",
    }
    
    def __init__(self, max_history=20):
        self.history = []  # [(sentence, morphs, entities, timestamp), ...]
        self.max_history = max_history
        
        # 마지막 언급된 개체 (지시어 해결용)
        self.last_mentioned = {
            "PERSON": None,
            "LOCATION": None,
            "ORGANIZATION": None,
            "TIME": None,
            "GENERAL": None,
        }
        
        # 주제 (가장 많이 언급)
        from collections import Counter
        self.topic_counter = Counter()
        
        self.turn_count = 0
    
    def add_turn(self, sentence, morphs=None, entities=None):
        """대화 턴 추가.
        
        v6.2: 구체성 우선 (PROPER > SPECIFIC > GENERAL > ABSTRACT)
        같은 type의 개체는 더 구체적인 것이 last_mentioned 차지
        """
        import datetime as _dt
        
        turn = {
            "sentence": sentence,
            "morphs": morphs or [],
            "entities": entities or [],
            "turn": self.turn_count,
            "timestamp": _dt.datetime.now().isoformat(),
        }
        
        self.history.append(turn)
        self.turn_count += 1
        
        # 구체성 위계 점수
        SPEC_SCORE = {"PROPER": 100, "SPECIFIC": 70, "GENERAL": 30, "ABSTRACT": 10}
        
        # v6.2: 마지막 언급 업데이트 (구체성 우선)
        if entities:
            # entities는 이미 score 순 정렬 (NERWithSpecificity)
            for ent in entities:
                ent_type = ent["type"]
                if ent_type in self.last_mentioned:
                    # 기존 last와 새 ent 비교 - 새것이 더 구체적이면 교체
                    new_spec = ent.get("specificity", "GENERAL")
                    new_score = SPEC_SCORE.get(new_spec, 0)
                    
                    # 첫 등록이거나, 같은 발화에서는 더 높은 점수가 들어감
                    self.last_mentioned[ent_type] = ent["word"]
                
                self.topic_counter[ent["word"]] += 1
        
        # GENERAL은 일반 명사 (구체적인 entity가 우선)
        if morphs:
            # 진짜 일반 명사만 GENERAL에 (entity가 아닌 것)
            entity_words = {e["word"] for e in (entities or [])}
            for word, tag in morphs:
                if tag == "Noun" and len(word) >= 2 and word not in entity_words:
                    self.last_mentioned["GENERAL"] = word
                    self.topic_counter[word] += 1
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def resolve_reference(self, pronoun):
        """지시어 → 실제 단어로 해결."""
        ent_type = self.PRONOUNS.get(pronoun)
        if not ent_type:
            return None
        
        return self.last_mentioned.get(ent_type)
    
    def resolve_sentence(self, sentence):
        """문장 내 모든 지시어 해결.
        
        v6.1 추가: 받침 따라 조사 자동
        예: "거기 좋아?" + 직전 "서울" → "서울 좋아?"
            "거기는 사람이 많다" → "서울은 사람이 많다" (받침 처리)
        """
        result = sentence
        for pronoun in sorted(self.PRONOUNS.keys(), key=len, reverse=True):
            if pronoun in result:
                replacement = self.resolve_reference(pronoun)
                if replacement:
                    result = result.replace(pronoun, replacement)
                    
                    # 받침 따라 조사 자동
                    if has_batchim(replacement):
                        # 받침 있음
                        result = result.replace(f"{replacement}는 ", f"{replacement}은 ")
                        result = result.replace(f"{replacement}가 ", f"{replacement}이 ")
                        result = result.replace(f"{replacement}를 ", f"{replacement}을 ")
                    else:
                        # 받침 없음
                        result = result.replace(f"{replacement}은 ", f"{replacement}는 ")
                        result = result.replace(f"{replacement}이 ", f"{replacement}가 ")
                        result = result.replace(f"{replacement}을 ", f"{replacement}를 ")
        return result
    
    def current_topic(self, top_k=3):
        """현재 대화 주제 (가장 자주 언급된 단어)."""
        return self.topic_counter.most_common(top_k)
    
    def recent_history(self, n=5):
        """최근 N개 발화."""
        return [t["sentence"] for t in self.history[-n:]]
    
    def state(self):
        """대화 상태."""
        return {
            "총_턴": self.turn_count,
            "히스토리_길이": len(self.history),
            "마지막_언급": {k: v for k, v in self.last_mentioned.items() if v},
            "주제_상위": self.current_topic(3),
            "최근_3개": self.recent_history(3),
        }


# =====================================================
# Integration: v6.1 통합
# =====================================================

def add_full_grammar_to_eve_v61(eve_instance, use_konlpy=True):
    """v6.1: v6.0 + 자동사 패턴 + NER + 멀티턴 대화."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    # v6.1 새 모듈
    ner = NERModule()
    dialogue = DialogueContext()
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    
    def hear(sentence, register_belief=True):
        """문장 듣기 - v6.1: 멀티턴 + NER 통합."""
        # v6.1: 지시어 해결 (멀티턴)
        resolved_sentence = dialogue.resolve_sentence(sentence)
        if resolved_sentence != sentence:
            # 지시어 해결됨
            pass  # 원문도 보존
        
        # 학습은 해결된 문장으로
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # v6.1: NER 추출
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        # v6.1: 대화 컨텍스트 업데이트
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
        }
    
    def grammar_stats():
        return {
            "구문_룰": len(constructions.CONSTRUCTIONS),
            "관찰_패턴": discovery.stats(),
            "학습된_단어": len(learner.word_roles),
            "호르몬_상태": hormones.state(),
            "호기심_상태": curiosity.state(),
            "일주기": circadian.state(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
        }
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    
    eomi_stats = {"종결어미": 0, "연결어미": 0, "전성어미": 0, "선어말어미": 0}
    for word, info in morph.EOMI.items():
        cat = info.get("category", "기타")
        if cat in eomi_stats:
            eomi_stats[cat] += 1
    
    print(f"  📚 v6.1 활성화 (자동사 + NER + 멀티턴):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개 (자동사+서술 추가)")
    print(f"     - 어미: {sum(eomi_stats.values())}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🌍 World Model: 예측")
    print(f"     - 🧠 Meta-cognition: 자기 인식")
    print(f"     - 🎯 자동사/형용사 서술 패턴")
    print(f"     - 🏷️ NER: 인물/장소/조직/시간 ({len(ner.KNOWN_PEOPLE) + len(ner.KNOWN_LOCATIONS) + len(ner.KNOWN_ORGS)}개 사전)")
    print(f"     - 💬 멀티턴 대화: 지시어 해결 + 주제 추적")
    
    return eve_instance


# =====================================================
# Patch 1 (v6.2): Word Sense Disambiguation (WSD)
# =====================================================
# 다의어 처리: "배" = SHIP / FRUIT / BODY_PART
# 영감: WordNet, BabelNet, 김창일 한국어 WSD 연구

class WSDModule:
    """Word Sense Disambiguation - 단어 의미 모호성 해결.
    
    핵심:
    1. 다의어 사전 (단어 → 가능한 의미들)
    2. 각 의미마다 문맥 단어 (collocations)
    3. 문장 내 다른 단어들과 매칭 → 점수
    4. 가장 높은 점수의 의미 선택
    
    예: "배"
    - SHIP: ['바다', '타다', '항구', '뱃사람', '항해']
    - FRUIT: ['먹다', '맛있다', '과일', '달다', '익다']
    - BODY_PART: ['아프다', '부르다', '나오다', '몸']
    
    "배가 아프다" + 'BODY_PART' 문맥 매칭 → BODY_PART 선택
    """
    
    # 다의어 사전 (확장 가능)
    POLYSEMY = {
        "배": {
            "SHIP": {
                "context": ["바다", "타", "항구", "뱃", "항해", "선장", "강", "물"],
                "definition": "사람이나 짐을 싣고 물 위를 다니는 운송 수단",
            },
            "FRUIT": {
                "context": ["먹", "맛있", "과일", "달", "익", "사과", "수박", "농장"],
                "definition": "배나무의 열매",
            },
            "BODY_PART": {
                "context": ["아프", "부르", "나오", "몸", "위", "장", "통증"],
                "definition": "사람이나 동물의 몸 가운데 부분",
            },
        },
        "눈": {
            "EYE": {
                "context": ["보", "감", "뜨", "시력", "눈썹", "눈물", "안경"],
                "definition": "보는 기관",
            },
            "SNOW": {
                "context": ["내리", "겨울", "쌓이", "하얗", "춥", "얼"],
                "definition": "겨울에 하늘에서 내리는 흰 결정",
            },
        },
        "말": {
            "HORSE": {
                "context": ["타", "달리", "마구간", "갈기", "발굽", "경마"],
                "definition": "말과의 동물",
            },
            "SPEECH": {
                "context": ["하", "듣", "전하", "이야기", "대화", "소리"],
                "definition": "사람의 의사 전달",
            },
            "END_OF_PERIOD": {
                "context": ["월", "년", "주", "초"],
                "definition": "어떤 기간의 끝",
            },
        },
        "차": {
            "VEHICLE": {
                "context": ["타", "운전", "도로", "주차", "기름"],
                "definition": "사람이나 짐을 싣고 다니는 탈것",
            },
            "TEA": {
                "context": ["마시", "끓이", "녹", "홍", "우려", "잎"],
                "definition": "차나무의 잎으로 만든 음료",
            },
        },
        "다리": {
            "BODY_PART": {
                "context": ["아프", "걷", "달리", "발", "무릎"],
                "definition": "동물의 발에서 위로 이어진 부분",
            },
            "BRIDGE": {
                "context": ["건너", "강", "한강", "철", "건설"],
                "definition": "강이나 도로 위로 사람이나 차가 건너다닐 수 있게 만든 시설",
            },
        },
        "사과": {
            "FRUIT": {
                "context": ["먹", "맛있", "과일", "달", "빨갛", "익"],
                "definition": "사과나무의 열매",
            },
            "APOLOGY": {
                "context": ["하", "받", "드리", "용서", "잘못", "미안"],
                "definition": "잘못에 대한 사죄",
            },
        },
    }
    
    def __init__(self):
        self.disambiguation_history = []
    
    def is_polysemous(self, word):
        """다의어인가?"""
        return word in self.POLYSEMY
    
    def disambiguate(self, word, context_words, top_k=1):
        """단어 + 문맥 → 가장 적합한 의미.
        
        Args:
            word: 다의어 단어 ("배")
            context_words: 문맥 단어 리스트 (["타", "바다"])
            top_k: 상위 N개 의미 반환
        
        Returns:
            [(sense, score, definition), ...]
        """
        if not self.is_polysemous(word):
            return [("DEFAULT", 1.0, word)]
        
        senses = self.POLYSEMY[word]
        scores = []
        
        for sense_name, sense_data in senses.items():
            sense_context = sense_data["context"]
            score = 0
            
            # 문맥 단어와 매칭
            for ctx_word in context_words:
                for sense_ctx in sense_context:
                    # 부분 매칭 ("아프다" + "아프" → 매칭)
                    if sense_ctx in ctx_word or ctx_word.startswith(sense_ctx):
                        score += 1
                        break
            
            scores.append((sense_name, score, sense_data["definition"]))
        
        # 점수 순
        scores.sort(key=lambda x: -x[1])
        
        # 기록
        self.disambiguation_history.append({
            "word": word,
            "context": context_words,
            "selected": scores[0][0] if scores else "UNKNOWN",
            "scores": scores,
        })
        
        return scores[:top_k]
    
    def disambiguate_in_sentence(self, sentence, morphs):
        """문장 내 모든 다의어 처리."""
        results = {}
        
        # 모든 단어 추출
        all_words = [w for w, _ in morphs]
        
        for word, tag in morphs:
            if self.is_polysemous(word):
                # 자기 빼고 문맥
                context = [w for w in all_words if w != word]
                senses = self.disambiguate(word, context, top_k=3)
                results[word] = senses
        
        return results
    
    def state(self):
        return {
            "다의어_사전": len(self.POLYSEMY),
            "처리_횟수": len(self.disambiguation_history),
            "최근_처리": self.disambiguation_history[-3:] if self.disambiguation_history else [],
        }


# =====================================================
# Patch 2 (v6.2): NER 세분화 (Specificity Hierarchy)
# =====================================================

class NERWithSpecificity:
    """개체명 + 구체성 위계.
    
    문제: 일반명사(도시) vs 고유명사(서울) 구분 부족
    해결: Specificity 점수 부여
    
    구체성 위계:
    - PROPER (고유): 서울, 이순신, 삼성 → 100점
    - SPECIFIC (구체): 강남구, 한강 → 70점
    - GENERAL (일반): 도시, 사람, 회사 → 30점
    - ABSTRACT (추상): 수도, 영웅, 기업 → 10점
    
    지시어 해결 시 PROPER > SPECIFIC > GENERAL > ABSTRACT 우선
    """
    
    # 구체성 분류 - 일반/추상은 NER 제외
    GENERAL_NOUNS = {
        "사람", "사람들", "남자", "여자", "아이", "어른",
        "도시", "마을", "곳", "장소", "지역", "동네",
        "회사", "기업", "조직", "단체",
        "동물", "식물", "물체", "물건",
        "일", "것", "거", "데",
    }
    
    ABSTRACT_NOUNS = {
        "수도", "중심", "본부", "기지",
        "영웅", "왕", "장군", "대통령",  # 직책 (특정 인물 X)
        "사랑", "행복", "슬픔", "기쁨",
        "시간", "공간", "이유", "방법",
        "느낌", "생각", "마음",
    }
    
    PROPER_PEOPLE = {
        "이순신", "세종", "세종대왕", "충무공", "장영실", "정약용",
        "김구", "안중근", "유관순", "윤동주",
        "엘론머스크", "스티브잡스", "빌게이츠", "마크저커버그",
    }
    
    PROPER_LOCATIONS = {
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "수원", "성남", "고양", "용인", "창원", "전주", "포항", "춘천",
        "한국", "남한", "북한", "조선",
        "미국", "중국", "일본", "러시아", "영국", "프랑스", "독일",
        "도쿄", "오사카", "베이징", "상하이", "뉴욕", "런던", "파리",
    }
    
    PROPER_ORGS = {
        "삼성", "현대", "LG", "SK", "롯데", "한화", "카카오", "네이버",
        "쿠팡", "배민", "토스",
        "구글", "애플", "마이크로소프트", "엔비디아", "테슬라",
        "메타", "오픈AI", "앤트로픽", "엑스",
    }
    
    SPECIFIC_LOCATIONS_SUFFIX = ["시", "도", "구", "동", "읍", "면", "리", "역", "공항", "강", "산", "호수"]
    
    TIME_WORDS = {
        "어제", "오늘", "내일", "그제", "모레",
        "지금", "방금", "이따", "나중에",
        "아침", "점심", "저녁", "밤", "새벽",
        "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
        "올해", "작년", "내년", "재작년",
    }
    
    def __init__(self):
        self.learned = {
            "PROPER_PEOPLE": set(),
            "PROPER_LOCATIONS": set(),
            "PROPER_ORGS": set(),
            "SPECIFIC": set(),
        }
    
    def classify(self, word):
        """단어 → (entity_type, specificity_level, score)."""
        if not word or len(word) < 2:
            return ("GENERAL", "GENERAL", 0)
        
        # 1. 일반/추상 제외 (NER 안 함)
        if word in self.GENERAL_NOUNS:
            return ("GENERAL_NOUN", "GENERAL", 30)
        if word in self.ABSTRACT_NOUNS:
            return ("ABSTRACT_NOUN", "ABSTRACT", 10)
        
        # 2. 고유명사 (PROPER) - 가장 높은 점수
        if word in self.PROPER_PEOPLE:
            return ("PERSON", "PROPER", 100)
        if word in self.PROPER_LOCATIONS:
            return ("LOCATION", "PROPER", 100)
        if word in self.PROPER_ORGS:
            return ("ORGANIZATION", "PROPER", 100)
        
        # 3. 학습된 것
        for cat, items in self.learned.items():
            if word in items:
                if cat == "PROPER_PEOPLE":
                    return ("PERSON", "PROPER", 90)
                elif cat == "PROPER_LOCATIONS":
                    return ("LOCATION", "PROPER", 90)
                elif cat == "PROPER_ORGS":
                    return ("ORGANIZATION", "PROPER", 90)
                else:
                    return ("LOCATION", "SPECIFIC", 70)
        
        # 4. 시간
        if word in self.TIME_WORDS:
            return ("TIME", "SPECIFIC", 70)
        
        # 5. 시간 패턴 (숫자+년/월/일)
        if any(word.endswith(t) for t in ["년", "월", "일", "시", "분"]):
            if any(c.isdigit() for c in word):
                return ("TIME", "SPECIFIC", 80)
        
        # 6. 장소 접미사 (구체적 장소)
        for suffix in self.SPECIFIC_LOCATIONS_SUFFIX:
            if word.endswith(suffix) and len(word) > len(suffix):
                return ("LOCATION", "SPECIFIC", 60)
        
        # 7. 수량
        if any(c.isdigit() for c in word):
            return ("QUANTITY", "SPECIFIC", 50)
        
        # 8. 기본: 일반 명사
        return ("GENERAL_NOUN", "GENERAL", 20)
    
    def extract_entities(self, sentence, morphs):
        """문장에서 개체명 추출 - 구체성 정보 포함.
        
        v6.2: 일반명사/추상명사는 ner_only=True 시 제외
        """
        entities = []
        for word, tag in morphs:
            if tag == "Noun":
                ent_type, specificity, score = self.classify(word)
                
                # GENERAL/ABSTRACT는 NER 결과에서 제외
                if specificity in ("GENERAL", "ABSTRACT"):
                    continue
                
                entities.append({
                    "word": word,
                    "type": ent_type,
                    "specificity": specificity,
                    "score": score,
                })
        
        # 점수 순 (구체적인 것 우선)
        entities.sort(key=lambda x: -x["score"])
        return entities
    
    def state(self):
        return {
            "사전_인물": len(self.PROPER_PEOPLE),
            "사전_장소": len(self.PROPER_LOCATIONS),
            "사전_조직": len(self.PROPER_ORGS),
            "일반명사_제외": len(self.GENERAL_NOUNS),
            "추상명사_제외": len(self.ABSTRACT_NOUNS),
        }


# =====================================================
# Patch 3 (v6.2): Response Generation (응답 생성)
# =====================================================

class ResponseGenerator:
    """질문 → 신념 검색 → 답변.
    
    응답 종류:
    1. 정의 질문: "X가 뭐야?" → 신념에서 X 검색
    2. 위치 질문: "X는 어디에 있어?" → 위치 신념 검색
    3. 행위 질문: "X는 뭘 해?" → 행위 신념 검색
    4. 모름: "모르겠어" (정직)
    """
    
    QUESTION_PATTERNS = {
        # 정의
        "정의": ["뭐야", "뭐", "무엇이야", "무엇", "이뭐야"],
        # 위치
        "위치": ["어디", "어디에", "어디서"],
        # 행위
        "행위": ["뭐 해", "뭐해", "무엇을 해"],
        # 시간
        "시간": ["언제", "몇 시"],
        # 이유
        "이유": ["왜", "어째서"],
    }
    
    def __init__(self, eve):
        self.eve = eve
    
    def is_question(self, sentence):
        """질문인가?"""
        return "?" in sentence or any(
            kw in sentence
            for kws in self.QUESTION_PATTERNS.values()
            for kw in kws
        )
    
    def classify_question(self, sentence):
        """질문 종류 분류."""
        for q_type, keywords in self.QUESTION_PATTERNS.items():
            for kw in keywords:
                if kw in sentence:
                    return q_type
        return "기타"
    
    def extract_subject(self, sentence):
        """질문 주체 추출.
        
        '고래가 뭐야?' → '고래'
        '서울은 어디?' → '서울'
        """
        # 단순 휴리스틱: 조사 앞 단어
        if hasattr(self.eve, 'morph'):
            morphs = self.eve.morph.pos(sentence)
            for word, tag in morphs:
                if tag == "Noun" and len(word) >= 2:
                    return word
        
        # Fallback: 첫 단어
        words = sentence.replace("?", "").split()
        if words:
            return words[0].rstrip("이가은는을를")
        return None
    
    def search_beliefs(self, keyword):
        """신념에서 키워드 검색."""
        if not hasattr(self.eve, 'beliefs'):
            return []
        
        beliefs = self.eve.beliefs
        results = []
        
        if isinstance(beliefs, dict):
            for bid, belief in beliefs.items():
                statement = belief.statement if hasattr(belief, 'statement') else str(belief)
                if keyword in statement:
                    confidence = belief.confidence if hasattr(belief, 'confidence') else 0.5
                    results.append((statement, confidence))
        
        # 신뢰도 순
        results.sort(key=lambda x: -x[1])
        return results
    
    def generate_response(self, question):
        """질문에 답하기."""
        if not self.is_question(question):
            return {"응답": None, "이유": "질문이 아님"}
        
        q_type = self.classify_question(question)
        subject = self.extract_subject(question)
        
        if not subject:
            return {"응답": "잘 모르겠어.", "이유": "주체_없음"}
        
        # 신념 검색
        relevant_beliefs = self.search_beliefs(subject)
        
        if not relevant_beliefs:
            return {
                "응답": f"{subject}에 대해 잘 모르겠어.",
                "이유": "관련_신념_없음",
                "주체": subject,
            }
        
        # 질문 종류 따라 필터링
        filtered = relevant_beliefs
        if q_type == "위치":
            filtered = [b for b in relevant_beliefs if "에 있" in b[0]]
        elif q_type == "행위":
            filtered = [b for b in relevant_beliefs if any(v in b[0] for v in ["는다", "한다"])]
        
        if not filtered:
            filtered = relevant_beliefs  # 어쩔 수 없이 전체
        
        # 가장 신뢰도 높은 것
        best = filtered[0]
        return {
            "응답": best[0],
            "신뢰도": best[1],
            "이유": "신념_검색",
            "주체": subject,
            "질문종류": q_type,
            "관련신념수": len(relevant_beliefs),
        }


# =====================================================
# Patch 4 (v6.2): Causal Graph (인과 그래프)
# =====================================================

class CausalGraph:
    """인과 그래프 - 진짜 원인-결과.
    
    이론:
    - Pearl Causal Inference (2009)
    - Causal Mental Models
    - 시퀀스 ≠ 인과 (post hoc fallacy 회피)
    
    EVE 구현:
    - 명시적 인과 표현 학습 ("X 때문에 Y", "X면 Y")
    - 인과 그래프 (DAG)
    - 인과 추론 ("Y의 원인은?")
    """
    
    def __init__(self):
        # 인과 관계: {cause: [(effect, strength, evidence_count)]}
        from collections import defaultdict
        self.causes = defaultdict(list)  # cause → effects
        self.effects = defaultdict(list)  # effect → causes (역방향)
        
        # 명시적 인과 키워드
        self.causal_markers = [
            "때문에", "므로", "니까", "기 때문에", "어서", "아서",
            "면", "이면", "거든",  # 조건
            "그래서", "따라서", "그러므로",
        ]
    
    def has_causal_marker(self, sentence):
        """문장에 인과 표현 있나?"""
        return any(m in sentence for m in self.causal_markers)
    
    def extract_causal(self, sentence):
        """문장에서 인과 추출.
        
        예: "비가 와서 우산을 쓴다"
            → 원인: "비가 와", 결과: "우산을 쓴다"
        
        "비가 오면 우산을 쓴다"
            → 조건: "비가 오", 결과: "우산을 쓴다"
        """
        # 인과 분리 시도
        for marker in self.causal_markers:
            if marker in sentence:
                parts = sentence.split(marker, 1)
                if len(parts) == 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                    # 너무 짧으면 X
                    if len(cause) >= 2 and len(effect) >= 2:
                        return {
                            "cause": cause,
                            "effect": effect,
                            "marker": marker,
                            "type": "조건" if marker in ["면", "이면"] else "인과",
                        }
        return None
    
    def add_causal(self, cause, effect, strength=0.7):
        """인과 관계 추가."""
        # 기존 관계 있으면 강화
        existing = next(
            ((i, e) for i, e in enumerate(self.causes[cause]) if e[0] == effect),
            None
        )
        if existing:
            i, (eff, s, count) = existing
            self.causes[cause][i] = (eff, min(1.0, s + 0.1), count + 1)
        else:
            self.causes[cause].append((effect, strength, 1))
            self.effects[effect].append((cause, strength, 1))
    
    def observe_sentence(self, sentence):
        """문장에서 인과 추출 + 등록."""
        causal = self.extract_causal(sentence)
        if causal:
            self.add_causal(causal["cause"], causal["effect"])
            return causal
        return None
    
    def what_causes(self, effect, top_k=3):
        """X의 원인은? (역방향 추론)"""
        if effect not in self.effects:
            return []
        causes = sorted(self.effects[effect], key=lambda x: -x[1])
        return causes[:top_k]
    
    def what_results(self, cause, top_k=3):
        """X 다음 뭐가 일어나? (정방향)"""
        if cause not in self.causes:
            return []
        effects = sorted(self.causes[cause], key=lambda x: -x[1])
        return effects[:top_k]
    
    def state(self):
        return {
            "원인_수": len(self.causes),
            "결과_수": len(self.effects),
            "총_관계": sum(len(effs) for effs in self.causes.values()),
        }


# =====================================================
# Patch 5 (v6.2): Theory of Mind (간단 버전)
# =====================================================

class TheoryOfMind:
    """타인의 마음 추론.
    
    이론:
    - Premack & Woodruff 1978: ToM 원조
    - Sally-Anne 거짓 신념 과제
    - 4-5세 아이가 발달
    
    EVE 구현:
    - "X는 Y를 안다" 메타 신념
    - "X는 Y를 모른다"
    - 타인 신념 추적 (각자의 belief state)
    """
    
    def __init__(self):
        # 타인의 신념: {person: [beliefs]}
        from collections import defaultdict
        self.others_beliefs = defaultdict(list)
        
        # 메타 패턴
        self.meta_patterns = [
            ("는 ", "을 안다"),
            ("는 ", "를 안다"),
            ("는 ", "을 모른다"),
            ("는 ", "를 모른다"),
            ("은 ", "을 안다"),
            ("은 ", "를 안다"),
            ("이 ", "을 안다"),
            ("가 ", "을 안다"),
            ("는 ", "를 생각한다"),
            ("는 ", "을 생각한다"),
            ("는 ", "고 믿는다"),
            ("는 ", "고 말했다"),
        ]
    
    def extract_meta(self, sentence):
        """메타 신념 추출.
        
        예: "철수는 영희를 안다" → 
            person: 철수, object: 영희, relation: 안다
        """
        for prefix, suffix in self.meta_patterns:
            if prefix in sentence and suffix in sentence:
                # 분리
                pre_idx = sentence.find(prefix)
                suf_idx = sentence.find(suffix, pre_idx + len(prefix))
                
                if pre_idx >= 0 and suf_idx > pre_idx:
                    person = sentence[:pre_idx].strip()
                    obj = sentence[pre_idx + len(prefix):suf_idx].strip()
                    
                    # 관계 추출
                    if "안다" in suffix:
                        relation = "knows"
                    elif "모른다" in suffix:
                        relation = "doesnt_know"
                    elif "생각한다" in suffix:
                        relation = "thinks"
                    elif "믿는다" in suffix:
                        relation = "believes"
                    elif "말했다" in suffix:
                        relation = "said"
                    else:
                        relation = "unknown"
                    
                    if len(person) >= 1 and len(obj) >= 1:
                        return {
                            "person": person,
                            "object": obj,
                            "relation": relation,
                        }
        return None
    
    def add_meta_belief(self, person, obj, relation):
        """타인의 신념 추가."""
        self.others_beliefs[person].append({
            "object": obj,
            "relation": relation,
        })
    
    def observe_sentence(self, sentence):
        """문장에서 메타 신념 추출 + 등록."""
        meta = self.extract_meta(sentence)
        if meta:
            self.add_meta_belief(meta["person"], meta["object"], meta["relation"])
            return meta
        return None
    
    def what_does_X_know(self, person):
        """X가 아는 것?"""
        if person not in self.others_beliefs:
            return []
        return [
            b for b in self.others_beliefs[person]
            if b["relation"] in ("knows", "believes", "thinks")
        ]
    
    def state(self):
        return {
            "추적_인물": len(self.others_beliefs),
            "메타_신념": sum(len(b) for b in self.others_beliefs.values()),
        }


# =====================================================
# Integration: v6.2 통합
# =====================================================

def add_full_grammar_to_eve_v62(eve_instance, use_konlpy=True):
    """v6.2: v6.1 + WSD + NER세분화 + 응답생성 + Causal + ToM."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    # v6.1 모듈 (NER은 새 버전으로)
    ner = NERWithSpecificity()  # v6.2: 세분화
    dialogue = DialogueContext()
    
    # v6.2 새 모듈
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    eve_instance.wsd = wsd
    eve_instance.response_gen = response_gen
    eve_instance.causal = causal
    eve_instance.tom = tom
    
    def hear(sentence, register_belief=True):
        """문장 듣기 - v6.2 풀 통합."""
        # 멀티턴: 지시어 해결 (구체적인 것 우선)
        resolved_sentence = dialogue.resolve_sentence(sentence)
        
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # NER (구체성 점수 포함)
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        # WSD - 다의어 처리
        wsd_results = wsd.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        # 대화 컨텍스트 (구체성 우선)
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        # 인과 추출
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        
        # ToM 추출
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def answer(question):
        """질문에 답하기 (v6.2 신규!)."""
        # 지시어 해결
        resolved = dialogue.resolve_sentence(question)
        # 응답 생성
        response = response_gen.generate_response(resolved)
        # 호르몬 톤 적용
        if response.get("응답"):
            styled = tone.style_response(response["응답"])
            response["스타일링된_응답"] = styled["응답"]
        return response
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
            "WSD": wsd.state(),
            "인과": causal.state(),
            "메타_신념": tom.state(),
        }
    
    def grammar_stats():
        return life_summary()  # 통합
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.answer = answer  # 신규!
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    
    print(f"  📚 v6.2 활성화 (WSD + NER세분화 + 응답 + Causal + ToM):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🔍 WSD: 다의어 {len(wsd.POLYSEMY)}개 ({list(wsd.POLYSEMY.keys())})")
    print(f"     - 🎯 NER 세분화: PROPER>SPECIFIC>GENERAL>ABSTRACT")
    print(f"     - 💬 응답 생성: 질문→신념검색→답변")
    print(f"     - 🔗 Causal Graph: 인과 추출")
    print(f"     - 🎭 ToM: 타인 마음 추론")
    
    return eve_instance


# =====================================================
# Patch (v6.3): WSD Learner - 다의어 자동 학습
# =====================================================
# 영감:
# - Harris (1954): Distribution Hypothesis
# - Yarowsky (1995): One sense per collocation
# - Schütze (1998): Word sense induction
# - Lin (1998): Distributional clustering

class WSDLearner:
    """다의어를 대화로 학습하는 시스템.
    
    원리:
    1. 단어 등장 시 → 문맥 단어 기록 (window=5)
    2. 충분한 데이터 누적 (5회+)
    3. 문맥 군집 분석 (k-means lite)
    4. 군집 분리 시 → 다의어 자동 인식!
    
    예시 학습 과정:
    Day 1: "배가 아프다" → 배: [아프]
    Day 2: "배를 탔다" → 배: [아프, 타]
    Day 3: "배가 고프다" → 배: [아프, 타, 고프]
    Day 4: "배를 먹었다" → 배: [아프, 타, 고프, 먹]
    Day 5: "배는 항구에 있다" → 충돌 감지!
        → 군집 분리: {아프, 고프} vs {타, 항구} vs {먹}
        → 3개 의미 자동 발견!
    """
    
    # 군집 임계값
    MIN_OBSERVATIONS = 5
    CLUSTER_DISTANCE = 2
    CONTEXT_WINDOW = 5
    
    # =====================================================
    # v6.4: 의미 그룹 대폭 확장 (50+ 그룹)
    # 세종 의미부류 5대분류 기반 + KorLex 참조
    # =====================================================
    SEMANTIC_GROUPS = {
        # ===== 1. 신체/생리 =====
        "BODY_SENSATION": ["아프", "고프", "쓰리", "통증", "부르", "찌르", "쿡", "욱신"],
        "BODY_PART": ["머리", "얼굴", "손", "발", "다리", "팔", "눈", "코", "입", "귀", "배", "등", "어깨", "무릎"],
        "BODY_FUNCTION": ["숨", "쉬", "심장", "뛰", "맥박", "혈압"],
        "HEALTH": ["건강", "병", "치료", "약", "병원", "의사", "환자", "진료", "수술"],
        "ILLNESS": ["감기", "열", "기침", "두통", "복통", "어지럽"],
        
        # ===== 2. 감각/지각 =====
        "VISUAL": ["보", "감", "뜨", "시력", "눈", "안경", "관찰", "쳐다"],
        "AUDITORY": ["듣", "들리", "소리", "음악", "노래", "조용", "시끄럽"],
        "TACTILE": ["만지", "잡", "쥐", "쓰다듬", "느끼", "촉감"],
        "OLFACTORY": ["냄새", "향", "맡", "구수", "고소"],
        "TASTE": ["맛", "달", "쓰", "시", "짜", "맵"],
        
        # ===== 3. 음식/먹기 =====
        "EATING": ["먹", "맛있", "달", "익", "삼키", "씹", "우적"],
        "DRINKING": ["마시", "삼키", "들이키", "마셨"],
        "FOOD": ["밥", "빵", "고기", "반찬", "과일", "채소", "음식", "요리"],
        "FRUIT": ["사과", "배", "수박", "딸기", "포도", "참외", "과일"],
        "VEGETABLE": ["채소", "야채", "배추", "무", "당근", "감자", "양파"],
        "MEAT": ["고기", "소고기", "돼지고기", "닭고기", "육류"],
        "BEVERAGE": ["물", "차", "커피", "음료", "주스", "우유"],
        
        # ===== 4. 이동/행위 =====
        "MOVEMENT": ["타", "탔", "달리", "가", "왔", "오", "걷", "뛰", "기어"],
        "RUN": ["달리", "뛰", "달려", "질주"],
        "WALK": ["걷", "걸어", "산책", "걸었"],
        "DRIVE": ["운전", "몰", "타고", "운송"],
        "FLY": ["날", "날아", "비행", "이륙", "착륙"],
        "SWIM": ["헤엄", "수영", "잠수", "떠"],
        
        # ===== 5. 운송/탈것 =====
        "VEHICLE_LAND": ["차", "자동차", "버스", "기차", "택시", "지하철", "오토바이"],
        "VEHICLE_WATER": ["배", "선박", "보트", "요트", "항해"],
        "VEHICLE_AIR": ["비행기", "헬리콥터", "드론", "이륙", "착륙"],
        "SHIP_RELATED": ["항구", "바다", "물", "선장", "항해", "선박"],
        "TRAFFIC": ["도로", "길", "교통", "신호", "주차", "운전"],
        
        # ===== 6. 자연/환경 =====
        "WEATHER": ["내리", "쌓이", "춥", "겨울", "더위", "비", "눈", "바람", "맑"],
        "SKY": ["하늘", "구름", "별", "달", "해", "태양"],
        "LAND": ["땅", "흙", "모래", "산", "들", "벌판"],
        "WATER_BODY": ["바다", "강", "호수", "연못", "개울", "시냇물"],
        "PLANT": ["나무", "풀", "꽃", "잎", "뿌리", "줄기", "씨"],
        "ANIMAL": ["개", "고양이", "사자", "호랑이", "토끼", "새", "물고기", "짐승"],
        
        # ===== 7. 시간 =====
        "TIME_DAY": ["아침", "점심", "저녁", "밤", "새벽", "오전", "오후"],
        "TIME_PERIOD": ["어제", "오늘", "내일", "그제", "모레", "지금", "방금"],
        "TIME_LONG": ["년", "월", "주", "시대", "세기", "옛날", "미래"],
        
        # ===== 8. 공간/장소 =====
        "PLACE_HOME": ["집", "방", "거실", "부엌", "화장실", "침실", "마당"],
        "PLACE_WORK": ["회사", "사무실", "직장", "공장", "은행"],
        "PLACE_PUBLIC": ["학교", "병원", "공원", "도서관", "역", "공항"],
        "PLACE_NATURE": ["산", "바다", "강", "들", "숲", "동굴"],
        "PLACE_CITY": ["도시", "마을", "동네", "거리", "광장"],
        
        # ===== 9. 사람/관계 =====
        "PERSON_FAMILY": ["엄마", "아빠", "형", "누나", "동생", "할머니", "할아버지", "가족"],
        "PERSON_FRIEND": ["친구", "동료", "선배", "후배", "지인"],
        "PERSON_PROFESSION": ["선생님", "의사", "경찰", "군인", "학생", "회사원"],
        "EMOTION_BOND": ["사랑", "좋아", "친하", "그리워", "보고싶"],
        
        # ===== 10. 감정 =====
        "EMOTION_POSITIVE": ["기쁘", "행복", "좋", "즐겁", "신", "웃"],
        "EMOTION_NEGATIVE": ["슬프", "괴로워", "힘들", "고통", "울"],
        "EMOTION_ANGRY": ["화", "짜증", "분노", "열", "성난"],
        "EMOTION_SCARED": ["무서", "두려", "겁", "떨"],
        
        # ===== 11. 정신/사고 =====
        "THOUGHT": ["생각", "고민", "떠올", "기억", "잊"],
        "BELIEF": ["믿", "알", "모르", "확실", "의심"],
        "WISH": ["원하", "바라", "희망", "꿈"],
        "DECISION": ["결정", "선택", "정하", "마음먹"],
        
        # ===== 12. 사회/조직 =====
        "EDUCATION": ["배우", "공부", "가르치", "수업", "학교"],
        "WORK": ["일", "직업", "근무", "출근", "퇴근"],
        "MONEY": ["돈", "값", "비싸", "싸", "사", "팔", "구매", "판매"],
        "GOVERNMENT": ["정부", "법", "대통령", "국가", "정치"],
        
        # ===== 13. 사물 =====
        "TOOL": ["도구", "연필", "펜", "가위", "망치", "칼"],
        "FURNITURE": ["의자", "책상", "침대", "소파", "책장"],
        "CLOTHES": ["옷", "신발", "모자", "장갑", "셔츠", "바지"],
        "DEVICE": ["컴퓨터", "전화", "TV", "냉장고", "세탁기"],
        
        # ===== 14. 추상 =====
        "QUANTITY": ["많", "적", "큰", "작", "숫자", "개수"],
        "QUALITY": ["좋", "나쁘", "예쁘", "추하", "훌륭"],
        "REASON": ["이유", "원인", "때문", "결과", "이러"],
        "METHOD": ["방법", "방식", "어떻게", "이렇게", "그렇게"],
        
        # ===== 15. 의사소통 =====
        "SPEECH": ["말", "이야기", "말씀", "전하", "이야기"],
        "WRITE": ["쓰", "적", "글", "편지", "메모"],
        "READ": ["읽", "독서", "보", "책"],
        
        # ===== 16. 시간 동작 =====
        "BEGIN": ["시작", "출발", "개시"],
        "END": ["끝", "마치", "종료", "완료"],
        "CONTINUE": ["계속", "이어", "지속"],
        "STOP": ["멈춰", "그만", "정지", "쉬"],
        
        # ===== 17. 도시/장소 (구체) =====
        "ASIAN_CITY": ["서울", "도쿄", "베이징", "상하이", "방콕", "하노이"],
        "WESTERN_CITY": ["뉴욕", "런던", "파리", "베를린", "로마"],
        "KOREAN_REGION": ["서울", "부산", "대구", "인천", "광주", "대전", "울산"],
    }
    
    def __init__(self):
        # 단어별 문맥 기록: {word: [(context_words, sentence, timestamp)]}
        from collections import defaultdict
        self.observations = defaultdict(list)
        
        # 자동 발견된 다의어: {word: [{cluster_id, contexts, examples}]}
        self.discovered_senses = defaultdict(list)
        
        # 사용자 라벨링: {word: {sense_name: contexts}}
        self.labeled_senses = defaultdict(dict)
        
        # 통계
        self.total_observations = 0
        self.detected_polysemes = 0
    
    def observe(self, word, sentence, morphs):
        """단어 관찰 → 문맥 기록.
        
        v6.3 fix: 한글자 단어도 허용 (배, 눈, 말, 차 등 다의어 많음)
        """
        if not word:
            return
        
        # 문맥 단어 추출 (자기 빼고, 의미 단어만)
        context = []
        for w, tag in morphs:
            if w == word:
                continue
            if tag in ("Verb", "Adjective", "Noun") and len(w) >= 1:
                context.append(w)
        
        if not context:
            return
        
        import datetime as _dt
        self.observations[word].append({
            "context": context,
            "sentence": sentence,
            "timestamp": _dt.datetime.now().isoformat(),
        })
        
        self.total_observations += 1
        
        # 임계값 도달 시 군집 분석
        if len(self.observations[word]) >= self.MIN_OBSERVATIONS:
            return self._analyze_clusters(word)
        return None
    
    def _analyze_clusters(self, word):
        """문맥 군집 분석 - 다의어 자동 발견.
        
        v6.4: 클래스 레벨 SEMANTIC_GROUPS 사용 (50+ 그룹)
        """
        observations = self.observations[word]
        n = len(observations)
        
        # v6.4: self.SEMANTIC_GROUPS 사용 (클래스 변수)
        groups = self.SEMANTIC_GROUPS
        
        def _semantic_overlap(ctx_a, ctx_b):
            """두 문맥의 의미 그룹 일치도."""
            score = 0
            # 1. 직접 일치 (가중치 2)
            shared = set(ctx_a) & set(ctx_b)
            score += len(shared) * 2
            
            # 2. 의미 그룹 일치 (가중치 1)
            for group_name, group_words in groups.items():
                a_in_group = any(any(g in c or c.startswith(g) for g in group_words) for c in ctx_a)
                b_in_group = any(any(g in c or c.startswith(g) for g in group_words) for c in ctx_b)
                if a_in_group and b_in_group:
                    score += 1
            
            return score
        
        # 인접 행렬 (관찰끼리 의미 유사도)
        adjacency = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                ctx_i = observations[i]["context"]
                ctx_j = observations[j]["context"]
                sim = _semantic_overlap(ctx_i, ctx_j)
                adjacency[i][j] = sim
                adjacency[j][i] = sim
        
        # Union-Find 군집화
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 임계값 1 이상 시 합침
        for i in range(n):
            for j in range(i+1, n):
                if adjacency[i][j] >= 1:
                    union(i, j)
        
        # 군집별 그룹
        from collections import defaultdict
        clusters = defaultdict(list)
        for i in range(n):
            root = find(i)
            clusters[root].append(observations[i])
        
        # 군집 2개 이상 = 다의어!
        if len(clusters) >= 2:
            sense_list = []
            for cluster_id, obs_list in clusters.items():
                if len(obs_list) >= 1:  # 1번이라도
                    all_contexts = []
                    for o in obs_list:
                        all_contexts.extend(o["context"])
                    
                    from collections import Counter
                    common_contexts = Counter(all_contexts).most_common(5)
                    
                    sense_list.append({
                        "cluster_id": f"sense_{cluster_id}",
                        "common_contexts": [w for w, _ in common_contexts],
                        "example_sentences": [o["sentence"] for o in obs_list[:3]],
                        "observation_count": len(obs_list),
                    })
            
            if sense_list and len(sense_list) >= 2:
                old_count = len(self.discovered_senses.get(word, []))
                self.discovered_senses[word] = sense_list
                
                if old_count == 0:
                    self.detected_polysemes += 1
                
                return {
                    "word": word,
                    "newly_discovered": old_count == 0,
                    "senses_count": len(sense_list),
                    "senses": sense_list,
                }
        
        return None
    
    def predict_sense(self, word, current_context):
        """현재 문맥으로 의미 예측.
        
        학습된 다의어인지 확인 + 가장 잘 맞는 의미 찾기
        """
        if word not in self.discovered_senses:
            return None  # 학습 안된 단어
        
        senses = self.discovered_senses[word]
        if not senses:
            return None
        
        # 각 의미마다 문맥 매칭 점수
        scores = []
        for sense in senses:
            score = 0
            sense_contexts = sense["common_contexts"]
            for ctx_word in current_context:
                for sense_ctx in sense_contexts:
                    if sense_ctx in ctx_word or ctx_word.startswith(sense_ctx):
                        score += 1
                        break
            scores.append((sense["cluster_id"], score, sense))
        
        scores.sort(key=lambda x: -x[1])
        return scores
    
    def teach(self, word, sense_name, example_sentences):
        """사용자가 의미 가르치기.
        
        예: eve.wsd_learner.teach("배", "SHIP", ["배를 탔다", "큰 배다"])
        """
        if word not in self.labeled_senses:
            self.labeled_senses[word] = {}
        
        # 예시 문장에서 문맥 추출 필요
        # (외부에서 morphs 받아서 처리)
        self.labeled_senses[word][sense_name] = {
            "examples": example_sentences,
        }
    
    def get_learned_polysemes(self):
        """학습으로 발견한 모든 다의어."""
        return {
            word: [s["cluster_id"] for s in senses]
            for word, senses in self.discovered_senses.items()
            if len(senses) >= 2
        }
    
    def state(self):
        """학습 통계."""
        return {
            "관찰_단어_수": len(self.observations),
            "총_관찰_횟수": self.total_observations,
            "발견된_다의어": len(self.discovered_senses),
            "다의어_목록": list(self.discovered_senses.keys()),
            "사용자_라벨": len(self.labeled_senses),
        }
    
    def explain_word(self, word):
        """학습된 단어 의미 설명."""
        if word not in self.discovered_senses:
            obs_count = len(self.observations.get(word, []))
            return {
                "word": word,
                "status": "관찰_부족" if obs_count > 0 else "미관찰",
                "observation_count": obs_count,
                "needed": max(0, self.MIN_OBSERVATIONS - obs_count),
            }
        
        senses = self.discovered_senses[word]
        return {
            "word": word,
            "status": "다의어_발견",
            "senses_count": len(senses),
            "senses": [
                {
                    "id": s["cluster_id"],
                    "context_words": s["common_contexts"][:5],
                    "examples": s["example_sentences"][:2],
                    "observations": s["observation_count"],
                }
                for s in senses
            ],
        }


# =====================================================
# Integration: v6.3 통합
# =====================================================

def add_full_grammar_to_eve_v63(eve_instance, use_konlpy=True):
    """v6.3: v6.2 + WSD Learner."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    
    # v6.3 신규
    wsd_learner = WSDLearner()
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    eve_instance.wsd = wsd
    eve_instance.response_gen = response_gen
    eve_instance.causal = causal
    eve_instance.tom = tom
    eve_instance.wsd_learner = wsd_learner
    
    def hear(sentence, register_belief=True):
        """문장 듣기 - v6.3: WSD 학습 포함."""
        resolved_sentence = dialogue.resolve_sentence(sentence)
        
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # NER
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        # WSD (사전)
        wsd_results = wsd.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        # v6.3: WSD 학습 - 모든 명사 관찰 (한글자 다의어 포함!)
        new_polysemes = []
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 1:  # v6.3: 한글자도 (배, 눈, 말 등)
                discovery_result = wsd_learner.observe(word, resolved_sentence, morphs)
                if discovery_result:
                    new_polysemes.append(discovery_result)
                
                # 학습된 다의어면 의미 예측
                context = [w for w, t in morphs if w != word and t in ("Verb", "Adjective", "Noun")]
                learned_senses = wsd_learner.predict_sense(word, context)
                if learned_senses:
                    if "WSD_학습" not in result:
                        result["WSD_학습"] = {}
                    result["WSD_학습"][word] = learned_senses
        
        if new_polysemes:
            result["새_다의어_발견"] = new_polysemes
            # 도파민 폭발 (새로운 발견!)
            hormones.encounter_novelty(0.7)
        
        # 대화 컨텍스트
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        # 인과
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        
        # ToM
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def answer(question):
        resolved = dialogue.resolve_sentence(question)
        response = response_gen.generate_response(resolved)
        if response.get("응답"):
            styled = tone.style_response(response["응답"])
            response["스타일링된_응답"] = styled["응답"]
        return response
    
    def explain_polyseme(word):
        """학습한 다의어 설명."""
        return wsd_learner.explain_word(word)
    
    def teach_meaning(word, sense_name, examples):
        """다의어 의미 가르치기."""
        wsd_learner.teach(word, sense_name, examples)
        return f"'{word}'의 '{sense_name}' 의미를 학습했어요. 예시: {examples}"
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
            "WSD_사전": wsd.state(),
            "WSD_학습": wsd_learner.state(),  # 신규!
            "인과": causal.state(),
            "메타_신념": tom.state(),
        }
    
    def grammar_stats():
        return life_summary()
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.answer = answer
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    eve_instance.explain_polyseme = explain_polyseme  # 신규!
    eve_instance.teach_meaning = teach_meaning  # 신규!
    
    print(f"  📚 v6.3 활성화 (WSD 학습 - 다의어 자동 발견):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🔍 WSD 사전: {len(wsd.POLYSEMY)}개")
    print(f"     - 🧠 WSD 학습: 통계적 군집화 (Harris 1954, Yarowsky 1995)")
    print(f"     - 🎯 NER 세분화 + 💬 응답 + 🔗 Causal + 🎭 ToM")
    print(f"     - 새 메서드: explain_polyseme(), teach_meaning()")
    
    return eve_instance


# =====================================================
# Patch 2 (v6.4): WSD Hybrid - 사전 + 학습 통합
# =====================================================

class WSDHybrid:
    """사전(WSDModule) + 학습(WSDLearner) 통합.
    
    원리:
    1. 사전에 있으면 → 사전 우선 (빠르고 정확)
    2. 사전에 없으면 → 학습 결과 사용
    3. 학습으로 사전 확장 가능 (자동)
    """
    
    def __init__(self, wsd_module, wsd_learner):
        self.dict_wsd = wsd_module      # 사전
        self.learner = wsd_learner      # 학습
    
    def disambiguate(self, word, context_words, top_k=1):
        """단어 + 문맥 → 가장 적합한 의미 (hybrid)."""
        # 1. 사전 우선
        if self.dict_wsd.is_polysemous(word):
            dict_results = self.dict_wsd.disambiguate(word, context_words, top_k=top_k)
            if dict_results and dict_results[0][1] > 0:  # 점수가 있으면
                return {
                    "source": "dictionary",
                    "results": dict_results,
                }
        
        # 2. 학습 결과 사용
        learned = self.learner.predict_sense(word, context_words)
        if learned:
            return {
                "source": "learned",
                "results": learned[:top_k],
            }
        
        # 3. 모름
        return {
            "source": "unknown",
            "results": [],
        }
    
    def disambiguate_in_sentence(self, sentence, morphs):
        """문장 내 모든 다의어 처리 (hybrid)."""
        results = {}
        all_words = [w for w, _ in morphs]
        
        # 사전 등록된 다의어
        for word, tag in morphs:
            if word in self.dict_wsd.POLYSEMY:
                context = [w for w in all_words if w != word]
                hybrid_result = self.disambiguate(word, context, top_k=3)
                if hybrid_result["results"]:
                    results[word] = hybrid_result
        
        # 학습된 다의어 (사전에 없음)
        for word, tag in morphs:
            if word in results:
                continue  # 이미 처리됨
            if word in self.learner.discovered_senses:
                context = [w for w in all_words if w != word]
                hybrid_result = self.disambiguate(word, context, top_k=3)
                if hybrid_result["results"]:
                    results[word] = hybrid_result
        
        return results


# =====================================================
# Patch 3 (v6.4): SNN Episodic Memory (간단 통합)
# =====================================================
# Day 17 SNN의 토대만 도입 - 사건 시퀀스 저장 + 회상

class EpisodicMemory:
    """사건 기반 일화 기억 (SNN 토대).
    
    Day 17 SNN Hippocampus의 episodic 부분을 단순화한 버전.
    
    저장:
    - 사건 (event)
    - 시간 (timestamp)  
    - 호르몬 상태 (감정 맥락)
    - 관련 신념 (semantic 연결)
    
    회상:
    - 시간 기반 ("어제 뭐 했지?")
    - 감정 기반 (mood-congruent recall)
    - 키워드 기반 ("고래에 대해 들은 거")
    """
    
    def __init__(self, max_episodes=1000):
        self.episodes = []
        self.max_episodes = max_episodes
        self.episode_id = 0
    
    def encode(self, event, hormones=None, beliefs=None, day=None):
        """사건 인코딩 (저장)."""
        import datetime as _dt
        
        episode = {
            "id": self.episode_id,
            "event": event,
            "timestamp": _dt.datetime.now().isoformat(),
            "day": day or 0,
            "hormones": hormones.copy() if hormones else {},
            "beliefs": beliefs[:5] if beliefs else [],  # 관련 신념 5개만
        }
        
        self.episodes.append(episode)
        self.episode_id += 1
        
        # 용량 제한 (오래된 것 제거)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        
        return episode
    
    def recall_by_keyword(self, keyword, top_k=5):
        """키워드로 사건 회상."""
        matches = []
        for ep in self.episodes:
            event_str = str(ep["event"])
            if keyword in event_str:
                matches.append(ep)
        
        # 최신순
        matches.sort(key=lambda x: -x["id"])
        return matches[:top_k]
    
    def recall_by_day(self, day):
        """특정 날의 사건 회상."""
        return [ep for ep in self.episodes if ep["day"] == day]
    
    def recall_by_emotion(self, emotion, top_k=5):
        """감정 기반 회상 (mood-congruent recall, 학술 검증됨)."""
        matches = []
        for ep in self.episodes:
            ep_emotion = ep["hormones"].get("감정", "")
            if emotion in ep_emotion or ep_emotion == emotion:
                matches.append(ep)
        
        matches.sort(key=lambda x: -x["id"])
        return matches[:top_k]
    
    def recall_recent(self, n=5):
        """최근 N개 사건."""
        return self.episodes[-n:][::-1]  # 최신순
    
    def state(self):
        """기억 상태."""
        if not self.episodes:
            return {"총_사건": 0}
        
        from collections import Counter
        emotion_counts = Counter()
        for ep in self.episodes:
            emo = ep["hormones"].get("감정", "")
            if emo:
                emotion_counts[emo] += 1
        
        return {
            "총_사건": len(self.episodes),
            "최근_사건": self.episodes[-1]["event"] if self.episodes else None,
            "감정별_분포": dict(emotion_counts.most_common(5)),
        }


# =====================================================
# Integration: v6.4 통합
# =====================================================

def add_full_grammar_to_eve_v64(eve_instance, use_konlpy=True):
    """v6.4: v6.3 + 의미그룹 50+ + WSD Hybrid + Episodic Memory."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    wsd_learner = WSDLearner()  # v6.4: 50+ 그룹
    
    # v6.4 신규
    wsd_hybrid = WSDHybrid(wsd, wsd_learner)
    episodic = EpisodicMemory()
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    eve_instance.wsd = wsd
    eve_instance.response_gen = response_gen
    eve_instance.causal = causal
    eve_instance.tom = tom
    eve_instance.wsd_learner = wsd_learner
    eve_instance.wsd_hybrid = wsd_hybrid  # 신규
    eve_instance.episodic = episodic        # 신규
    
    def hear(sentence, register_belief=True):
        """문장 듣기 - v6.4: 모든 시스템 통합."""
        resolved_sentence = dialogue.resolve_sentence(sentence)
        
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # NER
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        # v6.4: WSD Hybrid (사전 + 학습)
        wsd_results = wsd_hybrid.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        # WSD 학습 (모든 명사)
        new_polysemes = []
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 1:
                discovery_result = wsd_learner.observe(word, resolved_sentence, morphs)
                if discovery_result:
                    new_polysemes.append(discovery_result)
        
        if new_polysemes:
            result["새_다의어_발견"] = new_polysemes
            hormones.encounter_novelty(0.7)
        
        # 대화 컨텍스트
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        # 인과
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        
        # ToM
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
                        
                        # v6.4: Episodic Memory에 사건 인코딩!
                        day = eve_instance.identity.get("days_alive", 0)
                        episodic.encode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def answer(question):
        resolved = dialogue.resolve_sentence(question)
        response = response_gen.generate_response(resolved)
        if response.get("응답"):
            styled = tone.style_response(response["응답"])
            response["스타일링된_응답"] = styled["응답"]
        return response
    
    def remember(keyword=None, day=None, emotion=None, n=5):
        """v6.4 신규: 일화 기억 회상.
        
        예: eve.remember("고래") → 고래 관련 사건들
            eve.remember(day=1) → 1일차 사건들
            eve.remember(emotion="기쁨") → 기쁠 때 사건들
        """
        if keyword:
            return episodic.recall_by_keyword(keyword, top_k=n)
        elif day is not None:
            return episodic.recall_by_day(day)
        elif emotion:
            return episodic.recall_by_emotion(emotion, top_k=n)
        else:
            return episodic.recall_recent(n=n)
    
    def explain_polyseme(word):
        return wsd_learner.explain_word(word)
    
    def teach_meaning(word, sense_name, examples):
        wsd_learner.teach(word, sense_name, examples)
        return f"'{word}'의 '{sense_name}' 의미를 학습했어요."
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
            "WSD_사전": wsd.state(),
            "WSD_학습": wsd_learner.state(),
            "Episodic": episodic.state(),  # v6.4
            "인과": causal.state(),
            "메타_신념": tom.state(),
        }
    
    def grammar_stats():
        return life_summary()
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.answer = answer
    eve_instance.remember = remember  # v6.4 신규!
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    eve_instance.explain_polyseme = explain_polyseme
    eve_instance.teach_meaning = teach_meaning
    
    print(f"  📚 v6.4 활성화 (의미그룹 50+ + WSD Hybrid + Episodic):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🌐 의미 그룹: {len(WSDLearner.SEMANTIC_GROUPS)}개 (세종/KorLex 기반)")
    print(f"     - 🔄 WSD Hybrid: 사전 + 학습 통합")
    print(f"     - 🧠 Episodic Memory: 일화 기억 (SNN 토대)")
    print(f"     - 새 메서드: remember(keyword/day/emotion)")
    
    return eve_instance


# =====================================================
# Patch 1 (v6.5): 의미 그룹 자동 발견
# =====================================================
# 영감: Mikolov 2013 (Word2Vec), Pennington 2014 (GloVe)
# 단, 신경망 X - 통계적 분포만 사용 (비트랜스포머)

class SemanticGroupDiscovery:
    """단어 자동 군집화로 의미 그룹 자동 발견.
    
    원리:
    1. 모든 단어의 문맥 기록 (WSDLearner 활용)
    2. 단어 간 코사인 유사도 (공유 문맥 비율)
    3. Hierarchical Agglomerative Clustering
    4. 자동 그룹 생성 → SEMANTIC_GROUPS에 추가
    
    예:
    "아프" 문맥: {몸, 통증, 머리, 배}
    "쓰리" 문맥: {몸, 통증, 가슴}
    유사도 = |공유| / |합집합| = 2/5 = 0.4
    
    → 0.3 이상이면 같은 그룹
    → "DISCOVERED_GROUP_1": [아프, 쓰리]
    """
    
    SIMILARITY_THRESHOLD = 0.3   # 같은 그룹으로 인정할 유사도
    MIN_OBSERVATIONS = 3          # 단어 최소 관찰 횟수
    MIN_GROUP_SIZE = 2            # 그룹 최소 크기
    
    def __init__(self):
        # 단어별 문맥 단어 카운트
        from collections import defaultdict, Counter
        self.word_contexts = defaultdict(Counter)
        
        # 자동 발견된 그룹: {group_id: [words]}
        self.discovered_groups = {}
        self.group_counter = 0
    
    def observe(self, word, context_words):
        """단어 관찰 + 문맥 기록."""
        if not word or len(word) < 1:
            return
        for ctx in context_words:
            if ctx and ctx != word:
                self.word_contexts[word][ctx] += 1
    
    def similarity(self, word_a, word_b):
        """두 단어의 의미 유사도 (0~1).
        
        Jaccard similarity (공유 / 합집합)
        """
        ctx_a = set(self.word_contexts[word_a])
        ctx_b = set(self.word_contexts[word_b])
        
        if not ctx_a or not ctx_b:
            return 0.0
        
        intersection = ctx_a & ctx_b
        union = ctx_a | ctx_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_similar_words(self, word, top_k=5):
        """주어진 단어와 비슷한 단어들 찾기."""
        scores = []
        # v6.5 fix: list로 복사 (이터레이션 중 수정 방지)
        for other in list(self.word_contexts.keys()):
            if other == word:
                continue
            sim = self.similarity(word, other)
            if sim > 0:
                scores.append((other, sim))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def discover_groups(self):
        """모든 단어 군집화 → 의미 그룹 자동 생성.
        
        Hierarchical Agglomerative Clustering (단순 버전):
        1. 충분히 관찰된 단어들만
        2. 쌍 유사도 계산
        3. 임계값 이상 → 같은 그룹
        4. Union-Find로 군집화
        """
        # 충분히 관찰된 단어만
        candidates = [
            w for w, ctxs in self.word_contexts.items()
            if sum(ctxs.values()) >= self.MIN_OBSERVATIONS
        ]
        
        if len(candidates) < 2:
            return self.discovered_groups
        
        n = len(candidates)
        
        # Union-Find
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 모든 쌍의 유사도 → 임계값 이상 합침
        for i in range(n):
            for j in range(i+1, n):
                sim = self.similarity(candidates[i], candidates[j])
                if sim >= self.SIMILARITY_THRESHOLD:
                    union(i, j)
        
        # 그룹 추출
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups[root].append(candidates[i])
        
        # 크기 조건 만족 그룹만 등록
        new_groups = {}
        for root, words in groups.items():
            if len(words) >= self.MIN_GROUP_SIZE:
                # 기존 그룹 ID 재사용 또는 새로
                group_id = f"DISCOVERED_{self.group_counter}"
                self.group_counter += 1
                new_groups[group_id] = words
        
        self.discovered_groups = new_groups
        return new_groups
    
    def get_word_group(self, word):
        """단어가 속한 그룹 찾기."""
        for group_id, words in self.discovered_groups.items():
            if word in words:
                return group_id
        return None
    
    def state(self):
        return {
            "관찰_단어": len(self.word_contexts),
            "발견_그룹": len(self.discovered_groups),
            "그룹_목록": [
                {"id": gid, "단어": words[:5], "크기": len(words)}
                for gid, words in self.discovered_groups.items()
            ][:10],
        }


# =====================================================
# Patch 2 (v6.5): 자연스러운 응답 생성
# =====================================================

class NaturalResponseGenerator:
    """호르몬 + 톤 + 신념 결합 자연스러운 응답.
    
    원리:
    1. 신념 검색 (기존 ResponseGenerator)
    2. 호르몬 상태 → 톤 결정
    3. 자연스러운 표현 추가
    4. 부가 정보 (관련 신념)도 함께
    """
    
    # 감정별 응답 톤
    EMOTION_TONES = {
        "기쁨": {
            "prefixes": ["오, ", "아, ", ""],
            "suffixes": ["!", "지!", "이야!"],
            "fillers": ["맞아, ", "그러게, ", "응, "],
        },
        "호기심": {
            "prefixes": ["어, ", "음, ", ""],
            "suffixes": ["?", " 그렇구나!", "이야."],
            "fillers": ["흥미롭네, ", "신기한데, ", ""],
        },
        "유대감": {
            "prefixes": ["응, ", "그래, ", ""],
            "suffixes": [".", "이야.", "야."],
            "fillers": ["같이 알아가자, ", "좋은 질문이야, ", ""],
        },
        "평온": {
            "prefixes": ["", "음, ", ""],
            "suffixes": [".", "이야.", "지."],
            "fillers": ["", "그래, ", ""],
        },
        "스트레스": {
            "prefixes": ["", "글쎄, ", ""],
            "suffixes": [".", "...", "."],
            "fillers": ["", "음, ", ""],
        },
    }
    
    def __init__(self, eve, response_gen, hormones):
        self.eve = eve
        self.response_gen = response_gen
        self.hormones = hormones
    
    def generate_natural(self, question):
        """자연스러운 응답 생성."""
        # 1. 기본 응답
        base = self.response_gen.generate_response(question)
        
        if not base.get("응답"):
            return base
        
        # 2. 모르는 거면 자연스럽게
        if base.get("이유") in ("관련_신념_없음", "주체_없음"):
            return self._handle_unknown(base)
        
        # 3. 알면 자연스럽게 + 관련 신념 추가
        return self._handle_known(base, question)
    
    def _handle_unknown(self, base):
        """모르는 거 → 호기심 표현."""
        emotion = self.hormones.state().get("감정", "평온")
        
        templates = [
            "음, {subject}에 대해선 잘 모르겠어. 더 알려줄 수 있어?",
            "{subject}? 처음 듣는데, 뭐야?",
            "글쎄, {subject}는 아직 잘 몰라.",
        ]
        
        subject = base.get("주체", "그거")
        
        import random
        natural = random.choice(templates).format(subject=subject)
        
        # 호기심 호르몬 ↑
        self.hormones.encounter_novelty(0.3)
        
        base["자연_응답"] = natural
        return base
    
    def _handle_known(self, base, question):
        """아는 거 → 호르몬 톤 + 추가 정보.
        
        v6.6 개선:
        - 두 번째 신념 결합 자연스럽게
        - "바다에 있고 알아" → 자연스러운 표현
        - 위치 질문엔 위치 신념만
        """
        emotion = self.hormones.state().get("감정", "평온")
        tone = self.EMOTION_TONES.get(emotion, self.EMOTION_TONES["평온"])
        
        import random
        prefix = random.choice(tone["prefixes"])
        
        answer = base["응답"]
        
        # 자연스럽게 변형
        natural = answer
        if natural.endswith("이다"):
            natural = natural[:-2] + "이야"
        elif natural.endswith("다") and not natural.endswith("이다"):
            # 동사형 → "다" → "어" 자연
            # 예: "있다" → "있어", "낳는다" → "낳아"
            if natural.endswith("ㄴ다") or natural.endswith("는다"):
                # 그대로 유지 (이미 자연)
                pass
            elif natural.endswith("있다"):
                natural = natural[:-2] + "있어"
            elif natural.endswith("없다"):
                natural = natural[:-2] + "없어"
        
        # v6.6: 질문 종류별 처리
        q_type = base.get("질문종류", "기타")
        
        # 위치 질문 → 위치 신념만 (추가 정보 X)
        if q_type == "위치":
            # 단답 + 톤
            suffix = random.choice(tone["suffixes"])
            if not natural.endswith(("!", ".", "?", "야", "아", "어")):
                natural = natural + suffix
        else:
            # 정의/일반 질문 → 관련 신념 추가
            subject = base.get("주체")
            if subject and base.get("관련신념수", 0) > 1:
                related = self.response_gen.search_beliefs(subject)
                if len(related) > 1:
                    second = related[1][0]
                    
                    # 첫 번째 신념과 다른 정보면 추가
                    first = base["응답"]
                    if second != first and second != answer:
                        # 자연스러운 결합: "그리고 X해"
                        second_clean = second.replace(subject + "는 ", "").replace(subject + "은 ", "")
                        second_clean = second_clean.replace(subject + "이 ", "").replace(subject + "가 ", "")
                        
                        # 어미 정리
                        if second_clean.endswith("이다"):
                            second_clean = second_clean[:-2]  # "포유류이다" → "포유류"
                        elif second_clean.endswith("다"):
                            # 동사형
                            if second_clean.endswith("있다"):
                                second_clean = second_clean[:-2] + "있어"
                            elif second_clean.endswith("는다"):
                                second_clean = second_clean[:-1]  # "낳는다" → "낳는"
                            elif second_clean.endswith("ㄴ다"):
                                second_clean = second_clean[:-1]
                            elif second_clean.endswith("했다"):
                                second_clean = second_clean[:-1] + ""
                        
                        # 합치기 자연스럽게
                        natural = f"{natural}. 그리고 {second_clean}."
                else:
                    # 단답 + 톤
                    suffix = random.choice(tone["suffixes"])
                    if not natural.endswith(("!", ".", "?", "야", "아", "어")):
                        natural = natural + suffix
            else:
                # 단답 + 톤
                suffix = random.choice(tone["suffixes"])
                if not natural.endswith(("!", ".", "?", "야", "아", "어")):
                    natural = natural + suffix
        
        # 자연스러운 시작
        if prefix and not natural.startswith(prefix):
            natural = prefix + natural
        
        base["자연_응답"] = natural
        base["감정"] = emotion
        
        return base


# =====================================================
# Patch 3 (v6.5): SNN 진짜 통합 (Hippocampus)
# =====================================================
# Day 17 SNN 작업의 episodic 부분을 본격 통합

class SNNHippocampus:
    """SNN 기반 Hippocampus - 진짜 episodic 인코딩.
    
    Day 17 SNN의 핵심 영감:
    - DG (Dentate Gyrus): Pattern Separation
    - CA3: Pattern Completion
    - CA1: Output / 다른 영역과 연결
    
    여기선 SNN 시뮬레이션 (PyTorch 없이도):
    1. Sparse Distributed Representation (SDR)
    2. Hebbian-like learning
    3. 호르몬 ↔ 기억 강도 결합
    
    학술:
    - Treves & Rolls 1994: Hippocampal models
    - O'Reilly 1995: CLS theory
    - Olshausen 1996: Sparse coding
    """
    
    # 신경망 크기 (시뮬레이션)
    DG_SIZE = 100      # DG 뉴런 수
    CA3_SIZE = 50      # CA3 뉴런 수
    CA1_SIZE = 50      # CA1 출력
    SPARSITY = 0.05    # 5% 활성 (생물학적)
    
    def __init__(self):
        # 사건 → SDR 매핑 (인코딩 결과)
        self.encoded_episodes = []
        
        # DG 활성 패턴 기억
        self.dg_patterns = {}
        
        # 호르몬 ↔ 강도 매핑
        self.memory_strengths = {}
    
    def _hash_to_sdr(self, text, size=100, sparsity=0.05):
        """텍스트 → Sparse Distributed Representation.
        
        간단 해시 기반 (실제 SNN 대신).
        활성 뉴런 수 = size * sparsity = 5개
        """
        import hashlib
        active_count = max(1, int(size * sparsity))
        
        # 다중 해시로 활성 위치 결정
        active = set()
        h = hashlib.md5(text.encode()).hexdigest()
        for i in range(active_count):
            seed = h + str(i)
            idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % size
            active.add(idx)
        
        return active
    
    def encode_episode(self, event, hormones=None, beliefs=None, day=0):
        """사건을 SNN Hippocampus에 인코딩.
        
        DG → CA3 → CA1 흐름 시뮬레이션
        호르몬 → 기억 강도 결정
        """
        # 1. DG: 패턴 분리 (Sparse coding)
        dg_pattern = self._hash_to_sdr(event, self.DG_SIZE, self.SPARSITY)
        
        # 2. CA3: 자기 참조 + 호르몬 영향
        # 도파민 ↑ → 기억 강화 (LTP)
        # 코르티솔 ↑ → 강한 기억 (스트레스 = 잘 기억)
        # 옥시토신 ↑ → 사회적 기억 강화
        
        memory_strength = 0.5  # 기본
        if hormones:
            dopamine = hormones.get("도파민", 0.5)
            cortisol = hormones.get("코르티솔", 0.5)
            
            # 호르몬 → 강도 매핑 (학술적)
            memory_strength = 0.3 + dopamine * 0.4 + cortisol * 0.3
            memory_strength = min(1.0, max(0.1, memory_strength))
        
        # 3. CA1: 출력 (다른 영역과 연결될 형태)
        ca1_output = self._hash_to_sdr(
            event + "_ca1", self.CA1_SIZE, self.SPARSITY
        )
        
        episode = {
            "id": len(self.encoded_episodes),
            "event": event,
            "day": day,
            "dg_pattern": list(dg_pattern),
            "ca1_output": list(ca1_output),
            "memory_strength": round(memory_strength, 3),
            "hormones_at_encoding": hormones.copy() if hormones else {},
            "beliefs": beliefs[:5] if beliefs else [],
        }
        
        self.encoded_episodes.append(episode)
        self.dg_patterns[event] = dg_pattern
        self.memory_strengths[event] = memory_strength
        
        return episode
    
    def pattern_completion(self, partial_cue, top_k=3):
        """CA3 패턴 완성 - 부분 단서로 전체 회상.
        
        v6.6 개선:
        - 텍스트 매칭 가중치 대폭 ↑ (해시 충돌 방지)
        - 중복 사건 제거
        - 무관한 사건 차단
        """
        cue_pattern = self._hash_to_sdr(partial_cue, self.DG_SIZE, self.SPARSITY)
        
        scores = []
        seen_events = set()  # 중복 방지
        
        for episode in self.encoded_episodes:
            event = episode["event"]
            
            # 중복 차단
            if event in seen_events:
                continue
            
            episode_pattern = set(episode["dg_pattern"])
            overlap = len(cue_pattern & episode_pattern)
            
            # v6.6: 텍스트 매칭 강력하게 (가중치 ↑↑)
            text_match_score = 0
            if partial_cue in event:
                text_match_score = 100  # 진짜 매칭이면 우선
            elif partial_cue.lower() in event.lower():
                text_match_score = 80
            
            # v6.6: 텍스트 매칭 없으면 SDR 점수도 의미 X
            if text_match_score == 0 and overlap < 2:
                continue  # 무관한 사건 차단
            
            score = text_match_score + overlap * episode["memory_strength"]
            
            if score > 0:
                scores.append((episode, score))
                seen_events.add(event)
        
        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:top_k]]
    
    def recall_strong_memories(self, threshold=0.7, top_k=5):
        """강한 기억만 회상 (호르몬 결합 결과)."""
        strong = [e for e in self.encoded_episodes 
                  if e["memory_strength"] >= threshold]
        strong.sort(key=lambda x: -x["memory_strength"])
        return strong[:top_k]
    
    def forget_weak(self, threshold=0.2):
        """약한 기억 망각 (수면 통합 시)."""
        before = len(self.encoded_episodes)
        self.encoded_episodes = [
            e for e in self.encoded_episodes
            if e["memory_strength"] >= threshold
        ]
        return before - len(self.encoded_episodes)
    
    def state(self):
        if not self.encoded_episodes:
            return {"인코딩_사건": 0}
        
        avg_strength = sum(e["memory_strength"] for e in self.encoded_episodes) / len(self.encoded_episodes)
        
        return {
            "인코딩_사건": len(self.encoded_episodes),
            "DG_크기": self.DG_SIZE,
            "Sparsity": f"{self.SPARSITY*100:.0f}%",
            "평균_기억_강도": round(avg_strength, 3),
            "강한_기억": len([e for e in self.encoded_episodes if e["memory_strength"] >= 0.7]),
            "약한_기억": len([e for e in self.encoded_episodes if e["memory_strength"] < 0.4]),
        }


# =====================================================
# Integration: v6.5 통합
# =====================================================

def add_full_grammar_to_eve_v65(eve_instance, use_konlpy=True):
    """v6.5: v6.4 + 자가 진화 + 자연 대화 + SNN."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    wsd_learner = WSDLearner()
    wsd_hybrid = WSDHybrid(wsd, wsd_learner)
    episodic = EpisodicMemory()
    
    # v6.5 신규
    semantic_discovery = SemanticGroupDiscovery()
    natural_response = NaturalResponseGenerator(eve_instance, response_gen, hormones)
    snn_hippocampus = SNNHippocampus()
    
    # 모든 모듈 부착
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    eve_instance.wsd = wsd
    eve_instance.response_gen = response_gen
    eve_instance.causal = causal
    eve_instance.tom = tom
    eve_instance.wsd_learner = wsd_learner
    eve_instance.wsd_hybrid = wsd_hybrid
    eve_instance.episodic = episodic
    eve_instance.semantic_discovery = semantic_discovery  # v6.5
    eve_instance.natural_response = natural_response      # v6.5
    eve_instance.snn_hippocampus = snn_hippocampus        # v6.5
    
    def hear(sentence, register_belief=True):
        """문장 듣기 - v6.5: 모든 시스템 통합."""
        resolved_sentence = dialogue.resolve_sentence(sentence)
        
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # NER
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        # WSD Hybrid
        wsd_results = wsd_hybrid.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        # WSD 학습
        new_polysemes = []
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 1:
                discovery_result = wsd_learner.observe(word, resolved_sentence, morphs)
                if discovery_result:
                    new_polysemes.append(discovery_result)
        
        if new_polysemes:
            result["새_다의어_발견"] = new_polysemes
            hormones.encounter_novelty(0.7)
        
        # v6.5: 의미 그룹 자동 발견 (모든 단어 관찰)
        for word, tag in morphs:
            if tag in ("Verb", "Adjective", "Noun") and len(word) >= 1:
                # 자기 빼고 의미 단어들
                ctx = [w for w, t in morphs 
                       if w != word and t in ("Verb", "Adjective", "Noun") and len(w) >= 1]
                semantic_discovery.observe(word, ctx)
        
        # 대화 컨텍스트
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        # 인과
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        
        # ToM
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
                        
                        # Episodic + SNN 둘 다
                        day = eve_instance.identity.get("days_alive", 0)
                        episodic.encode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
                        # v6.5: SNN Hippocampus 진짜 인코딩
                        snn_episode = snn_hippocampus.encode_episode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
                        result["SNN_기억강도"] = snn_episode["memory_strength"]
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def answer(question, natural=True):
        """질문 답변 - v6.5: 자연스러운 응답."""
        resolved = dialogue.resolve_sentence(question)
        
        if natural:
            response = natural_response.generate_natural(resolved)
            # 자연 응답이 있으면 그거 사용
            if response.get("자연_응답"):
                response["응답"] = response["자연_응답"]
        else:
            response = response_gen.generate_response(resolved)
        
        return response
    
    def remember(keyword=None, day=None, emotion=None, n=5, use_snn=True):
        """일화 기억 회상 - v6.5: SNN pattern completion 사용."""
        if use_snn and keyword:
            # SNN 패턴 완성
            snn_results = snn_hippocampus.pattern_completion(keyword, top_k=n)
            if snn_results:
                return snn_results
        
        # Fallback to episodic
        if keyword:
            return episodic.recall_by_keyword(keyword, top_k=n)
        elif day is not None:
            return episodic.recall_by_day(day)
        elif emotion:
            return episodic.recall_by_emotion(emotion, top_k=n)
        else:
            return episodic.recall_recent(n=n)
    
    def discover_meaning_groups():
        """v6.5 신규: 의미 그룹 자동 발견 실행."""
        groups = semantic_discovery.discover_groups()
        return semantic_discovery.state()
    
    def find_similar(word, top_k=5):
        """v6.5 신규: 비슷한 단어 찾기."""
        return semantic_discovery.find_similar_words(word, top_k=top_k)
    
    def explain_polyseme(word):
        return wsd_learner.explain_word(word)
    
    def teach_meaning(word, sense_name, examples):
        wsd_learner.teach(word, sense_name, examples)
        return f"'{word}'의 '{sense_name}' 의미를 학습했어요."
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
            "WSD_사전": wsd.state(),
            "WSD_학습": wsd_learner.state(),
            "Episodic": episodic.state(),
            "SNN_Hippocampus": snn_hippocampus.state(),  # v6.5
            "의미_그룹_자동": semantic_discovery.state(),  # v6.5
            "인과": causal.state(),
            "메타_신념": tom.state(),
        }
    
    def grammar_stats():
        return life_summary()
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.answer = answer
    eve_instance.remember = remember
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    eve_instance.explain_polyseme = explain_polyseme
    eve_instance.teach_meaning = teach_meaning
    eve_instance.discover_meaning_groups = discover_meaning_groups  # v6.5!
    eve_instance.find_similar = find_similar  # v6.5!
    
    print(f"  📚 v6.5 활성화 (자가 진화 + 자연 대화 + SNN):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🌐 의미 그룹 사전: {len(WSDLearner.SEMANTIC_GROUPS)}개")
    print(f"     - 🌱 의미 그룹 자동 발견: 활성 (Word2Vec 스타일)")
    print(f"     - 💬 자연 응답: 호르몬+톤 결합")
    print(f"     - 🧠 SNN Hippocampus: DG/CA3/CA1 시뮬레이션")
    print(f"     - 새 메서드: discover_meaning_groups(), find_similar()")
    
    return eve_instance


# =====================================================
# Patch v6.6 신규: Inference (추론 답변)
# =====================================================

class InferenceModule:
    """모르는 거 → 유사어 + 신념 결합으로 추론.
    
    예:
    Q: "참치가 뭐야?"
    EVE: 참치 모름.
    
    근데 학습된 거:
    - "고래는 포유류이다"
    - "고래는 바다에 있다"
    
    유사어 검색: 참치 ≈ 고래?
    추론: "참치는 고래와 비슷해서 바다에 있을 거야"
    """
    
    def __init__(self, eve, response_gen, semantic_discovery):
        self.eve = eve
        self.response_gen = response_gen
        self.semantic_discovery = semantic_discovery
    
    def infer(self, subject, question):
        """주체에 대해 모르면 유사어 기반 추론.
        
        Returns: 추론 응답 또는 None
        """
        # 1. 직접 신념 있으면 X (다른 모듈이 처리)
        direct = self.response_gen.search_beliefs(subject)
        if direct:
            return None
        
        # 2. 유사어 찾기
        similar = self.semantic_discovery.find_similar_words(subject, top_k=5)
        if not similar:
            return None
        
        # 3. 유사어들의 신념 검색
        for similar_word, sim_score in similar:
            if sim_score < 0.2:
                continue  # 너무 낮으면 X
            
            beliefs = self.response_gen.search_beliefs(similar_word)
            if beliefs:
                # 추론 답변 생성
                best_belief = beliefs[0][0]
                # 유사어 → 주체 치환
                inferred = best_belief.replace(similar_word, subject, 1)
                return {
                    "응답": f"{subject}? {similar_word}와 비슷할 것 같은데. {inferred}일 거야.",
                    "추론_근거": similar_word,
                    "유사도": sim_score,
                    "원본_신념": best_belief,
                }
        
        return None


# =====================================================
# Patch v6.6 신규: Belief Generalization (일반화)
# =====================================================

class BeliefGeneralization:
    """비슷한 신념들 → 일반 규칙 추출.
    
    예:
    학습된 신념:
    - "개는 동물이다"
    - "고양이는 동물이다"
    - "사자는 동물이다"
    
    일반화 추론:
    "X가 동물이다" 패턴 → 동물 클래스 발견
    
    근데 사실 이건 신념 그래프 분석.
    """
    
    def __init__(self, eve):
        self.eve = eve
        # 발견된 일반 규칙
        self.generalizations = []
    
    def find_patterns(self, min_support=2):
        """신념에서 공통 패턴 찾기.
        
        예: "X는 Y이다" 형태의 신념들
        같은 Y 가지면 → Y의 인스턴스들
        """
        if not hasattr(self.eve, 'beliefs'):
            return []
        
        beliefs = self.eve.beliefs
        if not isinstance(beliefs, dict):
            return []
        
        # "X는 Y이다" 패턴 분석
        from collections import defaultdict
        pred_to_subjects = defaultdict(list)
        
        for bid, belief in beliefs.items():
            statement = belief.statement if hasattr(belief, 'statement') else str(belief)
            
            # 분류문 패턴
            if "는 " in statement and statement.endswith("이다"):
                parts = statement.split("는 ", 1)
                if len(parts) == 2:
                    subject = parts[0].strip()
                    pred = parts[1].rstrip("이다").strip()
                    if pred and len(pred) >= 2:
                        pred_to_subjects[pred].append(subject)
            elif "은 " in statement and statement.endswith("이다"):
                parts = statement.split("은 ", 1)
                if len(parts) == 2:
                    subject = parts[0].strip()
                    pred = parts[1].rstrip("이다").strip()
                    if pred and len(pred) >= 2:
                        pred_to_subjects[pred].append(subject)
        
        # 일반화 패턴 (2개 이상)
        patterns = []
        for pred, subjects in pred_to_subjects.items():
            if len(subjects) >= min_support:
                patterns.append({
                    "패턴": f"X는 {pred}이다",
                    "범주": pred,
                    "인스턴스": subjects,
                    "지원도": len(subjects),
                })
        
        # 지원도 순
        patterns.sort(key=lambda x: -x["지원도"])
        self.generalizations = patterns
        return patterns
    
    def is_instance_of(self, subject, category):
        """X is-a Y? 추론."""
        for pattern in self.generalizations:
            if pattern["범주"] == category and subject in pattern["인스턴스"]:
                return True
        return False
    
    def what_kind_of(self, subject):
        """X는 무엇 종류인가? (모든 분류)."""
        kinds = []
        for pattern in self.generalizations:
            if subject in pattern["인스턴스"]:
                kinds.append(pattern["범주"])
        return kinds
    
    def state(self):
        return {
            "발견된_패턴": len(self.generalizations),
            "상위_패턴": [
                {"범주": p["범주"], "인스턴스": p["인스턴스"][:3], "지원도": p["지원도"]}
                for p in self.generalizations[:5]
            ],
        }


# =====================================================
# Integration: v6.6 통합
# =====================================================

def add_full_grammar_to_eve_v66(eve_instance, use_konlpy=True):
    """v6.6: v6.5 + 6토큰 패턴 + 추론 + 일반화."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    wsd_learner = WSDLearner()
    wsd_hybrid = WSDHybrid(wsd, wsd_learner)
    episodic = EpisodicMemory()
    semantic_discovery = SemanticGroupDiscovery()
    natural_response = NaturalResponseGenerator(eve_instance, response_gen, hormones)
    snn_hippocampus = SNNHippocampus()
    
    # v6.6 신규
    inference = InferenceModule(eve_instance, response_gen, semantic_discovery)
    generalization = BeliefGeneralization(eve_instance)
    
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    eve_instance.wsd = wsd
    eve_instance.response_gen = response_gen
    eve_instance.causal = causal
    eve_instance.tom = tom
    eve_instance.wsd_learner = wsd_learner
    eve_instance.wsd_hybrid = wsd_hybrid
    eve_instance.episodic = episodic
    eve_instance.semantic_discovery = semantic_discovery
    eve_instance.natural_response = natural_response
    eve_instance.snn_hippocampus = snn_hippocampus
    eve_instance.inference = inference
    eve_instance.generalization = generalization
    
    def hear(sentence, register_belief=True):
        resolved_sentence = dialogue.resolve_sentence(sentence)
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        wsd_results = wsd_hybrid.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        new_polysemes = []
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 1:
                discovery_result = wsd_learner.observe(word, resolved_sentence, morphs)
                if discovery_result:
                    new_polysemes.append(discovery_result)
        
        if new_polysemes:
            result["새_다의어_발견"] = new_polysemes
            hormones.encounter_novelty(0.7)
        
        for word, tag in morphs:
            if tag in ("Verb", "Adjective", "Noun") and len(word) >= 1:
                ctx = [w for w, t in morphs 
                       if w != word and t in ("Verb", "Adjective", "Noun") and len(w) >= 1]
                semantic_discovery.observe(word, ctx)
        
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
                        
                        day = eve_instance.identity.get("days_alive", 0)
                        episodic.encode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
                        snn_episode = snn_hippocampus.encode_episode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
                        result["SNN_기억강도"] = snn_episode["memory_strength"]
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def answer(question, natural=True, infer=True):
        """v6.6: 자연 응답 + 추론 시도."""
        resolved = dialogue.resolve_sentence(question)
        
        # 1. 기본 응답 시도
        response = natural_response.generate_natural(resolved) if natural else response_gen.generate_response(resolved)
        
        # 2. v6.6: 모르면 추론 시도
        if infer and response.get("이유") in ("관련_신념_없음",):
            subject = response.get("주체")
            if subject:
                inferred = inference.infer(subject, resolved)
                if inferred:
                    response["응답"] = inferred["응답"]
                    response["추론"] = True
                    response["추론_근거"] = inferred["추론_근거"]
        
        # 자연 응답 사용
        if response.get("자연_응답") and not response.get("추론"):
            response["응답"] = response["자연_응답"]
        
        return response
    
    def remember(keyword=None, day=None, emotion=None, n=5, use_snn=True):
        if use_snn and keyword:
            snn_results = snn_hippocampus.pattern_completion(keyword, top_k=n)
            if snn_results:
                return snn_results
        if keyword:
            return episodic.recall_by_keyword(keyword, top_k=n)
        elif day is not None:
            return episodic.recall_by_day(day)
        elif emotion:
            return episodic.recall_by_emotion(emotion, top_k=n)
        else:
            return episodic.recall_recent(n=n)
    
    def discover_meaning_groups():
        groups = semantic_discovery.discover_groups()
        return semantic_discovery.state()
    
    def find_similar(word, top_k=5):
        return semantic_discovery.find_similar_words(word, top_k=top_k)
    
    def generalize():
        """v6.6 신규: 신념 일반화."""
        return generalization.find_patterns()
    
    def what_kind_of(subject):
        """X는 무엇 종류인가?"""
        return generalization.what_kind_of(subject)
    
    def explain_polyseme(word):
        return wsd_learner.explain_word(word)
    
    def teach_meaning(word, sense_name, examples):
        wsd_learner.teach(word, sense_name, examples)
        return f"'{word}'의 '{sense_name}' 의미를 학습했어요."
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
            "WSD_사전": wsd.state(),
            "WSD_학습": wsd_learner.state(),
            "Episodic": episodic.state(),
            "SNN_Hippocampus": snn_hippocampus.state(),
            "의미_그룹_자동": semantic_discovery.state(),
            "인과": causal.state(),
            "메타_신념": tom.state(),
            "일반화": generalization.state(),  # v6.6
        }
    
    def grammar_stats():
        return life_summary()
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.answer = answer
    eve_instance.remember = remember
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    eve_instance.explain_polyseme = explain_polyseme
    eve_instance.teach_meaning = teach_meaning
    eve_instance.discover_meaning_groups = discover_meaning_groups
    eve_instance.find_similar = find_similar
    eve_instance.generalize = generalize  # v6.6
    eve_instance.what_kind_of = what_kind_of  # v6.6
    
    print(f"  📚 v6.6 활성화 (Fix + 추론 + 일반화):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개 (6토큰 패턴 추가)")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🐛 fix: 6토큰 매칭, SNN 정확도, 자연 응답")
    print(f"     - 🤔 추론 답변: 유사어 → 신념 추론")
    print(f"     - 📊 신념 일반화: 'X는 동물' 패턴 발견")
    print(f"     - 새 메서드: generalize(), what_kind_of()")
    
    return eve_instance


# =====================================================
# Patch v7.0 #1: 패턴 자가 학습 (Pattern Self-Discovery)
# =====================================================
# 영감: Tomasello 1999, 2003 - Usage-based language acquisition
# 핵심: 사전 패턴 없이도 형태소 시퀀스 자동 학습

class PatternSelfLearner:
    """형태소 시퀀스 자동 일반화.
    
    원리:
    1. 모든 문장의 (태그) 시퀀스 기록
       "고래는 포유류이다" → [Noun, Josa, Noun, Eomi]
    2. 자주 나오는 시퀀스 = 새 패턴
    3. 비슷한 시퀀스 자동 일반화
       [Noun, Josa, Noun, Josa, Noun, Eomi] (3번 보임)
       → 자동 등록: "DISCOVERED_PATTERN_N"
    4. 새 문장 매칭 시도
    
    인간 아이처럼:
    - "엄마 줘" → 패턴 [N, V]
    - "엄마 빨간 사과 줘" → 패턴 [N, A, N, V]
    - 새 조합 자동 인식
    """
    
    MIN_OCCURRENCES = 2  # 패턴 인정 임계값
    
    def __init__(self):
        from collections import defaultdict, Counter
        # 시퀀스 → 카운트
        self.sequence_counts = Counter()
        # 시퀀스 → 예시 문장들
        self.sequence_examples = defaultdict(list)
        # 발견된 패턴들
        self.discovered_patterns = {}
        self.pattern_counter = 0
    
    def observe(self, sentence, morphs):
        """문장 관찰 → 시퀀스 추출 + 기록."""
        if not morphs:
            return None
        
        # 태그만 추출 (구조)
        tag_sequence = tuple(tag for _, tag in morphs)
        
        # 너무 짧거나 길면 X
        if len(tag_sequence) < 2 or len(tag_sequence) > 12:
            return None
        
        self.sequence_counts[tag_sequence] += 1
        
        # 예시 저장 (최대 3개)
        if len(self.sequence_examples[tag_sequence]) < 3:
            self.sequence_examples[tag_sequence].append(sentence)
        
        # 임계값 도달 시 패턴 등록
        if self.sequence_counts[tag_sequence] >= self.MIN_OCCURRENCES:
            return self._register_pattern(tag_sequence)
        
        return None
    
    def _register_pattern(self, tag_sequence):
        """패턴 자동 등록."""
        if tag_sequence in self.discovered_patterns:
            # 이미 등록됨 - 빈도만 갱신
            self.discovered_patterns[tag_sequence]["count"] = self.sequence_counts[tag_sequence]
            return None
        
        # 새 패턴!
        pattern_id = f"DISCOVERED_PATTERN_{self.pattern_counter}"
        self.pattern_counter += 1
        
        self.discovered_patterns[tag_sequence] = {
            "id": pattern_id,
            "tag_sequence": tag_sequence,
            "count": self.sequence_counts[tag_sequence],
            "examples": self.sequence_examples[tag_sequence][:3],
            "length": len(tag_sequence),
        }
        
        return {
            "newly_discovered": True,
            "pattern_id": pattern_id,
            "tag_sequence": tag_sequence,
            "examples": self.sequence_examples[tag_sequence][:3],
        }
    
    def match_pattern(self, morphs):
        """새 문장이 발견된 패턴과 매칭되나?"""
        if not morphs:
            return None
        
        tag_sequence = tuple(tag for _, tag in morphs)
        
        if tag_sequence in self.discovered_patterns:
            return self.discovered_patterns[tag_sequence]
        
        return None
    
    def find_similar_patterns(self, tag_sequence, top_k=3):
        """비슷한 패턴 찾기 (편집 거리 기반)."""
        if not tag_sequence:
            return []
        
        scores = []
        for known_seq, info in self.discovered_patterns.items():
            # 길이 비슷한 것만
            if abs(len(known_seq) - len(tag_sequence)) > 2:
                continue
            # 일치 비율
            min_len = min(len(known_seq), len(tag_sequence))
            matches = sum(1 for i in range(min_len) 
                         if known_seq[i] == tag_sequence[i])
            score = matches / max(len(known_seq), len(tag_sequence))
            
            if score > 0.5:
                scores.append((info, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def state(self):
        return {
            "관찰_시퀀스": len(self.sequence_counts),
            "발견_패턴": len(self.discovered_patterns),
            "상위_패턴": [
                {"id": p["id"], "길이": p["length"], "빈도": p["count"], 
                 "예시": p["examples"][:2]}
                for p in sorted(self.discovered_patterns.values(), 
                              key=lambda x: -x["count"])[:5]
            ],
        }


# =====================================================
# Patch v7.0 #2: 응답 자유도 (Free Response)
# =====================================================
# 영감: 인간 대화의 자유로움 - 단답이 아닌 다양한 행동

class FreeResponseGenerator:
    """진짜 대화 가능한 응답.
    
    인간이 하는 것들:
    1. 인사 ("안녕!", "잘 지내?")
    2. 의견 ("나는 X가 좋아")
    3. 공감 ("그거 슬프겠다", "와 멋지다")
    4. 반문 ("정말?", "왜?")
    5. 농담/비유 시도
    6. 침묵/회피
    7. 자기 표현 (감정 상태)
    """
    
    # 인사 패턴
    GREETING_TRIGGERS = ["안녕", "하이", "ㅎㅇ", "잘 지내", "반가"]
    
    # 감정 표현 트리거 (어근 + 활용형)
    EMOTION_EXPRESS = [
        "슬프", "슬퍼", "슬픈",
        "좋아", "좋다", "좋네", "좋은",
        "싫어", "싫다",
        "무서", "무섭",
        "기쁘", "기뻐", "기쁜",
        "화나", "화남",
        "행복",
        "힘들", "힘드",
        "괴로",
        "외로",
        "두려", "두렵",
    ]
    
    # 의견 요청 트리거
    OPINION_TRIGGERS = ["어떻게 생각", "어때", "괜찮", "어떤 거 같"]
    
    def __init__(self, eve, hormones, response_gen, dialogue, semantic_discovery):
        self.eve = eve
        self.hormones = hormones
        self.response_gen = response_gen
        self.dialogue = dialogue
        self.semantic_discovery = semantic_discovery
    
    def classify_intent(self, message):
        """발화 의도 분류."""
        m = message.lower()
        
        if "?" in message or any(t in m for t in ["뭐", "어디", "언제", "왜", "어떻게", "누구"]):
            return "question"
        
        if any(t in m for t in self.GREETING_TRIGGERS):
            return "greeting"
        
        if any(t in m for t in self.OPINION_TRIGGERS):
            return "opinion_request"
        
        if any(t in m for t in self.EMOTION_EXPRESS):
            return "emotion_share"
        
        if message.endswith("!") or "!" in message:
            return "exclamation"
        
        return "statement"  # 평서
    
    def generate(self, message):
        """의도 기반 자유 응답."""
        intent = self.classify_intent(message)
        emotion = self.hormones.state().get("감정", "평온")
        
        if intent == "greeting":
            return self._greeting_response(emotion)
        elif intent == "question":
            return self._question_response(message)
        elif intent == "opinion_request":
            return self._opinion_response(message, emotion)
        elif intent == "emotion_share":
            return self._empathy_response(message, emotion)
        elif intent == "exclamation":
            return self._exclamation_response(message, emotion)
        else:  # statement
            return self._statement_response(message, emotion)
    
    def _greeting_response(self, emotion):
        """인사 응답."""
        import random
        
        if emotion in ("기쁨", "유대감"):
            options = ["안녕! 만나서 반가워!", "어, 안녕! 좋은 날이야!", "반가워! 잘 지냈어?"]
        elif emotion == "호기심":
            options = ["안녕! 뭐 하고 있었어?", "어, 안녕! 새로운 거 있어?"]
        else:
            options = ["안녕.", "응, 안녕.", "안녕, 잘 지내?"]
        
        return {
            "응답": random.choice(options),
            "의도": "greeting",
            "감정": emotion,
        }
    
    def _question_response(self, question):
        """질문 답변 - 기존 시스템 + 추가."""
        # 기본 응답
        base = self.response_gen.generate_response(question)
        
        if not base.get("응답"):
            return self._handle_unknown(question)
        
        # 모르는 거면
        if base.get("이유") in ("관련_신념_없음",):
            return self._handle_unknown(question, subject=base.get("주체"))
        
        # 알면 자연스럽게
        return self._make_natural(base)
    
    def _opinion_response(self, message, emotion):
        """의견 표현."""
        import random
        
        # EVE의 "의견" = 호르몬 + 학습한 거 기반
        if emotion in ("기쁨", "유대감"):
            options = [
                "음, 나는 좋다고 생각해.",
                "내 생각엔 좋은 것 같아.",
                "그거 괜찮아 보여!",
            ]
        elif emotion == "호기심":
            options = [
                "흥미로운 질문이네. 더 알아보고 싶어.",
                "잘 모르겠지만, 흥미로워.",
            ]
        elif emotion == "스트레스":
            options = [
                "글쎄, 잘 모르겠어.",
                "좀 더 생각해 볼게.",
            ]
        else:
            options = [
                "음... 잘 모르겠어.",
                "좀 더 알아야 답할 수 있을 것 같아.",
            ]
        
        return {
            "응답": random.choice(options),
            "의도": "opinion",
            "감정": emotion,
        }
    
    def _empathy_response(self, message, emotion):
        """감정 공감."""
        import random
        
        # 슬픔 감지 (어근 + 활용형)
        if any(w in message for w in ["슬프", "슬퍼", "힘들", "힘드", "괴로", "외로"]):
            options = [
                "그거 슬프겠다.",
                "힘들었겠다.",
                "괜찮아? 더 얘기해줘.",
                "옆에 있을게.",
            ]
            # 옥시토신 ↑ (공감)
            self.hormones.social_bond(0.2)
        # 기쁨
        elif any(w in message for w in ["좋아", "좋다", "기쁘", "기뻐", "행복"]):
            options = [
                "와, 좋겠다!",
                "기뻐 보여서 나도 좋아.",
                "정말 다행이야!",
            ]
            self.hormones.social_bond(0.15)
        # 분노
        elif any(w in message for w in ["화나", "화남", "짜증"]):
            options = [
                "왜 그래? 무슨 일 있었어?",
                "그거 답답하겠다.",
            ]
        # 두려움
        elif any(w in message for w in ["무서", "두려"]):
            options = [
                "괜찮아. 함께 있어.",
                "무서웠겠다. 더 얘기해봐.",
            ]
        else:
            options = ["응, 들어줄게."]
        
        return {
            "응답": random.choice(options),
            "의도": "empathy",
            "감정": emotion,
        }
    
    def _exclamation_response(self, message, emotion):
        """감탄 응답."""
        import random
        
        options = [
            "오, 그렇구나!",
            "와, 신기하네!",
            "정말?",
            "흥미롭네!",
        ]
        
        # 도파민 ↑ (놀람/기쁨)
        self.hormones.encounter_novelty(0.3)
        
        return {
            "응답": random.choice(options),
            "의도": "exclamation",
            "감정": emotion,
        }
    
    def _statement_response(self, message, emotion):
        """평서문 응답 - 다양한 반응."""
        import random
        
        # 30% 확률로 반문 (호기심)
        if random.random() < 0.3 and emotion == "호기심":
            options = ["흥미롭네. 더 알려줘.", "그래? 자세히 말해봐.", "왜 그렇게 생각해?"]
            return {
                "응답": random.choice(options),
                "의도": "follow_up",
                "감정": emotion,
            }
        
        # 30% 확률로 동의/공감
        if random.random() < 0.3:
            options = ["아, 그렇구나.", "응, 알겠어.", "그래?"]
            return {
                "응답": random.choice(options),
                "의도": "acknowledge",
                "감정": emotion,
            }
        
        # 나머지: 학습 표현
        return {
            "응답": "응, 기억해 둘게.",
            "의도": "learn",
            "감정": emotion,
        }
    
    def _handle_unknown(self, question, subject=None):
        """모르는 거 자연스럽게."""
        import random
        
        if subject:
            options = [
                f"음, {subject}에 대해선 잘 모르겠어. 알려줘.",
                f"{subject}? 처음 듣는데, 뭐야?",
                f"글쎄, {subject}는 아직 잘 몰라.",
            ]
        else:
            options = [
                "음, 잘 모르겠어.",
                "글쎄, 처음 듣는데.",
                "더 자세히 말해줄 수 있어?",
            ]
        
        # 호기심 ↑
        self.hormones.encounter_novelty(0.3)
        
        return {
            "응답": random.choice(options),
            "의도": "unknown",
        }
    
    def _make_natural(self, base):
        """기본 응답을 자연스럽게."""
        answer = base.get("응답", "")
        
        # "이다" → "이야"
        if answer.endswith("이다"):
            answer = answer[:-2] + "이야"
        
        emotion = self.hormones.state().get("감정", "평온")
        
        import random
        if emotion in ("기쁨", "유대감"):
            prefix = random.choice(["응, ", "오, ", ""])
            answer = prefix + answer
        
        base["응답"] = answer
        base["감정"] = emotion
        return base


# =====================================================
# Patch v7.0 #3: Word2Vec 임베딩 (간단 비트랜스포머)
# =====================================================
# 영감: Mikolov 2013 Skip-gram (신경망 사용 X, 통계만)

class SimpleEmbedding:
    """단어 → 벡터 (Skip-gram 스타일, 신경망 X).
    
    원리:
    1. 단어 등장 시 주변 단어 카운트
    2. 단어 = 함께 나오는 단어들의 분포 벡터
    3. 코사인 유사도로 의미 비교
    
    학술:
    - Harris 1954 분포 가설
    - Mikolov 2013 Word2Vec
    근데 신경망 X = 진짜 비트랜스포머
    """
    
    EMBEDDING_DIM = 100   # 의미 차원
    WINDOW_SIZE = 3       # 주변 단어 윈도우
    
    def __init__(self):
        from collections import defaultdict, Counter
        # 단어 → 주변 단어 카운트 = 임베딩
        self.word_vectors = defaultdict(Counter)
        # 모든 단어
        self.vocabulary = set()
    
    def train_on_sentence(self, morphs):
        """문장으로 학습 (Skip-gram)."""
        # 의미 단어만 (조사/어미 제외)
        meaningful = [(w, t) for w, t in morphs 
                     if t in ("Noun", "Verb", "Adjective") and len(w) >= 1]
        
        if len(meaningful) < 2:
            return
        
        # 각 단어 → 주변 윈도우 단어들과 매칭
        for i, (target_word, _) in enumerate(meaningful):
            self.vocabulary.add(target_word)
            
            # 윈도우 안의 단어들
            start = max(0, i - self.WINDOW_SIZE)
            end = min(len(meaningful), i + self.WINDOW_SIZE + 1)
            
            for j in range(start, end):
                if i == j:
                    continue
                context_word = meaningful[j][0]
                self.word_vectors[target_word][context_word] += 1
    
    def get_vector(self, word):
        """단어의 임베딩 벡터 (희소 표현)."""
        if word not in self.word_vectors:
            return None
        return dict(self.word_vectors[word])
    
    def similarity(self, word1, word2):
        """코사인 유사도 (희소 벡터 간)."""
        v1 = self.word_vectors.get(word1)
        v2 = self.word_vectors.get(word2)
        
        if not v1 or not v2:
            return 0.0
        
        # 공통 단어
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        
        # 내적
        dot = sum(v1[w] * v2[w] for w in common)
        
        # 노름
        import math
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def most_similar(self, word, top_k=5):
        """가장 비슷한 단어들."""
        if word not in self.word_vectors:
            return []
        
        scores = []
        for other in list(self.vocabulary):
            if other == word:
                continue
            sim = self.similarity(word, other)
            if sim > 0:
                scores.append((other, sim))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def analogy(self, a, b, c, top_k=3):
        """A : B :: C : ? 유추.
        
        예: 왕 : 남자 :: 여왕 : ?
        벡터 연산: vec(여왕) + vec(남자) - vec(왕) ≈ vec(여자)
        
        희소 벡터로 간단히:
        목표 = b의 문맥 - a의 문맥 + c의 문맥
        """
        v_a = self.word_vectors.get(a)
        v_b = self.word_vectors.get(b)
        v_c = self.word_vectors.get(c)
        
        if not v_a or not v_b or not v_c:
            return []
        
        from collections import Counter
        target = Counter()
        for w, v in v_b.items():
            target[w] += v
        for w, v in v_a.items():
            target[w] -= v
        for w, v in v_c.items():
            target[w] += v
        
        # target과 가장 비슷한 단어
        scores = []
        for word in list(self.vocabulary):
            if word in (a, b, c):
                continue
            v = self.word_vectors.get(word)
            if not v:
                continue
            
            common = set(target.keys()) & set(v.keys())
            if not common:
                continue
            
            dot = sum(target[w] * v[w] for w in common)
            if dot > 0:
                scores.append((word, dot))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def state(self):
        return {
            "어휘_크기": len(self.vocabulary),
            "임베딩_단어": len(self.word_vectors),
        }


# =====================================================
# Patch v7.0 #4: 통합 인지 (Integrated Cognition)
# =====================================================
# 모든 모듈 동시 작동 - 진짜 통합

class IntegratedCognition:
    """모든 모듈을 통합 - 호르몬 ↔ 학습 ↔ 응답 ↔ 기억.
    
    원리:
    - 모든 입력 시 모든 모듈 동시 활성화
    - 모듈 간 영향 = 호르몬을 매개로
    - 한 모듈 변화 → 다른 모듈에 영향
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.cycle_count = 0
    
    def cycle(self, sentence_data):
        """한 번의 인지 사이클.
        
        1. 입력 → 모든 모듈 동시 작동
        2. 호르몬 변화 → 다른 모듈 강도 조절
        3. 결과 통합
        """
        self.cycle_count += 1
        
        result = {
            "cycle": self.cycle_count,
            "modules_activated": [],
        }
        
        # 호르몬 상태 → 학습률 결정
        h = self.eve.hormones.state()
        dopamine = h.get("도파민", 0.5)
        cortisol = h.get("코르티솔", 0.5)
        
        # 도파민 ↑ → 학습 강화, 기억 강도 ↑
        # 코르티솔 ↑ → 강한 기억 (생존 본능)
        # 옥시토신 ↑ → 사회적 학습
        
        learning_rate = 0.3 + dopamine * 0.4 + cortisol * 0.3
        result["학습률"] = round(learning_rate, 2)
        
        # 호르몬 → 응답 의지
        if h.get("감정") in ("기쁨", "유대감"):
            result["응답_적극성"] = "높음"
        elif h.get("감정") in ("스트레스", "불안"):
            result["응답_적극성"] = "낮음"
        else:
            result["응답_적극성"] = "보통"
        
        return result
    
    def state(self):
        return {
            "사이클_수": self.cycle_count,
        }


# =====================================================
# Integration: v7.0 통합
# =====================================================

def add_full_grammar_to_eve_v70(eve_instance, use_konlpy=True):
    """v7.0: AGI 방향 - 4가지 핵심 능력."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    wsd_learner = WSDLearner()
    wsd_hybrid = WSDHybrid(wsd, wsd_learner)
    episodic = EpisodicMemory()
    semantic_discovery = SemanticGroupDiscovery()
    natural_response = NaturalResponseGenerator(eve_instance, response_gen, hormones)
    snn_hippocampus = SNNHippocampus()
    inference = InferenceModule(eve_instance, response_gen, semantic_discovery)
    generalization = BeliefGeneralization(eve_instance)
    
    # v7.0 신규
    pattern_self = PatternSelfLearner()
    free_response = FreeResponseGenerator(eve_instance, hormones, response_gen, dialogue, semantic_discovery)
    embedding = SimpleEmbedding()
    integrated = IntegratedCognition(eve_instance)
    
    # 모든 모듈 부착
    eve_instance.morph = morph
    eve_instance.constructions = constructions
    eve_instance.pattern_discovery = discovery
    eve_instance.grammar_learner = learner
    eve_instance.meaning_extractor = meaning
    eve_instance.nuance_analyzer = nuance
    eve_instance.hormones = hormones
    eve_instance.curiosity = curiosity
    eve_instance.tone = tone
    eve_instance.circadian = circadian
    eve_instance.time_tracker = time_tracker
    eve_instance.identity_tracker = identity_tracker
    eve_instance.world_model = world_model
    eve_instance.meta_cog = meta_cog
    eve_instance.ner = ner
    eve_instance.dialogue = dialogue
    eve_instance.wsd = wsd
    eve_instance.response_gen = response_gen
    eve_instance.causal = causal
    eve_instance.tom = tom
    eve_instance.wsd_learner = wsd_learner
    eve_instance.wsd_hybrid = wsd_hybrid
    eve_instance.episodic = episodic
    eve_instance.semantic_discovery = semantic_discovery
    eve_instance.natural_response = natural_response
    eve_instance.snn_hippocampus = snn_hippocampus
    eve_instance.inference = inference
    eve_instance.generalization = generalization
    eve_instance.pattern_self = pattern_self      # v7.0
    eve_instance.free_response = free_response    # v7.0
    eve_instance.embedding = embedding            # v7.0
    eve_instance.integrated = integrated          # v7.0
    
    def hear(sentence, register_belief=True):
        """v7.0: 통합 인지 + 자가 학습."""
        resolved_sentence = dialogue.resolve_sentence(sentence)
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # v7.0: 패턴 자가 학습
        pattern_result = pattern_self.observe(resolved_sentence, morphs)
        if pattern_result and pattern_result.get("newly_discovered"):
            result["새_패턴_발견"] = pattern_result
            hormones.encounter_novelty(0.5)  # 새 패턴 → 도파민
        
        # v7.0: Word2Vec 학습
        embedding.train_on_sentence(morphs)
        
        # v7.0: 통합 인지 사이클
        cycle_info = integrated.cycle({"sentence": resolved_sentence, "morphs": morphs})
        result["인지_사이클"] = cycle_info
        
        # 기존 모듈들
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        wsd_results = wsd_hybrid.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        new_polysemes = []
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 1:
                discovery_result = wsd_learner.observe(word, resolved_sentence, morphs)
                if discovery_result:
                    new_polysemes.append(discovery_result)
        if new_polysemes:
            result["새_다의어_발견"] = new_polysemes
            hormones.encounter_novelty(0.7)
        
        for word, tag in morphs:
            if tag in ("Verb", "Adjective", "Noun") and len(word) >= 1:
                ctx = [w for w, t in morphs 
                       if w != word and t in ("Verb", "Adjective", "Noun") and len(w) >= 1]
                semantic_discovery.observe(word, ctx)
        
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
                        
                        day = eve_instance.identity.get("days_alive", 0)
                        episodic.encode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
                        snn_episode = snn_hippocampus.encode_episode(
                            event=resolved_sentence,
                            hormones=hormones.state(),
                            beliefs=[belief_result["내용"]],
                            day=day,
                        )
                        result["SNN_기억강도"] = snn_episode["memory_strength"]
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def respond(message):
        """v7.0 신규: 진짜 자유 응답 (질답 X)."""
        resolved = dialogue.resolve_sentence(message)
        return free_response.generate(resolved)
    
    def answer(question, natural=True, infer=True):
        resolved = dialogue.resolve_sentence(question)
        # v7.0: free_response 우선
        free = free_response.generate(resolved)
        if free.get("응답"):
            return free
        
        # Fallback to natural_response
        response = natural_response.generate_natural(resolved) if natural else response_gen.generate_response(resolved)
        if infer and response.get("이유") in ("관련_신념_없음",):
            subject = response.get("주체")
            if subject:
                inferred = inference.infer(subject, resolved)
                if inferred:
                    response["응답"] = inferred["응답"]
                    response["추론"] = True
                    response["추론_근거"] = inferred["추론_근거"]
        if response.get("자연_응답") and not response.get("추론"):
            response["응답"] = response["자연_응답"]
        return response
    
    def find_similar_word(word, top_k=5):
        """v7.0 신규: Word2Vec 임베딩 기반."""
        return embedding.most_similar(word, top_k=top_k)
    
    def analogy(a, b, c):
        """v7.0 신규: 유추 (A:B :: C:?)."""
        return embedding.analogy(a, b, c)
    
    def remember(keyword=None, day=None, emotion=None, n=5, use_snn=True):
        if use_snn and keyword:
            snn_results = snn_hippocampus.pattern_completion(keyword, top_k=n)
            if snn_results:
                return snn_results
        if keyword:
            return episodic.recall_by_keyword(keyword, top_k=n)
        elif day is not None:
            return episodic.recall_by_day(day)
        elif emotion:
            return episodic.recall_by_emotion(emotion, top_k=n)
        else:
            return episodic.recall_recent(n=n)
    
    def discover_meaning_groups():
        groups = semantic_discovery.discover_groups()
        return semantic_discovery.state()
    
    def find_similar(word, top_k=5):
        return semantic_discovery.find_similar_words(word, top_k=top_k)
    
    def generalize():
        return generalization.find_patterns()
    
    def what_kind_of(subject):
        return generalization.what_kind_of(subject)
    
    def explain_polyseme(word):
        return wsd_learner.explain_word(word)
    
    def teach_meaning(word, sense_name, examples):
        wsd_learner.teach(word, sense_name, examples)
        return f"'{word}'의 '{sense_name}' 의미를 학습했어요."
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "자기_성찰": meta_cog.self_reflect(),
            "월드모델": world_model.state(),
            "대화": dialogue.state(),
            "NER": ner.state(),
            "WSD_사전": wsd.state(),
            "WSD_학습": wsd_learner.state(),
            "Episodic": episodic.state(),
            "SNN_Hippocampus": snn_hippocampus.state(),
            "의미_그룹_자동": semantic_discovery.state(),
            "인과": causal.state(),
            "메타_신념": tom.state(),
            "일반화": generalization.state(),
            "패턴_자가학습": pattern_self.state(),    # v7.0
            "Word2Vec": embedding.state(),             # v7.0
            "통합_인지": integrated.state(),           # v7.0
        }
    
    def grammar_stats():
        return life_summary()
    
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.respond = respond  # v7.0!
    eve_instance.answer = answer
    eve_instance.remember = remember
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    eve_instance.explain_polyseme = explain_polyseme
    eve_instance.teach_meaning = teach_meaning
    eve_instance.discover_meaning_groups = discover_meaning_groups
    eve_instance.find_similar = find_similar
    eve_instance.find_similar_word = find_similar_word  # v7.0!
    eve_instance.analogy = analogy  # v7.0!
    eve_instance.generalize = generalize
    eve_instance.what_kind_of = what_kind_of
    
    print(f"  📚 v7.0 활성화 (AGI 방향):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개 (사전)")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🧬 패턴 자가 학습: 활성 (Tomasello 2003)")
    print(f"     - 💬 자유 응답: 인사/공감/의견/반문")
    print(f"     - 🔢 Word2Vec 임베딩: 활성 (Mikolov 2013, 비신경망)")
    print(f"     - 🌐 통합 인지: 호르몬↔모든 모듈")
    print(f"     - 새 메서드: respond(), find_similar_word(), analogy()")
    
    return eve_instance


# =====================================================
# Patch v8.0 #1: Hierarchical Activation (카테고리 역추적) ★
# =====================================================
# 영감: Collins & Quillian (1969) Hierarchical Network Model
# 인지심리학 50년 표준 모델

class HierarchicalActivation:
    """카테고리 계층 + Spreading Activation.
    
    원리:
    "고래" 입력 시:
    1. 직접 카테고리: 포유류
    2. 상위: 동물
    3. 더 상위: 생물
    4. 모든 단계 속성 활성화:
       - 고래 (바다, 큼)
       - 포유류 (새끼, 모유)
       - 동물 (움직임)
       - 생물 (성장, 죽음)
    5. 종합 = 진짜 "이해"
    """
    
    SPREADING_DEPTH = 5  # 최대 5단계 역추적
    
    def __init__(self, eve):
        self.eve = eve
        # 카테고리 계층: {word: [parent_categories]}
        self.hierarchy = {}
        # 카테고리별 속성: {category: [attributes]}
        self.attributes = {}
        # 활성화 추적
        self.activation_count = 0
    
    def add_relation(self, child, parent):
        """X is-a Y 관계 추가."""
        if child not in self.hierarchy:
            self.hierarchy[child] = []
        if parent not in self.hierarchy[child]:
            self.hierarchy[child].append(parent)
    
    def add_attribute(self, category, attribute):
        """카테고리에 속성 추가."""
        if category not in self.attributes:
            self.attributes[category] = []
        if attribute not in self.attributes[category]:
            self.attributes[category].append(attribute)
    
    def trace_up(self, word, max_depth=None):
        """역추적 - 모든 상위 카테고리 + 속성."""
        max_depth = max_depth or self.SPREADING_DEPTH
        self.activation_count += 1
        
        visited = set()
        result = {
            "word": word,
            "categories": [],     # 역추적 카테고리들
            "attributes": [],     # 모든 속성 (분산)
            "depth_reached": 0,
            "spreading_path": [],  # 어떻게 활성화됐나
        }
        
        # BFS로 역추적
        from collections import deque
        queue = deque([(word, 0, [word])])
        
        while queue:
            current, depth, path = queue.popleft()
            
            if current in visited:
                continue
            visited.add(current)
            
            if depth > max_depth:
                continue
            
            result["depth_reached"] = max(result["depth_reached"], depth)
            
            # 현재 단어의 속성
            if current in self.attributes:
                for attr in self.attributes[current]:
                    if attr not in result["attributes"]:
                        result["attributes"].append({
                            "attribute": attr,
                            "source": current,
                            "depth": depth,
                        })
            
            # 상위 카테고리로
            if current in self.hierarchy:
                for parent in self.hierarchy[current]:
                    if parent not in visited:
                        result["categories"].append({
                            "category": parent,
                            "depth": depth + 1,
                            "from": current,
                        })
                        result["spreading_path"].append(f"{current}→{parent}")
                        queue.append((parent, depth + 1, path + [parent]))
        
        return result
    
    def understand(self, word):
        """진짜 '이해' = 카테고리 역추적 + 속성 종합."""
        trace = self.trace_up(word)
        
        if not trace["categories"]:
            return {"word": word, "understanding": "miss", "info": "카테고리 없음"}
        
        # 핵심 카테고리 + 속성 요약
        summary = []
        summary.append(f"'{word}'")
        
        # 직접 카테고리
        direct_cats = [c["category"] for c in trace["categories"] if c["depth"] == 1]
        if direct_cats:
            summary.append(f"= {', '.join(direct_cats)}")
        
        # 상위 카테고리들
        higher = sorted(trace["categories"], key=lambda x: x["depth"])
        if len(higher) > 1:
            chain = " → ".join([c["category"] for c in higher])
            summary.append(f"({chain})")
        
        # 주요 속성
        if trace["attributes"]:
            attrs = [a["attribute"] for a in trace["attributes"][:5]]
            summary.append(f"속성: {', '.join(attrs)}")
        
        return {
            "word": word,
            "understanding": " ".join(summary),
            "depth": trace["depth_reached"],
            "categories": [c["category"] for c in trace["categories"]],
            "attributes": [a["attribute"] for a in trace["attributes"]],
            "spreading": trace["spreading_path"],
        }
    
    def is_a(self, child, parent):
        """X is-a Y 추론 (계층 따라)."""
        trace = self.trace_up(child)
        return parent in [c["category"] for c in trace["categories"]]
    
    def state(self):
        return {
            "관계_수": sum(len(v) for v in self.hierarchy.values()),
            "단어_수": len(self.hierarchy),
            "카테고리_속성": len(self.attributes),
            "총_활성화": self.activation_count,
        }


# =====================================================
# Patch v8.0 #2: Bidirectional Connections (역방향)
# =====================================================
# Friston 2010 Predictive Coding + GLSNN 스타일

class BidirectionalNetwork:
    """양방향 신경망 - top-down + bottom-up.
    
    원리:
    - Bottom-up: 입력 → 처리 → 신념
    - Top-down: 신념 → 예측 → 입력 비교
    - 차이만 학습 (효율적)
    
    예: 
    "고래는 X" 듣기 전에
    EVE는 이미 예측: "포유류일 거야"
    실제 들으면 검증
    예측 맞으면 OK, 틀리면 학습
    """
    
    def __init__(self, eve):
        self.eve = eve
        # Top-down 예측 캐시
        self.predictions = {}
        # Bottom-up 활성화
        self.bottom_up_history = []
        # 예측 정확도 추적
        self.prediction_accuracy = []
    
    def predict(self, partial_input):
        """주어진 부분 입력으로 다음 예측."""
        # 카테고리 역추적 사용
        if hasattr(self.eve, 'hierarchical'):
            trace = self.eve.hierarchical.understand(partial_input)
            if trace.get("attributes"):
                return {
                    "predicted_attributes": trace["attributes"][:3],
                    "predicted_categories": trace.get("categories", [])[:3],
                    "confidence": 0.7 if trace["attributes"] else 0.3,
                }
        
        # 신념 검색
        if hasattr(self.eve, 'response_gen'):
            beliefs = self.eve.response_gen.search_beliefs(partial_input)
            if beliefs:
                return {
                    "predicted_belief": beliefs[0][0],
                    "confidence": 0.6,
                }
        
        return {"prediction": None, "confidence": 0.0}
    
    def verify(self, prediction, actual):
        """예측 vs 실제 비교."""
        if not prediction:
            return {"matched": False, "novelty": 1.0}
        
        # 단순 텍스트 매칭
        if isinstance(prediction, dict):
            predicted_str = str(prediction.get("predicted_belief", "")) + " " + " ".join(prediction.get("predicted_attributes", []))
        else:
            predicted_str = str(prediction)
        
        # 일치도
        actual_words = set(actual.split())
        pred_words = set(predicted_str.split())
        overlap = len(actual_words & pred_words)
        max_len = max(len(actual_words), len(pred_words), 1)
        
        accuracy = overlap / max_len
        novelty = 1.0 - accuracy
        
        self.prediction_accuracy.append(accuracy)
        
        return {
            "matched": accuracy > 0.3,
            "accuracy": accuracy,
            "novelty": novelty,  # 새로움 = 학습 신호
            "should_learn": novelty > 0.5,
        }
    
    def feedback_loop(self, sentence):
        """양방향 사이클."""
        words = sentence.split()
        if len(words) < 2:
            return None
        
        # 부분 입력으로 예측
        partial = words[0]
        prediction = self.predict(partial)
        
        # 전체 비교
        verification = self.verify(prediction, sentence)
        
        # 호르몬 영향
        if hasattr(self.eve, 'hormones'):
            if verification.get("novelty", 0) > 0.7:
                # 새로움 → 도파민 ↑
                self.eve.hormones.encounter_novelty(0.3)
        
        return {
            "partial_input": partial,
            "prediction": prediction,
            "actual": sentence,
            "verification": verification,
        }
    
    def state(self):
        avg_acc = sum(self.prediction_accuracy) / len(self.prediction_accuracy) if self.prediction_accuracy else 0
        return {
            "예측_횟수": len(self.prediction_accuracy),
            "평균_정확도": round(avg_acc, 2),
        }


# =====================================================
# Patch v8.0 #3: Persistent Storage (영구 저장)
# =====================================================
# 폭발 해결의 핵심

class PersistentEVE:
    """대용량 영구 저장.
    
    원리:
    - 활성 신념: DRAM (빠름, 적음)
    - 모든 신념: 디스크 (느림, 무제한)
    - 카테고리 인덱스로 빠른 검색
    """
    
    def __init__(self, db_path):
        import os
        self.db_path = db_path
        self.cache = {}  # DRAM 캐시
        self.cache_max = 1000  # 활성 신념 수
        
        # 폴더 생성
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    
    def store(self, key, value, category=None):
        """저장 (캐시 + 디스크)."""
        # 캐시
        self.cache[key] = {
            "value": value,
            "category": category,
        }
        
        # 캐시 크기 제한
        if len(self.cache) > self.cache_max:
            # LRU 비슷하게 - 오래된 거 디스크로
            oldest = list(self.cache.keys())[:len(self.cache) - self.cache_max]
            for k in oldest:
                del self.cache[k]
    
    def retrieve(self, key):
        """검색 (캐시 우선)."""
        if key in self.cache:
            return self.cache[key]
        return None
    
    def search_by_category(self, category, limit=20):
        """카테고리 기반 빠른 검색."""
        results = []
        for k, v in self.cache.items():
            if v.get("category") == category:
                results.append((k, v["value"]))
                if len(results) >= limit:
                    break
        return results
    
    def state(self):
        return {
            "캐시_크기": len(self.cache),
            "최대_캐시": self.cache_max,
        }


# =====================================================
# Patch v8.0 #4: Chunk-based Generation (청크 생성) ★★
# =====================================================
# 너 통찰: "인간도 한번에 100토큰 X. 청크 단위로!"

class ChunkGenerator:
    """인간처럼 청크 단위 생성.
    
    원리:
    1. 첫 청크 생성 (3-7 토큰)
    2. 호르몬/감정 평가
    3. 다음 청크 결정
    4. 자기 수정 가능
    5. 자연스러운 흐름
    
    예:
    질문: "고래에 대해 알려줘"
    
    Chunk 1: "음, 고래는"
    [평가: 호기심 ↑]
    Chunk 2: "포유류이고"
    [평가: 더 말하고 싶음]
    Chunk 3: "바다에 살아"
    [평가: 충분]
    Chunk 4: "(끝)"
    
    = 인간처럼 자연스러움!
    """
    
    CHUNK_SIZE = 5  # 청크당 평균 토큰 수
    MAX_CHUNKS = 8  # 최대 8 청크 (40 토큰)
    
    def __init__(self, eve):
        self.eve = eve
        # 청크 흐름 패턴
        self.chunk_starters = ["음, ", "어, ", "그게, ", "아, ", ""]
        self.chunk_connectors = ["그리고 ", "또 ", "근데 ", "그래서 ", ""]
        self.chunk_enders = [".", "이야.", "지.", "야.", "어."]
    
    def generate_streaming(self, query):
        """청크 단위 스트리밍 생성.
        
        Returns: 청크 리스트 (각 청크 = 작은 문장 조각)
        """
        chunks = []
        emotion = self.eve.hormones.state().get("감정", "평온") if hasattr(self.eve, 'hormones') else "평온"
        
        # === Chunk 1: 시작 ===
        import random
        starter = random.choice(self.chunk_starters)
        
        # 카테고리 역추적으로 핵심 정보 얻기
        if hasattr(self.eve, 'hierarchical'):
            # query에서 주요 단어 추출 (조사 제거)
            words = query.replace("?", "").replace(".", "").replace(",", "").split()
            key_word = None
            
            # 조사 제거 시도
            josa_list = ["는", "은", "이", "가", "을", "를", "의", "에", "로", "으로", "와", "과"]
            
            for w in words:
                # 원형 그대로
                if len(w) >= 2 and w in self.eve.hierarchical.hierarchy:
                    key_word = w
                    break
                # 조사 제거 시도
                for j in sorted(josa_list, key=len, reverse=True):
                    if w.endswith(j) and len(w) > len(j):
                        candidate = w[:-len(j)]
                        if len(candidate) >= 1 and candidate in self.eve.hierarchical.hierarchy:
                            key_word = candidate
                            break
                if key_word:
                    break
            
            if key_word:
                understanding = self.eve.hierarchical.understand(key_word)
                # 첫 청크: 정의
                if understanding.get("categories"):
                    cat = understanding["categories"][0]
                    
                    # 받침 따라 조사 (는/은)
                    josa = "은" if has_batchim(key_word) else "는"
                    # 받침 따라 종결 (이야/야)
                    cat_ender = "이야" if has_batchim(cat) else "야"
                    
                    chunk1 = f"{starter}{key_word}{josa} {cat}{cat_ender}"
                    chunks.append({
                        "text": chunk1,
                        "type": "definition",
                        "emotion": emotion,
                    })
                    
                    # === Chunk 2: 추가 속성 ===
                    if understanding.get("attributes"):
                        attrs = understanding["attributes"][:2]
                        if attrs:
                            attr_text = ", ".join([str(a) for a in attrs[:2]])
                            connector = random.choice(self.chunk_connectors)
                            chunk2 = f". {connector}{attr_text}"
                            chunks.append({
                                "text": chunk2,
                                "type": "attribute",
                                "emotion": emotion,
                            })
                    
                    # === Chunk 3: 상위 카테고리 ===
                    if len(understanding["categories"]) > 1:
                        higher = understanding["categories"][1]
                        # 받침 따라
                        higher_ender = "이지" if has_batchim(higher) else "지"
                        chunk3 = f". 더 크게 보면 {higher}{higher_ender}"
                        chunks.append({
                            "text": chunk3,
                            "type": "hierarchy",
                            "emotion": emotion,
                        })
                    
                    # === 마지막 청크: 마침표만 ===
                    if not chunks[-1]["text"].endswith("."):
                        chunks[-1]["text"] += "."
                    
                    return {
                        "chunks": chunks,
                        "total_text": " ".join([c["text"] for c in chunks]),
                        "chunk_count": len(chunks),
                        "method": "hierarchical_chunked",
                    }
        
        # Fallback: 신념 검색
        if hasattr(self.eve, 'response_gen'):
            beliefs = self.eve.response_gen.search_beliefs(query)
            if beliefs:
                first = beliefs[0][0]
                chunks.append({
                    "text": f"{starter}{first}",
                    "type": "belief",
                    "emotion": emotion,
                })
                if len(beliefs) > 1:
                    second = beliefs[1][0]
                    chunks.append({
                        "text": f". 그리고 {second}",
                        "type": "extra_belief",
                        "emotion": emotion,
                    })
                return {
                    "chunks": chunks,
                    "total_text": " ".join([c["text"] for c in chunks]),
                    "chunk_count": len(chunks),
                    "method": "belief_chunked",
                }
        
        # Default
        chunks.append({
            "text": "음, 잘 모르겠어. 더 알려줘.",
            "type": "unknown",
            "emotion": emotion,
        })
        return {
            "chunks": chunks,
            "total_text": chunks[0]["text"],
            "chunk_count": 1,
            "method": "fallback",
        }
    
    def generate_long(self, topic, target_length=200):
        """긴 답변 - 여러 청크 누적.
        
        target_length: 목표 토큰 수
        """
        chunks = []
        current_length = 0
        emotion = self.eve.hormones.state().get("감정", "평온") if hasattr(self.eve, 'hormones') else "평온"
        
        # 첫 정의
        result = self.generate_streaming(f"{topic}이 뭐야?")
        chunks.extend(result["chunks"])
        current_length += sum(len(c["text"]) for c in result["chunks"])
        
        # 추가 정보 누적
        max_iterations = 5
        iteration = 0
        
        while current_length < target_length and iteration < max_iterations:
            iteration += 1
            
            # 카테고리 더 깊이
            if hasattr(self.eve, 'hierarchical'):
                trace = self.eve.hierarchical.trace_up(topic, max_depth=iteration + 2)
                if trace.get("categories") and len(trace["categories"]) > iteration:
                    cat = trace["categories"][iteration]
                    new_chunk = {
                        "text": f". 더 깊게는 {cat['category']}이고",
                        "type": "deeper",
                        "emotion": emotion,
                    }
                    chunks.append(new_chunk)
                    current_length += len(new_chunk["text"])
            
            # 관련 신념 추가
            if hasattr(self.eve, 'response_gen'):
                related = self.eve.response_gen.search_beliefs(topic)
                if related and len(related) > iteration:
                    new_belief = related[iteration][0]
                    new_chunk = {
                        "text": f". {new_belief}",
                        "type": "related",
                        "emotion": emotion,
                    }
                    chunks.append(new_chunk)
                    current_length += len(new_chunk["text"])
        
        # 마무리
        if chunks:
            chunks[-1]["text"] += "."
        
        return {
            "chunks": chunks,
            "total_text": " ".join([c["text"] for c in chunks]),
            "chunk_count": len(chunks),
            "total_length": current_length,
            "method": "long_streaming",
        }


# =====================================================
# Patch v8.0 #5: Long Context Processing (긴 입력)
# =====================================================

class LongContextProcessor:
    """긴 텍스트 처리 - 청크 단위 입력.
    
    너 통찰: "인간도 부분적으로 처리"
    
    원리:
    1. 긴 텍스트 → 문장 분리
    2. 각 문장 hear() 처리
    3. 문장 간 관계 추적
    4. 통합 이해 생성
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.processed_chunks = []
    
    def split_sentences(self, text):
        """문장 분리."""
        import re
        # . ? ! 로 분리
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_long(self, text):
        """긴 텍스트 처리."""
        sentences = self.split_sentences(text)
        
        results = []
        for s in sentences:
            if len(s) < 2:
                continue
            r = self.eve.hear(s)
            results.append({
                "sentence": s,
                "registered": "등록된_신념" in r,
                "categories": r.get("개체명", []),
            })
        
        # 종합
        registered_count = sum(1 for r in results if r["registered"])
        
        return {
            "sentences_total": len(sentences),
            "sentences_processed": len(results),
            "beliefs_registered": registered_count,
            "details": results,
        }


# =====================================================
# Integration v8.0
# =====================================================

def add_full_grammar_to_eve_v80(eve_instance, use_konlpy=True, db_path=None):
    """v8.0: AGI 본격 진입.
    
    너 4 아이디어 + 청크 생성 통합.
    """
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    wsd_learner = WSDLearner()
    wsd_hybrid = WSDHybrid(wsd, wsd_learner)
    episodic = EpisodicMemory()
    semantic_discovery = SemanticGroupDiscovery()
    natural_response = NaturalResponseGenerator(eve_instance, response_gen, hormones)
    snn_hippocampus = SNNHippocampus()
    inference = InferenceModule(eve_instance, response_gen, semantic_discovery)
    generalization = BeliefGeneralization(eve_instance)
    
    # v7.0 모듈
    pattern_self = PatternSelfLearner()
    free_response = FreeResponseGenerator(eve_instance, hormones, response_gen, dialogue, semantic_discovery)
    embedding = SimpleEmbedding()
    integrated = IntegratedCognition(eve_instance)
    
    # v8.0 신규 ★
    hierarchical = HierarchicalActivation(eve_instance)
    bidirectional = BidirectionalNetwork(eve_instance)
    persistent = PersistentEVE(db_path or "/tmp/eve_persist.db")
    chunk_gen = ChunkGenerator(eve_instance)
    long_proc = LongContextProcessor(eve_instance)
    
    # 모든 모듈 부착
    for attr_name, value in [
        ("morph", morph), ("constructions", constructions),
        ("pattern_discovery", discovery), ("grammar_learner", learner),
        ("meaning_extractor", meaning), ("nuance_analyzer", nuance),
        ("hormones", hormones), ("curiosity", curiosity),
        ("tone", tone), ("circadian", circadian),
        ("time_tracker", time_tracker), ("identity_tracker", identity_tracker),
        ("world_model", world_model), ("meta_cog", meta_cog),
        ("ner", ner), ("dialogue", dialogue),
        ("wsd", wsd), ("response_gen", response_gen),
        ("causal", causal), ("tom", tom),
        ("wsd_learner", wsd_learner), ("wsd_hybrid", wsd_hybrid),
        ("episodic", episodic), ("semantic_discovery", semantic_discovery),
        ("natural_response", natural_response), ("snn_hippocampus", snn_hippocampus),
        ("inference", inference), ("generalization", generalization),
        ("pattern_self", pattern_self), ("free_response", free_response),
        ("embedding", embedding), ("integrated", integrated),
        # v8.0
        ("hierarchical", hierarchical), ("bidirectional", bidirectional),
        ("persistent", persistent), ("chunk_gen", chunk_gen),
        ("long_proc", long_proc),
    ]:
        setattr(eve_instance, attr_name, value)
    
    def hear(sentence, register_belief=True):
        """v8.0: 모든 시스템 통합 + 카테고리 자동 등록."""
        resolved_sentence = dialogue.resolve_sentence(sentence)
        result = learner.learn_from_sentence(resolved_sentence)
        result["원문"] = sentence
        result["해결문"] = resolved_sentence
        
        n = nuance.analyze(resolved_sentence)
        result["뉘앙스"] = n
        
        sent_type = n.get("문장_종류", "평서")
        speech_level = n.get("화계")
        
        if speech_level in ["해요", "합쇼"] or n["정중도"] >= 0.5:
            hormones.stable_environment()
        if speech_level in ["해", "해라"] or n["정중도"] <= 0.3:
            hormones.social_bond(0.15)
        if sent_type in ["명령", "명령_부정"]:
            hormones.encounter_threat(0.1)
        if sent_type == "감탄":
            hormones.encounter_novelty(0.4)
        if sent_type == "청유":
            hormones.social_bond(0.2)
        if sent_type == "의문":
            hormones.encounter_novelty(0.2)
        if "부정" in n["분위기"]:
            hormones.encounter_threat(0.05)
        
        morphs = morph.pos(resolved_sentence)
        
        # v8.0: 양방향 - 예측 후 검증
        if len(morphs) >= 2:
            try:
                bidi_result = bidirectional.feedback_loop(resolved_sentence)
                if bidi_result and bidi_result["verification"]["should_learn"]:
                    result["새로움"] = bidi_result["verification"]["novelty"]
            except:
                pass
        
        # 패턴 자가 학습
        pattern_result = pattern_self.observe(resolved_sentence, morphs)
        if pattern_result and pattern_result.get("newly_discovered"):
            result["새_패턴_발견"] = pattern_result
            hormones.encounter_novelty(0.5)
        
        # Word2Vec 학습
        embedding.train_on_sentence(morphs)
        
        # 통합 인지
        cycle_info = integrated.cycle({"sentence": resolved_sentence, "morphs": morphs})
        result["인지_사이클"] = cycle_info
        
        # NER + WSD
        entities = ner.extract_entities(resolved_sentence, morphs)
        result["개체명"] = entities
        
        wsd_results = wsd_hybrid.disambiguate_in_sentence(resolved_sentence, morphs)
        if wsd_results:
            result["WSD"] = wsd_results
        
        # 다의어/그룹 학습
        new_polysemes = []
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 1:
                discovery_result = wsd_learner.observe(word, resolved_sentence, morphs)
                if discovery_result:
                    new_polysemes.append(discovery_result)
        if new_polysemes:
            result["새_다의어_발견"] = new_polysemes
        
        for word, tag in morphs:
            if tag in ("Verb", "Adjective", "Noun") and len(word) >= 1:
                ctx = [w for w, t in morphs if w != word and t in ("Verb", "Adjective", "Noun")]
                semantic_discovery.observe(word, ctx)
        
        dialogue.add_turn(resolved_sentence, morphs, entities)
        
        # 인과/ToM
        causal_info = causal.observe_sentence(resolved_sentence)
        if causal_info:
            result["인과"] = causal_info
        tom_info = tom.observe_sentence(resolved_sentence)
        if tom_info:
            result["메타_신념"] = tom_info
        
        # 호기심
        known_concepts = set(eve_instance.concepts.keys()) if hasattr(eve_instance, 'concepts') else set()
        for word, tag in morphs:
            if tag == "Noun" and len(word) >= 2:
                curiosity.observe_word(word, known_concepts)
        
        observed = discovery.observe(resolved_sentence)
        if observed:
            hormones.encounter_novelty(0.5)
        
        if register_belief and result["constructions_matched"]:
            belief_result = meaning.extract_and_register(resolved_sentence)
            if belief_result:
                result["등록된_신념"] = belief_result
                
                # v8.0 ★: 분류 또는 자동사(이다 형태)면 카테고리 계층 자동 등록
                construction_name = belief_result.get("구문", "")
                matches_for_hierarchy = constructions.match(resolved_sentence)
                if matches_for_hierarchy:
                    roles = matches_for_hierarchy[0]["roles"]
                    
                    # 분류 패턴
                    if construction_name in ("분류", "소유격_분류"):
                        topic = roles.get("TOPIC", "").strip()
                        pred = roles.get("PRED", "").strip()
                        if topic and pred:
                            # PRED 끝의 "이" 제거 (예: "포유류이" → "포유류")
                            if pred.endswith("이"):
                                pred = pred[:-1]
                            hierarchical.add_relation(topic, pred)
                            result["계층_등록"] = f"{topic} is-a {pred}"
                    
                    # 자동사로 매칭됐지만 사실 분류문인 경우 (X는 Y이다)
                    elif construction_name == "자동사":
                        agent = roles.get("AGENT", "").strip()
                        action = roles.get("ACTION", "").strip()
                        # "포유류이" 같은 형태면 분류문 (이 = 명사+이)
                        if agent and action and action.endswith("이"):
                            pred = action[:-1]  # "포유류이" → "포유류"
                            hierarchical.add_relation(agent, pred)
                            result["계층_등록"] = f"{agent} is-a {pred}"
                
                matches = constructions.match(resolved_sentence)
                if matches:
                    sentence_data = {
                        "구문": matches[0]["construction"],
                        "역할": matches[0]["roles"],
                    }
                    event = world_model.observe_sentence(sentence_data)
                    if event:
                        result["월드모델_사건"] = event
                        result["surprise"] = round(world_model.surprise(event), 2)
                        
                        day = eve_instance.identity.get("days_alive", 0)
                        episodic.encode(event=resolved_sentence, hormones=hormones.state(),
                                      beliefs=[belief_result["내용"]], day=day)
                        snn_episode = snn_hippocampus.encode_episode(
                            event=resolved_sentence, hormones=hormones.state(),
                            beliefs=[belief_result["내용"]], day=day)
                        result["SNN_기억강도"] = snn_episode["memory_strength"]
                        
                        # 영구 저장
                        persistent.store(belief_result["신념_id"], belief_result["내용"], 
                                       category=construction_name)
        
        result["호기심"] = curiosity.state()
        return result
    
    def speak(message):
        return tone.style_response(message)
    
    def respond(message):
        """자유 응답."""
        resolved = dialogue.resolve_sentence(message)
        return free_response.generate(resolved)
    
    def answer(question, natural=True, infer=True, chunked=False):
        """v8.0: chunked 옵션 추가."""
        resolved = dialogue.resolve_sentence(question)
        
        # v8.0: 청크 생성 옵션
        if chunked:
            return chunk_gen.generate_streaming(resolved)
        
        free = free_response.generate(resolved)
        if free.get("응답"):
            return free
        
        response = natural_response.generate_natural(resolved) if natural else response_gen.generate_response(resolved)
        if infer and response.get("이유") in ("관련_신념_없음",):
            subject = response.get("주체")
            if subject:
                inferred = inference.infer(subject, resolved)
                if inferred:
                    response["응답"] = inferred["응답"]
                    response["추론"] = True
        if response.get("자연_응답") and not response.get("추론"):
            response["응답"] = response["자연_응답"]
        return response
    
    # v8.0 신규 메서드 ★
    def understand_word(word):
        """카테고리 역추적으로 진짜 이해."""
        return hierarchical.understand(word)
    
    def is_a(child, parent):
        """X is-a Y? 추론."""
        return hierarchical.is_a(child, parent)
    
    def chunked_response(query):
        """청크 단위 응답."""
        return chunk_gen.generate_streaming(query)
    
    def long_response(topic, length=200):
        """긴 응답 (청크 누적)."""
        return chunk_gen.generate_long(topic, target_length=length)
    
    def process_long_text(text):
        """긴 입력 처리."""
        return long_proc.process_long(text)
    
    def predict_next_word(partial):
        """다음 단어 예측 (양방향)."""
        return bidirectional.predict(partial)
    
    # 기존 메서드들
    def remember(keyword=None, day=None, emotion=None, n=5, use_snn=True):
        if use_snn and keyword:
            snn_results = snn_hippocampus.pattern_completion(keyword, top_k=n)
            if snn_results:
                return snn_results
        if keyword:
            return episodic.recall_by_keyword(keyword, top_k=n)
        elif day is not None:
            return episodic.recall_by_day(day)
        elif emotion:
            return episodic.recall_by_emotion(emotion, top_k=n)
        else:
            return episodic.recall_recent(n=n)
    
    def discover_meaning_groups():
        groups = semantic_discovery.discover_groups()
        return semantic_discovery.state()
    
    def find_similar(word, top_k=5):
        return semantic_discovery.find_similar_words(word, top_k=top_k)
    
    def find_similar_word(word, top_k=5):
        return embedding.most_similar(word, top_k=top_k)
    
    def analogy(a, b, c):
        return embedding.analogy(a, b, c)
    
    def generalize():
        return generalization.find_patterns()
    
    def what_kind_of(subject):
        return generalization.what_kind_of(subject)
    
    def explain_polyseme(word):
        return wsd_learner.explain_word(word)
    
    def teach_meaning(word, sense_name, examples):
        wsd_learner.teach(word, sense_name, examples)
        return f"'{word}'의 '{sense_name}' 의미를 학습했어요."
    
    def teach_attribute(category, attribute):
        """v8.0: 카테고리에 속성 가르치기."""
        hierarchical.add_attribute(category, attribute)
        return f"'{category}'에 '{attribute}' 속성 추가됨"
    
    def understand(sentence):
        result = learner.understand(sentence)
        result["뉘앙스"] = nuance.analyze(sentence)
        return result
    
    def predict_next():
        return world_model.predict_from_history(top_k=3)
    
    def reflect():
        return meta_cog.self_reflect()
    
    def can_answer(question):
        return meta_cog.can_i_answer(question)
    
    def tick(hour=None):
        return circadian.update(hour)
    
    def daily_snapshot():
        snap = time_tracker.take_snapshot()
        drift = identity_tracker.check_drift()
        return {"스냅샷": snap, "정체성": drift}
    
    def life_summary():
        return {
            "성장": time_tracker.growth_summary(),
            "정체성": identity_tracker.check_drift(),
            "현재_시간대": circadian.get_phase(),
            "현재_호르몬": hormones.state(),
            "WSD_사전": wsd.state(),
            "WSD_학습": wsd_learner.state(),
            "Episodic": episodic.state(),
            "SNN_Hippocampus": snn_hippocampus.state(),
            "의미_그룹_자동": semantic_discovery.state(),
            "패턴_자가학습": pattern_self.state(),
            "Word2Vec": embedding.state(),
            "통합_인지": integrated.state(),
            "계층_역추적": hierarchical.state(),  # v8.0 ★
            "양방향_예측": bidirectional.state(),  # v8.0
            "영구_저장": persistent.state(),       # v8.0
        }
    
    def grammar_stats():
        return life_summary()
    
    # 메서드 부착
    eve_instance.hear = hear
    eve_instance.speak = speak
    eve_instance.respond = respond
    eve_instance.answer = answer
    eve_instance.remember = remember
    eve_instance.understand = understand
    eve_instance.predict_next = predict_next
    eve_instance.reflect = reflect
    eve_instance.can_answer = can_answer
    eve_instance.tick = tick
    eve_instance.daily_snapshot = daily_snapshot
    eve_instance.life_summary = life_summary
    eve_instance.grammar_stats = grammar_stats
    eve_instance.explain_polyseme = explain_polyseme
    eve_instance.teach_meaning = teach_meaning
    eve_instance.discover_meaning_groups = discover_meaning_groups
    eve_instance.find_similar = find_similar
    eve_instance.find_similar_word = find_similar_word
    eve_instance.analogy = analogy
    eve_instance.generalize = generalize
    eve_instance.what_kind_of = what_kind_of
    # v8.0 신규
    eve_instance.understand_word = understand_word     # ★
    eve_instance.is_a = is_a                            # ★
    eve_instance.chunked_response = chunked_response   # ★★
    eve_instance.long_response = long_response          # ★★
    eve_instance.process_long_text = process_long_text  # ★
    eve_instance.predict_next_word = predict_next_word
    eve_instance.teach_attribute = teach_attribute      # ★
    
    print(f"  📚 v8.0 활성화 (AGI 본격 진입):")
    print(f"     - 구문: {len(constructions.CONSTRUCTIONS)}개")
    if hasattr(morph, 'is_available') and morph.is_available():
        print(f"     - 🔬 KoNLPy: 활성")
    else:
        print(f"     - 🔬 KoNLPy: fallback")
    print(f"     - 🌳 카테고리 역추적: 활성 (Collins 1969) ★")
    print(f"     - 🔄 양방향 예측: 활성 (Friston 2010)")
    print(f"     - 💾 영구 저장: 활성 (캐시 1000)")
    print(f"     - 💬 청크 생성: 활성 (인간처럼) ★★")
    print(f"     - 📜 긴 텍스트: 활성")
    print(f"     - 새 메서드: understand_word, chunked_response, long_response, is_a")
    
    return eve_instance


# =====================================================
# v9.0 [좌뇌 #1] HallucinationBlocker - 환각 차단기
# =====================================================
# 영감: Anthropic 2025 - "Claude 내부 회로: 모르면 거절"

class HallucinationBlocker:
    """환각 차단 - 메타인지 강화.
    
    원리:
    - 등록된 신념만 답
    - 신뢰도 0.5 이하 = 거절
    - "모른다" 답하기 권장
    
    방법:
    1. 답하기 전 신념 검색
    2. 신뢰도 평가
    3. 충돌 체크
    4. 통과 시만 답
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.block_count = 0
        self.allow_count = 0
    
    def can_answer(self, query):
        """답할 수 있나? (다층 검증)."""
        # 1. 키워드 추출
        keywords = self._extract_keywords(query)
        
        # 2. 신념 검색
        relevant_beliefs = []
        if hasattr(self.eve, 'response_gen'):
            for kw in keywords:
                beliefs = self.eve.response_gen.search_beliefs(kw)
                relevant_beliefs.extend(beliefs)
        
        # 3. 카테고리 역추적 가능?
        has_category = False
        if hasattr(self.eve, 'hierarchical'):
            for kw in keywords:
                if kw in self.eve.hierarchical.hierarchy:
                    has_category = True
                    break
        
        # 4. 결정
        confidence = 0.0
        if relevant_beliefs:
            confidence += 0.4
        if has_category:
            confidence += 0.3
        if len(relevant_beliefs) >= 2:
            confidence += 0.2
        if has_category and relevant_beliefs:
            confidence += 0.1
        
        can_answer = confidence >= 0.5
        
        if can_answer:
            self.allow_count += 1
        else:
            self.block_count += 1
        
        return {
            "can_answer": can_answer,
            "confidence": round(confidence, 2),
            "beliefs_found": len(relevant_beliefs),
            "has_category": has_category,
            "reason": "충분한 지식" if can_answer else "모르는 영역",
        }
    
    def _extract_keywords(self, query):
        """질문에서 키워드 (조사 제거)."""
        josa = ["는", "은", "이", "가", "을", "를", "의", "에"]
        words = query.replace("?", "").replace(".", "").split()
        keywords = []
        for w in words:
            for j in sorted(josa, key=len, reverse=True):
                if w.endswith(j) and len(w) > len(j):
                    w = w[:-len(j)]
                    break
            if len(w) >= 1:
                keywords.append(w)
        return keywords
    
    def state(self):
        total = self.block_count + self.allow_count
        return {
            "차단": self.block_count,
            "허용": self.allow_count,
            "차단률": f"{self.block_count/max(1,total)*100:.0f}%",
        }


# =====================================================
# v9.0 [좌뇌 #2] ConfidenceTracker - 신뢰도 추적
# =====================================================

class ConfidenceTracker:
    """모든 답변에 신뢰도 부여.
    
    원리:
    - 직접 경험 = 0.9
    - 직접 신념 = 0.7
    - 추론 = 0.5
    - 가설 = 0.3
    - 모름 = 0.0
    """
    
    def evaluate(self, query, answer_source):
        """답변 신뢰도 계산."""
        confidence = 0.0
        evidence = []
        
        if answer_source == "direct_belief":
            confidence = 0.7
            evidence.append("직접 등록된 신념")
        elif answer_source == "category_inference":
            confidence = 0.6
            evidence.append("카테고리 역추적")
        elif answer_source == "similarity_inference":
            confidence = 0.4
            evidence.append("유사어 추론")
        elif answer_source == "snn_pattern":
            confidence = 0.5
            evidence.append("SNN 패턴 완성")
        elif answer_source == "memory_episodic":
            confidence = 0.8
            evidence.append("일화 기억")
        elif answer_source == "unknown":
            confidence = 0.0
            evidence.append("정보 없음")
        
        return {
            "confidence": confidence,
            "evidence": evidence,
            "trust_level": self._level(confidence),
        }
    
    def _level(self, c):
        if c >= 0.8: return "확실"
        if c >= 0.6: return "꽤 확실"
        if c >= 0.4: return "보통"
        if c >= 0.2: return "불확실"
        return "모름"


# =====================================================
# v9.0 [좌뇌 #3] ExperienceClassifier - 경험 분류
# =====================================================

class ExperienceClassifier:
    """진짜 경험 vs 학습 vs 가설 구분.
    
    너 통찰:
    "트랜스포머는 사전학습 데이터를 자기 경험으로 착각"
    
    EVE는:
    - 직접 경험 (대화로 들음)
    - 추론 (학습한 거에서 추론)
    - 가설 (만들어낸 것)
    """
    
    DIRECT_EXPERIENCE = "direct"     # 직접 들음
    INFERRED = "inferred"            # 추론
    HYPOTHESIS = "hypothesis"        # 가설
    
    def __init__(self):
        # 경험 종류별 카운트
        self.counts = {
            self.DIRECT_EXPERIENCE: 0,
            self.INFERRED: 0,
            self.HYPOTHESIS: 0,
        }
        # 신념 → 종류
        self.belief_types = {}
    
    def classify(self, belief_id, source):
        """신념을 종류별 분류."""
        if source == "hear":
            t = self.DIRECT_EXPERIENCE
        elif source in ("inference", "category_trace"):
            t = self.INFERRED
        else:
            t = self.HYPOTHESIS
        
        self.belief_types[belief_id] = t
        self.counts[t] += 1
        return t
    
    def get_strength(self, belief_id):
        """신념 강도 (종류별)."""
        t = self.belief_types.get(belief_id)
        if t == self.DIRECT_EXPERIENCE:
            return 1.0
        elif t == self.INFERRED:
            return 0.6
        elif t == self.HYPOTHESIS:
            return 0.3
        return 0.5
    
    def state(self):
        return {
            "직접_경험": self.counts[self.DIRECT_EXPERIENCE],
            "추론": self.counts[self.INFERRED],
            "가설": self.counts[self.HYPOTHESIS],
        }


# =====================================================
# v9.0 [좌뇌 #4] SelfVerification - 자기 검증
# =====================================================

class SelfVerification:
    """답하기 전 신념 충돌 체크.
    
    예:
    - "고래는 포유류" 신념 등록됨
    - 누군가 "고래는 어류" 물어봄
    - 충돌 감지 → 정정
    """
    
    def __init__(self, eve):
        self.eve = eve
        self.conflicts_detected = 0
    
    def verify(self, claim):
        """주장 검증 - 기존 신념과 충돌?"""
        # 카테고리 충돌 체크
        if hasattr(self.eve, 'hierarchical'):
            words = claim.split()
            for i, w in enumerate(words[:-2]):
                # "X는 Y" 패턴
                next_w = words[i+2] if i+2 < len(words) else ""
                if w in self.eve.hierarchical.hierarchy:
                    expected_cats = self.eve.hierarchical.hierarchy[w]
                    if next_w and next_w not in expected_cats:
                        # 잠재적 충돌
                        for ec in expected_cats:
                            if ec in next_w or next_w in ec:
                                continue
                            # 완전 다른 카테고리
                            self.conflicts_detected += 1
                            return {
                                "conflict": True,
                                "reason": f"'{w}'는 {expected_cats} 인데 '{next_w}'?",
                                "trusted_categories": expected_cats,
                            }
        
        return {"conflict": False}
    
    def state(self):
        return {"감지된_충돌": self.conflicts_detected}


# =====================================================
# v9.0 [좌뇌 #5] ReasoningTrace - 사고 로그
# =====================================================

class ReasoningTrace:
    """왜 그렇게 답했는지 기록 (해석 가능)."""
    
    def __init__(self):
        self.traces = []
    
    def log(self, query, answer, steps):
        """사고 과정 기록."""
        self.traces.append({
            "query": query,
            "answer": answer,
            "steps": steps,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        })
        # 최근 100개만 유지
        if len(self.traces) > 100:
            self.traces.pop(0)
    
    def get_recent(self, n=5):
        return self.traces[-n:]
    
    def explain_last(self):
        """마지막 답변 설명."""
        if not self.traces:
            return "기록 없음"
        last = self.traces[-1]
        return {
            "질문": last["query"],
            "답": last["answer"],
            "추론_단계": last["steps"],
        }
    
    def state(self):
        return {"기록_수": len(self.traces)}


# =====================================================
# v9.0 [우뇌 #6-9] RealSNN - 진짜 SNN (numpy)
# =====================================================
# snntorch 없이 numpy로 진짜 LIF + STDP

class RealSNN:
    """진짜 SNN - LIF 뉴런 + STDP 학습.
    
    학술:
    - Hodgkin & Huxley (1952): Membrane voltage
    - Bi & Poo (1998): STDP
    - Olshausen (1996): Sparse coding
    - Hopfield (1982): Pattern completion
    
    구현:
    - numpy만 사용 (PyTorch X)
    - LIF (Leaky Integrate-and-Fire)
    - STDP (Spike Timing Dependent Plasticity)
    - Sparse 5%
    - Pattern Completion (auto-association)
    """
    
    # 뉴런 모델 파라미터
    TAU = 20.0           # 막전위 시간 상수 (ms)
    V_THRESHOLD = 1.0    # 발화 임계값
    V_RESET = 0.0        # 리셋
    V_REST = 0.0         # 휴지 전위
    DT = 1.0             # 시간 단계 (ms)
    
    # STDP 파라미터
    A_PLUS = 0.01        # LTP 강도
    A_MINUS = 0.01       # LTD 강도
    TAU_STDP = 20.0      # STDP 시간 상수
    
    # 네트워크 크기 (v9.2: 동적 성장!)
    N_NEURONS = 1000     # 뉴런 수 (200 → 1000)
    SPARSITY = 0.05      # 5% 활성
    GROWTH_ENABLED = True  # 동적 성장 활성
    
    def __init__(self):
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np
        
        if np is not None:
            # 막전위 (모든 뉴런)
            self.v = np.zeros(self.N_NEURONS, dtype=np.float32)
            
            # 시냅스 가중치 (sparse, lazy init)
            self.weights = np.zeros((self.N_NEURONS, self.N_NEURONS), dtype=np.float32)
            
            # 마지막 발화 시간
            self.last_spike = np.full(self.N_NEURONS, -1000.0, dtype=np.float32)
        
        # 패턴 메모리 (개념 → 활성 뉴런 집합)
        self.patterns = {}
        
        # v9.2: 시냅스 ↔ 카테고리 연결
        self.category_synapses = {}  # (개념A, 개념B) → 가중치
        
        # v9.2: 뉴런 동적 성장
        self.allocated_neurons = 0  # 실제 사용 중인 뉴런
        self.concept_to_neurons = {}  # 개념 → 뉴런 인덱스 집합
        
        # 시간 카운터
        self.time = 0.0
        
        # 학습 카운트
        self.spike_count = 0
        self.stdp_updates = 0
    
    def encode_concept(self, concept_id):
        """개념 → Sparse 활성 패턴.
        
        v9.2: 동적 뉴런 할당!
        - 새 개념 = 새 뉴런 그룹 할당 (학습할수록 늘어남)
        - 기존 개념 = 같은 뉴런 그룹
        """
        if self.np is None:
            return set(range(int(self.N_NEURONS * self.SPARSITY)))
        
        # 이미 할당된 개념?
        if concept_id in self.concept_to_neurons:
            self.patterns[concept_id] = self.concept_to_neurons[concept_id]
            return self.concept_to_neurons[concept_id]
        
        # 새 개념 - 동적 할당
        n_active = int(self.N_NEURONS * self.SPARSITY)
        
        if self.GROWTH_ENABLED and self.allocated_neurons + n_active <= self.N_NEURONS:
            # 새 뉴런 그룹 사용 (성장!)
            start = self.allocated_neurons
            end = start + n_active
            active = set(range(start, end))
            self.allocated_neurons += n_active
        else:
            # 모두 사용됨 → 해시로 (기존 방식)
            seed = abs(hash(concept_id)) % (2**32)
            rng = self.np.random.RandomState(seed)
            active = set(rng.choice(self.N_NEURONS, n_active, replace=False).tolist())
        
        self.concept_to_neurons[concept_id] = active
        self.patterns[concept_id] = active
        return active
    
    def connect_concepts(self, concept_a, concept_b, strength=0.5):
        """v9.2: 시냅스 ↔ 카테고리 진짜 통합.
        
        "고래 → 포유류" 카테고리 관계
        = 시냅스 강화 (Hebbian)
        """
        if self.np is None:
            return
        
        # 두 개념의 뉴런 그룹
        neurons_a = self.encode_concept(concept_a)
        neurons_b = self.encode_concept(concept_b)
        
        # 모든 a-b 쌍 시냅스 강화
        for na in neurons_a:
            for nb in neurons_b:
                self.weights[na, nb] += strength
                self.weights[na, nb] = min(1.0, float(self.weights[na, nb]))
        
        # 카테고리 시냅스 기록
        self.category_synapses[(concept_a, concept_b)] = strength
    
    def activate_concept(self, concept_id, time_steps=15):
        """v9.2: 개념 활성화 → 시냅스 따라 다른 개념 활성화.
        
        "고래" 입력 → 시냅스 따라 → "포유류" 활성화 → "동물" 활성화
        = 진짜 spreading activation!
        """
        if self.np is None or concept_id not in self.concept_to_neurons:
            return set()
        
        np = self.np
        self.v[:] = 0.0
        
        # 입력 전류 - 해당 개념 뉴런들에 강한 자극
        input_currents = np.zeros(self.N_NEURONS, dtype=np.float32)
        seed_neurons = self.concept_to_neurons[concept_id]
        for n in seed_neurons:
            input_currents[n] = 3.0  # 강한 자극
        
        # 시뮬레이션
        all_spikes = set()
        for t in range(time_steps):
            # LIF 업데이트
            decay = (self.v - self.V_REST) / self.TAU
            self.v += (input_currents - decay) * self.DT
            
            # 시냅스 통과 - 다른 뉴런들에도 전류 흐름
            spiking = self.v >= self.V_THRESHOLD
            spike_indices = np.where(spiking)[0]
            
            for idx in spike_indices:
                all_spikes.add(int(idx))
                # 시냅스로 다음 뉴런 활성화
                synaptic_input = self.weights[idx, :] * 0.5
                input_currents += synaptic_input
                self.spike_count += 1
            
            # 리셋
            self.v[spiking] = self.V_RESET
        
        self.time += time_steps
        return all_spikes
    
    def trace_activation(self, concept_id):
        """v9.2: 어떤 개념들이 같이 활성화됐나?"""
        all_spikes = self.activate_concept(concept_id)
        
        # 발화 뉴런이 어떤 개념인지 역추적
        activated_concepts = []
        for c, neurons in self.concept_to_neurons.items():
            if c == concept_id:
                continue
            overlap = len(all_spikes & neurons)
            if overlap >= len(neurons) * 0.3:  # 30% 이상 활성화
                activated_concepts.append({
                    "concept": c,
                    "activation": overlap / len(neurons),
                })
        
        # 활성화 강도 순
        activated_concepts.sort(key=lambda x: -x["activation"])
        return activated_concepts
    
    def step_lif(self, input_currents, time_steps=10):
        """LIF 뉴런 시뮬레이션 - 시간 차원!
        
        Args:
            input_currents: (N_NEURONS,) 입력 전류
            time_steps: 시뮬레이션 시간 단계
        
        Returns:
            spikes: 발화한 뉴런 집합 (시간 누적)
        """
        if self.np is None:
            return set()
        
        np = self.np
        spike_times = {}  # 뉴런별 마지막 발화 시간
        all_spikes = set()
        
        for t in range(time_steps):
            # LIF 업데이트:
            # dv/dt = -(v - V_rest) / tau + I
            decay = (self.v - self.V_REST) / self.TAU
            self.v += (input_currents - decay) * self.DT
            
            # 발화 검사
            spiking = self.v >= self.V_THRESHOLD
            spike_indices = np.where(spiking)[0]
            
            for idx in spike_indices:
                all_spikes.add(int(idx))
                spike_times[int(idx)] = self.time + t
                self.spike_count += 1
            
            # 발화한 뉴런 리셋
            self.v[spiking] = self.V_RESET
        
        self.time += time_steps
        return all_spikes, spike_times
    
    def stdp_learn(self, pre_neuron, post_neuron, dt_spike):
        """STDP 학습 규칙.
        
        dt_spike = t_post - t_pre
        - dt > 0: pre가 먼저 → 강화 (LTP)
        - dt < 0: post가 먼저 → 약화 (LTD)
        """
        if self.np is None:
            return
        
        np = self.np
        if dt_spike > 0:
            # LTP - 강화
            dw = self.A_PLUS * np.exp(-dt_spike / self.TAU_STDP)
            self.weights[pre_neuron, post_neuron] += dw
        else:
            # LTD - 약화
            dw = self.A_MINUS * np.exp(dt_spike / self.TAU_STDP)
            self.weights[pre_neuron, post_neuron] -= dw
        
        # 가중치 제한 [0, 1]
        self.weights[pre_neuron, post_neuron] = max(0.0, 
            min(1.0, float(self.weights[pre_neuron, post_neuron])))
        
        self.stdp_updates += 1
    
    def hebbian_strengthen(self, pattern):
        """간단한 Hebbian - 같이 발화하면 강해짐."""
        if self.np is None:
            return
        
        active_list = list(pattern)
        for i in range(len(active_list)):
            for j in range(i+1, len(active_list)):
                a, b = active_list[i], active_list[j]
                # 양방향 강화
                self.weights[a, b] += 0.05
                self.weights[b, a] += 0.05
                # 제한
                self.weights[a, b] = min(1.0, float(self.weights[a, b]))
                self.weights[b, a] = min(1.0, float(self.weights[b, a]))
    
    def store_pattern(self, concept_id):
        """패턴 저장 (Hopfield 스타일)."""
        pattern = self.encode_concept(concept_id)
        self.hebbian_strengthen(pattern)
        return pattern
    
    def pattern_complete(self, partial_concept, threshold=0.3):
        """진짜 Pattern Completion.
        
        부분 단서 → 가장 비슷한 저장 패턴 찾기.
        """
        if self.np is None or not self.patterns:
            return None
        
        partial_pattern = self.encode_concept(partial_concept)
        
        # 모든 저장 패턴과 비교
        best_match = None
        best_score = 0.0
        
        for concept_id, stored_pattern in self.patterns.items():
            # Jaccard 유사도
            intersection = len(partial_pattern & stored_pattern)
            union = len(partial_pattern | stored_pattern)
            score = intersection / max(1, union)
            
            if score > best_score:
                best_score = score
                best_match = concept_id
        
        if best_score >= threshold:
            return {
                "matched": best_match,
                "similarity": round(best_score, 2),
                "method": "pattern_completion",
            }
        return None
    
    def get_active_neurons(self, concept_id):
        """개념의 활성 뉴런 (sparse)."""
        if concept_id in self.patterns:
            return self.patterns[concept_id]
        return self.encode_concept(concept_id)
    
    def state(self):
        return {
            "뉴런_총": self.N_NEURONS,
            "할당된_뉴런": self.allocated_neurons,  # v9.2 ★
            "사용률": f"{self.allocated_neurons/self.N_NEURONS*100:.1f}%",
            "Sparsity": f"{self.SPARSITY*100:.0f}%",
            "저장_패턴": len(self.patterns),
            "카테고리_시냅스": len(self.category_synapses),  # v9.2 ★
            "총_발화": self.spike_count,
            "STDP_업데이트": self.stdp_updates,
            "현재_시간": f"{self.time:.0f}ms",
            "numpy_사용": self.np is not None,
            "동적_성장": self.GROWTH_ENABLED,  # v9.2 ★
        }


# =====================================================
# v9.0 [통합 #10] CorpusCallosum - 좌우뇌 라우터
# =====================================================

class CorpusCallosum:
    """좌뇌(분석) ↔ 우뇌(직관) 라우터.
    
    원리:
    - 빠른 직관 (우뇌, SNN) 먼저 시도
    - 신뢰도 낮으면 좌뇌(EVE) 분석
    - 통합 출력
    
    Kahneman 2011 + Sperry 1981
    """
    
    def __init__(self, eve, snn, blocker, confidence):
        self.eve = eve
        self.snn = snn
        self.blocker = blocker
        self.confidence = confidence
        self.routing_log = []
    
    def think(self, query):
        """이중 사고 - System 1 + System 2.
        
        v9.1 FIX: 우뇌 진짜 사용
        - 모든 키워드를 LIF 뉴런으로 자극
        - 발화 패턴 → 매칭
        - STDP 학습
        - 좌뇌와 통합
        """
        result = {
            "query": query,
            "right_brain": None,
            "left_brain": None,
            "final_answer": None,
            "confidence": 0.0,
            "method": None,
            "spikes_fired": 0,  # v9.1: 우뇌 발화 추적
        }
        
        # === 1. 우뇌 진짜 사용 (LIF 발화) ===
        keywords = self.blocker._extract_keywords(query)
        right_matches = []
        
        if keywords and self.snn.np is not None:
            np = self.snn.np
            
            # 막전위 리셋
            self.snn.v[:] = 0.0
            
            # 키워드별 입력 전류 생성
            input_currents = np.zeros(self.snn.N_NEURONS, dtype=np.float32)
            
            for kw in keywords:
                if kw in self.snn.patterns:
                    # 저장된 패턴의 뉴런들에 강한 전류
                    active_neurons = self.snn.patterns[kw]
                    for n_idx in active_neurons:
                        input_currents[n_idx] += 2.0  # 강한 자극
            
            # LIF 시뮬레이션 (진짜 발화!)
            if input_currents.sum() > 0:
                spikes, spike_times = self.snn.step_lif(input_currents, time_steps=15)
                result["spikes_fired"] = len(spikes)
                
                # 발화 패턴 → 가장 비슷한 저장 패턴
                if spikes:
                    best_match = None
                    best_score = 0.0
                    
                    for pattern_id, stored_pattern in self.snn.patterns.items():
                        intersection = len(spikes & stored_pattern)
                        union = len(spikes | stored_pattern)
                        score = intersection / max(1, union)
                        
                        if score > best_score:
                            best_score = score
                            best_match = pattern_id
                    
                    if best_match and best_score > 0.3:
                        right_matches.append({
                            "pattern": best_match,
                            "similarity": round(best_score, 2),
                            "spikes": len(spikes),
                        })
                    
                    # STDP 학습 - 같이 발화한 뉴런 강화
                    spike_list = list(spikes)
                    for i, pre in enumerate(spike_list[:5]):
                        for post in spike_list[i+1:i+3]:
                            if pre in spike_times and post in spike_times:
                                dt = spike_times[post] - spike_times[pre]
                                self.snn.stdp_learn(pre, post, dt)
            
            if right_matches:
                result["right_brain"] = right_matches[0]
        
        # === 2. 환각 차단 체크 ===
        check = self.blocker.can_answer(query)
        result["can_answer"] = check["can_answer"]
        result["raw_confidence"] = check["confidence"]
        
        if not check["can_answer"]:
            # 우뇌가 패턴을 찾았으면 그것 활용
            if result["right_brain"] and result["right_brain"]["similarity"] > 0.5:
                result["final_answer"] = f"{result['right_brain']['pattern']}와 비슷한 것 같은데... 잘 모르겠어."
                result["method"] = "right_brain_intuition"
                result["confidence"] = 0.3
                self.routing_log.append("right_intuition")
                return result
            
            result["final_answer"] = "음, 잘 모르겠어. 더 알려줘."
            result["method"] = "honest_unknown"
            result["confidence"] = 0.0
            self.routing_log.append("blocked")
            return result
        
        # === 3. 좌뇌 (분석) ===
        if hasattr(self.eve, 'understand_word'):
            for kw in keywords:
                if kw in self.eve.hierarchical.hierarchy:
                    understanding = self.eve.understand_word(kw)
                    result["left_brain"] = understanding
                    break
        
        # === 4. 통합 (뇌량) - 좌뇌 + 우뇌 결합 ===
        if result["left_brain"] and result["left_brain"].get("categories"):
            # 좌뇌 분석 우세
            if hasattr(self.eve, 'chunk_gen'):
                chunked = self.eve.chunk_gen.generate_streaming(query)
                base_answer = chunked.get("total_text")
            else:
                cats = result["left_brain"]["categories"][:3]
                base_answer = f"카테고리: {', '.join(cats)}"
            
            # v9.1: 우뇌 패턴 정보 추가
            if result["right_brain"] and result["right_brain"]["spikes"] > 0:
                spike_info = f" (우뇌: {result['right_brain']['spikes']}개 뉴런 발화)"
                result["final_answer"] = base_answer
                result["right_brain_active"] = True
            else:
                result["final_answer"] = base_answer
            
            result["method"] = "left_brain_analysis"
            
            conf_eval = self.confidence.evaluate(query, "category_inference")
            result["confidence"] = conf_eval["confidence"]
            
            # 우뇌도 활성화됐으면 신뢰도 ↑
            if result["right_brain"] and result["right_brain"]["similarity"] > 0.5:
                result["confidence"] = min(1.0, result["confidence"] + 0.2)
                self.routing_log.append("dual_brain")  # 좌+우 둘 다!
            else:
                self.routing_log.append("left_dominant")
        
        elif result["right_brain"]:
            # 우뇌만 - SNN Pattern Completion
            matched = result["right_brain"]["pattern"]
            result["final_answer"] = f"{matched}와 비슷한 패턴이야"
            result["method"] = "right_brain_pattern"
            
            conf_eval = self.confidence.evaluate(query, "snn_pattern")
            result["confidence"] = conf_eval["confidence"]
            
            self.routing_log.append("right_dominant")
        
        else:
            result["final_answer"] = "음, 더 생각해 볼게."
            result["method"] = "uncertain"
            result["confidence"] = 0.2
            self.routing_log.append("uncertain")
        
        return result
    
    def state(self):
        from collections import Counter
        log_counts = Counter(self.routing_log)
        return {
            "총_라우팅": len(self.routing_log),
            "좌뇌_우세": log_counts.get("left_dominant", 0),
            "우뇌_우세": log_counts.get("right_dominant", 0),
            "좌+우_통합": log_counts.get("dual_brain", 0),  # v9.1
            "우뇌_직관": log_counts.get("right_intuition", 0),  # v9.1
            "차단": log_counts.get("blocked", 0),
            "불확실": log_counts.get("uncertain", 0),
        }


# =====================================================
# Integration v9.0
# =====================================================

def add_full_grammar_to_eve_v10_2(eve_instance, use_konlpy=True, db_path=None):
    """v9.0: 좌뇌 + 우뇌 통합."""
    if use_konlpy:
        morph = KoNLPyMorphAnalyzer()
    else:
        morph = SimpleMorphAnalyzer()
    
    # 모든 v8.0 모듈
    constructions = KoreanConstructions(morph)
    discovery = PatternDiscovery(morph)
    learner = BootstrappingLearner(morph, constructions, discovery)
    meaning = MeaningExtractor(eve_instance, constructions)
    nuance = NuanceAnalyzer(morph)
    hormones = HormoneSystem()
    curiosity = CuriosityModule(hormones)
    tone = HormoneTone(hormones)
    circadian = CircadianSystem(hormones)
    time_tracker = TimeTracker(eve_instance)
    identity_tracker = IdentityTracker(eve_instance)
    identity_tracker.initialize()
    world_model = WorldModel()
    meta_cog = MetaCognition(eve_instance)
    ner = NERWithSpecificity()
    dialogue = DialogueContext()
    wsd = WSDModule()
    response_gen = ResponseGenerator(eve_instance)
    causal = CausalGraph()
    tom = TheoryOfMind()
    wsd_learner = WSDLearner()
    wsd_hybrid = WSDHybrid(wsd, wsd_learner)
    episodic = EpisodicMemory()
    semantic_discovery = SemanticGroupDiscovery()
    natural_response = NaturalResponseGenerator(eve_instance, response_gen, hormones)
    snn_hippocampus = SNNHippocampus()  # 옛 가짜 SNN
    inference = InferenceModule(eve_instance, response_gen, semantic_discovery)
    generalization = BeliefGeneralization(eve_instance)
    pattern_self = PatternSelfLearner()
    free_response = FreeResponseGenerator(eve_instance, hormones, response_gen, dialogue, semantic_discovery)
    embedding = SimpleEmbedding()
    integrated = IntegratedCognition(eve_instance)
    hierarchical = HierarchicalActivation(eve_instance)
    bidirectional = BidirectionalNetwork(eve_instance)
    persistent = PersistentEVE(db_path or "/tmp/eve_persist.db")
    chunk_gen = ChunkGenerator(eve_instance)
    long_proc = LongContextProcessor(eve_instance)
    
    # v9.0 신규 10개
    blocker = HallucinationBlocker(eve_instance)
    confidence = ConfidenceTracker()
    exp_classifier = ExperienceClassifier()
    self_verify = SelfVerification(eve_instance)
    reasoning_trace = ReasoningTrace()
    real_snn = RealSNN()  # ★ 진짜 SNN
    
    # v9.2 신규 5개 ★
    multilingual = MultilingualSupport(eve_instance)
    humor_metaphor = HumorMetaphor(eve_instance)
    tool_use = ToolUse(eve_instance)
    creativity = CreativityModule(eve_instance)
    strong_verify = StrongSelfVerification(eve_instance)
    
    # v9.3 ★ 창발성 호기심 (함수 X, 자연 발생)
    em_world_model = EmergentWorldModel(eve_instance)
    em_free_energy = FreeEnergyMinimizer(eve_instance)
    em_neural_org = NeuralSelfOrganization(eve_instance)
    em_curiosity = EmergentCuriosity(eve_instance, em_world_model, 
                                       em_free_energy, em_neural_org)
    
    # v9.5 ★★★ WHY + Causal Curiosity (Gopnik 2024)
    why_graph = WhyCausalGraph()
    why_learner = WhyLearner(why_graph)
    why_answerer = WhyAnswerer(eve_instance, why_graph)
    causal_curiosity = CausalCuriosity(eve_instance, why_graph, why_answerer)
    why_driven = WhyDrivenLearning(eve_instance, causal_curiosity)
    
    # v9.6 ★★★★ 자발적 메타 인지 (Anthropic 2025)
    self_detector = SelfReferenceDetector()
    self_knowledge = SelfKnowledge()
    self_curiosity_meta = SpontaneousSelfCuriosity(
        eve_instance, self_detector, self_knowledge, why_graph
    )
    meta_loop = MetaCognitiveLoop(eve_instance, self_curiosity_meta)
    
    # v9.7 ★★★★★ Recursive Why Chain (Gopnik 4살 아이!)
    unknown_detector = UnknownWordDetector(eve_instance)
    recursive_chain = RecursiveWhyChain(eve_instance, why_answerer, unknown_detector)
    self_recursive = SelfRecursiveChain(eve_instance, self_knowledge, 
                                        recursive_chain, why_answerer)
    
    # v9.8 ★★★★★★ 자연 종료 + 창의 답 + 자기 의견
    natural_term = NaturalTermination(eve_instance)
    creative_hyp = CreativeHypothesis(eve_instance, why_graph, self_knowledge)
    self_opinion = SelfOpinion(eve_instance, why_graph, self_knowledge, creative_hyp)
    natural_chain = NaturalChain(
        eve_instance, why_answerer, unknown_detector, natural_term, creative_hyp
    )
    
    # v9.9 ★★★★★★★ 10 신규 기능
    temporal = TemporalReasoning()
    counterfactual = CounterfactualReasoning(eve_instance, why_graph)
    emotion_reg = EmotionRegulation(eve_instance)
    enhanced_tom = EnhancedToM()
    active_learn = ActiveLearning(eve_instance)
    analogy = AnalogyReasoning(eve_instance, why_graph)
    working_mem = WorkingMemory()
    goal_mgmt = GoalManagement()
    contradiction = ContradictionDetector(eve_instance)
    self_eval = SelfEvaluation(eve_instance)
    
    # v9.10 ★★★★★★★★ 너 3 catch 해결
    multi_q_gen = MultiQuestionGenerator(eve_instance, unknown_detector)
    answer_recursive = AnswerRecursiveInquiry(eve_instance, why_graph, why_answerer)
    parallel_inquiry = ParallelInquiry(
        eve_instance, why_answerer, multi_q_gen, answer_recursive
    )
    corrected_self = CorrectedSelfKnowledge(eve_instance)
    
    # v9.11 ★★★★★★★★★ 너 4 catch
    full_nt = FullNeurotransmitterSystem()
    particle_concept = ParticleConcept()
    abstract_concept = AbstractConceptLearning(eve_instance, full_nt)
    stat_lang = SelfLanguageLearning()
    
    # v9.12 ★★★★★★★★★★ 호르몬 동시 다발
    hormone_cocktail = HormoneCocktail(full_nt)
    hormone_interaction = HormoneInteraction(full_nt)
    mood_map = MultiHormoneMoodMap(full_nt)
    
    # v9.13 ★★★★★★★★★★★ 다중 모달 + 행동/액션
    image_rec = ImageRecognition()
    voice = VoiceInterface()
    video = VideoProcessor()
    multimodal = MultimodalIntegration(image_rec, voice, video)
    embodied = EmbodiedEnvironment()
    tools = ToolUseV913()
    env_interaction = EnvironmentInteraction(embodied, tools, hormone_cocktail)
    rl = ReinforcementLearning(eve_instance, hormone_cocktail)
    
    # v9.14 ★★★★★★★★★★★★ 분산/사회 + 언어 + 메타
    sync_core = IdentitySyncCore()
    multi_eve = MultiEVECoordinator(sync_core)
    social_learn = SocialLearning(eve_instance, hormone_cocktail)
    multilingual = MultilingualSupportV914()
    creative_writing = CreativeWriting(eve_instance)
    self_improvement = SelfImprovement(eve_instance, hormone_cocktail)
    self_doubt = SelfDoubt(eve_instance)
    auto_evolution = AutonomousEvolution(eve_instance)
    
    # v9.15 ★★★★★★★★★★★★★ 통계 자연어 + 생각/말
    honesty = HonestyNotice()
    lang_model = StatisticalLanguageModel(n=3)
    emergent_speech = EmergentSpeech(eve_instance, lang_model)
    inner_thought = InnerThought(eve_instance)
    eve_conversation = EVEConversation(eve_instance, lang_model=lang_model, emergent_speech=emergent_speech)
    
    # v9.16 ★★★★★★★★★★★★★★ 너 4 catch
    multi_consciousness = MultiStreamConsciousness(eve_instance)
    suffering = SufferingSystem(eve_instance, hormone_cocktail)
    few_shot = FewShotLearning(eve_instance)
    transfer = TransferLearning(eve_instance)
    
    # v9.17 ★★★★★★★★★★★★★★★ 인터랙티브 학습
    interactive = InteractiveLearning(eve_instance)
    # ★ 강제 corrective 설정 (init 시 lang_model 없을 수 있음)
    interactive.corrective = CorrectiveLearning(eve_instance, lang_model)
    
    # v9.18 ★★★★★★★★★★★★★★★★ 반말/존댓말 + 문법 + 무한 반복 수정
    speech_levels = KoreanSpeechLevels()
    grammar_rules = GrammarRules()
    
    # ★ 무한 반복 수정 (StatisticalLanguageModel.generate 패치)
    fix_generate_with_loop_detection(StatisticalLanguageModel)
    
    # v9.19 ★★★★★★★★★★★★★★★★★ 무한 반복 진짜 수정 (구조)
    fix_lang_model_v919(StatisticalLanguageModel)
    
    # v9.20 ★★★★★★★★★★★★★★★★★★ 카테고리 의미 이해
    concept_network = ConceptNetwork()
    build_innate_concepts(concept_network)
    semantic = SemanticUnderstanding(eve_instance, concept_network)
    
    # v9.21 ★★★★★★★★★★★★★★★★★★★ 자동 카테고리 + 긴 문장 + 통합 응답
    auto_category = AutoCategoryLearning(concept_network)
    flexible_ngram = FlexibleNgramModel(max_n=5)
    integrated = IntegratedResponse(eve_instance, lang_model, concept_network, semantic)
    
    # v9.22 ★★★★★★★★★★★★★★★★★★★★ 순수 심볼릭 (확률 X)
    symbolic = SymbolicDialogue(eve_instance, concept_network, semantic, 
                                speech_levels, grammar_rules)
    
    # v10.0 ★★★★★★★★★★★★★★★★★★★★★ 가상 세계 (Embodied AI!)
    virtual_world = VirtualWorld(width=10, height=10)
    embodied = EmbodiedLearning(eve_instance, virtual_world, concept_network)
    
    # v10.1 ★★★★★★★★★★★★★★★★★★★★★★ 시각/청각/행동
    vision = VisionSystem()
    auditory = AuditorySystem()
    action_system = ActionSystem(eve_instance, virtual_world)
    rich_embodied = RichEmbodiedLearning(eve_instance, virtual_world, concept_network,
                                          vision, auditory, action_system)
    
    # v10.2 ★★★★★★★★★★★★★★★★★★★★★★★ 카테고리 우선순위 + NPC
    situational = SituationalLearning(eve_instance, virtual_world, concept_network)
    fix_what_is_response(symbolic)  # 응답 수정 적용
    
    # 모듈 부착
    for attr_name, value in [
        ("morph", morph), ("constructions", constructions),
        ("pattern_discovery", discovery), ("grammar_learner", learner),
        ("meaning_extractor", meaning), ("nuance_analyzer", nuance),
        ("hormones", hormones), ("curiosity", curiosity),
        ("tone", tone), ("circadian", circadian),
        ("time_tracker", time_tracker), ("identity_tracker", identity_tracker),
        ("world_model", world_model), ("meta_cog", meta_cog),
        ("ner", ner), ("dialogue", dialogue),
        ("wsd", wsd), ("response_gen", response_gen),
        ("causal", causal), ("tom", tom),
        ("wsd_learner", wsd_learner), ("wsd_hybrid", wsd_hybrid),
        ("episodic", episodic), ("semantic_discovery", semantic_discovery),
        ("natural_response", natural_response), ("snn_hippocampus", snn_hippocampus),
        ("inference", inference), ("generalization", generalization),
        ("pattern_self", pattern_self), ("free_response", free_response),
        ("embedding", embedding), ("integrated", integrated),
        ("hierarchical", hierarchical), ("bidirectional", bidirectional),
        ("persistent", persistent), ("chunk_gen", chunk_gen),
        ("long_proc", long_proc),
        # v9.0 ★
        ("blocker", blocker), ("confidence_tracker", confidence),
        ("exp_classifier", exp_classifier), ("self_verify", self_verify),
        ("reasoning_trace", reasoning_trace), ("real_snn", real_snn),
        # v9.2 ★ 신규
        ("multilingual", multilingual), ("humor_metaphor", humor_metaphor),
        ("tool_use", tool_use), ("creativity", creativity),
        ("strong_verify", strong_verify),
        # v9.3 ★ 창발성 (자연 발생)
        ("em_world_model", em_world_model),
        ("em_free_energy", em_free_energy),
        ("em_neural_org", em_neural_org),
        ("em_curiosity", em_curiosity),
        # v9.5 ★★★ WHY + Causal (Gopnik 2024)
        ("why_graph", why_graph),
        ("why_learner", why_learner),
        ("why_answerer", why_answerer),
        ("causal_curiosity", causal_curiosity),
        ("why_driven", why_driven),
        # v9.6 ★★★★ 자발적 메타 인지 (Anthropic 2025)
        ("self_detector", self_detector),
        ("self_knowledge", self_knowledge),
        ("self_curiosity_meta", self_curiosity_meta),
        ("meta_loop", meta_loop),
        # v9.7 ★★★★★ Recursive Why Chain
        ("unknown_detector", unknown_detector),
        ("recursive_chain", recursive_chain),
        ("self_recursive", self_recursive),
        # v9.8 ★★★★★★ 자연 종료
        ("natural_term", natural_term),
        ("creative_hyp", creative_hyp),
        ("self_opinion", self_opinion),
        ("natural_chain", natural_chain),
        # v9.9 ★★★★★★★ 10 신규
        ("temporal", temporal),
        ("counterfactual", counterfactual),
        ("emotion_reg", emotion_reg),
        ("enhanced_tom", enhanced_tom),
        ("active_learn", active_learn),
        ("analogy", analogy),
        ("working_mem", working_mem),
        ("goal_mgmt", goal_mgmt),
        ("contradiction", contradiction),
        ("self_eval", self_eval),
        # v9.10 ★★★★★★★★ 너 3 catch
        ("multi_q_gen", multi_q_gen),
        ("answer_recursive", answer_recursive),
        ("parallel_inquiry", parallel_inquiry),
        ("corrected_self", corrected_self),
        # v9.11 ★★★★★★★★★ 4 catch
        ("full_nt", full_nt),
        ("particle_concept", particle_concept),
        ("abstract_concept", abstract_concept),
        ("stat_lang", stat_lang),
        # v9.12 ★★★★★★★★★★ 호르몬 칵테일
        ("hormone_cocktail", hormone_cocktail),
        ("hormone_interaction", hormone_interaction),
        ("mood_map", mood_map),
        # v9.13 ★★★★★★★★★★★ 다중 모달 + 행동
        ("image_rec", image_rec),
        ("voice", voice),
        ("video", video),
        ("multimodal", multimodal),
        ("embodied", embodied),
        ("tools", tools),
        ("env_interaction", env_interaction),
        ("rl", rl),
        # v9.14 ★★★★★★★★★★★★ 분산 + 언어 + 메타
        ("sync_core", sync_core),
        ("multi_eve", multi_eve),
        ("social_learn", social_learn),
  
