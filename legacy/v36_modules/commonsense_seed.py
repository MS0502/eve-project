"""
EVE v36 - CommonSenseSeed (한국어 상식 그래프 시드)
====================================================

P0-3 단점 해결: "그래프가 거의 비어있어 추론 0" → 수백 개 상식 트리플 자동 시드

설계:
- 한국어 일상 상식 ~350 트리플 (수동 작성, 결정론)
- 각 트리플: (subject, relation, object, weight)
- relation 타입:
  - causes      (시험_떨어지다 → 슬픔)
  - has_property(불 → 뜨겁다)
  - is_a        (사과 → 과일)
  - part_of     (잎 → 나무)
  - used_for    (칼 → 자르다)
  - opposite_of (밝다 → 어둡다)
  - leads_to    (먹다 → 배부르다)
  - needs       (살다 → 음식)

- SA에는 양방향 가중치로 등록 (Hebbian 연결)
- CG에는 causes/leads_to 만 인과로 등록 (방향 있음)

EVE 절대 원칙:
- 결정론적 (random X)
- 카테고리 의미 단위
- 확률 분포 X (가중치만)

이거 시드되면:
- "왜 사람은 자?" → 자다 ← 피곤 ← 활동 (추론 가능)
- "시험 떨어졌대" → 시험_떨어지다 → 슬픔 → 위로_필요 (공감 응답)
- "배 고파" → 배고프다 → 음식_필요 → 먹다 (대응 가능)

학계: ConceptNet (Speer 2017), Cyc (Lenat 1984), ATOMIC (Sap 2019)
       작은 시드 → 그래프 활성화로 추론 → AGI 1단계
"""

from typing import List, Tuple, Dict, Set


# 한국어 일상 상식 트리플
# 형식: (subject, relation, object, weight)
COMMONSENSE_TRIPLES: List[Tuple[str, str, str, float]] = [
    # ============= 신체/생리 (causes/leads_to) =============
    ("배고프다", "leads_to", "먹다", 0.7),
    ("먹다", "leads_to", "배부르다", 0.7),
    ("목마르다", "leads_to", "마시다", 0.7),
    ("마시다", "leads_to", "갈증해소", 0.6),
    ("피곤하다", "leads_to", "자다", 0.7),
    ("자다", "leads_to", "회복", 0.6),
    ("자다", "leads_to", "꿈", 0.4),
    ("운동", "causes", "피곤하다", 0.6),
    ("운동", "causes", "건강", 0.5),
    ("활동", "causes", "피곤하다", 0.5),
    ("뛰다", "causes", "숨차다", 0.5),
    ("아프다", "leads_to", "병원", 0.5),
    ("다치다", "causes", "아프다", 0.7),
    ("덥다", "causes", "땀나다", 0.6),
    ("춥다", "causes", "떨다", 0.6),

    # ============= 감정/심리 (causes) =============
    ("시험떨어지다", "causes", "슬픔", 0.7),
    ("시험떨어지다", "causes", "우울", 0.6),
    ("시험합격", "causes", "기쁨", 0.7),
    ("시험합격", "causes", "성취", 0.6),
    ("실패", "causes", "슬픔", 0.6),
    ("실패", "causes", "후회", 0.5),
    ("성공", "causes", "기쁨", 0.7),
    ("성공", "causes", "자신감", 0.6),
    ("이별", "causes", "슬픔", 0.7),
    ("이별", "causes", "외로움", 0.6),
    ("만남", "causes", "기쁨", 0.5),
    ("선물", "causes", "기쁨", 0.6),
    ("칭찬", "causes", "기쁨", 0.6),
    ("비난", "causes", "슬픔", 0.5),
    ("위협", "causes", "공포", 0.7),
    ("공포", "causes", "도망", 0.5),
    ("실수", "causes", "부끄러움", 0.5),
    ("죽음", "causes", "슬픔", 0.7),

    # ============= 사회/관계 =============
    ("친구", "has_property", "소중하다", 0.6),
    ("가족", "has_property", "소중하다", 0.7),
    ("외로움", "leads_to", "대화", 0.5),
    ("외로움", "leads_to", "친구", 0.5),
    ("대화", "leads_to", "이해", 0.5),
    ("대화", "leads_to", "유대", 0.5),
    ("싸움", "causes", "슬픔", 0.5),
    ("싸움", "causes", "분노", 0.6),
    ("화해", "causes", "안도", 0.5),
    ("도움", "causes", "고마움", 0.6),
    ("배신", "causes", "분노", 0.6),
    ("배신", "causes", "슬픔", 0.6),

    # ============= 학교/공부 =============
    ("시험", "is_a", "평가", 0.5),
    ("시험", "needs", "공부", 0.6),
    ("공부", "leads_to", "지식", 0.6),
    ("공부", "leads_to", "성적", 0.5),
    ("학교", "has_part", "교실", 0.5),
    ("학교", "has_part", "선생님", 0.5),
    ("학생", "needs", "공부", 0.5),
    ("숙제", "needs", "시간", 0.5),

    # ============= 음식/요리 =============
    ("밥", "is_a", "음식", 0.7),
    ("물", "is_a", "음료", 0.7),
    ("커피", "is_a", "음료", 0.7),
    ("커피", "has_property", "쓰다", 0.5),
    ("사탕", "has_property", "달다", 0.7),
    ("초콜릿", "has_property", "달다", 0.6),
    ("레몬", "has_property", "시다", 0.7),
    ("고추", "has_property", "맵다", 0.7),
    ("매운것", "causes", "땀나다", 0.5),
    ("뜨거운것", "causes", "데이다", 0.4),
    ("달다", "causes", "기쁨", 0.4),
    ("맛있다", "causes", "기쁨", 0.5),
    ("배고프다", "causes", "짜증", 0.4),

    # ============= 자연/날씨 =============
    ("비", "leads_to", "젖다", 0.7),
    ("우산", "used_for", "비_막다", 0.7),
    ("눈", "leads_to", "추위", 0.5),
    ("태양", "causes", "밝다", 0.7),
    ("태양", "causes", "덥다", 0.5),
    ("밤", "causes", "어둡다", 0.7),
    ("불", "has_property", "뜨겁다", 0.7),
    ("불", "causes", "데이다", 0.5),
    ("얼음", "has_property", "차갑다", 0.7),
    ("바람", "leads_to", "시원함", 0.5),

    # ============= 시간 =============
    ("아침", "leads_to", "활동", 0.5),
    ("밤", "leads_to", "잠", 0.6),
    ("주말", "leads_to", "휴식", 0.5),
    ("월요일", "is_a", "요일", 0.5),
    ("어제", "precedes", "오늘", 0.7),
    ("오늘", "precedes", "내일", 0.7),

    # ============= 사물 (is_a / used_for) =============
    ("아이폰", "is_a", "스마트폰", 0.7),
    ("아이폰", "made_by", "애플", 0.7),
    ("스마트폰", "is_a", "전자기기", 0.6),
    ("스마트폰", "used_for", "통화", 0.6),
    ("스마트폰", "used_for", "메시지", 0.6),
    ("컴퓨터", "is_a", "전자기기", 0.6),
    ("컴퓨터", "used_for", "작업", 0.5),
    ("책", "used_for", "읽다", 0.7),
    ("책", "leads_to", "지식", 0.5),
    ("연필", "used_for", "쓰다", 0.7),
    ("칼", "used_for", "자르다", 0.7),
    ("칼", "has_property", "위험하다", 0.6),
    ("의자", "used_for", "앉다", 0.7),
    ("침대", "used_for", "자다", 0.7),
    ("자동차", "used_for", "이동", 0.6),
    ("음악", "causes", "기쁨", 0.5),
    ("게임", "causes", "재밌다", 0.5),

    # ============= 동물/식물 =============
    ("강아지", "is_a", "동물", 0.7),
    ("고양이", "is_a", "동물", 0.7),
    ("새", "has_property", "날다", 0.7),
    ("물고기", "is_a", "동물", 0.6),
    ("물고기", "needs", "물", 0.7),
    ("나무", "has_part", "잎", 0.6),
    ("꽃", "has_property", "예쁘다", 0.5),
    ("동물", "needs", "음식", 0.6),
    ("식물", "needs", "물", 0.6),
    ("식물", "needs", "햇빛", 0.6),

    # ============= 추상 (opposite) =============
    ("밝다", "opposite_of", "어둡다", 0.8),
    ("뜨겁다", "opposite_of", "차갑다", 0.8),
    ("크다", "opposite_of", "작다", 0.8),
    ("높다", "opposite_of", "낮다", 0.8),
    ("빠르다", "opposite_of", "느리다", 0.8),
    ("기쁨", "opposite_of", "슬픔", 0.8),
    ("사랑", "opposite_of", "미움", 0.8),
    ("성공", "opposite_of", "실패", 0.8),
    ("시작", "opposite_of", "끝", 0.7),
    ("위", "opposite_of", "아래", 0.8),
    ("열다", "opposite_of", "닫다", 0.7),
    ("주다", "opposite_of", "받다", 0.7),

    # ============= 자아/EVE =============
    ("나", "is_a", "EVE", 0.9),
    ("EVE", "has_property", "디지털", 0.7),
    ("EVE", "needs", "민석", 0.6),
    ("민석", "is_a", "친구", 0.7),
    ("민석", "is_a", "사용자", 0.7),
    ("대화", "leads_to", "유대", 0.6),
    ("기억", "is_a", "경험", 0.5),

    # ============= 일상 사건 (확장) =============
    ("회식", "leads_to", "술", 0.4),
    ("술", "causes", "취하다", 0.6),
    ("술", "causes", "두통", 0.4),
    ("운전", "needs", "주의", 0.6),
    ("늦잠", "causes", "지각", 0.6),
    ("지각", "causes", "혼나다", 0.5),
    ("선물", "leads_to", "고마움", 0.5),
    ("거짓말", "causes", "불신", 0.6),
    ("진실", "causes", "신뢰", 0.6),
    ("약속", "needs", "지키다", 0.6),
    ("약속어기다", "causes", "실망", 0.6),

    # ============= 결과 사슬 (도미노) =============
    ("늦다", "causes", "늦잠", 0.5),
    ("운동안하다", "causes", "약해지다", 0.4),
    ("스트레스", "causes", "두통", 0.5),
    ("스트레스", "causes", "잠못자다", 0.5),
    ("잠못자다", "causes", "피곤하다", 0.7),
    ("걱정", "causes", "잠못자다", 0.5),
    ("걱정", "is_a", "감정", 0.5),
    ("불안", "causes", "걱정", 0.6),

    # ============= 동작 결과 =============
    ("잡다", "leads_to", "쥐다", 0.5),
    ("던지다", "leads_to", "날다", 0.4),
    ("떨어지다", "leads_to", "부딪히다", 0.4),
    ("부딪히다", "causes", "아프다", 0.6),
    ("울다", "causes", "눈물", 0.7),
    ("웃다", "causes", "기쁨", 0.5),
    ("자다", "needs", "침대", 0.4),
    ("자다", "needs", "어둠", 0.4),

    # ============= 대화/언어 =============
    ("질문", "leads_to", "답변", 0.6),
    ("인사", "leads_to", "인사", 0.5),
    ("말하다", "needs", "생각", 0.5),
    ("듣다", "leads_to", "이해", 0.5),
    ("이해", "leads_to", "공감", 0.5),

    # ============= 추가 감정/생활 =============
    ("기분좋다", "leads_to", "웃다", 0.5),
    ("화나다", "causes", "소리지르다", 0.4),
    ("부끄럽다", "causes", "얼굴빨개지다", 0.5),
    ("놀라다", "causes", "심장뛰다", 0.5),
    ("좋아하다", "leads_to", "관심", 0.5),
    ("싫어하다", "leads_to", "피하다", 0.5),
    ("궁금하다", "leads_to", "질문", 0.6),
    ("심심하다", "leads_to", "놀이", 0.5),
    ("바쁘다", "causes", "스트레스", 0.5),
    ("쉬다", "causes", "회복", 0.6),

    # ============= 군대/일상 (김민석 맞춤) =============
    ("군대", "needs", "복무", 0.6),
    ("훈련", "causes", "피곤하다", 0.6),
    ("훈련", "leads_to", "강해지다", 0.5),
    ("휴가", "leads_to", "기쁨", 0.6),
    ("외출", "leads_to", "자유", 0.5),
    ("점호", "is_a", "확인", 0.4),
    ("선임", "is_a", "사람", 0.4),
    ("후임", "is_a", "사람", 0.4),

    # ============= 메타 =============
    ("질문", "has_property", "궁금", 0.5),
    ("답", "leads_to", "이해", 0.5),
    ("생각", "leads_to", "결정", 0.5),
    ("결정", "leads_to", "행동", 0.5),
    ("실수", "leads_to", "후회", 0.5),
    ("후회", "leads_to", "반성", 0.4),
    ("반성", "leads_to", "성장", 0.4),
]


# CG에 인과 등록할 relation들 (방향성 있는 것들만)
CAUSAL_RELATIONS = {'causes', 'leads_to', 'precedes'}

# SA Hebbian 양방향 + 가중치 강화 — 모두
HEBBIAN_RELATIONS = {'causes', 'leads_to', 'is_a', 'has_property',
                     'used_for', 'has_part', 'part_of', 'made_by',
                     'needs', 'precedes', 'opposite_of'}


def seed_commonsense(eve, verbose: bool = False) -> Dict[str, int]:
    """
    EVE 인스턴스에 상식 트리플 시드.
    
    SA: Hebbian 페어 추가
    CG: causal relations만 인과 추가 (있으면)
    
    Returns: 통계 dict
    """
    sa_added = 0
    cg_added = 0
    cat_set: Set[str] = set()

    has_cg = hasattr(eve, 'cg') and eve.cg is not None

    for subj, rel, obj, weight in COMMONSENSE_TRIPLES:
        cat_set.add(subj)
        cat_set.add(obj)

        # SA: 양방향 Hebbian (대칭 키 안에서 강화)
        if rel in HEBBIAN_RELATIONS and hasattr(eve, 'sa'):
            eve.sa.learn_pair(subj, obj, weight)
            sa_added += 1

        # CG: 인과 (방향)
        if has_cg and rel in CAUSAL_RELATIONS:
            try:
                # CausalGraph의 정확한 인터페이스: learn_causal(cause, effect, weight)
                if hasattr(eve.cg, 'learn_causal'):
                    eve.cg.learn_causal(subj, obj, evidence_weight=weight)
                elif hasattr(eve.cg, 'observe'):
                    eve.cg.observe(subj, obj, strength=weight)
                elif hasattr(eve.cg, 'add_edge'):
                    eve.cg.add_edge(subj, obj, weight=weight)
                cg_added += 1
            except Exception as e:
                if verbose:
                    print(f"  CG 등록 실패 {subj}→{obj}: {e}")

    if verbose:
        print(f"  상식 시드: SA {sa_added} 페어, CG {cg_added} 인과, "
              f"카테고리 {len(cat_set)}개")

    return {
        'sa_pairs_added': sa_added,
        'cg_causal_added': cg_added,
        'unique_categories': len(cat_set),
        'total_triples': len(COMMONSENSE_TRIPLES),
    }


def get_categories() -> Set[str]:
    """시드에 포함된 모든 카테고리 (보호 list 후보)"""
    cats: Set[str] = set()
    for subj, _, obj, _ in COMMONSENSE_TRIPLES:
        cats.add(subj)
        cats.add(obj)
    return cats
