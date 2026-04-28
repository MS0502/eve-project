"""
EVE v15 SYNAPTIC - 카테고리 시냅스 처음부터
=========================================
- add_belief 자체 패치 (등록 시 자동 카테고리)
- 신념 ↔ 신념 그래프 (NetworkX)
- 카테고리 동적 생성/삭제
- Hebbian 강화 (같이 활성 → 연결)
- EVE 자기 판단

사용:
exec(open('/content/drive/MyDrive/eve_v15_synaptic.py').read())
"""

import time, random
from collections import defaultdict, Counter

print("=" * 60)
print("EVE v15 SYNAPTIC - 카테고리 시냅스")
print("=" * 60)

# 1. 카테고리 분류 함수
def classify_belief(bid, statement=''):
    if any(k in bid for k in ['_명사', '_동사', '_형용사', 'punct_', 'space_', '어미_', '겉표지']):
        return 'lexical'
    if 'intent_' in bid:
        return 'intent'
    if any(k in bid for k in ['q_', 'p_', 'ctx_', 'meta_', 'talk_']):
        return 'meta'
    if '나는_' in bid or 'EVE이다' in bid:
        return 'identity'
    if any(k in bid for k in ['learned_', 'taught_']):
        return 'learned'
    if any(k in bid for k in ['_정의', '_특징', '_감정', '_색', '_맛', '_모양']):
        return 'semantic'
    if 'episode_' in bid:
        return 'episodic'
    return 'semantic'

eve._categories = defaultdict(set)
eve._belief_graph = defaultdict(set)
eve._category_meta = {}

# 2. 기존 4,970 분류
for bid, b in eve.beliefs.items():
    cat = classify_belief(bid, b.statement)
    eve._categories[cat].add(bid)

print(f"\n[기존 분류 결과]")
for cat, bids in sorted(eve._categories.items(), key=lambda x: -len(x[1])):
    print(f"  {cat}: {len(bids)}")

# 3. add_belief 패치 - 등록 시 자동 분류 + 그래프 연결
original_add = eve.add_belief

def synaptic_add(bid, statement, confidence=0.5, source='unknown'):
    result = original_add(bid, statement, confidence, source)
    cat = classify_belief(bid, statement)
    eve._categories[cat].add(bid)
    
    words = set(statement.split() + bid.split('_'))
    for other_bid in list(eve.beliefs.keys()):
        if other_bid == bid:
            continue
        other = eve.beliefs[other_bid]
        other_words = set(other.statement.split() + other_bid.split('_'))
        overlap = len(words & other_words)
        if overlap >= 2:
            eve._belief_graph[bid].add(other_bid)
            eve._belief_graph[other_bid].add(bid)
    
    return result

eve.add_belief = synaptic_add

# 4. 그래프 초기 구축 (기존 신념끼리)
print(f"\n[신념 그래프 구축]")
for bid in list(eve.beliefs.keys())[:500]:
    b = eve.beliefs[bid]
    words = set(b.statement.split() + bid.split('_'))
    for other_bid in list(eve.beliefs.keys()):
        if other_bid == bid or other_bid in eve._belief_graph[bid]:
            continue
        other = eve.beliefs[other_bid]
        other_words = set(other.statement.split() + other_bid.split('_'))
        if len(words & other_words) >= 2:
            eve._belief_graph[bid].add(other_bid)
            eve._belief_graph[other_bid].add(bid)

total_edges = sum(len(v) for v in eve._belief_graph.values()) // 2
print(f"  시냅스 연결: {total_edges}개")

# 5. 카테고리 인출
eve.recall = lambda cat, limit=10: [(bid, eve.beliefs[bid].statement) for bid in list(eve._categories.get(cat, set()))[:limit]]

# 6. 신념끼리 활성화 (Hebbian)
def activate_belief(bid):
    if bid not in eve.beliefs:
        return []
    activated = {bid}
    for connected in eve._belief_graph.get(bid, set()):
        activated.add(connected)
        b = eve.beliefs[connected]
        b.evidence_count = getattr(b, 'evidence_count', 1) + 1
    return list(activated)

eve.activate = activate_belief

# 7. EVE 자기 판단 - 카테고리 생성
def eve_create_category(name, criteria_func, parent=None):
    eve._category_meta[name] = {'criteria': criteria_func, 'parent': parent, 'created': time.time(), 'auto': True}
    matched = [bid for bid in eve.beliefs if criteria_func(bid, eve.beliefs[bid].statement)]
    eve._categories[name] = set(matched)
    return f"카테고리 '{name}' 생성. {len(matched)}개 신념 포함"

eve.create_category = eve_create_category

# 8. 카테고리 삭제 (자기 판단)
def eve_remove_category(name):
    if name in ['lexical', 'intent', 'meta', 'identity', 'learned', 'semantic', 'episodic']:
        return f"기본 카테고리 '{name}'은 삭제 X"
    if name in eve._categories:
        del eve._categories[name]
        if name in eve._category_meta:
            del eve._category_meta[name]
        return f"카테고리 '{name}' 삭제"
    return f"'{name}' 없음"

eve.remove_category = eve_remove_category

# 9. 시냅스 추론 - 그래프 따라 N단계
def synaptic_recall(seed_word, depth=2, limit=10):
    seed_beliefs = [bid for bid in eve.beliefs if seed_word in bid or seed_word in eve.beliefs[bid].statement]
    activated = set(seed_beliefs)
    for _ in range(depth):
        new_active = set()
        for bid in activated:
            new_active.update(eve._belief_graph.get(bid, set()))
        activated.update(new_active)
    return [(bid, eve.beliefs[bid].statement) for bid in list(activated)[:limit] if bid in eve.beliefs]

eve.synaptic_recall = synaptic_recall

# 10. 통계
eve.synaptic_stats = lambda: {
    'categories': len(eve._categories),
    'total_beliefs': len(eve.beliefs),
    'total_synapses': sum(len(v) for v in eve._belief_graph.values()) // 2,
    'auto_categories': len(eve._category_meta),
    'avg_synapses_per_belief': round(sum(len(v) for v in eve._belief_graph.values()) / max(1, len(eve.beliefs)), 2)
}

print("\n" + "=" * 60)
print("✅ EVE v15 SYNAPTIC 부착 완료!")
print("=" * 60)
print(f"\n통계:")
for k, v in eve.synaptic_stats().items():
    print(f"  {k}: {v}")

print(f"\n사용:")
print(f"  eve.recall('semantic', 5)       # 카테고리 인출")
print(f"  eve.activate('강아지_정의')      # 시냅스 활성화")
print(f"  eve.synaptic_recall('강아지', 2) # 그래프 추론")
print(f"  eve.create_category('동물', lambda bid, s: '동물' in s)")
print(f"  eve.remove_category('동물')")
print(f"  eve.add_belief(...)             # 자동 분류 + 그래프")
