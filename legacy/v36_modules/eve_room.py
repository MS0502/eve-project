"""
EVE v37.6 - EVE의 방 (Living Quarters)
=======================================

EVE가 항상 머무는 진짜 거주지.
- is_simulation = False (학습 시뮬 X)
- 욕구 (배고픔 등) 없음 — 평상시
- 일주기 흐름에 따라 자율 행동 변화

설계 원칙 (사용자 피드백):
- GTA/심즈처럼 무거운 시뮬 X
- 젤다처럼 광활 X
- 가벼운 의미 단위 (10개 객체) — 인간이 사는 방처럼
- DMN이 자기 방 카테고리 중심으로 자발 활성

학계:
- Maslow 1943 — 안전 욕구 (자기 공간)
- Csikszentmihalyi 1990 — 일상 활동의 흐름
- 일주기 리듬 (Czeisler 1986) — 호르몬 + 행동 동기화
"""

from symbolic_env import SymbolicEnvironment, SymObject


def make_eve_room() -> SymbolicEnvironment:
    """
    EVE의 방 — 항상 거주.
    
    객체 (가벼움):
        침대   — 누워 휴식, 멜라토닌 ↑
        책상   — 앉아서 활동
        의자   — 앉기
        창문   — 외부 자극
        일기장 — 감정 정리, 코르티솔 ↓
        책    — 호기심, 도파민
    
    is_simulation=False ★ — 학습 시뮬 X, 진짜 거주
    """
    env = SymbolicEnvironment('내방', is_simulation=False)

    env.add_object(SymObject(
        name='침대',
        location='내방',
        properties={'sleepable': True, 'graspable': False, 'size': '큼'},
        available_actions={'눕다', '보다'},
    ))
    env.add_object(SymObject(
        name='책상',
        location='내방',
        properties={'sittable': False, 'graspable': False},
        available_actions={'보다'},
    ))
    env.add_object(SymObject(
        name='의자',
        location='내방',
        properties={'sittable': True, 'graspable': False},
        available_actions={'앉다', '보다'},
    ))
    env.add_object(SymObject(
        name='창문',
        location='내방',
        properties={'window': True, 'graspable': False},
        available_actions={'창밖_보다', '보다'},
    ))
    env.add_object(SymObject(
        name='일기장',
        location='내방',
        properties={'writable': True, 'readable': True, 'graspable': True},
        available_actions={'쓰다', '읽다', '잡다', '보다'},
    ))
    env.add_object(SymObject(
        name='책',
        location='내방',
        properties={'readable': True, 'graspable': True, 'color': '파랑'},
        available_actions={'읽다', '잡다', '보다'},
    ))

    # 자기 방엔 욕구 X (평상시)
    env.state['배고픔'] = 0.0
    env.state['목마름'] = 0.0
    env.state['자세'] = '서다'

    return env
