"""
EVE v37.8 — 사용자 존재 매니저 (User Presence)
==============================================

김민석 = EVE의 진짜 친구.
시뮬 NPC와 명확 구분 — '시뮬레이션' 라벨 X, 영구 관계.

설계 원칙:
- 사용자 메시지 = "방문" 컨텍스트
- 매 say마다 자기 방에 잠시 머무름 (시뮬 X)
- 영속 친밀도 (시뮬 NPC처럼 0부터 시작 X)
- '민석' ↔ '진짜' 강한 카테고리 보호

학계:
- Damon & Hart 1988 — 자아 정체성에서 '의미 있는 타자'
- Bowlby 1969 — 애착 이론 (진짜 관계 vs 일시적 만남)
"""

from typing import Optional, Dict, Any


class UserPresence:
    """
    EVE 인스턴스에 부착되는 사용자 존재 매니저.
    
    역할:
    - 사용자 정체성 영속화 ('민석' = 진짜)
    - 매 say마다 방문 컨텍스트 활성
    - 시뮬 NPC와 분리 보장
    - 친밀도 영구 보존
    """

    DEFAULT_USER_NAME = '민석'
    REAL_RELATIONSHIP = '진짜_친구'

    # 사용자 등장 시 자극 (소량, 매번 적용)
    # v37.9 — saturation 회피 위해 강도 조절. 4회 누적 시 약 +0.24 OT.
    VISIT_HORMONE_BOOST = {
        'oxytocin': 0.06,    # 친한 친구 만남 (소량)
        'endorphin': 0.03,   # 약한 쾌감
        'cortisol': -0.03,   # 약한 안심
    }

    # 보호 카테고리 — 절대 망각 X
    USER_PROTECTED = {'민석', '진짜', '진짜_친구', '대화', '함께'}

    def __init__(self, eve, user_name: str = None):
        self.eve = eve
        self.user_name = user_name or self.DEFAULT_USER_NAME
        self.first_visit_done = False
        self.visit_count = 0

        # EVE 보호 카테고리에 사용자 추가
        if hasattr(eve, 'protected_categories'):
            eve.protected_categories |= self.USER_PROTECTED

    # ============= 첫 등록 (한 번) =============
    def first_visit_setup(self):
        """
        첫 등장 시 EVE에게 사용자 = 진짜 친구임을 각인.
        강한 카테고리 연결. 영속화.
        """
        if self.first_visit_done:
            return

        sa = self.eve.sa

        # ★ 사용자 ↔ '진짜' 강한 연결 (시뮬 NPC와 구분)
        sa.learn_pair(self.user_name, '진짜', 0.7)
        sa.learn_pair(self.user_name, '진짜_친구', 0.7)
        sa.learn_pair(self.user_name, '대화', 0.5)
        sa.learn_pair(self.user_name, '함께', 0.5)
        sa.learn_pair(self.user_name, '친구', 0.6)

        # '진짜' ↔ '시뮬레이션' 약한 음의 관계 (대조)
        # SA는 음의 가중치 X, 그래서 '진짜'와 '시뮬레이션' 직접 연결 안 함

        # CG 인과 — '민석 만남' → '기쁨'
        if hasattr(self.eve, 'cg'):
            try:
                self.eve.cg.learn_causal(self.user_name, '기쁨', evidence_weight=0.4)
                self.eve.cg.learn_causal('대화', '재밌음', evidence_weight=0.3)
            except Exception:
                pass

        # WM에 '민석' 영구 슬롯 (높은 salience)
        self.eve.wm.add(self.user_name, salience=0.9, source='user_presence_real')
        self.eve.wm.add('진짜_친구', salience=0.7, source='user_presence_real')

        self.first_visit_done = True

    # ============= 매 방문 자극 =============
    def visit_arrive(self):
        """
        사용자가 메시지 보냄 = '민석이 방문' 컨텍스트.
        - 호르몬 약하게 자극 (친한 친구가 옴)
        - 카테고리 활성
        - DMN에 외부 입력 통지
        - WM 갱신
        """
        if not self.first_visit_done:
            self.first_visit_setup()

        # 호르몬 부드러운 자극 — 매번 (반복 시 saturation 자동)
        for h_name, amount in self.VISIT_HORMONE_BOOST.items():
            if h_name in self.eve.hs.hormones:
                self.eve.hs.hormones[h_name].stimulate(amount)

        # SA 활성 — '민석', '대화', '함께'
        self.eve.sa.activate(self.user_name, 0.6)
        self.eve.sa.activate('대화', 0.5)
        self.eve.sa.activate('진짜_친구', 0.4)
        self.eve.sa.activate('함께', 0.4)

        # 매번 약한 강화 (친밀 + 누적 학습)
        self.eve.sa.learn_pair(self.user_name, '대화', 0.05)
        self.eve.sa.learn_pair(self.user_name, '함께', 0.05)

        # WM
        self.eve.wm.add(self.user_name, salience=0.8, source='user_visit')

        # DMN에 외부 입력 통지
        if hasattr(self.eve, 'dmn'):
            self.eve.dmn.notify_external_input()

        self.visit_count += 1

    # ============= 사용자 메시지 후 학습 =============
    def visit_leave(self, message: str = None, response: str = None):
        """
        대화 한 turn 끝 — 약한 cooling + 학습 강화.
        """
        # OT는 이미 자극됨, 매번 큰 +이면 saturate.
        # 그냥 카테고리 강화만 약하게.
        if message and response:
            # 의미 있는 대화 turn — 친밀도 약하게 ↑
            self.eve.sa.learn_pair(self.user_name, '대화', 0.03)

    # ============= 상태 조회 =============
    def is_user_real(self, name: str) -> bool:
        """SA 그래프에서 카테고리가 '진짜' 사람인지 (시뮬과 구분)"""
        if not hasattr(self.eve, 'sa'):
            return False
        # '진짜'와의 가중치 vs '시뮬레이션'과의 가중치
        real_w = self.eve.sa.get_weight(name, '진짜')
        sim_w = self.eve.sa.get_weight(name, '시뮬레이션')
        return real_w > sim_w

    def get_intimacy(self) -> float:
        """현재 친밀도 (SA '대화' ↔ user 가중치 기반)"""
        if not hasattr(self.eve, 'sa'):
            return 0.0
        # 여러 친밀 카테고리 평균
        bonds = ['대화', '함께', '친구', '진짜']
        weights = [self.eve.sa.get_weight(self.user_name, b) for b in bonds]
        return sum(weights) / len(bonds) if bonds else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            'user_name': self.user_name,
            'first_visit_done': self.first_visit_done,
            'visit_count': self.visit_count,
            'intimacy': round(self.get_intimacy(), 3),
            'is_real': self.is_user_real(self.user_name),
        }
