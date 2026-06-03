"""
EVE v36 - Persistence (체크포인트 save/load)
================================================

P0-1 단점 해결: "매번 처음부터" → "1년 누적"

설계 철학:
- pickle 사용 (Python 표준, numpy 호환)
- 모든 모듈의 핵심 상태 직렬화 (호르몬 26 상태 포함)
- 버전 체크 (구버전 호환 fallback)
- 무손실: save → load 후 행동 동일

저장 대상 (최소 충분 set):
  HS  - 26 호르몬 level/baseline/sim_hour/active set
  SA  - activations, weights, neighbors, history, group cache
  WM  - slots, listeners 메타 (callable 제외)
  EM  - episodes (전체)
  SD  - baseline, prediction history
  NL  - beliefs, belief_index, discovered, refusal_count
  CG  - graph (edges, weights)
  AI  - hormone-outcome weights, history
  Agency / Goal / Norm / Allostatic - 학습된 패턴
  meta - tick_count, time, dialogue_history, version

저장 안 함 (직렬화 어렵거나 불필요):
  callable listeners (broadcast 시 다시 등록)
  scratch buffers (deque maxlen)
  Verbose flags

학계: Continual Learning, Catastrophic Forgetting (McCloskey 1989)
       → 체크포인트는 인공 신경망의 'memory consolidation'에 해당
"""

import os
import pickle
import gzip
import time
from typing import Any, Dict, Optional, Set, List
from collections import deque

# 호환성 경계 — 이전 버전과 못 읽으면 명시 에러
EVE_CHECKPOINT_VERSION = "v36.0"
EVE_CHECKPOINT_MAGIC = b"EVE_CKPT_V36\x00"


class Persistence:
    """
    EVE 체크포인트 매니저.
    
    EVE 인스턴스에 부착해서 사용:
        eve.save("/path/eve.ckpt")
        eve = EVE_TierAB.load("/path/eve.ckpt")
    """

    # 모듈별 직렬화/복원 매핑
    # (module_attr, state_keys) - 단순 attribute copy 가능한 keys
    SIMPLE_STATE_KEYS = {
        'sa': [
            'activations', 'weights', 'neighbors', 'time',
            'activation_history', 'hormone_modulation',
            'base_threshold', 'decay_rate', 'spread_factor',
        ],
        'wm': [
            'slots', 'current_capacity', 'base_capacity',
            'broadcast_count', 'last_focus',
        ],
        'em': [
            'episodes', 'next_id', 'consolidation_threshold',
            'encoded_count', 'recalled_count', 'reminded_count',
            'forgot_count',
        ],
        'sd': [
            'baseline', 'baseline_init', 'doubt_history',
            'prediction_history', 'pending_predictions',
            'fp_count', 'fn_count', 'evaluation_count',
            'last_doubt_level', 'last_action',
        ],
        'nl': [
            'beliefs', 'belief_index', 'discovered_count',
            'refusal_count',
        ],
        'cg': [
            'graph', 'reverse_graph', 'edge_weights',
            'observation_count',
        ],
        'ai': [
            'hormone_weights', 'observation_history',
            'prediction_history', 'pending_predictions',
            'update_count',
        ],
        'ag': [
            'recent_actions', 'agency_history',
        ],
        'mc': [
            'recent_thoughts', 'thought_count',
        ],
        'gm': [
            'goals', 'next_goal_id', 'completed_count',
            'abandoned_count',
        ],
        'cf': [
            'recent_counterfactuals',
        ],
        'tm': [
            'recall_count',
        ],
        'er': [
            'regulation_history', 'strategy_effects',
        ],
        'an': [
            'analogy_count',
        ],
        'wm_world': [
            'overview_count',
        ],
        'ds': [
            'somatic_state', 'baseline',
        ],
    }

    # 호르몬 시스템은 별도 처리 (26 호르몬 nested)
    HORMONE_PER_KEYS = ['level', 'baseline', 'last_stimulus_time']

    @staticmethod
    def collect_state(eve) -> Dict[str, Any]:
        """
        EVE 인스턴스 → 상태 dict.
        모든 모듈 attribute 직렬화 가능한 형태로.
        """
        state: Dict[str, Any] = {
            'version': EVE_CHECKPOINT_VERSION,
            'saved_at': time.time(),
        }

        # 메타 ─────────────────────
        state['meta'] = {
            'time': getattr(eve, 'time', 0.0),
            'tick_count': getattr(eve, 'tick_count', 0),
            'broadcast_count': getattr(eve, 'broadcast_count', 0),
            'forget_runs': getattr(eve, 'forget_runs', 0),
            'protected_categories': set(getattr(eve, 'protected_categories', set())),
            'dialogue_history': list(getattr(eve, 'dialogue_history', [])),
            'event_log': list(getattr(eve, 'event_log', [])[-200:]),  # 최근 200
        }

        # HormoneSystem ────────────
        if hasattr(eve, 'hs') and eve.hs is not None:
            hormones_state: Dict[str, Dict] = {}
            for name, h in eve.hs.hormones.items():
                hormones_state[name] = {
                    k: getattr(h, k, None) for k in Persistence.HORMONE_PER_KEYS
                    if hasattr(h, k)
                }
            state['hs'] = {
                'hormones': hormones_state,
                'sim_hour': getattr(eve.hs, 'sim_hour', 0.0),
                'active_hormones': set(getattr(eve.hs, 'active_hormones', set())),
                'phase': getattr(eve.hs, 'phase', 4),
                'developmental_stage': getattr(eve.hs, 'developmental_stage', 'adult'),
            }

        # DMN ─────────────────────
        if hasattr(eve, 'dmn') and eve.dmn is not None:
            state['dmn'] = {
                'time': eve.dmn.time,
                'last_external_input_time': eve.dmn.last_external_input_time,
                'wandering_history': list(eve.dmn.wandering_history),
                'spontaneous_count': eve.dmn.spontaneous_count,
                'self_intent_count': eve.dmn.self_intent_count,
                'current_mode': eve.dmn.current_mode,
                'current_intent': eve.dmn.current_intent,
            }

        # 단순 모듈들 ──────────────
        for attr, keys in Persistence.SIMPLE_STATE_KEYS.items():
            mod = getattr(eve, attr, None)
            if mod is None:
                continue
            mod_state: Dict[str, Any] = {}
            for k in keys:
                if not hasattr(mod, k):
                    continue
                v = getattr(mod, k)
                # deque → list로 변환 (pickle 안전)
                if isinstance(v, deque):
                    v = list(v)
                # set은 그대로 (pickle 가능)
                # defaultdict는 dict로 변환
                from collections import defaultdict
                if isinstance(v, defaultdict):
                    # neighbors처럼 defaultdict(set)이면 dict로
                    v = dict(v)
                mod_state[k] = v
            state[attr] = mod_state

        # NormInternalization (있으면)
        if hasattr(eve, 'norm') and eve.norm is not None:
            state['norm'] = {
                'norms': dict(getattr(eve.norm, 'norms', {})),
                'enforcement_count': getattr(eve.norm, 'enforcement_count', 0),
            }

        # AllostaticLearning (있으면)
        if hasattr(eve, 'allo') and eve.allo is not None:
            state['allo'] = {
                'cocktails': list(getattr(eve.allo, 'cocktails', [])),
                'experience_count': getattr(eve.allo, 'experience_count', 0),
            }

        # ★ v37.12 — UserPresence (사용자 정체성 영속)
        if hasattr(eve, 'user_presence') and eve.user_presence is not None:
            state['user_presence'] = {
                'user_name': eve.user_presence.user_name,
                'first_visit_done': eve.user_presence.first_visit_done,
                'visit_count': eve.user_presence.visit_count,
            }

        # ★ v37.13 — IntegratedSelf history (메모리/대시보드용)
        if hasattr(eve, 'self_view') and eve.self_view is not None:
            # snapshot history는 너무 커서 최근 20개만
            state['self_view'] = {
                'history_count': len(eve.self_view.history),
                # 세부 history는 EM/log로 충분, 영속 X
            }

        # ★ v37.12 — env_adapter 메타 (현재 환경 + 외출 history)
        if hasattr(eve, '_env_adapter') and eve._env_adapter is not None:
            adapter = eve._env_adapter
            cur_env = adapter.current_env
            state['env_adapter'] = {
                # 환경 자체는 재생성 가능 — 이름만 저장
                'current_env_name': cur_env.name if cur_env else None,
                'current_env_is_sim': getattr(cur_env, 'is_simulation', None) if cur_env else None,
                'enter_time': adapter.enter_time,
                'env_dwell_total': dict(adapter.env_dwell_total),
                'outing_history': list(adapter.outing_history),
                'episode_count': adapter.episode_count,
                'action_count': adapter.action_count,
                'success_count': adapter.success_count,
            }

        return state

    @staticmethod
    def apply_state(eve, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        상태 dict → EVE 인스턴스에 복원.
        EVE는 이미 __init__으로 모듈 인스턴스화된 상태여야 함.
        """
        report = {'restored': [], 'skipped': []}

        version = state.get('version', 'unknown')
        if version != EVE_CHECKPOINT_VERSION:
            # 호환 시도 (구버전이면 경고만)
            report['version_mismatch'] = (version, EVE_CHECKPOINT_VERSION)

        # 메타 ─────────────────────
        meta = state.get('meta', {})
        if meta:
            eve.time = meta.get('time', 0.0)
            eve.tick_count = meta.get('tick_count', 0)
            eve.broadcast_count = meta.get('broadcast_count', 0)
            eve.forget_runs = meta.get('forget_runs', 0)
            eve.protected_categories = set(meta.get('protected_categories', set()))
            eve.dialogue_history = list(meta.get('dialogue_history', []))
            eve.event_log = list(meta.get('event_log', []))
            report['restored'].append('meta')

        # HormoneSystem ────────────
        if 'hs' in state and hasattr(eve, 'hs'):
            hs_state = state['hs']
            eve.hs.sim_hour = hs_state.get('sim_hour', 0.0)
            saved_active = hs_state.get('active_hormones')
            if saved_active:
                eve.hs.active_hormones = set(saved_active)
            for name, h_state in hs_state.get('hormones', {}).items():
                if name in eve.hs.hormones:
                    h = eve.hs.hormones[name]
                    for k, v in h_state.items():
                        if v is not None and hasattr(h, k):
                            setattr(h, k, v)
            report['restored'].append('hs')

        # DMN ─────────────────────
        if 'dmn' in state and hasattr(eve, 'dmn'):
            dmn_state = state['dmn']
            for k, v in dmn_state.items():
                if hasattr(eve.dmn, k):
                    setattr(eve.dmn, k, v)
            report['restored'].append('dmn')

        # 단순 모듈들 ──────────────
        for attr, keys in Persistence.SIMPLE_STATE_KEYS.items():
            if attr not in state:
                continue
            mod = getattr(eve, attr, None)
            if mod is None:
                report['skipped'].append(attr)
                continue
            mod_state = state[attr]
            for k, v in mod_state.items():
                if k in keys and hasattr(mod, k):
                    # neighbors처럼 defaultdict라면 보존
                    cur = getattr(mod, k)
                    from collections import defaultdict
                    if isinstance(cur, defaultdict) and isinstance(v, dict):
                        # 기존 default_factory 살리기
                        cur.clear()
                        for kk, vv in v.items():
                            cur[kk] = vv
                    elif isinstance(cur, deque) and isinstance(v, list):
                        cur.clear()
                        for item in v:
                            cur.append(item)
                    else:
                        setattr(mod, k, v)
            report['restored'].append(attr)

        # NormInternalization
        if 'norm' in state and hasattr(eve, 'norm') and eve.norm is not None:
            norm_state = state['norm']
            if hasattr(eve.norm, 'norms'):
                eve.norm.norms = dict(norm_state.get('norms', {}))
            if hasattr(eve.norm, 'enforcement_count'):
                eve.norm.enforcement_count = norm_state.get('enforcement_count', 0)
            report['restored'].append('norm')

        # Allostatic
        if 'allo' in state and hasattr(eve, 'allo') and eve.allo is not None:
            allo_state = state['allo']
            if hasattr(eve.allo, 'cocktails'):
                eve.allo.cocktails = list(allo_state.get('cocktails', []))
            if hasattr(eve.allo, 'experience_count'):
                eve.allo.experience_count = allo_state.get('experience_count', 0)
            report['restored'].append('allo')

        # ★ v37.12 — UserPresence 복원
        if 'user_presence' in state and hasattr(eve, 'user_presence'):
            up_state = state['user_presence']
            eve.user_presence.user_name = up_state.get('user_name', '민석')
            eve.user_presence.first_visit_done = up_state.get('first_visit_done', False)
            eve.user_presence.visit_count = up_state.get('visit_count', 0)
            report['restored'].append('user_presence')

        # ★ v37.12 — env_adapter 복원 (환경 재생성)
        if 'env_adapter' in state:
            ea_state = state['env_adapter']
            # adapter 없으면 lazy 생성
            if not hasattr(eve, '_env_adapter') or eve._env_adapter is None:
                try:
                    from env_adapter import EnvironmentAdapter
                    eve._env_adapter = EnvironmentAdapter(eve)
                except Exception:
                    eve._env_adapter = None
            adapter = getattr(eve, '_env_adapter', None)
            if adapter is None:
                report['skipped'].append('env_adapter')
            else:
                adapter.enter_time = ea_state.get('enter_time')
                adapter.env_dwell_total = dict(ea_state.get('env_dwell_total', {}))
                adapter.outing_history = list(ea_state.get('outing_history', []))
                adapter.episode_count = ea_state.get('episode_count', 0)
                adapter.action_count = ea_state.get('action_count', 0)
                adapter.success_count = ea_state.get('success_count', 0)

                saved_env_name = ea_state.get('current_env_name')
                if saved_env_name:
                    new_env = None
                    try:
                        if saved_env_name == '내방':
                            from eve_room import make_eve_room
                            new_env = make_eve_room()
                        elif saved_env_name == '거실':
                            from social_env import make_social_living_room
                            new_env = make_social_living_room()
                        elif saved_env_name == '주방':
                            from symbolic_env import make_kitchen_env
                            new_env = make_kitchen_env()
                    except Exception:
                        pass
                    if new_env is not None:
                        adapter.current_env = new_env
                report['restored'].append('env_adapter')

        return report

    # ============= 파일 입출력 =============
    @staticmethod
    def save(eve, path: str, compress: bool = True) -> Dict[str, Any]:
        """
        EVE → 파일 (.ckpt 또는 .ckpt.gz).
        
        Returns:
            메타 정보 (size, modules_saved 등)
        """
        state = Persistence.collect_state(eve)
        # magic header + pickle
        data = EVE_CHECKPOINT_MAGIC + pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        
        if compress:
            data = gzip.compress(data, compresslevel=6)
            if not path.endswith('.gz'):
                path = path + '.gz'

        # atomic write (tmp → rename)
        tmp_path = path + '.tmp'
        with open(tmp_path, 'wb') as f:
            f.write(data)
        os.replace(tmp_path, path)

        return {
            'path': path,
            'size_bytes': len(data),
            'compressed': compress,
            'modules_saved': len([k for k in state.keys()
                                if k not in ('version', 'saved_at', 'meta')]),
            'tick_count': state['meta'].get('tick_count', 0),
            'sim_time': state['meta'].get('time', 0.0),
            'dialogue_turns': len(state['meta'].get('dialogue_history', [])),
        }

    @staticmethod
    def load_state(path: str) -> Dict[str, Any]:
        """파일 → state dict (인스턴스화 안 함)"""
        if not os.path.exists(path):
            # .gz 자동 추가
            if os.path.exists(path + '.gz'):
                path = path + '.gz'
            else:
                raise FileNotFoundError(f"체크포인트 없음: {path}")

        with open(path, 'rb') as f:
            data = f.read()

        # gzip 자동 감지
        if data[:2] == b'\x1f\x8b':
            data = gzip.decompress(data)

        # magic 체크
        if not data.startswith(EVE_CHECKPOINT_MAGIC):
            raise ValueError(
                f"체크포인트 magic header 불일치 (잘못된 파일 또는 손상)"
            )
        data = data[len(EVE_CHECKPOINT_MAGIC):]
        state = pickle.loads(data)
        return state

    @staticmethod
    def load_into(eve, path: str) -> Dict[str, Any]:
        """파일 → 기존 EVE 인스턴스 복원"""
        state = Persistence.load_state(path)
        report = Persistence.apply_state(eve, state)
        report['version'] = state.get('version')
        report['saved_at'] = state.get('saved_at')
        return report


# ============= EVE 클래스에 부착할 메소드 =============
def attach_persistence(eve_class):
    """
    EVE 클래스에 save/load 메소드 동적 부착.
    
    사용:
        from persistence import attach_persistence
        from eve_main_abc import EVE_TierAB
        attach_persistence(EVE_TierAB)
        
        eve = EVE_TierAB()
        eve.save("/path/eve.ckpt")
        
        eve2 = EVE_TierAB()
        eve2.load("/path/eve.ckpt")
    """
    def save(self, path: str, compress: bool = True) -> Dict[str, Any]:
        """체크포인트 파일 저장"""
        return Persistence.save(self, path, compress=compress)

    def load(self, path: str) -> Dict[str, Any]:
        """체크포인트 파일에서 상태 복원 (in-place)"""
        return Persistence.load_into(self, path)

    @classmethod
    def from_checkpoint(cls, path: str, **init_kwargs) -> 'EVE_TierAB':
        """새 EVE 인스턴스 생성 + 체크포인트 자동 로드"""
        eve = cls(**init_kwargs)
        eve.load(path)
        return eve

    eve_class.save = save
    eve_class.load = load
    eve_class.from_checkpoint = from_checkpoint
    return eve_class
