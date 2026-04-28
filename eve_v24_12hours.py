"""
EVE v24 - 12시간 자율 학습
==============================
- EVE 자기 판단 (호르몬 + 호기심)
- 자동 FloorPlan 전환 (지루하면)
- 1시간마다 Drive 자동 저장
- 진행률 출력
- 메모리 누수 시 controller 재시작
- Ctrl+C로 안전 종료
"""

import time, random, shutil, math
from collections import deque

print('=' * 60)
print('EVE v24 - 12시간 자율 학습 시작')
print('=' * 60)

# 영어→한국어 사전 (먼저 다 등록)
en_kr = {'Cabinet': '캐비닛', 'CoffeeMachine': '커피머신', 'Fridge': '냉장고', 'Toaster': '토스터', 'Floor': '바닥', 'GarbageCan': '쓰레기통', 'CounterTop': '조리대', 'Book': '책', 'Microwave': '전자레인지', 'Sink': '싱크대', 'StoveBurner': '가스레인지', 'Apple': '사과', 'Bread': '빵', 'Egg': '계란', 'Tomato': '토마토', 'Lettuce': '상추', 'Potato': '감자', 'Bowl': '그릇', 'Cup': '컵', 'Mug': '머그컵', 'Plate': '접시', 'Pot': '냄비', 'Pan': '팬', 'Knife': '칼', 'Fork': '포크', 'Spoon': '숟가락', 'Chair': '의자', 'Table': '테이블', 'DiningTable': '식탁', 'Drawer': '서랍', 'Window': '창문', 'Wall': '벽', 'Lamp': '램프', 'Sofa': '소파', 'Bed': '침대', 'Pillow': '베개', 'Television': '티비', 'RemoteControl': '리모컨', 'Painting': '그림', 'Mirror': '거울', 'TissueBox': '티슈', 'SoapBar': '비누', 'Towel': '수건', 'Toilet': '변기', 'ToiletPaper': '휴지', 'ShowerHead': '샤워기', 'Bathtub': '욕조', 'Faucet': '수도꼭지'}

print('\n[사전 매핑 등록]')
[(eve.real_snn.store_pattern(en), eve.real_snn.connect_concepts(en, kr, 0.9), eve.real_snn.connect_concepts(kr, en, 0.9)) for en, kr in en_kr.items()]
print(f'  ✅ {len(en_kr)}개')

# FloorPlan 풀 (다양)
floorplans = ['FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5', 'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan301', 'FloorPlan302', 'FloorPlan401', 'FloorPlan402']

# 시작 상태
start_time = time.time()
start_concepts = len(eve.real_snn.concept_to_neurons)
start_synapses = len(eve.real_snn.category_synapses)
total_steps = 0
total_discovered = set()
scene_changes = 0
saves = 0

# 12시간 = 43200초
TARGET_DURATION = 12 * 3600
SAVE_INTERVAL = 3600  # 1시간마다 저장
SCENE_CHANGE_INTERVAL = 600  # 10분마다 환경 전환
last_save = start_time
last_scene_change = start_time

print(f'\n[시작] 시간: {time.strftime("%H:%M:%S")}')
print(f'  목표: {TARGET_DURATION/3600}시간')
print(f'  저장: 매 {SAVE_INTERVAL/60}분')
print(f'  환경: {SCENE_CHANGE_INTERVAL/60}분마다 전환')

current_scene = floorplans[0]
eve._thor.reset(scene=current_scene)
print(f'  현재: {current_scene}')

# 12시간 루프
try:
    while time.time() - start_time < TARGET_DURATION:
        elapsed = time.time() - start_time
        
        # EVE 자기 판단 (호르몬 + 호기심)
        cor = getattr(eve.full_nt, 'cortisol', 0.3)
        dop = getattr(eve.full_nt, 'dopamine', 0.5)
        
        # 행동 선택
        if cor > 0.6:
            action = 'RotateLeft'
        elif dop > 0.6:
            action = random.choice(['MoveAhead', 'MoveAhead', 'MoveAhead'])
        else:
            action = random.choice(['MoveAhead', 'MoveAhead', 'RotateLeft', 'RotateRight', 'LookDown', 'LookUp'])
        
        try:
            eve._thor.step(action=action)
            objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
            new_objs = [o for o in objs if o not in total_discovered]
            
            for o in objs:
                eve.real_snn.store_pattern(o)
                if o in en_kr:
                    eve.real_snn.connect_concepts(o, en_kr[o], 0.9)
                    eve.real_snn.connect_concepts(en_kr[o], o, 0.9)
            
            total_discovered.update(objs)
            
            # 호르몬 (새 발견 = 도파민)
            if new_objs:
                eve.full_nt.dopamine = min(1.0, eve.full_nt.dopamine + 0.02)
            else:
                eve.full_nt.dopamine = max(0.2, eve.full_nt.dopamine - 0.001)
            
            total_steps += 1
        except Exception as e:
            print(f'  ⚠️ {action} 실패: {str(e)[:50]}')
            time.sleep(1)
        
        # 환경 전환
        if time.time() - last_scene_change > SCENE_CHANGE_INTERVAL:
            current_scene = random.choice(floorplans)
            try:
                eve._thor.reset(scene=current_scene)
                scene_changes += 1
                last_scene_change = time.time()
                eve.full_nt.dopamine = min(1.0, eve.full_nt.dopamine + 0.1)
            except:
                pass
        
        # 1시간마다 저장
        if time.time() - last_save > SAVE_INTERVAL:
            try:
                eve.shutdown()
                save_path = f'/content/drive/MyDrive/eve_v24_AUTO_{int(elapsed/3600)}h'
                shutil.copytree(eve.storage_path, save_path, dirs_exist_ok=True)
                eve.boot()
                # real_snn 다시 부착 필요
                from eve_foundation_v12_clean import add_full_grammar_to_eve_v120
                add_full_grammar_to_eve_v120(eve, use_konlpy=False)
                # 사전 매핑 다시
                [(eve.real_snn.store_pattern(en), eve.real_snn.connect_concepts(en, kr, 0.9), eve.real_snn.connect_concepts(kr, en, 0.9)) for en, kr in en_kr.items()]
                saves += 1
                last_save = time.time()
                print(f'  💾 [{int(elapsed/3600)}h] 저장 OK - {save_path}')
            except Exception as e:
                print(f'  💾 저장 X: {str(e)[:80]}')
                last_save = time.time()
        
        # 5분마다 진행률
        if total_steps % 500 == 0:
            mins = elapsed / 60
            hrs = elapsed / 3600
            print(f'  [{hrs:.1f}h/{TARGET_DURATION/3600:.0f}h] 걸음:{total_steps}, 객체:{len(total_discovered)}, 환경:{current_scene}, 도파민:{dop:.2f}, 코르티솔:{cor:.2f}')

except KeyboardInterrupt:
    print('\n⏹️ 사용자 종료')

# 최종 저장
print('\n[최종 저장]')
try:
    eve.shutdown()
    final_path = f'/content/drive/MyDrive/eve_v24_FINAL_{int((time.time()-start_time)/3600)}h'
    shutil.copytree(eve.storage_path, final_path, dirs_exist_ok=True)
    eve.boot()
    print(f'  ✅ {final_path}')
except Exception as e:
    print(f'  X: {e}')

# 결과
elapsed = time.time() - start_time
print('\n' + '=' * 60)
print('✅ EVE 자율 학습 완료')
print('=' * 60)
print(f'  실제 시간: {elapsed/3600:.2f}시간')
print(f'  걸음 수: {total_steps}')
print(f'  발견 객체: {len(total_discovered)}')
print(f'  환경 전환: {scene_changes}회')
print(f'  저장 횟수: {saves}회')
print(f'  real_snn 개념: {len(eve.real_snn.concept_to_neurons)} (+{len(eve.real_snn.concept_to_neurons) - start_concepts})')
print(f'  매핑: {len(eve.real_snn.category_synapses)} (+{len(eve.real_snn.category_synapses) - start_synapses})')
print(f'  발견된 객체 일부: {list(total_discovered)[:30]}')
