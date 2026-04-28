"""
EVE v23 AUTONOMOUS - 대규모 자동 학습 + 자동 저장
=====================================================
- AI2-THOR 자율 탐험 (50걸음)
- 발견 객체 자동 영어↔한국어 매핑
- 카테고리 자동 학습
- 50걸음마다 Drive 자동 저장
- multi-hop 검증
"""

import time, random, shutil

print('=' * 60)
print('EVE v23 AUTONOMOUS - 대규모 자동 학습')
print('=' * 60)

# 영어→한국어 사전 (확장 100+)
en_kr = {'Cabinet': '캐비닛', 'CoffeeMachine': '커피머신', 'Fridge': '냉장고', 'Toaster': '토스터', 'Floor': '바닥', 'GarbageCan': '쓰레기통', 'CounterTop': '조리대', 'Book': '책', 'Microwave': '전자레인지', 'Sink': '싱크대', 'StoveBurner': '가스레인지', 'StoveKnob': '가스레인지손잡이', 'Apple': '사과', 'Bread': '빵', 'Egg': '계란', 'Tomato': '토마토', 'Lettuce': '상추', 'Potato': '감자', 'Bowl': '그릇', 'Cup': '컵', 'Mug': '머그컵', 'Plate': '접시', 'Pot': '냄비', 'Pan': '팬', 'Knife': '칼', 'Fork': '포크', 'Spoon': '숟가락', 'ButterKnife': '버터칼', 'Chair': '의자', 'Table': '테이블', 'DiningTable': '식탁', 'Drawer': '서랍', 'Window': '창문', 'Wall': '벽', 'Lamp': '램프', 'LightSwitch': '전등스위치', 'Sofa': '소파', 'Bed': '침대', 'Pillow': '베개', 'Television': '티비', 'RemoteControl': '리모컨', 'Painting': '그림', 'Mirror': '거울', 'TissueBox': '티슈', 'SoapBar': '비누', 'Towel': '수건', 'Toilet': '변기', 'ToiletPaper': '휴지', 'ShowerHead': '샤워기', 'Bathtub': '욕조', 'Faucet': '수도꼭지', 'PaperTowelRoll': '키친타올', 'SaltShaker': '소금통', 'PepperShaker': '후추통', 'WineBottle': '와인병', 'Bottle': '병', 'GarbageBag': '쓰레기봉투', 'DishSponge': '수세미', 'SprayBottle': '분무기', 'Kettle': '주전자'}

# 카테고리 매핑
cat = {'Cabinet': '가구', 'Drawer': '가구', 'Chair': '가구', 'Table': '가구', 'DiningTable': '가구', 'Sofa': '가구', 'Bed': '가구', 'Fridge': '부엌가전', 'CoffeeMachine': '부엌가전', 'Toaster': '부엌가전', 'Microwave': '부엌가전', 'Kettle': '부엌가전', 'Apple': '과일', 'Tomato': '과일', 'Bread': '음식', 'Egg': '음식', 'Lettuce': '야채', 'Potato': '야채', 'Bowl': '식기', 'Cup': '식기', 'Mug': '식기', 'Plate': '식기', 'Pot': '식기', 'Pan': '식기', 'Knife': '식기', 'Fork': '식기', 'Spoon': '식기', 'ButterKnife': '식기', 'Television': '가전', 'RemoteControl': '가전', 'Lamp': '가전', 'Toilet': '욕실', 'Bathtub': '욕실', 'ShowerHead': '욕실', 'Faucet': '욕실', 'SoapBar': '욕실', 'Towel': '욕실', 'ToiletPaper': '욕실', 'Pillow': '침구', 'Bed': '침구', 'Window': '건축', 'Wall': '건축', 'Floor': '건축', '부엌가전': '가전', '가구': '사물', '식기': '사물', '가전': '사물', '욕실': '공간', '침구': '사물', '건축': '사물'}

# 시작 상태
auto_start_objects = len(eve.real_snn.concept_to_neurons)
auto_start_synapses = len(eve.real_snn.category_synapses)
print(f'\n[시작] 개념: {auto_start_objects}, 매핑: {auto_start_synapses}')

# 1. 자율 탐험 50걸음 (FloorPlan1 부엌)
print('\n[1] 자율 탐험 50걸음 (부엌)')
discovered = set()
for i in range(50):
    eve._thor.step(action=random.choice(['MoveAhead', 'MoveAhead', 'RotateLeft', 'RotateRight', 'LookDown', 'LookUp']))
    objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
    [eve.real_snn.store_pattern(o) for o in objs]
    discovered.update(objs)
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/50 - 발견: {len(discovered)}개')

print(f'  ✅ 부엌 객체: {len(discovered)}개')

# 2. 거실 탐험 (FloorPlan201)
print('\n[2] 거실 탐험 30걸음')
try:
    eve._thor.reset(scene='FloorPlan201')
    for i in range(30):
        eve._thor.step(action=random.choice(['MoveAhead', 'MoveAhead', 'RotateLeft', 'RotateRight']))
        objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
        [eve.real_snn.store_pattern(o) for o in objs]
        discovered.update(objs)
    print(f'  ✅ 누적: {len(discovered)}개')
except Exception as e:
    print(f'  거실 X: {e}')

# 3. 욕실 탐험 (FloorPlan401)
print('\n[3] 욕실 탐험 30걸음')
try:
    eve._thor.reset(scene='FloorPlan401')
    for i in range(30):
        eve._thor.step(action=random.choice(['MoveAhead', 'MoveAhead', 'RotateLeft', 'RotateRight']))
        objs = [o['objectType'] for o in eve._thor.last_event.metadata.get('objects', []) if o.get('visible', False)]
        [eve.real_snn.store_pattern(o) for o in objs]
        discovered.update(objs)
    print(f'  ✅ 누적: {len(discovered)}개')
except Exception as e:
    print(f'  욕실 X: {e}')

# 4. 영어→한국어 매핑 자동
print('\n[4] 영어→한국어 매핑')
mapped = [(eve.real_snn.connect_concepts(en, kr, 0.9), eve.real_snn.connect_concepts(kr, en, 0.9), eve.real_snn.store_pattern(kr)) for en, kr in en_kr.items() if en in eve.real_snn.concept_to_neurons]
print(f'  ✅ {len(mapped)}개 매핑')

# 5. 카테고리 매핑 자동
print('\n[5] 카테고리 학습')
cat_learned = [eve.learn_pair(en if en in eve.real_snn.concept_to_neurons else en_kr.get(en, en), c) for en, c in cat.items() if en in eve.real_snn.concept_to_neurons or en in en_kr]
print(f'  ✅ {len(cat_learned)}개 카테고리 매핑')

# 6. multi-hop 검증
print('\n[6] multi-hop 추론 검증')
print(f'  Fridge: {eve.true_speak("Fridge")}')
print(f'  냉장고: {eve.true_speak("냉장고")}')
print(f'  Apple: {eve.true_speak("Apple")}')
print(f'  사과: {eve.true_speak("사과")}')

# 7. 자동 저장
print('\n[7] 자동 저장')
try:
    eve.shutdown()
    save_path = '/content/drive/MyDrive/eve_v23_AUTO_28th'
    shutil.copytree(eve.storage_path, save_path, dirs_exist_ok=True)
    print(f'  ✅ Drive: {save_path}')
    eve.boot()
except Exception as e:
    print(f'  저장 X: {e}')

# 결과
print('\n' + '=' * 60)
print('✅ EVE v23 AUTO 완료')
print('=' * 60)
print(f'  발견 객체: {len(discovered)}개')
print(f'  real_snn 개념: {len(eve.real_snn.concept_to_neurons)} (시작 {auto_start_objects}, +{len(eve.real_snn.concept_to_neurons) - auto_start_objects})')
print(f'  매핑: {len(eve.real_snn.category_synapses)} (시작 {auto_start_synapses}, +{len(eve.real_snn.category_synapses) - auto_start_synapses})')
print(f'  Drive 저장: eve_v23_AUTO_28th')
