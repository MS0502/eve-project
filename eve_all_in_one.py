"""
EVE 통합 실행 - 한 파일로 다 됨
=====================================
들여쓰기 오류 X, 한 번에 다 됨

사용법 (Colab):

# 1) 설치 (한 번만)
!apt-get install -y openjdk-11-jdk
!pip install konlpy

# 2) Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 3) 실행 (이거 한 줄!)
exec(open('/content/drive/MyDrive/eve_all_in_one.py').read())

= 끝!
"""

import os
import sys
import json

print("=" * 60)
print("EVE 통합 실행 - All-in-One")
print("=" * 60)

# ============================================
# Step 1: EVE 기본 코드 로드
# ============================================
print("\n[1/5] EVE 기본 로드...")

eve_foundation_path = '/content/drive/MyDrive/eve_foundation_v12_clean.py'

if os.path.exists(eve_foundation_path):
    exec(open(eve_foundation_path).read())
    print("  ✅ eve_foundation_v12_clean.py 로드")
else:
    print(f"  ❌ {eve_foundation_path} 없음!")
    print("     Drive에 EVE 기본 파일 업로드 필요")
    sys.exit(1)

# ============================================
# Step 2: EVE 인스턴스 생성
# ============================================
print("\n[2/5] EVE 인스턴스 생성...")

eve = EmbodiedEVE(storage_path='/content/eve_state')
eve.boot()
print(f"  ✅ EVE 부팅: {len(eve.concept_network.concepts)}개 개념")

# ============================================
# Step 3: 19,440 학습 데이터 로드 (있으면)
# ============================================
print("\n[3/5] 19,440 학습 데이터 로드...")

learning_data_path = '/content/drive/MyDrive/eve_v12_full_final.json'

if os.path.exists(learning_data_path):
    with open(learning_data_path, 'r') as f:
        data = json.load(f)
    
    loaded = 0
    for word, info in data.items():
        cats = info.get('categories', ['한국어'])
        props = info.get('properties', [])
        try:
            eve.teach_concept(word, categories=cats, properties=props)
            loaded += 1
        except:
            pass
    
    print(f"  ✅ {loaded}개 추가됨")
    print(f"  EVE 총: {len(eve.concept_network.concepts)}개")
else:
    print(f"  ⚠️ {learning_data_path} 없음 (skip)")

# ============================================
# Step 4: 한국어 대규모 학습 코드 로드
# ============================================
print("\n[4/5] 한국어 대규모 학습 코드 로드...")

massive_path = '/content/drive/MyDrive/eve_massive_korean.py'

if os.path.exists(massive_path):
    exec(open(massive_path).read())
    print("  ✅ massive_train 함수 로드")
else:
    print(f"  ❌ {massive_path} 없음!")
    print("     Drive에 eve_massive_korean.py 업로드 필요")
    sys.exit(1)

# ============================================
# Step 5: EVE 변수 사용 가능 확인
# ============================================
print("\n[5/5] 준비 완료!")
print(f"  eve 변수: ✅ ({len(eve.concept_network.concepts)}개 개념)")
print(f"  massive_train 함수: ✅")

print("\n" + "=" * 60)
print("이제 다음 셀에서 실행:")
print("=" * 60)
print("""
# 빠른 테스트 (30분)
massive_train(eve, level='quick')

# 중간 (1-2시간)
massive_train(eve, level='medium')

# 풀 (3-4시간)
massive_train(eve, level='full')
""")
