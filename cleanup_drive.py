"""
Drive 정리 스크립트 - Colab에서 실행
======================================

용도: 옛 EVE 버전 폴더들을 _archive_legacy/ 로 이동.
      삭제 X (안전), 이동만.

사용:
  1. dry_run=True로 먼저 돌려서 뭐가 옮겨지는지 확인
  2. 결과 OK면 dry_run=False로 다시 돌려서 실제 이동

옮길 폴더 (16개):
  - eve_v14_*, eve_v19_*, eve_v21_*, eve_v23_*, eve_v24_*
  - eve_v25_*, eve_v26_*, eve_v27_*, eve_v28_*
  - eve_v31, eve_v31_1
  - eve_state_*, eve_data_v4_backup, eve_code_backups, eve-project

유지할 폴더 (현재 작업용):
  - eve_v32 (메인)
  - eve_data (beliefs.json)
  - _archive_legacy (옮긴 것들 안전 보관소)
"""

import os
import shutil
from pathlib import Path

# ============= 설정 =============
DRIVE_ROOT = '/content/drive/MyDrive'
ARCHIVE_NAME = '_archive_legacy'

# 보관할 폴더 (정리에서 제외)
KEEP = {
    'eve_v32',           # 현재 메인 작업
    'eve_data',          # beliefs.json 들어있음
    ARCHIVE_NAME,        # archive 자체
}

# 이동할 폴더 패턴 (이 prefix로 시작하는 eve_* 폴더)
ARCHIVE_PATTERNS = [
    'eve_v14_', 'eve_v19_', 'eve_v21_', 'eve_v23_', 'eve_v24_',
    'eve_v25_', 'eve_v26_', 'eve_v27_', 'eve_v28_',
    'eve_v31',           # eve_v31, eve_v31_1
    'eve_state_',
    'eve_data_v4_backup',
    'eve_code_backups',
    'eve-project',
]


def should_archive(folder_name: str) -> bool:
    """이 폴더를 archive로 옮길지 판단"""
    if folder_name in KEEP:
        return False
    if not folder_name.startswith('eve'):
        return False
    for pat in ARCHIVE_PATTERNS:
        if folder_name.startswith(pat):
            return True
    return False


def get_size_str(path: Path) -> str:
    """폴더 크기 사람이 읽기 쉽게 (대략, du 없이)"""
    try:
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        if total > 1e9:
            return f"{total/1e9:.1f} GB"
        elif total > 1e6:
            return f"{total/1e6:.1f} MB"
        else:
            return f"{total/1e3:.1f} KB"
    except Exception:
        return "?"


def main(dry_run: bool = True):
    drive = Path(DRIVE_ROOT)
    archive_dir = drive / ARCHIVE_NAME

    print("=" * 70)
    print(f"Drive 정리 스크립트 ({'DRY RUN (실제 이동 X)' if dry_run else '실제 실행'})")
    print("=" * 70)
    print(f"Drive root: {drive}")
    print(f"Archive 위치: {archive_dir}")
    print()

    if not drive.exists():
        print(f"❌ Drive 마운트 안 됨. 먼저 drive.mount() 실행하세요.")
        return

    # archive 폴더 만들기 (실제 실행 시만)
    if not dry_run and not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 {archive_dir} 생성됨")
        print()

    # MyDrive 직속 폴더만 스캔 (재귀 X, 안전)
    folders = [f for f in drive.iterdir() if f.is_dir()]

    to_archive = []
    to_keep = []
    other = []  # eve로 시작 안 하거나 KEEP에 있는 것

    for f in folders:
        if should_archive(f.name):
            to_archive.append(f)
        elif f.name in KEEP:
            to_keep.append(f)
        else:
            other.append(f)

    # === 보고 ===
    print(f"📊 스캔 결과")
    print(f"  유지(메인): {len(to_keep)}개")
    for f in to_keep:
        print(f"    ✓ {f.name}")
    print(f"  옮길(archive): {len(to_archive)}개")
    for f in to_archive:
        size = get_size_str(f) if dry_run else "?"
        print(f"    → {f.name}  ({size})")
    print(f"  기타(건드리지 않음): {len(other)}개")
    if len(other) <= 10:
        for f in other[:10]:
            print(f"    - {f.name}")
    print()

    if not to_archive:
        print("✅ 정리할 폴더 없음. 이미 깔끔함.")
        return

    # === 실제 이동 ===
    if dry_run:
        print("=" * 70)
        print("⚠️  DRY RUN 모드. 실제 이동하려면:")
        print("    main(dry_run=False)")
        print("=" * 70)
        return

    print("=" * 70)
    print(f"실제 이동 시작 ({len(to_archive)}개)")
    print("=" * 70)

    moved = 0
    failed = []
    for f in to_archive:
        target = archive_dir / f.name
        # 이미 archive에 같은 이름 있으면 skip
        if target.exists():
            print(f"  ⚠️  {f.name}: archive에 이미 있음, skip")
            failed.append((f.name, 'already exists'))
            continue
        try:
            shutil.move(str(f), str(target))
            print(f"  ✅ {f.name} → {ARCHIVE_NAME}/")
            moved += 1
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
            failed.append((f.name, str(e)))

    print()
    print("=" * 70)
    print(f"완료: {moved} 이동, {len(failed)} 실패")
    print("=" * 70)

    if failed:
        print("\n실패 목록:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

    print(f"\n📁 archive 위치: {archive_dir}")
    print("   필요 없으면 나중에 직접 삭제하면 됨.")
    print("\n복원 방법 (만약 필요하면):")
    print(f"   shutil.move('{archive_dir}/<폴더명>', '{drive}/<폴더명>')")


# ============= 실행 =============
# 1단계: 먼저 dry_run으로 확인
main(dry_run=True)

# 2단계: 결과 확인 후 위 줄 주석 처리하고 아래 줄 실행
# main(dry_run=False)
