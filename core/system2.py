"""
System 2 (검증) - Kahneman 2011

eve2.md 3.5:
- 논리 위반 제거
- 자아/성격 위반 제거
- 중복 제거

v41 첫 버전: 길이/중복 검증.
나중에 v40의 self_doubt.py로 신념 충돌 검증, integrated_self.py로 정체성 정렬 검증.
"""


class System2:

    def __init__(self, min_len: int = 3, max_len: int = 80, sd_adapter=None):
        self.min_len = min_len
        self.max_len = max_len
        self.sd_adapter = sd_adapter

    def filter(self, candidates: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        seen_texts: set[str] = set()

        for c in candidates:
            text = c.get("text", "").strip()
            if not text:
                continue
            if len(text) < self.min_len or len(text) > self.max_len:
                continue
            if text in seen_texts:                  # 중복
                continue
            seen_texts.add(text)
            filtered.append(c)

        # SD 어댑터 있으면 신념 검증
        if self.sd_adapter is not None:
            filtered = self.sd_adapter.filter_candidates(filtered)

        # 점수 내림차순
        filtered.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return filtered
