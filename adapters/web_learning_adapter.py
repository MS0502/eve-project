"""
WebLearningAdapter — 라운드 22

EVE가 인터넷에서 텍스트 받아 학습:
- fetch_url(url) → HTML 텍스트 추출 → corpus
- search_wiki(topic) → 위키피디아 한국어 → corpus
- youtube_transcript(video_id) → 자막 → corpus
- daily_news(rss_url) → 뉴스 RSS → corpus

원칙:
- Lazy load (라이브러리 없어도 EVE 안 죽음)
- Graceful degrade
- 결과는 항상 corpus_adapter.learn_text 통과
- safety: 화이트리스트 도메인만 (옵션)

EVE 결정론 안 깸 (인터넷에서 텍스트 가져오기만, 사고는 EVE가).
"""

import re
from typing import Optional


# 안전한 도메인 화이트리스트 (옵션, 주의)
DEFAULT_ALLOWED_DOMAINS = {
    "ko.wikipedia.org", "en.wikipedia.org",
    "namu.wiki",
    "youtube.com", "www.youtube.com",
}


def _strip_html(html: str) -> str:
    """가장 가벼운 HTML → 텍스트 (외부 의존 X). 라이브러리 있으면 더 좋음."""
    # script/style 제거
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # 태그 제거
    text = re.sub(r"<[^>]+>", " ", html)
    # entity 일부
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WebLearningAdapter:

    def __init__(self, engine,
                 allowed_domains: Optional[set] = None,
                 max_text_length: int = 50000):
        self.engine = engine
        self.allowed_domains = allowed_domains  # None이면 모두 허용
        self.max_text_length = max_text_length
        # Lazy
        self._requests = None
        self._wiki = None
        self._yt = None
        self._bs4 = None

    # ===== 의존성 로드 =====

    def _load_requests(self):
        if self._requests is not None:
            return self._requests
        try:
            import requests
            self._requests = requests
            return self._requests
        except ImportError:
            self._requests = "unavailable"
            return None

    def _load_wikipedia(self):
        if self._wiki is not None:
            return self._wiki
        try:
            import wikipedia
            wikipedia.set_lang("ko")
            self._wiki = wikipedia
            return self._wiki
        except ImportError:
            self._wiki = "unavailable"
            return None

    def _load_youtube_transcript(self):
        if self._yt is not None:
            return self._yt
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            self._yt = YouTubeTranscriptApi
            return self._yt
        except ImportError:
            self._yt = "unavailable"
            return None

    def _load_bs4(self):
        if self._bs4 is not None:
            return self._bs4
        try:
            from bs4 import BeautifulSoup
            self._bs4 = BeautifulSoup
            return self._bs4
        except ImportError:
            self._bs4 = "unavailable"
            return None

    def status(self) -> dict:
        return {
            "requests": self._load_requests() is not None and self._requests != "unavailable",
            "wikipedia": self._load_wikipedia() is not None and self._wiki != "unavailable",
            "youtube_transcript": self._load_youtube_transcript() is not None and self._yt != "unavailable",
            "bs4": self._load_bs4() is not None and self._bs4 != "unavailable",
        }

    # ===== Safety =====

    def _is_allowed(self, url: str) -> bool:
        if self.allowed_domains is None:
            return True
        try:
            from urllib.parse import urlparse
            host = urlparse(url).hostname or ""
            return any(host == d or host.endswith("." + d) for d in self.allowed_domains)
        except Exception:
            return False

    # ===== fetch_url =====

    def fetch_url(self, url: str, timeout: float = 10.0) -> str:
        """URL → 텍스트 (HTML 제거). 실패 시 ""."""
        if not self._is_allowed(url):
            return ""
        req = self._load_requests()
        if req is None:
            return ""
        try:
            r = req.get(url, timeout=timeout, headers={
                "User-Agent": "EVE-Learning/1.0",
            })
            if r.status_code != 200:
                return ""
            html = r.text
        except Exception:
            return ""

        # bs4 있으면 더 깔끔
        bs4 = self._load_bs4()
        if bs4 is not None and self._bs4 != "unavailable":
            try:
                soup = bs4(html, "html.parser")
                # script/style 제거
                for s in soup(["script", "style"]):
                    s.decompose()
                text = soup.get_text(" ", strip=True)
            except Exception:
                text = _strip_html(html)
        else:
            text = _strip_html(html)

        return text[: self.max_text_length]

    def fetch_and_learn(self, url: str) -> dict:
        """URL → 텍스트 → corpus 학습. trace dict 반환."""
        text = self.fetch_url(url)
        if not text:
            return {"url": url, "text_length": 0, "learned": False}
        result = {"url": url, "text_length": len(text), "learned": False}
        if self.engine.corpus_adapter is not None:
            try:
                self.engine.corpus_adapter.learn_text(text)
                result["learned"] = True
            except Exception:
                pass
        return result

    # ===== Wikipedia =====

    def search_wiki(self, topic: str, sentences: int = 10) -> str:
        """위키피디아 한국어에서 주제 요약."""
        wiki = self._load_wikipedia()
        if wiki is None:
            return ""
        try:
            return wiki.summary(topic, sentences=sentences)
        except wiki.exceptions.DisambiguationError as e:
            # 여러 항목 있으면 첫 번째 시도
            try:
                return wiki.summary(e.options[0], sentences=sentences)
            except Exception:
                return ""
        except Exception:
            return ""

    def wiki_and_learn(self, topic: str, sentences: int = 10) -> dict:
        text = self.search_wiki(topic, sentences=sentences)
        result = {"topic": topic, "text_length": len(text), "learned": False}
        if text and self.engine.corpus_adapter is not None:
            try:
                self.engine.corpus_adapter.learn_text(text)
                result["learned"] = True
            except Exception:
                pass
        return result

    # ===== YouTube 자막 =====

    def youtube_transcript(self, video_id_or_url: str) -> str:
        """YouTube 자막 받기 (한국어 우선)."""
        yt = self._load_youtube_transcript()
        if yt is None:
            return ""
        # URL이면 video_id 추출
        video_id = self._extract_video_id(video_id_or_url)
        if not video_id:
            return ""
        try:
            transcripts = yt.list_transcripts(video_id)
            try:
                t = transcripts.find_transcript(["ko", "ko-KR"])
            except Exception:
                # 한국어 없으면 자동 번역 가능한 거
                try:
                    t = transcripts.find_transcript(["en"]).translate("ko")
                except Exception:
                    t = list(transcripts)[0] if transcripts else None
            if t is None:
                return ""
            data = t.fetch()
            return " ".join(item["text"] for item in data if item.get("text"))
        except Exception:
            return ""

    def _extract_video_id(self, s: str) -> Optional[str]:
        # 11자 ID 직접
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
            return s
        # URL 패턴
        m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", s)
        return m.group(1) if m else None

    def youtube_and_learn(self, video_id_or_url: str) -> dict:
        text = self.youtube_transcript(video_id_or_url)
        result = {"video": video_id_or_url, "text_length": len(text), "learned": False}
        if text and self.engine.corpus_adapter is not None:
            try:
                self.engine.corpus_adapter.learn_text(text)
                result["learned"] = True
            except Exception:
                pass
        return result
