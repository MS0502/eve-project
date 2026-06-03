"""
VisionAdapter — 라운드 22

CLIP 기반 이미지 → 한국어 카테고리.
사진/스크린샷 보여주면 "여기 뭐 있어?" 답하게 됨.

원칙:
- Lazy load (transformers/torch 없어도 EVE 안 죽음)
- 카테고리 후보 한국어 사전 (확장 가능)
- 결과 corpus_adapter로 흘러서 EVE 학습
- 폰에서 무거움 — Z Fold 6은 가능하나 느림 (ViT-B/32 추천)

대안:
- transformers + open_clip — 풀 기능
- onnx-runtime + 양자화 모델 — 폰 친화 (선택)

EVE 결정론 안 깸 (인식만, 사고는 EVE가).
"""

from typing import Optional


# 한국어 카테고리 후보 (확장 가능, 필요시 사용자가 더 추가)
DEFAULT_KOR_LABELS = [
    "사람", "남자", "여자", "아이", "노인",
    "강아지", "고양이", "동물", "새", "물고기",
    "음식", "라면", "김치", "밥", "고기", "과일", "커피", "물",
    "건물", "집", "방", "침대", "책상", "의자", "창문", "문",
    "차", "도로", "나무", "꽃", "하늘", "구름", "산", "바다",
    "옷", "신발", "가방", "핸드폰", "컴퓨터", "책", "TV",
    "운동", "PT", "헬스장", "달리기", "축구", "농구",
    "군대", "사역", "훈련", "막사",
    "글자", "책상위", "스크린샷", "그림", "사진",
    "낮", "밤", "실내", "실외",
]


class VisionAdapter:

    def __init__(self, engine, labels: Optional[list[str]] = None,
                 model_name: str = "openai/clip-vit-base-patch32"):
        self.engine = engine
        self.labels = labels if labels is not None else list(DEFAULT_KOR_LABELS)
        self.model_name = model_name
        # Lazy
        self._clip = None       # (model, processor)
        self._top_k = 5

    def _load_clip(self):
        if self._clip is not None:
            return self._clip
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            model = CLIPModel.from_pretrained(self.model_name)
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model.eval()
            self._clip = (model, processor, torch)
            return self._clip
        except Exception:
            self._clip = "unavailable"
            return None

    def available(self) -> bool:
        if self._clip is None:
            self._load_clip()
        return self._clip is not None and self._clip != "unavailable"

    def classify_image(self, image_path: str, top_k: int = 5) -> list[tuple[str, float]]:
        """이미지 → top-k 카테고리 (label, score)."""
        if not self.available():
            return []
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return []
        try:
            model, processor, torch = self._clip
            # CLIP은 영어 prompt가 더 정확 → "a photo of {korean}" 패턴
            prompts = [f"a photo of {label}" for label in self.labels]
            inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # image-text similarity
            logits = outputs.logits_per_image[0]  # [num_labels]
            probs = logits.softmax(dim=-1)
            # top-k
            top_indices = probs.argsort(descending=True)[:top_k]
            return [(self.labels[i.item()], probs[i].item()) for i in top_indices]
        except Exception:
            return []

    def describe_image(self, image_path: str, top_k: int = 3) -> str:
        """이미지 → 한국어 한 줄."""
        results = self.classify_image(image_path, top_k=top_k)
        if not results:
            return ""
        labels = [name for name, score in results if score > 0.05]
        if not labels:
            return ""
        if len(labels) == 1:
            return f"{labels[0]} 같은 거 보여"
        return f"{', '.join(labels[:-1])}이랑 {labels[-1]} 같은 거 보여"

    def see_and_learn(self, image_path: str, top_k: int = 5) -> dict:
        """이미지 → 카테고리 → corpus 학습."""
        results = self.classify_image(image_path, top_k=top_k)
        result = {"image": image_path, "labels": results, "learned": False}
        if results and self.engine.corpus_adapter is not None:
            # 카테고리들을 자연어로 묶어서 학습
            text = " ".join(name for name, _ in results)
            try:
                self.engine.corpus_adapter.learn_text(text)
                result["learned"] = True
            except Exception:
                pass
        return result

    def add_label(self, label: str):
        """사용자가 새 카테고리 추가."""
        if label not in self.labels:
            self.labels.append(label)

    def status(self) -> dict:
        return {
            "clip": self.available(),
            "model": self.model_name,
            "labels_count": len(self.labels),
        }
