"""
Card Classifier — local HOG+SVM inference for full cards.

Loads two pre-trained classifiers (rank + suit) and returns the full card:
    classifier = CardClassifier(
        rank_model_path="vision/models/card_rank_classifier.joblib",
        suit_model_path="vision/models/card_suit_classifier.joblib",
    )
    card, confidence = classifier.predict_card(card_crop)

The input crop should contain a single card (BGR or grayscale). It is resized to
the standard 60x80 size, split into rank/suit ROIs matching the training script,
and each ROI is classified independently.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import joblib
from skimage.feature import hog


HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)

RANKS = set("A23456789TJQK")
SUITS = set("shdc")

# Same ROI percentages used by train_full_card_classifier.py
RANK_Y0, RANK_Y1 = 0.0, 0.22
RANK_X0, RANK_X1 = 0.0, 0.22
SUIT_Y0, SUIT_Y1 = 0.20, 0.45
SUIT_X0, SUIT_X1 = 0.0, 0.22


class CardClassifier:
    """Predict full card (rank+suit) from a single card crop."""

    FULL_SIZE = (60, 80)  # (width, height)

    def __init__(
        self,
        rank_model_path: str,
        suit_model_path: str,
        confidence_threshold: float = 0.5,
    ):
        self.rank_model_path = Path(rank_model_path)
        self.suit_model_path = Path(suit_model_path)
        self.confidence_threshold = confidence_threshold

        self._rank_model: Optional[dict] = None
        self._suit_model: Optional[dict] = None

    def _load(self, path: Path) -> Optional[dict]:
        if not path.exists():
            print(f"[card_classifier] model not found: {path}")
            return None
        try:
            payload = joblib.load(path)
            print(
                f"[card_classifier] loaded {path.name}: "
                f"labels={payload['labels']}, img_size={payload['img_size']}"
            )
            return payload
        except Exception as e:
            print(f"[card_classifier] failed to load {path}: {e}")
            return None

    def _ensure_loaded(self) -> bool:
        if self._rank_model is None:
            self._rank_model = self._load(self.rank_model_path)
        if self._suit_model is None:
            self._suit_model = self._load(self.suit_model_path)
        return self._rank_model is not None and self._suit_model is not None

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Empty image")
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _prepare_full(self, img: np.ndarray) -> np.ndarray:
        gray = self._to_gray(img)
        if gray.shape != (self.FULL_SIZE[1], self.FULL_SIZE[0]):
            gray = cv2.resize(gray, self.FULL_SIZE, interpolation=cv2.INTER_AREA)
        return gray

    @staticmethod
    def _split(gray_full: np.ndarray, class_size: tuple):
        h, w = gray_full.shape
        rank_roi = gray_full[int(h * RANK_Y0):int(h * RANK_Y1), int(w * RANK_X0):int(w * RANK_X1)]
        suit_roi = gray_full[int(h * SUIT_Y0):int(h * SUIT_Y1), int(w * SUIT_X0):int(w * SUIT_X1)]
        rank_roi = cv2.resize(rank_roi, class_size, interpolation=cv2.INTER_AREA)
        suit_roi = cv2.resize(suit_roi, class_size, interpolation=cv2.INTER_AREA)
        return rank_roi, suit_roi

    def _predict_part(self, roi: np.ndarray, payload: dict) -> Tuple[Optional[str], float]:
        target_size = payload.get("img_size", (32, 32))
        if roi.shape != (target_size[1], target_size[0]):
            roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

        feat = hog(roi, **HOG_PARAMS)
        feat_scaled = payload["scaler"].transform(feat.reshape(1, -1))
        clf = payload["classifier"]

        scores = clf.decision_function(feat_scaled)[0]
        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()
        top_idx = int(np.argmax(probs))
        label = payload["labels"][top_idx]
        confidence = float(probs[top_idx])
        return label, confidence

    def predict_card(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict the full card from a crop.

        Returns:
            (card_str, confidence) e.g. ("As", 0.97) or (None, 0.0).
            confidence is the lower of the rank and suit confidences.
        """
        if not self._ensure_loaded():
            return (None, 0.0)

        try:
            gray_full = self._prepare_full(img)
        except Exception:
            return (None, 0.0)

        rank_size = self._rank_model.get("img_size", (32, 32))
        suit_size = self._suit_model.get("img_size", (32, 32))
        rank_roi, suit_roi = self._split(gray_full, rank_size)

        rank, rank_conf = self._predict_part(rank_roi, self._rank_model)
        suit, suit_conf = self._predict_part(suit_roi, self._suit_model)

        if rank is None or suit is None:
            return (None, 0.0)

        confidence = min(rank_conf, suit_conf)
        if confidence < self.confidence_threshold:
            return (None, confidence)

        card = f"{rank.upper()}{suit.lower()}"
        return (card, confidence)

    def predict_cards(self, imgs: list[np.ndarray]) -> list[Tuple[Optional[str], float]]:
        """Batch version; order preserved."""
        return [self.predict_card(img) for img in imgs]

    @property
    def is_ready(self) -> bool:
        return self._ensure_loaded()


# --------------------------------------------------------------------------- #
#  CLI smoke test
# --------------------------------------------------------------------------- #

def _cli_test():
    here = Path(__file__).parent
    rank_path = here / "models" / "card_rank_classifier.joblib"
    suit_path = here / "models" / "card_suit_classifier.joblib"

    if not rank_path.exists() or not suit_path.exists():
        print("[card_classifier] SKIP: models not found.")
        print("[card_classifier] Run: python vision/train_full_card_classifier.py")
        return

    clf = CardClassifier(str(rank_path), str(suit_path), confidence_threshold=0.5)

    templates_dir = here / "templates"
    if not templates_dir.exists():
        print("[card_classifier] No templates to test on.")
        return

    correct = 0
    total = 0
    print("\n[card_classifier] Testing on templates:")
    for fname in sorted(templates_dir.glob("*.png")):
        label = label_from_filename(fname.name)
        if label is None:
            continue
        expected = f"{label[0].upper()}{label[1].lower()}"
        img = cv2.imread(str(fname))
        pred, conf = clf.predict_card(img)
        ok = pred == expected
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  expected={expected}  predicted={pred}  conf={conf:.2f}  -- {fname.name}")
        if ok:
            correct += 1
        total += 1

    if total:
        print(f"\n[card_classifier] {correct}/{total} correct ({100*correct/total:.0f}%)")


def label_from_filename(name: str) -> tuple[str, str] | None:
    stem = Path(name).stem
    if len(stem) >= 2:
        r, s = stem[0].upper(), stem[1].lower()
        if r in RANKS and s in SUITS:
            return (r, s)
    return None


if __name__ == "__main__":
    _cli_test()
