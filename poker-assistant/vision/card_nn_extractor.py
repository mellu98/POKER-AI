"""
Card NN Extractor — scikit-learn HOG + SVM inference.

Loads a pre-trained card rank classifier and provides a clean API:
    extractor = CardNNExtractor(model_path)
    rank, confidence = extractor.predict_rank(gray_crop_60x80)

The model is trained on 12 ranks (A, 2-9, T, J, Q, K — no '8' in training set).
For card images that don't match any known rank with high confidence,
the extractor returns (None, 0.0) so the caller can fall back to template matching.

Clean-room implementation: no code from external sources.
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


class CardNNExtractor:
    """Lazy-loaded rank classifier for card crops (60x80 grayscale)."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._scaler = None
        self._labels = None
        self._img_size = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_path.exists():
            print(f"[card_nn] model not found at {self.model_path}")
            return False
        try:
            payload = joblib.load(self.model_path)
            self._model = payload["classifier"]
            self._scaler = payload["scaler"]
            self._labels = payload["labels"]
            self._img_size = payload.get("img_size", (60, 80))
            print(f"[card_nn] loaded model: ranks={self._labels}, "
                  f"img_size={self._img_size}, threshold={self.confidence_threshold}")
            return True
        except Exception as e:
            print(f"[card_nn] failed to load model: {e}")
            return False

    def _prepare(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Convert BGR or grayscale image to 60x80 grayscale and extract HOG features."""
        if img is None or img.size == 0:
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        target = (self._img_size[1], self._img_size[0])  # (h, w) for cv2.resize
        if img.shape != target:
            img = cv2.resize(img, (self._img_size[0], self._img_size[1]),
                             interpolation=cv2.INTER_AREA)
        return hog(img, **HOG_PARAMS)

    def predict_rank(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict the rank of a card crop.

        Returns:
            (rank, confidence) where rank is one of 'A','2'-'9','T','J','Q','K' or None,
            and confidence is in [0.0, 1.0] (margin between top-1 and top-2
            decision function values, normalized).
        """
        if not self._ensure_loaded():
            return (None, 0.0)
        feat = self._prepare(img)
        if feat is None:
            return (None, 0.0)
        feat_scaled = self._scaler.transform(feat.reshape(1, -1))

        # decision_function returns (1, n_classes)
        scores = self._model.decision_function(feat_scaled)[0]
        # Convert to softmax-like probabilities for nicer confidence
        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()
        top_idx = int(np.argmax(probs))
        rank = self._labels[top_idx]
        confidence = float(probs[top_idx])

        if confidence < self.confidence_threshold:
            return (None, confidence)
        return (rank, confidence)

    def predict_ranks(self, imgs: list[np.ndarray]) -> list[Tuple[Optional[str], float]]:
        """Batch version of predict_rank. Order preserved."""
        return [self.predict_rank(img) for img in imgs]

    @property
    def is_ready(self) -> bool:
        return self.model_path.exists() and self._ensure_loaded()


# --------------------------------------------------------------------------- #
#  CLI smoke test
# --------------------------------------------------------------------------- #

def _cli_tests() -> bool:
    here = Path(__file__).parent
    model_path = here / "models" / "card_classifier.joblib"
    if not model_path.exists():
        print(f"[card_nn] SKIP: no trained model at {model_path}")
        print("[card_nn] Run: python vision/train_card_classifier.py")
        return False

    extractor = CardNNExtractor(str(model_path), confidence_threshold=0.4)

    # Test on templates
    templates_dir = here / "templates"
    print("[card_nn] Testing on templates:")
    correct, total = 0, 0
    for fname in sorted(templates_dir.glob("*.png")):
        img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        expected_rank = fname.stem[0]
        predicted, conf = extractor.predict_rank(img)
        ok = predicted == expected_rank
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  expected={expected_rank}  predicted={predicted}  conf={conf:.2f}  -- {fname.name}")
        if ok:
            correct += 1
        total += 1
    print(f"[card_nn] Templates: {correct}/{total} correct ({100*correct/total:.0f}%)")

    # Test on debug crops
    debug_dir = here / "debug"
    print("\n[card_nn] Testing on debug crops:")
    correct, total = 0, 0
    for fname in sorted(debug_dir.glob("board_crop_*.png")):
        img = cv2.imread(str(fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # extract rank from filename
        stem = fname.stem
        expected_rank = None
        for r in "A23456789TJQK":
            for s in "shdc":
                if f"_{r}{s}_" in stem:
                    expected_rank = r
                    break
            if expected_rank:
                break
        if expected_rank is None:
            continue
        predicted, conf = extractor.predict_rank(img)
        ok = predicted == expected_rank
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  expected={expected_rank}  predicted={predicted}  conf={conf:.2f}  -- {fname.name}")
        if ok:
            correct += 1
        total += 1
    print(f"[card_nn] Debug crops: {correct}/{total} correct ({100*correct/total:.0f}%)")

    return True


if __name__ == "__main__":
    _cli_tests()
