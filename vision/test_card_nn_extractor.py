"""
Tests for vision/card_nn_extractor.py

Run with:  python vision/test_card_nn_extractor.py
"""
import sys
from pathlib import Path

# Make vision/ importable as a package when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from card_nn_extractor import CardNNExtractor


HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "card_classifier.joblib"
TEMPLATES_DIR = HERE / "templates"
DEBUG_DIR = HERE / "debug"


def test_extractor_loads():
    if not MODEL_PATH.exists():
        print("[test] SKIP  model not found (run vision/train_card_classifier.py first)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.4)
    assert ext._ensure_loaded(), "model should load"
    assert ext._model is not None
    assert ext._labels is not None
    assert len(ext._labels) >= 9, f"expected at least 9 ranks, got {len(ext._labels)}"
    print(f"[test] PASS  extractor loaded: {len(ext._labels)} ranks")


def test_predict_rank_returns_tuple():
    if not MODEL_PATH.exists():
        print("[test] SKIP  predict_rank (no model)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.4)
    img = np.zeros((80, 60), dtype=np.uint8)
    rank, conf = ext.predict_rank(img)
    assert rank is None or isinstance(rank, str)
    assert 0.0 <= conf <= 1.0
    print(f"[test] PASS  predict_rank returns valid tuple: rank={rank}, conf={conf:.2f}")


def test_predict_rank_handles_bgr():
    """BGR images should be converted to grayscale automatically."""
    if not MODEL_PATH.exists():
        print("[test] SKIP  predict_rank_bgr (no model)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.4)
    img = np.zeros((80, 60, 3), dtype=np.uint8)  # BGR
    rank, conf = ext.predict_rank(img)
    print(f"[test] PASS  predict_rank accepts BGR: rank={rank}, conf={conf:.2f}")


def test_templates_classified_correctly():
    """The model should correctly classify all 21 training templates."""
    if not MODEL_PATH.exists() or not TEMPLATES_DIR.exists():
        print("[test] SKIP  templates_classified (no model or templates)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.4)
    correct, total = 0, 0
    for f in sorted(TEMPLATES_DIR.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        expected = f.stem[0]
        pred, conf = ext.predict_rank(img)
        if pred == expected:
            correct += 1
        total += 1
    assert total == 21, f"expected 21 templates, got {total}"
    assert correct == total, f"only {correct}/{total} templates classified correctly"
    print(f"[test] PASS  templates classified: {correct}/{total}")


def test_debug_board_crops_classified():
    """The model should classify the 5 board crops correctly."""
    if not MODEL_PATH.exists() or not DEBUG_DIR.exists():
        print("[test] SKIP  debug_board_crops (no model or debug dir)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.4)
    correct, total = 0, 0
    for f in sorted(DEBUG_DIR.glob("board_crop_*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Parse expected rank from filename
        stem = f.stem
        expected = None
        for r in "A23456789TJQK":
            for s in "shdc":
                if f"_{r}{s}_" in stem:
                    expected = r
                    break
            if expected:
                break
        if expected is None:
            continue
        pred, conf = ext.predict_rank(img)
        if pred == expected:
            correct += 1
        total += 1
    assert total == 5, f"expected 5 board crops, got {total}"
    assert correct == total, f"only {correct}/{total} board crops classified correctly"
    print(f"[test] PASS  debug board crops classified: {correct}/{total}")


def test_low_confidence_returns_none():
    """A black/white image should return None (low confidence)."""
    if not MODEL_PATH.exists():
        print("[test] SKIP  low_confidence (no model)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.99)  # very high
    img = np.random.randint(0, 255, (80, 60), dtype=np.uint8)  # noise
    rank, conf = ext.predict_rank(img)
    assert rank is None, f"random noise should not match any rank with high confidence, got {rank} conf={conf:.2f}"
    print(f"[test] PASS  noise rejected at high threshold: rank={rank}, conf={conf:.2f}")


def test_batch_predict_preserves_order():
    if not MODEL_PATH.exists():
        print("[test] SKIP  batch_predict (no model)")
        return
    ext = CardNNExtractor(str(MODEL_PATH), confidence_threshold=0.4)
    imgs = [np.zeros((80, 60), dtype=np.uint8) for _ in range(3)]
    results = ext.predict_ranks(imgs)
    assert len(results) == 3
    print(f"[test] PASS  batch predict returns {len(results)} results")


def main():
    tests = [
        test_extractor_loads,
        test_predict_rank_returns_tuple,
        test_predict_rank_handles_bgr,
        test_templates_classified_correctly,
        test_debug_board_crops_classified,
        test_low_confidence_returns_none,
        test_batch_predict_preserves_order,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[test] FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[test] ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n[test] Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
