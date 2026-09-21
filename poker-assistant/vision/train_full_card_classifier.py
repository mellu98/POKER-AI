"""
Full Card Classifier Trainer — scikit-learn HOG + linear SVM.

Trains two classifiers on real card crops:
  - rank classifier: 13 classes (A, 2-9, T, J, Q, K)
  - suit classifier: 4 classes  (s, h, d, c)

Data layout expected:
  - vision/dataset/cards/<CARD>*.png   (full 60x80 card crops)
  - vision/templates/<CARD>.png        (seed templates)

Each full card is split into rank/suit ROIs matching ocr_cards.py, resized to
32x32, augmented, and fed to a LinearSVC.

Output:
  - vision/models/card_rank_classifier.joblib
  - vision/models/card_suit_classifier.joblib

Run:
    python vision/train_full_card_classifier.py
"""
import os
import sys
import random
import time
from pathlib import Path

import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix


VISION_DIR = Path(__file__).parent
DATASET_DIR = VISION_DIR / "dataset"
CARDS_DIR = DATASET_DIR / "cards"
TEMPLATES_DIR = VISION_DIR / "templates"
MODELS_DIR = VISION_DIR / "models"

RANKS = list("A23456789TJQK")
SUITS = list("shdc")

FULL_SIZE = (60, 80)   # width x height of a full card crop
CLASS_SIZE = (32, 32)  # width x height fed to each classifier

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)

# ROI percentages used by ocr_cards.py to isolate rank/suit in top-left corner
RANK_Y0, RANK_Y1 = 0.0, 0.22
RANK_X0, RANK_X1 = 0.0, 0.22
SUIT_Y0, SUIT_Y1 = 0.20, 0.45
SUIT_X0, SUIT_X1 = 0.0, 0.22


def label_from_filename(name: str) -> tuple[str, str] | None:
    """Extract (rank, suit) from names like 'As.png', 'th_123.png'."""
    stem = Path(name).stem
    # Try first two chars
    if len(stem) >= 2:
        r, s = stem[0].upper(), stem[1].lower()
        if r in RANKS and s in SUITS:
            return (r, s)
    # Fallback: search for any Rs pattern
    for r in RANKS:
        for s in SUITS:
            if f"{r}{s}" in stem.upper() or f"{r.lower()}{s}" in stem:
                return (r, s)
    return None


def split_rank_suit(full_gray: np.ndarray):
    """Return rank and suit ROIs from a full card grayscale image."""
    h, w = full_gray.shape
    rank = full_gray[int(h * RANK_Y0):int(h * RANK_Y1), int(w * RANK_X0):int(w * RANK_X1)]
    suit = full_gray[int(h * SUIT_Y0):int(h * SUIT_Y1), int(w * SUIT_X0):int(w * SUIT_X1)]
    return rank, suit


def load_full_card_samples() -> list[tuple[np.ndarray, str, str]]:
    """Load full card images and return (gray_60x80, rank, suit)."""
    samples = []
    sources = []
    if CARDS_DIR.exists():
        sources += sorted(CARDS_DIR.glob("*.png"))
    if TEMPLATES_DIR.exists():
        sources += sorted(TEMPLATES_DIR.glob("*.png"))

    for path in sources:
        label = label_from_filename(path.name)
        if label is None:
            continue
        rank, suit = label
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (FULL_SIZE[1], FULL_SIZE[0]):
            img = cv2.resize(img, FULL_SIZE, interpolation=cv2.INTER_AREA)
        samples.append((img, rank, suit))
    return samples


def augment(img: np.ndarray, n_augments: int = 20, rng: random.Random | None = None) -> list[np.ndarray]:
    """Generate geometric/photometric augmentations of a small grayscale image."""
    if rng is None:
        rng = random.Random()
    out = [img]
    h, w = img.shape

    for _ in range(n_augments):
        aug = img.copy().astype(np.float32)

        # Scale + rotation + translation in one affine matrix
        scale = rng.uniform(0.88, 1.12)
        angle = rng.uniform(-12.0, 12.0)
        tx = rng.uniform(-2, 2)
        ty = rng.uniform(-2, 2)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Brightness + contrast
        alpha = rng.uniform(0.80, 1.20)
        beta = rng.uniform(-25, 25)
        aug = np.clip(alpha * aug + beta, 0, 255)

        # Gaussian noise
        if rng.random() < 0.5:
            noise = np.random.normal(0, 6, aug.shape)
            aug = np.clip(aug + noise, 0, 255)

        out.append(aug.astype(np.uint8))

    return out


def extract_features(images: list[np.ndarray]) -> np.ndarray:
    """Return HOG feature matrix for a list of same-size grayscale images."""
    feats = []
    for img in images:
        f = hog(img, **HOG_PARAMS)
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Per-class accuracy dictionary."""
    acc = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() == 0:
            acc[label] = 0.0
        else:
            acc[label] = float((y_pred[mask] == label).mean())
    return acc


def train_classifier(
    name: str,
    crops: list[np.ndarray],
    labels: list[str],
    augmentations: int = 20,
    seed: int = 42,
) -> dict:
    """Train one HOG+SVM classifier and return a saveable payload."""
    print(f"\n[train] --- {name} classifier ---")
    print(f"[train] Source images: {len(crops)}")

    rng = random.Random(seed)

    # Augment
    aug_images = []
    aug_labels = []
    for img, label in zip(crops, labels):
        for v in augment(img, n_augments=augmentations, rng=rng):
            aug_images.append(v)
            aug_labels.append(label)
    print(f"[train] Augmented to {len(aug_images)} samples")

    # Features
    X = extract_features(aug_images)
    y = np.array(aug_labels)
    print(f"[train] Feature matrix: {X.shape}")

    # Scale + train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LinearSVC(C=1.0, max_iter=8000, dual="auto")
    clf.fit(X_scaled, y)
    train_acc = float(clf.score(X_scaled, y))
    print(f"[train] Train accuracy: {train_acc:.3f}")

    # Cross-validation
    n_classes = len(set(y))
    if len(y) >= 10 and n_classes >= 2:
        try:
            cv = StratifiedKFold(n_splits=min(5, n_classes))
            scores = cross_val_score(
                LinearSVC(C=1.0, max_iter=8000, dual="auto"),
                X_scaled, y, cv=cv
            )
            print(f"[train] CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"[train] CV skipped: {e}")

    # Confusion matrix on augmented training data (biased but useful for spotting bad classes)
    preds = clf.predict(X_scaled)
    per_class = class_accuracy(y, preds, sorted(set(labels)))
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:3]
    print(f"[train] Worst classes: {worst}")

    return {
        "classifier": clf,
        "scaler": scaler,
        "labels": sorted(set(y)),
        "img_size": CLASS_SIZE,
        "hog_params": HOG_PARAMS,
        "train_acc": train_acc,
        "n_samples": len(aug_images),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Train rank+suit card classifiers (HOG+SVM).")
    p.add_argument("--augments", type=int, default=20,
                   help="Augmentations per source image (default 20).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("[train] Loading card samples...")
    samples = load_full_card_samples()
    if not samples:
        print("[train] ERROR: no training images found.")
        print("[train] Capture some cards first with: python vision/collect_card_data.py")
        sys.exit(1)

    print(f"[train] Loaded {len(samples)} full-card samples")
    missing_ranks = set(RANKS) - {r for _, r, _ in samples}
    missing_suits = set(SUITS) - {s for _, _, s in samples}
    if missing_ranks:
        print(f"[train] WARNING: missing ranks: {sorted(missing_ranks)}")
    if missing_suits:
        print(f"[train] WARNING: missing suits: {sorted(missing_suits)}")

    # Build rank and suit crops
    rank_crops, rank_labels = [], []
    suit_crops, suit_labels = [], []
    for full, rank, suit in samples:
        rank_roi, suit_roi = split_rank_suit(full)
        rank_roi = cv2.resize(rank_roi, CLASS_SIZE, interpolation=cv2.INTER_AREA)
        suit_roi = cv2.resize(suit_roi, CLASS_SIZE, interpolation=cv2.INTER_AREA)
        rank_crops.append(rank_roi)
        rank_labels.append(rank)
        suit_crops.append(suit_roi)
        suit_labels.append(suit)

    # Train
    t0 = time.time()
    rank_payload = train_classifier("rank", rank_crops, rank_labels,
                                    augmentations=args.augments, seed=args.seed)
    suit_payload = train_classifier("suit", suit_crops, suit_labels,
                                    augmentations=args.augments, seed=args.seed)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rank_path = MODELS_DIR / "card_rank_classifier.joblib"
    suit_path = MODELS_DIR / "card_suit_classifier.joblib"
    joblib.dump(rank_payload, rank_path)
    joblib.dump(suit_payload, suit_path)

    print(f"\n[train] Saved rank model -> {rank_path}")
    print(f"[train] Saved suit model -> {suit_path}")
    print(f"[train] Total time: {time.time() - t0:.1f}s")
    print("[train] DONE. Update config.yaml: vision.nn.enabled=true")


if __name__ == "__main__":
    main()
