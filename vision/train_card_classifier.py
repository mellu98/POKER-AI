"""
Card Rank Classifier — scikit-learn HOG + linear SVM.

Trains a 13-class rank classifier (A, 2-9, T, J, Q, K) on real card crops
extracted from 247 Free Poker screenshots.

Data sources:
  - vision/templates/  (21 real card images, 60x80, named like 'As.png')
  - vision/debug/board_crop_X_<RANK><SUIT>_*.png  (5 board cards)
  - vision/debug/hole_<RANK><SUIT>_*  (6 hole card crops)

Augmentation per image (12x):
  - rotation ±10°
  - scale 0.9x-1.1x
  - brightness ±25%
  - contrast ±15%
  - Gaussian noise σ=5
  - small translation ±2px

Output:
  - vision/models/card_classifier.joblib  (the trained SVM + scaler + label map)

Run:
    python vision/train_card_classifier.py
"""
import os
import sys
import random
import joblib
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

VISION_DIR = Path(__file__).parent
TEMPLATES_DIR = VISION_DIR / "templates"
DEBUG_DIR = VISION_DIR / "debug"
MODELS_DIR = VISION_DIR / "models"
MODEL_PATH = MODELS_DIR / "card_classifier.joblib"

RANKS = list("A23456789TJQK")
IMG_SIZE = (60, 80)  # width x height (must match template extraction)

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def card_rank_from_filename(name: str) -> str:
    """Extract the rank character from a filename like 'As.png' or 'board_crop_0_Qs_...'."""
    name = name.replace(".png", "")
    if len(name) >= 2 and name[0] in RANKS and name[1].lower() in "shdc":
        return name[0]
    # debug format: board_crop_0_Qs_530_380_145_200 -> find <RANK><SUIT> substring
    for r in RANKS:
        idx = name.find(f"{r}s")
        if idx >= 0:
            return r
        idx = name.find(f"{r}h")
        if idx >= 0:
            return r
        idx = name.find(f"{r}d")
        if idx >= 0:
            return r
        idx = name.find(f"{r}c")
        if idx >= 0:
            return r
    return None


def load_images() -> list[tuple[np.ndarray, str]]:
    """Load all card images with their rank labels.

    Returns:
        List of (grayscale_image_resized_to_IMG_SIZE, rank_char) tuples.
    """
    samples = []

    # Templates: vision/templates/As.png etc.
    if TEMPLATES_DIR.exists():
        for fname in sorted(os.listdir(TEMPLATES_DIR)):
            if not fname.endswith(".png"):
                continue
            rank = card_rank_from_filename(fname)
            if rank is None:
                print(f"[train] SKIP template (no rank): {fname}")
                continue
            img = cv2.imread(str(TEMPLATES_DIR / fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE[1], IMG_SIZE[0]):
                img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            samples.append((img, rank))
            print(f"[train] template: {fname} -> rank={rank}")

    # Debug crops: vision/debug/board_crop_0_Qs_*.png, hole_5d_*.png
    if DEBUG_DIR.exists():
        for fname in sorted(os.listdir(DEBUG_DIR)):
            if not fname.endswith(".png"):
                continue
            rank = card_rank_from_filename(fname)
            if rank is None:
                continue
            img = cv2.imread(str(DEBUG_DIR / fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE[1], IMG_SIZE[0]):
                img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            samples.append((img, rank))
            print(f"[train] debug:    {fname} -> rank={rank}")

    return samples


# --------------------------------------------------------------------------- #
#  Augmentation
# --------------------------------------------------------------------------- #

def augment(img: np.ndarray, n_augments: int = 12, rng: random.Random = None) -> list[np.ndarray]:
    """Return n_augments augmented versions of the input image."""
    if rng is None:
        rng = random.Random()
    out = [img]
    h, w = img.shape

    for _ in range(n_augments):
        aug = img.copy()

        # Rotation
        angle = rng.uniform(-10.0, 10.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Scale + translation
        scale = rng.uniform(0.9, 1.1)
        tx = rng.uniform(-2, 2)
        ty = rng.uniform(-2, 2)
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Brightness + contrast
        alpha = rng.uniform(0.85, 1.15)  # contrast
        beta = rng.uniform(-25, 25)       # brightness
        aug = np.clip(alpha * aug.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Gaussian noise
        if rng.random() < 0.5:
            noise = np.random.normal(0, 5, aug.shape).astype(np.float32)
            aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        out.append(aug)

    return out


# --------------------------------------------------------------------------- #
#  Feature extraction
# --------------------------------------------------------------------------- #

def extract_hog_features(images: list[np.ndarray]) -> np.ndarray:
    """Return a (N, F) matrix of HOG features for each image."""
    feats = []
    for img in images:
        f = hog(img, **HOG_PARAMS)
        feats.append(f)
    return np.array(feats, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(augmentations_per_image: int = 12, seed: int = 42):
    print("[train] Loading images...")
    samples = load_images()
    if not samples:
        print("[train] ERROR: no training images found.")
        return False
    print(f"[train] Loaded {len(samples)} labeled images.")

    rng = random.Random(seed)

    # Augment
    print(f"[train] Augmenting ({augmentations_per_image}x per image)...")
    aug_images = []
    aug_labels = []
    for img, rank in samples:
        versions = augment(img, n_augments=augmentations_per_image, rng=rng)
        for v in versions:
            aug_images.append(v)
            aug_labels.append(rank)
    print(f"[train] Augmented dataset: {len(aug_images)} samples.")

    # Extract HOG features
    print("[train] Extracting HOG features...")
    X = extract_hog_features(aug_images)
    y = np.array(aug_labels)
    print(f"[train] Feature matrix: {X.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train linear SVM
    print("[train] Training LinearSVC...")
    clf = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    clf.fit(X_scaled, y)
    train_acc = clf.score(X_scaled, y)
    print(f"[train] Train accuracy: {train_acc:.3f}")

    # Cross-validation (5-fold) — honest accuracy estimate
    if len(set(y)) >= 2 and len(y) >= 10:
        try:
            scores = cross_val_score(LinearSVC(C=1.0, max_iter=5000, dual="auto"),
                                     X_scaled, y, cv=min(5, len(set(y))))
            print(f"[train] CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"[train] CV skipped: {e}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": clf,
        "scaler": scaler,
        "labels": sorted(set(y)),
        "img_size": IMG_SIZE,
        "train_acc": train_acc,
        "n_samples": len(aug_images),
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[train] Saved model to {MODEL_PATH}")
    return True


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    import argparse
    p = argparse.ArgumentParser(description="Train card rank classifier (HOG+SVM).")
    p.add_argument("--augments", type=int, default=12,
                   help="Augmentations per source image (default 12).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ok = train(augmentations_per_image=args.augments, seed=args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
