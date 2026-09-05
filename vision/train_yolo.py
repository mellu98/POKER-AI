"""
Fine-tune the pretrained 52-class card YOLO model on the collected Goldbet
dataset produced by ``vision/data_collector.py``.

Usage:
    python vision/train_yolo.py validate --data vision/dataset/yolo_goldbet
    python vision/train_yolo.py split    --data vision/dataset/yolo_goldbet
    python vision/train_yolo.py train    --data vision/dataset/yolo_goldbet
    python vision/train_yolo.py export   --best vision/models/runs/goldbet_cards/weights/best.pt
    python vision/train_yolo.py eval     --model vision/models/poker_yolo.pt
    python vision/train_yolo.py all      --data vision/dataset/yolo_goldbet

Flow (``all`` runs the whole pipeline):
1. ``validate``  — pre-flight check of every YOLO label file. HARD GATE:
                   refuses to train while any placeholder class id (52, meaning
                   "not yet reviewed") is present, and rejects out-of-range
                   class ids or malformed lines. Run
                   ``python vision/review_labels.py --dir <data>`` first.
2. ``split``     — moves images+labels into ``<data>/train/images|labels`` and
                   ``<data>/val/images|labels`` (deterministic seed) and
                   rewrites ``<data>/data.yaml`` to point at the split dirs.
3. ``train``     — fine-tunes ``vision/models/yolov8m_cards_synthetic.pt``
                   with ultralytics (Apple Silicon MPS when available) into
                   ``vision/models/runs/goldbet_cards/``, then prints mAP.
4. ``export``    — copies the best weights to ``vision/models/poker_yolo.pt``
                   (the path the bot's state extractor expects) and optionally
                   exports CoreML for Apple Silicon inference speed.
5. ``eval``      — bonus: runs the trained model on the labeled crops in
                   ``vision/dataset/cards/`` (filename = ``<card>_..._slot.png``)
                   and prints top-1 accuracy, so the improvement over the
                   previous 0% baseline is visible.

Class mapping (must match data_collector.py / review_labels.py):
52 real classes in TeogopK order — ranks ['10','2','3','4','5','6','7','8',
'9','A','J','K','Q'] x suits ['c','d','h','s'] — plus sentinel id 52 for
unresolved placeholders. Internal crop filenames use 'T' for ten ('Ts'), the
YOLO naming uses '10' ('10s').
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Class mapping — duplicated (not imported) from data_collector.py on purpose:
# this script must stay lightweight and importable even where cv2/tesseract
# are unavailable. Keep the two in sync.
# ---------------------------------------------------------------------------
YOLO_RANKS = ["10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "K", "Q"]
YOLO_SUITS = ["c", "d", "h", "s"]
CLASS_NAMES = [r + s for r in YOLO_RANKS for s in YOLO_SUITS]  # 52 entries

# Placeholder class written by data_collector.py for uncertain crops.
PLACEHOLDER_CLASS_ID = len(CLASS_NAMES)  # 52 — must NEVER reach training.
NUM_CLASSES = len(CLASS_NAMES)           # 52

# Internal crop-code ranks ('Ts') -> YOLO naming ('10s').
_RANK_10 = "T"

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA = "vision/dataset/yolo_goldbet"
DEFAULT_WEIGHTS = "vision/models/yolov8m_cards_synthetic.pt"
DEFAULT_RUNS_DIR = "vision/models/runs"
DEFAULT_RUN_NAME = "goldbet_cards"
DEFAULT_OUTPUT_MODEL = "vision/models/poker_yolo.pt"
DEFAULT_CROPS_DIR = "vision/dataset/cards"

# Reproducibility everywhere.
SEED = 42


def card_code_to_yolo_name(card: str) -> str:
    """Internal crop code ('7d', 'Ts') -> YOLO class name ('7d', '10s')."""
    rank, suit = card[0].upper(), card[1].lower()
    return ("10" if rank == _RANK_10 else rank) + suit


# ---------------------------------------------------------------------------
# ultralytics import — guarded so validate/split work without it installed.
# ---------------------------------------------------------------------------

def _import_ultralytics():
    """Import YOLO with a clear error message if ultralytics is missing."""
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:
        print("[train] ERROR: ultralytics is not installed.")
        print("[train] Install it with:  pip install ultralytics")
        print(f"[train] (import failed with: {exc})")
        sys.exit(2)


def _resolve_device(requested: str = "auto") -> str:
    """Pick the compute device: 'mps' on Apple Silicon when available, else cpu."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            print("[train] Apple Silicon MPS detected — using device 'mps'.")
            return "mps"
    except Exception as exc:
        print(f"[train] Could not probe MPS ({exc}); falling back to CPU.")
    print("[train] MPS not available — using device 'cpu'.")
    return "cpu"


# ---------------------------------------------------------------------------
# 1. validate — pre-flight gate
# ---------------------------------------------------------------------------

def validate_dataset(data_dir: str) -> int:
    """Check every YOLO label file. Returns 0 if safe to train, non-zero otherwise.

    Hard gates (refuse to train):
      * any class id 52 (placeholder = unreviewed crop),
      * any class id outside [0, 51],
      * any malformed line (not 5 whitespace-separated numeric fields, or
        coordinates not in [0, 1]).
    Warnings (non-fatal): classes with very few examples (< 5).
    """
    data = Path(data_dir)
    labels_dir = data / "labels"
    images_dir = data / "images"
    if not labels_dir.is_dir():
        print(f"[train] ERROR: labels dir not found: {labels_dir}")
        return 2

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"[train] ERROR: no label files in {labels_dir}")
        return 2

    image_files = sorted(images_dir.glob("*")) if images_dir.is_dir() else []
    image_stems = {p.stem for p in image_files}

    placeholder_files: dict[str, int] = {}   # file -> count of class-52 lines
    out_of_range: list[str] = []             # "file:line -> bad id"
    malformed: list[str] = []                # "file:line -> reason"
    label_stems: set[str] = set()
    per_class: Counter = Counter()
    total_instances = 0
    images_with_labels = 0

    for lf in label_files:
        label_stems.add(lf.stem)
        lines = lf.read_text().splitlines()
        if lines:
            images_with_labels += 1
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            where = f"{lf.name}:{i + 1}"
            if len(parts) != 5:
                malformed.append(f"{where} -> expected 5 fields, got {len(parts)}")
                continue
            try:
                values = [float(p) for p in parts]
            except ValueError:
                malformed.append(f"{where} -> non-numeric field")
                continue
            class_id = int(values[0])
            if values[0] != class_id:
                malformed.append(f"{where} -> class id is not an integer")
                continue
            if not all(0.0 <= v <= 1.0 for v in values[1:]):
                malformed.append(f"{where} -> coords out of [0, 1]")
                continue

            total_instances += 1
            per_class[class_id] += 1
            if class_id == PLACEHOLDER_CLASS_ID:
                placeholder_files[lf.name] = placeholder_files.get(lf.name, 0) + 1
            elif not 0 <= class_id < NUM_CLASSES:
                out_of_range.append(f"{where} -> class id {class_id}")

    # --- Report -------------------------------------------------------------
    n_placeholders = sum(placeholder_files.values())
    print("\n[train] === Dataset validation ===")
    print(f"[train] Data dir            : {data}")
    print(f"[train] Label files         : {len(label_files)}")
    print(f"[train] Image files         : {len(image_files)}")
    print(f"[train] Images with labels  : {images_with_labels}")
    labeled_without_image = sorted(label_stems - image_stems)
    images_without_labels = sorted(image_stems - label_stems)
    if labeled_without_image:
        print(f"[train] WARN: {len(labeled_without_image)} label file(s) have no "
              f"matching image (first: {labeled_without_image[0]})")
    if images_without_labels:
        print(f"[train] NOTE: {len(images_without_labels)} image(s) have no labels "
              f"(empty frames — usable for training as background)")
    print(f"[train] Total card instances: {total_instances}")
    print(f"[train] Classes present     : "
          f"{len([c for c in per_class if c != PLACEHOLDER_CLASS_ID])}/{NUM_CLASSES}")

    # Per-class distribution, compact: only classes with data (or missing ones).
    if per_class:
        present = sorted(c for c in per_class
                         if 0 <= c < NUM_CLASSES)  # ignore bad ids in the report
        missing = [c for c in range(NUM_CLASSES) if c not in per_class]
        low = [c for c in present if per_class[c] < 5]
        top = [(c, n) for c, n in per_class.most_common()
               if 0 <= c < NUM_CLASSES][:8]
        top_str = ", ".join(f"{CLASS_NAMES[c]}:{n}" for c, n in top
                            if c != PLACEHOLDER_CLASS_ID)
        print(f"[train] Top classes         : {top_str}")
        if missing:
            print(f"[train] WARN: {len(missing)} class(es) with NO examples: "
                  f"{', '.join(CLASS_NAMES[c] for c in missing[:12])}"
                  f"{' ...' if len(missing) > 12 else ''}")
        if low:
            print(f"[train] WARN: {len(low)} class(es) with < 5 examples: "
                  f"{', '.join(f'{CLASS_NAMES[c]}({per_class[c]})' for c in low[:12])}"
                  f"{' ...' if len(low) > 12 else ''}")

    # --- Hard gates ---------------------------------------------------------
    failed = False
    if n_placeholders:
        failed = True
        print(f"\n[train] REFUSING TO TRAIN: {n_placeholders} placeholder line(s) "
              f"(class {PLACEHOLDER_CLASS_ID} = 'needs review') in "
              f"{len(placeholder_files)} file(s):")
        for name, count in sorted(placeholder_files.items()):
            print(f"[train]   {name}: {count} placeholder line(s)")
        print(f"\n[train] Resolve them first with:")
        print(f"[train]   python vision/review_labels.py --dir {data}")
    if out_of_range:
        failed = True
        print(f"\n[train] REFUSING TO TRAIN: {len(out_of_range)} line(s) with class "
              f"ids outside [0, {NUM_CLASSES - 1}]:")
        for item in out_of_range[:20]:
            print(f"[train]   {item}")
    if malformed:
        failed = True
        print(f"\n[train] REFUSING TO TRAIN: {len(malformed)} malformed label line(s):")
        for item in malformed[:20]:
            print(f"[train]   {item}")

    if failed:
        print("\n[train] Validation FAILED — fix the issues above and re-run.")
        return 1
    print("\n[train] Validation PASSED — dataset is safe to train.")
    return 0


# ---------------------------------------------------------------------------
# 2. split — deterministic train/val split + data.yaml rewrite
# ---------------------------------------------------------------------------

def _load_existing_names(data_yaml: Path) -> list[str] | None:
    """Return the names list from an existing data.yaml, if it looks valid."""
    if not data_yaml.exists():
        return None
    try:
        import yaml

        cfg = yaml.safe_load(data_yaml.read_text()) or {}
        names = cfg.get("names")
        if isinstance(names, list) and len(names) == NUM_CLASSES:
            return [str(n) for n in names]
        if isinstance(names, dict):
            ordered = [names[k] for k in sorted(names)]
            if len(ordered) == NUM_CLASSES:
                return [str(n) for n in ordered]
    except Exception as exc:
        print(f"[train] Could not parse existing {data_yaml} ({exc}); "
              f"rebuilding the canonical names list.")
    return None


def _write_data_yaml(data_dir: Path, names: list[str]) -> Path:
    """Rewrite data.yaml to point train/val at the split subdirectories."""
    yaml_path = data_dir / "data.yaml"
    names_str = ", ".join(names)
    content = (
        f"# YOLOv8 dataset config — regenerated by vision/train_yolo.py (split).\n"
        f"# Reviewed dataset: no placeholder class-{PLACEHOLDER_CLASS_ID} lines.\n"
        f"path: {data_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: [{names_str}]\n"
    )
    yaml_path.write_text(content)
    print(f"[train] Wrote {yaml_path}")
    return yaml_path


def split_dataset(data_dir: str, val_fraction: float = 0.15,
                  seed: int = SEED) -> int:
    """Split images+labels into <data>/train and <data>/val (deterministic).

    Plain random split with a fixed seed. Class balance is left to the volume
    of collected frames (each frame carries up to 7 cards, so per-image
    stratification buys little); the pre-flight report already warns about
    scarce classes. Re-running re-uses already-split files as the pool.
    """
    data = Path(data_dir)
    images_dir = data / "images"
    labels_dir = data / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Re-split support: if a previous split exists, pull everything back.
    previously_split = any((data / s / "images").is_dir() for s in ("train", "val"))
    if previously_split and not any(images_dir.iterdir()):
        for split in ("train", "val"):
            for sub, dest in (("images", images_dir), ("labels", labels_dir)):
                prev = data / split / sub
                if prev.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    for f in prev.iterdir():
                        shutil.move(str(f), dest / f.name)
        print(f"[train] Re-split: reassembled previous train/val into {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in
                    {".png", ".jpg", ".jpeg", ".webp", ".bmp"})
    if not images:
        print(f"[train] ERROR: no images found in {images_dir}")
        return 2

    pairs = []  # (image_path, label_path or None)
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        pairs.append((img, lbl if lbl.exists() else None))

    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(pairs) * val_fraction))) if len(pairs) > 1 else 0
    val_set = set(indices[:n_val])

    # Clean/create target dirs (idempotent re-split).
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = data / split / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    n_moved_labels = {"train": 0, "val": 0}
    for idx, (img, lbl) in enumerate(pairs):
        split = "val" if idx in val_set else "train"
        shutil.move(str(img), data / split / "images" / img.name)
        if lbl is not None:
            shutil.move(str(lbl), data / split / "labels" / lbl.name)
            n_moved_labels[split] += 1

    print(f"[train] Split {len(pairs)} images (seed={seed}, val={val_fraction:.0%}):")
    print(f"[train]   train: {len(pairs) - n_val} images, "
          f"{n_moved_labels['train']} label files")
    print(f"[train]   val  : {n_val} images, {n_moved_labels['val']} label files")

    names = _load_existing_names(data / "data.yaml") or CLASS_NAMES
    _write_data_yaml(data, names)
    return 0


# ---------------------------------------------------------------------------
# 3. train — fine-tune with ultralytics
# ---------------------------------------------------------------------------

def train(data_dir: str, weights: str = DEFAULT_WEIGHTS, epochs: int = 100,
          imgsz: int = 640, batch: int = 8, patience: int = 20,
          device: str = "auto", project: str = DEFAULT_RUNS_DIR,
          name: str = DEFAULT_RUN_NAME) -> Path:
    """Fine-tune the base model and return the path to best.pt."""
    YOLO = _import_ultralytics()
    data_yaml = Path(data_dir) / "data.yaml"
    if not data_yaml.exists():
        print(f"[train] ERROR: {data_yaml} not found — run the split step first.")
        sys.exit(2)
    if not Path(weights).exists():
        print(f"[train] ERROR: base weights not found: {weights}")
        sys.exit(2)

    dev = _resolve_device(device)
    print(f"[train] Base weights: {weights}")
    print(f"[train] Data config : {data_yaml}")
    print(f"[train] epochs={epochs} imgsz={imgsz} batch={batch} "
          f"patience={patience} device={dev}")

    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=dev,
        project=project,
        name=name,
        seed=SEED,
        deterministic=True,
        exist_ok=True,  # resume into the same run dir instead of goldbet_cards2
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    if not best_path.exists():
        print(f"[train] ERROR: expected best weights at {best_path} but they are "
              f"missing — check the training log above.")
        sys.exit(1)

    # Final validation on the held-out split.
    print(f"\n[train] Validating {best_path} on the val split...")
    best = YOLO(str(best_path))
    metrics = best.val(data=str(data_yaml), device=dev)
    print(f"[train] mAP50    : {metrics.box.map50:.4f}")
    print(f"[train] mAP50-95 : {metrics.box.map:.4f}")
    return best_path


# ---------------------------------------------------------------------------
# 4. export — publish best.pt as poker_yolo.pt (+ optional CoreML)
# ---------------------------------------------------------------------------

def export(best_path: str = None, output: str = DEFAULT_OUTPUT_MODEL,
           coreml: bool = True) -> int:
    """Copy best.pt to vision/models/poker_yolo.pt; try CoreML export (optional)."""
    if best_path is None:
        best_path = str(Path(DEFAULT_RUNS_DIR) / DEFAULT_RUN_NAME / "weights" / "best.pt")
    best = Path(best_path)
    if not best.exists():
        print(f"[train] ERROR: best weights not found: {best} — train first.")
        return 2

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best), str(out))
    print(f"[train] Copied {best} -> {out}")

    if coreml:
        try:
            YOLO = _import_ultralytics()
            print("[train] Exporting CoreML (Apple Silicon acceleration)...")
            exported = YOLO(str(out)).export(format="coreml", imgsz=640)
            print(f"[train] CoreML export OK: {exported}")
        except Exception as exc:
            # Non-fatal: the .pt model works fine; CoreML is a speed bonus.
            print(f"[train] WARN: CoreML export failed ({exc}); "
                  f"continuing with the .pt model only.")
    return 0


# ---------------------------------------------------------------------------
# 5. eval — accuracy on the labeled crops in vision/dataset/cards/
# ---------------------------------------------------------------------------

def evaluate_on_crops(model_path: str = DEFAULT_OUTPUT_MODEL,
                      crops_dir: str = DEFAULT_CROPS_DIR,
                      conf_threshold: float = 0.25) -> int:
    """Run the model on labeled card crops and print top-1 accuracy.

    Crops are named ``<card>_<timestamp>_<slot>.png`` (e.g. ``7d_..._hole1.png``);
    files starting with ``XX`` are unlabeled and skipped. Same convention as
    ``vision/dataset/cards/``. Lets you compare a model against the 0% baseline
    previously measured with the untuned detector.
    """
    YOLO = _import_ultralytics()
    crops = Path(crops_dir)
    if not crops.is_dir():
        print(f"[train] ERROR: crops dir not found: {crops}")
        return 2
    if not Path(model_path).exists():
        print(f"[train] ERROR: model not found: {model_path}")
        return 2

    files = sorted(p for p in crops.glob("*.png")
                   if not p.name.startswith("XX"))
    if not files:
        print(f"[train] ERROR: no labeled crops in {crops}")
        return 2

    model = YOLO(model_path)
    correct = 0
    per_rank_correct: Counter = Counter()
    per_rank_total: Counter = Counter()
    failures: list[str] = []

    for f in files:
        card_code = f.name.split("_")[0]          # '7d', 'Ts', ...
        expected_name = card_code_to_yolo_name(card_code)
        try:
            expected_id = CLASS_NAMES.index(expected_name)
        except ValueError:
            print(f"[train] WARN: skipping {f.name} — unrecognized code "
                  f"'{card_code}'")
            continue

        results = model.predict(str(f), imgsz=640, conf=conf_threshold,
                                verbose=False)
        if results and len(results[0].boxes):
            top_id = int(results[0].boxes.cls[0].item())
        else:
            top_id = -1  # no detection

        rank = expected_name[:-1]
        per_rank_total[rank] += 1
        if top_id == expected_id:
            correct += 1
            per_rank_correct[rank] += 1
        elif len(failures) < 15:
            got = CLASS_NAMES[top_id] if 0 <= top_id < NUM_CLASSES else "no-detect"
            failures.append(f"{f.name}: expected {expected_name}, got {got}")

    n = sum(per_rank_total.values())
    acc = correct / n if n else 0.0
    print(f"\n[train] === Crop evaluation: {model_path} ===")
    print(f"[train] Crops evaluated : {n} (skipped XX/unrecognized)")
    print(f"[train] Top-1 accuracy  : {acc:.2%} ({correct}/{n})")
    print("[train] Per-rank accuracy:")
    for rank in sorted(per_rank_total):
        c, t = per_rank_correct[rank], per_rank_total[rank]
        print(f"[train]   {rank:>2}: {c / t:6.1%} ({c}/{t})")
    if failures:
        print("[train] Sample failures:")
        for line in failures:
            print(f"[train]   {line}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the 52-class card YOLO model on the Goldbet dataset."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data(p, required=True):
        p.add_argument("--data", type=str, default=None if required else DEFAULT_DATA,
                       help=f"Dataset directory (default: {DEFAULT_DATA})")

    p_val = sub.add_parser("validate", help="Pre-flight label check (placeholder gate)")
    add_data(p_val)

    p_split = sub.add_parser("split", help="Split into train/val + rewrite data.yaml")
    add_data(p_split)
    p_split.add_argument("--val-fraction", type=float, default=0.15,
                         help="Fraction of images for validation (default: 0.15)")
    p_split.add_argument("--seed", type=int, default=SEED,
                         help=f"Shuffle seed (default: {SEED})")

    p_train = sub.add_parser("train", help="Fine-tune with ultralytics")
    add_data(p_train)
    p_train.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                         help=f"Base model (default: {DEFAULT_WEIGHTS})")
    p_train.add_argument("--epochs", type=int, default=100,
                         help="Training epochs (default: 100)")
    p_train.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    p_train.add_argument("--batch", type=int, default=8,
                         help="Batch size; use -1 for auto (default: 8)")
    p_train.add_argument("--patience", type=int, default=20,
                         help="Early-stopping patience in epochs (default: 20)")
    p_train.add_argument("--device", type=str, default="auto",
                         help="'auto' (mps->cpu), 'mps', 'cpu' or a GPU index")
    p_train.add_argument("--project", type=str, default=DEFAULT_RUNS_DIR,
                         help=f"Runs root dir (default: {DEFAULT_RUNS_DIR})")
    p_train.add_argument("--name", type=str, default=DEFAULT_RUN_NAME,
                         help=f"Run name (default: {DEFAULT_RUN_NAME})")

    p_export = sub.add_parser("export", help="Publish best.pt as poker_yolo.pt")
    p_export.add_argument("--best", type=str, default=None,
                          help="Path to best.pt (default: latest goldbet_cards run)")
    p_export.add_argument("--output", type=str, default=DEFAULT_OUTPUT_MODEL,
                          help=f"Output model path (default: {DEFAULT_OUTPUT_MODEL})")
    p_export.add_argument("--no-coreml", action="store_true",
                          help="Skip the optional CoreML export")

    p_eval = sub.add_parser("eval", help="Top-1 accuracy on labeled card crops")
    p_eval.add_argument("--model", type=str, default=DEFAULT_OUTPUT_MODEL,
                        help=f"Model to evaluate (default: {DEFAULT_OUTPUT_MODEL})")
    p_eval.add_argument("--crops", type=str, default=DEFAULT_CROPS_DIR,
                        help=f"Crops dir (default: {DEFAULT_CROPS_DIR})")
    p_eval.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")

    p_all = sub.add_parser("all", help="validate -> split -> train -> export")
    add_data(p_all)
    p_all.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p_all.add_argument("--epochs", type=int, default=100)
    p_all.add_argument("--imgsz", type=int, default=640)
    p_all.add_argument("--batch", type=int, default=8)
    p_all.add_argument("--patience", type=int, default=20)
    p_all.add_argument("--device", type=str, default="auto")
    p_all.add_argument("--val-fraction", type=float, default=0.15)
    p_all.add_argument("--no-coreml", action="store_true")

    args = parser.parse_args()
    data = getattr(args, "data", None) or DEFAULT_DATA

    if args.command == "validate":
        sys.exit(validate_dataset(data))
    if args.command == "split":
        sys.exit(split_dataset(data, val_fraction=args.val_fraction,
                               seed=args.seed))
    if args.command == "train":
        train(data, weights=args.weights, epochs=args.epochs, imgsz=args.imgsz,
              batch=args.batch, patience=args.patience, device=args.device,
              project=args.project, name=args.name)
        sys.exit(0)
    if args.command == "export":
        sys.exit(export(best_path=args.best, output=args.output,
                        coreml=not args.no_coreml))
    if args.command == "eval":
        sys.exit(evaluate_on_crops(model_path=args.model, crops_dir=args.crops,
                                   conf_threshold=args.conf))
    if args.command == "all":
        rc = validate_dataset(data)
        if rc != 0:
            sys.exit(rc)  # hard gate — never train on unreviewed placeholders
        rc = split_dataset(data, val_fraction=args.val_fraction)
        if rc != 0:
            sys.exit(rc)
        train(data, weights=args.weights, epochs=args.epochs, imgsz=args.imgsz,
              batch=args.batch, patience=args.patience, device=args.device)
        sys.exit(export(coreml=not args.no_coreml))

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
