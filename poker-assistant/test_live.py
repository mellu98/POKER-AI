"""Test live del modello vision (gemini-3.5-flash-lite) sul tavolo reale.

Uso:
    python3 test_live.py                          # 10 cicli, finestra dal config
    python3 test_live.py --cycles 30 --title Poker --interval 2

Requisiti: il tavolo deve essere aperto e VISIBLE (non minimizzato):
il capture su macOS usa Quartz con filtro OnScreenOnly.
"""
import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vision"))

_capture = importlib.import_module("capture")
_lve = importlib.import_module("llm_vision_extractor")
screenshot = _capture.screenshot
LLMVisionExtractor = _lve.LLMVisionExtractor

CACHED_THRESHOLD_S = 0.2  # extract() sotto questa soglia = cache/cooldown


def load_window_title(override: str | None) -> str:
    if override:
        return override
    import yaml

    try:
        with open(ROOT / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return "Poker"
    return cfg.get("vision", {}).get("window_title", "Poker")


def _int_or_zero(value) -> int:
    """Conversione difensiva per i bound delle finestre Quartz."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _list_visible_window_titles() -> list[str]:
    """Titoli delle finestre visibili via Quartz (PyObjC: costanti dinamiche,
    accessibili via getattr per non confondere i type checker)."""
    titles: list[str] = []
    try:
        import Quartz
    except ImportError:
        return titles

    list_windows = getattr(Quartz, "CGWindowListCopyWindowInfo", None)
    if list_windows is None:
        return titles
    opts = getattr(Quartz, "kCGWindowListExcludeDesktopElements", 0) | getattr(
        Quartz, "kCGWindowListOptionOnScreenOnly", 0
    )
    null_id = getattr(Quartz, "kCGNullWindowID", None)
    key_name = getattr(Quartz, "kCGWindowName", "kCGWindowName")
    key_owner = getattr(Quartz, "kCGWindowOwnerName", "kCGWindowOwnerName")
    key_bounds = getattr(Quartz, "kCGWindowBounds", "kCGWindowBounds")

    try:
        wins = list_windows(opts, null_id)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        print(f"[live] enumerazione finestre fallita: {exc}")
        return titles

    for w in wins:
        name = (w.get(key_name, "") or "").strip()
        owner = w.get(key_owner, "") or ""
        bounds = w.get(key_bounds, {}) or {}
        if not name:
            continue
        if _int_or_zero(bounds.get("Width", 0)) < 300:
            continue
        if _int_or_zero(bounds.get("Height", 0)) < 200:
            continue
        titles.append(f"[{owner}] {name!r}")
    return titles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    title = load_window_title(args.title)

    # Verifica finestra PRIMA di bruciare chiamate API: senza finestra il
    # capture degrada a fullscreen e l'estrazione sarebbe spazzatura.
    rect = _capture._find_window_rect_mac(title)
    if rect is None:
        titles = _list_visible_window_titles()
        raise SystemExit(
            f"Finestra {title!r} non trovata.\n"
            "Apri il tavolo poker (visibile, non minimizzato) e rilancia.\n"
            "Finestre visibili ora:\n  " + "\n  ".join(titles[:20])
        )

    print(f">>> LIVE TEST — finestra {title!r} {rect['width']}x{rect['height']}")
    extractor = LLMVisionExtractor(
        config_path=str(ROOT / "config.yaml"), window_title=title
    )
    print(f">>> modello: {extractor.model} — {args.cycles} cicli, ogni {args.interval}s\n")

    api_lats: list[float] = []
    cached = 0
    valid = 0
    for i in range(args.cycles):
        frame = screenshot(window_title=title)
        if frame is None:
            print(f"[{i + 1:2d}] capture FALLITA")
            continue
        t0 = time.time()
        state = extractor.extract(frame)
        dt = time.time() - t0

        if dt < CACHED_THRESHOLD_S:
            cached += 1
            note = "(cache: frame invariato)"
        else:
            api_lats.append(dt)
            note = ""
        cards = list(state.get("hole", [])) + list(state.get("board", []))
        ok = len(cards) == len(set(cards))
        if ok:
            valid += 1
        flag = "" if ok else "  ⚠️ CARTE DUPLICATE"
        print(
            f"[{i + 1:2d}] {dt:5.2f}s {note} hole={state.get('hole')} "
            f"board={state.get('board')} pot={state.get('pot')} "
            f"to_call={state.get('to_call')} stage={state.get('stage')} "
            f"pos={state.get('position')}{flag}"
        )
        if i < args.cycles - 1:
            time.sleep(args.interval)

    total = len(api_lats) + cached
    print(f"\n=== RISULTATO: {valid}/{total} estrazioni valide, {cached} da cache ===")
    if api_lats:
        print(
            f"=== LATENZA API (nuove letture): media {statistics.mean(api_lats):.2f}s"
            f" | min {min(api_lats):.2f}s | max {max(api_lats):.2f}s ==="
        )


if __name__ == "__main__":
    main()
