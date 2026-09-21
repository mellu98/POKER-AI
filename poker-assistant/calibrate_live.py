"""Calibrazione live dei ROI sulla finestra poker reale.

Cattura frame ripetuti mentre una mano procede (le board cards appaiono
progressivamente), accumula i box delle carte bianche rilevate via CV
(filtri proporzionali al frame, non pixel assoluti), clusterizza per
posizione e aggiorna chirurgicamente config.yaml preservando ancore
(&id001...) alias e commenti.

Uso:  python3 calibrate_live.py [--seconds 90] [--title Poker]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vision"))

_capture = __import__("capture")
screenshot = _capture.screenshot

FRAME_GAP_S = 3.0
HOLE_Y_MIN, HOLE_Y_MAX = 0.55, 0.80   # fascia verticale delle hole (frazioni)
HOLE_X_MIN, HOLE_X_MAX = 0.30, 0.70
BOARD_Y_MIN, BOARD_Y_MAX = 0.30, 0.52  # fascia del board
BOARD_X_MIN, BOARD_X_MAX = 0.25, 0.75
CLUSTER_TOL = 0.02  # tolleranza cluster su x_center (frazione larghezza)


def detect_hole_boxes(frame: np.ndarray) -> list[dict]:
    """Blob nella fascia hole: carte singole (aspect 0.5-0.95) o coppie
    sovrapposte (aspect fino a 1.7), poi splittate in due ROI."""
    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 100, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    singles = detect_card_boxes(frame)
    single_w = statistics.median([b["w"] for b in singles]) if singles else fw * 0.045

    out: list[dict] = []
    for cnt in contours:
        try:
            x, y, w, h = cv2.boundingRect(cnt)
        except cv2.error:
            continue
        cy = (y + h / 2) / fh
        cx = (x + w / 2) / fw
        if not (HOLE_Y_MIN <= cy <= HOLE_Y_MAX and HOLE_X_MIN <= cx <= HOLE_X_MAX):
            continue
        if h <= 0:
            continue
        # stessi filtri dimensionali proporzionali delle carte del board
        area_frac = (w * h) / (fw * fh)
        if not (0.025 * fw <= w <= 0.08 * fw):
            continue
        if not (0.002 <= area_frac <= 0.012):
            continue
        aspect = w / h
        if 0.5 < aspect <= 0.95:
            out.append({"x": x, "y": y, "w": w, "h": h,
                        "cx": (x + w / 2) / fw, "cy": cy})
        elif 0.95 < aspect <= 1.7 and w > single_w * 1.4:
            # coppia sovrapposta: due ROI larghe quanto una carta, il secondo
            # spostato a destra di ~62% della larghezza (angolo rank/suit visibile)
            try:
                sw = int(single_w)
                x2 = int(x + single_w * 0.62)
            except (TypeError, ValueError, OverflowError):
                continue
            out.append({"x": x, "y": y, "w": sw, "h": h,
                        "cx": (x + sw / 2) / fw, "cy": cy})
            out.append({"x": x2, "y": y, "w": sw, "h": h,
                        "cx": (x2 + sw / 2) / fw, "cy": cy})
    return out


def detect_card_boxes(frame: np.ndarray) -> list[dict]:
    """Carte bianche con filtri PROPORZIONALI al frame (robusti a 1440p/913p)."""
    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 100, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        try:
            x, y, w, h = cv2.boundingRect(cnt)
        except cv2.error:
            continue
        aspect = w / h if h > 0 else 0.0
        area_frac = (w * h) / (fw * fh)
        # carta tipica: w 2.5-8% della larghezza, aspect 0.5-0.95,
        # area 0.2-1.2% del frame (proporzionale: regge 913p e 1440p)
        if not (0.025 * fw <= w <= 0.08 * fw):
            continue
        if not (0.5 < aspect < 0.95):
            continue
        if not (0.002 <= area_frac <= 0.012):
            continue
        if x <= 20 or y <= 20 or x + w >= fw - 20 or y + h >= fh - 20:
            continue
        boxes.append(
            {"x": x, "y": y, "w": w, "h": h,
             "cx": (x + w / 2) / fw, "cy": (y + h / 2) / fh}
        )
    return boxes


def cluster_by_x(boxes: list[dict]) -> list[list[dict]]:
    """Clusterizza i box per x_center con tolleranza relativa."""
    clusters: list[list[dict]] = []
    for b in sorted(boxes, key=lambda d: d["cx"]):
        placed = False
        for cl in clusters:
            center = statistics.median(d["cx"] for d in cl)
            if abs(b["cx"] - center) <= CLUSTER_TOL:
                cl.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])
    return clusters


def median_box(cluster: list[dict], fw: int, fh: int) -> dict:
    """Box mediano del cluster, in pixel (zeros se il cluster e' corrotto)."""
    try:
        return {
            "x": int(statistics.median(d["x"] for d in cluster)),
            "y": int(statistics.median(d["y"] for d in cluster)),
            "w": int(statistics.median(d["w"] for d in cluster)),
            "h": int(statistics.median(d["h"] for d in cluster)),
        }
    except (TypeError, ValueError, statistics.StatisticsError):
        return {"x": 0, "y": 0, "w": 0, "h": 0}


def roi_lines(anchor: str, rois: list[dict], indent: str = "    ") -> list[str]:
    """Genera le righe YAML di una sequenza ROI con anchor, formato config."""
    lines = [f"{indent}{anchor}"]
    for r in rois:
        lines.append(f"{indent}  - x: {r['x']}")
        lines.append(f"      y: {r['y']}")
        lines.append(f"      w: {r['w']}")
        lines.append(f"      h: {r['h']}")
        lines.append("      rel: true")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    try:
        with open(ROOT / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError as exc:
        raise SystemExit(f"config.yaml non leggibile: {exc}") from exc
    title = args.title or cfg.get("vision", {}).get("window_title", "Poker")

    print(f">>> Calibrazione live su {title!r} per {args.seconds:.0f}s")
    print(">>> Gioca normalmente: hole cards + board verranno accumulate.")

    hole_boxes: list[dict] = []
    board_boxes: list[dict] = []
    frames = 0
    frame: np.ndarray | None = None
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        frame = screenshot(window_title=title)
        if frame is None:
            time.sleep(FRAME_GAP_S)
            continue
        frames += 1
        fh, fw = frame.shape[:2]
        hole_boxes.extend(detect_hole_boxes(frame))
        board_boxes.extend(
            b for b in detect_card_boxes(frame)
            if BOARD_Y_MIN <= b["cy"] <= BOARD_Y_MAX and BOARD_X_MIN <= b["cx"] <= BOARD_X_MAX
        )
        n_hc = len(cluster_by_x(hole_boxes))
        n_bc = len(cluster_by_x(board_boxes))
        print(f"  [{frames}] hole cluster: {n_hc}, board cluster: {n_bc} (box: {len(hole_boxes)}/{len(board_boxes)})", end="\r")
        if n_hc >= 2 and n_bc >= 5:
            print("\n  >>> 5 slot board + 2 hole visti: fermo prima")
            break
        time.sleep(FRAME_GAP_S)
    print()

    if frames == 0 or frame is None:
        raise SystemExit("Nessun frame catturato: finestra non trovata")

    fh, fw = frame.shape[:2]
    hole_clusters = sorted(cluster_by_x(hole_boxes), key=lambda cl: statistics.median(d["cx"] for d in cl))
    board_clusters = sorted(cluster_by_x(board_boxes), key=lambda cl: statistics.median(d["cx"] for d in cl))

    # board: max 5 cluster, scarta rumore (cluster con 1 sola osservazione
    # se abbiamo abbondanza di dati)
    if len(board_clusters) > 5:
        board_clusters = sorted(
            board_clusters, key=lambda cl: -len(cl)
        )[:5]
        board_clusters.sort(key=lambda cl: statistics.median(d["cx"] for d in cl))
    board_clusters = [cl for cl in board_clusters if len(cl) >= 2] or board_clusters[:5]

    if len(board_clusters) < 3:
        print(f"Rilevazione insufficiente: board={len(board_clusters)}")
        print("Riprova con --seconds 120 quando c'e' una mano con board completo.")
        raise SystemExit(1)

    board_px = [median_box(cl, fw, fh) for cl in board_clusters[:5]]

    hole_clusters = [cl for cl in hole_clusters if len(cl) >= 2]
    if len(hole_clusters) >= 2 and board_px:
        bcx_px = sum(c["x"] + c["w"] / 2 for c in board_px) / len(board_px)
        bcx = bcx_px / fw
        hole_clusters.sort(
            key=lambda cl: abs(statistics.median(d["cx"] for d in cl) - bcx)
        )
        hole_px = [median_box(cl, fw, fh) for cl in sorted(
            hole_clusters[:2], key=lambda cl: statistics.median(d["cx"] for d in cl)
        )]
    else:
        # fallback geometrico dal board: delta verticale e proporzioni
        # ereditati dalla calibrazione originale del client Goldbet
        by = min(c["y"] for c in board_px)
        bcx = sum(c["x"] + c["w"] / 2 for c in board_px) / len(board_px)
        try:
            hw = int(fw * 0.045)
            hh = int(fh * 0.085)
            hy = int(by + fh * 0.228)
            gap = int(fw * 0.008)
            hole_px = [
                {"x": int(bcx - hw - gap), "y": hy, "w": hw, "h": hh},
                {"x": int(bcx + gap), "y": hy, "w": hw, "h": hh},
            ]
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            print("  hole: fallback geometrico fallito, uso offset del vecchio config")
            hole_px = [
                {"x": int(fw * 0.454), "y": int(fh * 0.665), "w": int(fw * 0.045), "h": int(fh * 0.085)},
                {"x": int(fw * 0.499), "y": int(fh * 0.665), "w": int(fw * 0.045), "h": int(fh * 0.085)},
            ]
        print(f"  hole: fallback geometrico dal board (cluster rilevati: {len(hole_clusters)})")

    # slot board mancanti (fino a 5): interpola dalla spaziatura rilevata
    if len(board_px) >= 2 and len(board_px) < 5:
        step = board_px[1]["x"] - board_px[0]["x"]
        first = board_px[0]
        template = board_px[0]
        while len(board_px) < 5:
            nxt = dict(template)
            nxt["x"] = board_px[-1]["x"] + step
            board_px.append(nxt)
        _ = first

    pot_px = None
    if board_px:
        try:
            bcx = sum(c["x"] + c["w"] / 2 for c in board_px) / len(board_px)
            by = min(c["y"] for c in board_px)
            pot_px = {
                "x": int(bcx - fw * 0.08), "y": max(0, by - int(fh * 0.10)),
                "w": int(fw * 0.16), "h": int(fh * 0.08),
            }
        except (TypeError, ValueError, ZeroDivisionError):
            pot_px = None

    def rel(px: dict) -> dict:
        try:
            return {
                "x": round(px["x"] / fw, 3), "y": round(px["y"] / fh, 3),
                "w": round(px["w"] / fw, 3), "h": round(px["h"] / fh, 3), "rel": True,
            }
        except (TypeError, ValueError, ZeroDivisionError):
            return {"x": 0, "y": 0, "w": 0, "h": 0, "rel": True}

    hole_rel = [rel(c) for c in hole_px]
    board_rel = [rel(c) for c in board_px]
    pot_rel = rel(pot_px) if pot_px else None

    print("\nROI rilevati (relativi):")
    print(f"  hole : {hole_rel}")
    print(f"  board: {board_rel}")
    print(f"  pot  : {pot_rel}")

    # --- aggiornamento chirurgico del config (preserva ancore/alias) ---
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    lines = text.split("\n")

    def replace_block(lines: list[str], start_marker: str, new_block: list[str],
                      end_markers: list[str]) -> list[str]:
        try:
            start = next(i for i, l in enumerate(lines) if l.startswith(start_marker))
        except StopIteration:
            print(f"  !! marker {start_marker!r} non trovato, salto")
            return lines
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if any(lines[i].startswith(m) for m in end_markers):
                end = i
                break
        return lines[:start] + new_block + lines[end:]

    lines = replace_block(
        lines, "    hole: &id001", roi_lines("    hole: &id001", hole_rel),
        ["    board:", "    pot:", "    to_call:"],
    )
    lines = replace_block(
        lines, "    board: &id002", roi_lines("    board: &id002", board_rel),
        ["    pot:"],
    )
    if pot_rel:
        pot_block = [
            "    pot: &id003",
            f"      x: {pot_rel['x']}", f"      y: {pot_rel['y']}",
            f"      w: {pot_rel['w']}", f"      h: {pot_rel['h']}", "      rel: true",
        ]
        lines = replace_block(lines, "    pot: &id003", pot_block, ["    to_call:"])

    new_text = "\n".join(lines)
    new_cfg = yaml.safe_load(new_text)
    assert new_cfg["sites"]["golbet"]["hole"] == new_cfg["vision"]["rois"]["hole"], "alias rotti!"
    assert new_cfg["sites"]["golbet"]["board"] == new_cfg["vision"]["rois"]["board"], "alias rotti!"

    backup = ROOT / "config.yaml.bak"
    backup.write_text(text, encoding="utf-8")
    (ROOT / "config.yaml").write_text(new_text, encoding="utf-8")
    print(f"\n✓ config.yaml aggiornato (backup: {backup.name})")

    # debug overlay
    debug = frame.copy()
    for c in hole_px:
        cv2.rectangle(debug, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 255, 0), 3)
    for c in board_px:
        cv2.rectangle(debug, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (255, 0, 0), 3)
    if pot_px:
        cv2.rectangle(debug, (pot_px["x"], pot_px["y"]),
                      (pot_px["x"] + pot_px["w"], pot_px["y"] + pot_px["h"]), (0, 0, 255), 3)
    Path(ROOT / "captures").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "captures/calib_live_debug.png"), debug)
    print("✓ overlay: captures/calib_live_debug.png")


if __name__ == "__main__":
    main()
