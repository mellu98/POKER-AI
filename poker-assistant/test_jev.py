"""Test del layer Jev: audit di coerenza sullo stato estratto dalla pipeline reale.

1. Estrae lo stato dal tavolo (pipeline completa, modello vision gemini-3.5-flash-lite)
2. Audita lo stato con Jev (chiamata sincrona, per mostrare latenza e output)
3. Audita uno stato VOLONTARIAMENTE rotto: Jev deve segnare le incoerenze

Uso:  python3 test_jev.py
"""
import importlib
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vision"))

_lve = importlib.import_module("llm_vision_extractor")
_jev = importlib.import_module("jev_decisions")
LLMVisionExtractor = _lve.LLMVisionExtractor
JevClient = _jev.JevClient

IMAGE = "auto_calib_debug.png"


def print_report(title: str, answers: dict | None) -> None:
    print(f"\n--- {title} ---")
    if answers is None:
        print("  NESSUN REPORT (errore o API key mancante)")
        return
    for qid, ans in answers.items():
        choice = ans.get("choice")
        conf = ans.get("confidence")
        print(f"  {qid:20s} -> {choice!r:8s} (confidence {conf})")


def main() -> None:
    # 1) pipeline reale
    frame = cv2.imread(str(ROOT / IMAGE))
    if frame is None:
        raise SystemExit(f"{IMAGE} non trovata")
    print(f">>> extract() su {IMAGE} (pipeline completa)")
    extractor = LLMVisionExtractor(config_path=str(ROOT / "config.yaml"))
    t0 = time.time()
    state = extractor.extract(frame)
    print(f"\nStato estratto in {time.time() - t0:.2f}s:")
    for k in ("hole", "board", "pot", "to_call", "stack", "stage", "position"):
        print(f"  {k:10s}: {state.get(k)}")

    # 2) audit Jev sincrono sullo stato reale
    client = JevClient()
    print("\n>>> Jev consistency_check sullo stato REALE")
    t0 = time.time()
    report = client.consistency_check(state)
    dt = time.time() - t0
    print(f"    (audit completato in {dt:.2f}s)")
    print_report("STATO REALE", report)

    # 3) audit su stato rotto: stage flop con 5 board, to_call impossibile
    broken = {
        "hole": ["Th", "2s"],
        "board": ["Td", "Ts", "3c", "7d", "Kd"],
        "stage": "flop",  # SBAGLIATO: 5 carte = river
        "pot": 98,
        "to_call": 5000,  # IMPOSSIBILE: stack effettivo 950
        "stack": 784,
        "effective_stack": 950,
        "button_seat": 1,
    }
    print("\n>>> Jev consistency_check su stato ROTTO (stage flop + 5 board,")
    print("    to_call 5000 > effective_stack 950)")
    t0 = time.time()
    report_broken = client.consistency_check(broken)
    dt = time.time() - t0
    print(f"    (audit completato in {dt:.2f}s)")
    print_report("STATO ROTTO", report_broken)

    ok_real = report is not None and all(
        a.get("choice") in ("yes", "no") for a in report.values()
    )
    catches = (
        report_broken is not None
        and report_broken.get("stage_consistent", {}).get("choice") == "no"
        and report_broken.get("numbers_plausible", {}).get("choice") == "no"
    )
    print(f"\n=== ESITO: audit reale risponde={ok_real}, rotto intercettato={catches} ===")


if __name__ == "__main__":
    main()
