"""
Poker AI Assistant — main entry point.

Usage:
    python main.py --manual
    python main.py --screenshot
    python main.py --webcam --device 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "engine"))
sys.path.insert(0, str(Path(__file__).parent / "vision"))
sys.path.insert(0, str(Path(__file__).parent / "ui"))

from ui.controller import AssistantController


def parse_args():
    parser = argparse.ArgumentParser(description="Poker AI Real-Time Assistant")
    parser.add_argument(
        "--manual", action="store_true", help="Manual input mode (no vision)"
    )
    parser.add_argument(
        "--screenshot", action="store_true", help="Screenshot capture mode"
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Webcam capture mode"
    )
    parser.add_argument(
        "--llm-vision", action="store_true", help="Use LLM vision (OpenRouter) instead of template matching"
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Hybrid mode: local template OCR for cards/numbers + LLM for position (cached 60s)"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Webcam device index (default 0)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.5, help="Update interval in seconds"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    return parser.parse_args()


def manual_loop(controller: AssistantController):
    """CLI loop where the user types in the current hand state."""
    print("=" * 60)
    print("Poker AI Assistant — MANUAL MODE")
    print("=" * 60)
    print("Commands:")
    print("  hole=As Kh board=Qd Jh 2c pot=120 call=20 pos=BTN stage=flop")
    print("  (or type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            raw = input("\nState> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if raw.lower() in ("quit", "exit", "q"):
            break

        if not raw:
            continue

        try:
            state = _parse_manual_input(raw)
            controller.set_manual_state(state)
            print("  -> State updated. Check the overlay window.")
        except Exception as e:
            print(f"  Error parsing input: {e}")

    controller.stop()


def _parse_manual_input(text: str) -> dict:
    """Parse a compact state string into a dict."""
    state = {
        "hole": [],
        "board": [],
        "pot": 0,
        "to_call": 0,
        "position": "BTN",
        "stage": "preflop",
        "stack": 1000,
        "big_blind": 2,
    }

    # Collect multi-token values (e.g. hole=As Kh board=Qd Jh 2c)
    tokens = text.split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "=" not in tok:
            i += 1
            continue
        key, val = tok.split("=", 1)
        key = key.lower()

        # Gather trailing tokens without "=" as continuation of this value
        j = i + 1
        while j < len(tokens) and "=" not in tokens[j]:
            val += " " + tokens[j]
            j += 1
        i = j

        if key == "hole":
            state["hole"] = val.split()
        elif key == "board":
            state["board"] = val.split()
        elif key == "pot":
            state["pot"] = int(val)
        elif key in ("call", "tocall", "to_call"):
            state["to_call"] = int(val)
        elif key == "pos":
            state["position"] = val.upper()
        elif key == "stage":
            state["stage"] = val.lower()
        elif key == "stack":
            state["stack"] = int(val)
        elif key == "bb":
            state["big_blind"] = int(val)

    return state


def main():
    args = parse_args()

    if args.manual:
        mode = "manual"
    elif args.llm_vision:
        mode = "llm"
    elif args.hybrid:
        mode = "hybrid"
    elif args.screenshot:
        mode = "screenshot"
    elif args.webcam:
        mode = "webcam"
    else:
        print("No mode selected. Use --manual, --hybrid, --llm-vision, --screenshot, or --webcam.")
        sys.exit(1)

    controller = AssistantController(
        mode=mode,
        update_interval=args.interval,
        config_path=args.config,
    )

    if mode == "manual":
        # Overlay runs on main thread (Tkinter requirement);
        # controller loop and CLI input run in background threads.
        controller._running = True
        import threading
        t = threading.Thread(target=controller._loop, daemon=True)
        t.start()
        cli_t = threading.Thread(target=manual_loop, args=(controller,), daemon=True)
        cli_t.start()
        controller.overlay.run()
    else:
        controller.start()


if __name__ == "__main__":
    main()
