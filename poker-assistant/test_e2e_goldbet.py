"""
End-to-end test su screenshot Goldbet salvati.
Simula il loop del controller: Vision LLM → Equity → Engine.
"""
import sys
from pathlib import Path
import yaml
import cv2

sys.path.insert(0, str(Path(__file__).parent / "engine"))
sys.path.insert(0, str(Path(__file__).parent / "vision"))
sys.path.insert(0, str(Path(__file__).parent / "ui"))

from llm_vision_extractor import LLMVisionExtractor
from assistant_engine import AssistantEngine
from equity_service import calculate_equity, get_hand_strength_class


def test_image(img_name: str):
    img_path = Path.home() / "Desktop" / img_name
    if not img_path.exists():
        print(f"[SKIP] {img_name} not found")
        return

    print(f"\n{'='*60}")
    print(f"TEST: {img_name}")
    print(f"{'='*60}")

    frame = cv2.imread(str(img_path))
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    llm_cfg = cfg.get("vision", {}).get("llm", {})

    # 1. Vision LLM
    extractor = LLMVisionExtractor(
        api_key=llm_cfg.get("api_key") or None,
        model=llm_cfg.get("model", "google/gemini-3.1-flash-lite"),
        cooldown_seconds=0,
        window_title=cfg.get("vision", {}).get("window_title"),
    )
    state = extractor.extract(frame)

    hole = state.get("hole", [])
    board = state.get("board", [])
    pot = state.get("pot", 0)
    to_call = state.get("to_call", 0)
    stack = state.get("stack", 1000)
    position = state.get("position", "BTN")
    stage = state.get("stage", "preflop")

    print(f"\n[VISION]")
    print(f"  Hole:     {' '.join(hole) if hole else '---'}")
    print(f"  Board:    {' '.join(board) if board else '---'}")
    print(f"  Pot:      {pot}")
    print(f"  To call:  {to_call}")
    print(f"  Stack:    {stack}")
    print(f"  Position: {position}")
    print(f"  Stage:    {stage}")

    # Guard: need 2 hole cards
    if len(hole) != 2:
        print("[ENGINE] Skipped: need exactly 2 hole cards")
        return

    # 2. Equity
    equity = calculate_equity(hole, board, n=1000)
    strength = get_hand_strength_class(hole, board)
    print(f"\n[EQUITY]")
    print(f"  Equity:   {equity:.1%}")
    print(f"  Strength: {strength}")

    # 3. Engine recommendation
    print(f"\n[ENGINE RECOMMENDATION]")
    if stage == "postflop" and len(board) not in (3, 4, 5):
        print(f"  SKIPPED: board has {len(board)} cards, expected 3/4/5 for postflop")
        return

    try:
        engine = AssistantEngine()
        history = [f"b{to_call}"] if to_call > 0 else []
        rec = engine.recommend(
            hole=hole,
            board=board,
            history=history,
            pot=pot,
            stack=stack,
            big_blind=cfg.get("big_blind", 2),
            is_dealer=(position == "BTN"),
            to_call=to_call,
        )

        action = rec["action"]
        if action.startswith("b"):
            pretty = f"RAISE {action[1:]}"
        elif action == "c":
            pretty = "CALL"
        elif action == "f":
            pretty = "FOLD"
        elif action == "k":
            pretty = "CHECK"
        else:
            pretty = action.upper()

        print(f"  Action:   {pretty}")
        print(f"  Strategy: {rec['strategy']}")
        print(f"  Stage:    {rec['stage']}")
        if "preflop_equity" in rec:
            print(f"  Preflop Eq: {rec['preflop_equity']:.1%}")
        if "equity" in rec and rec["equity"] is not None:
            print(f"  Postflop Eq: {rec['equity']:.1%}")
        if "draw_outs" in rec:
            print(f"  Draw Outs: {rec['draw_outs']} ({rec.get('draw_type', '')})")
        if "fold_reason" in rec:
            print(f"  Fold Reason: {rec['fold_reason']}")
        if "value_bet_override" in rec:
            print(f"  [Override] Value bet triggered")
    except Exception as e:
        print(f"  ENGINE ERROR: {e}")


if __name__ == "__main__":
    test_image("screenshotgoldbet.jpeg")
    test_image("screenshotgoldbet2.jpeg")
    print("\n" + "="*60)
    print("GOLDBET E2E TEST COMPLETE")
    print("="*60)
