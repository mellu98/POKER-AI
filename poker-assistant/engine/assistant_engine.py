"""
Assistant Engine — CFR Blueprint Lookup Wrapper.

Loads pre-trained CFR strategies and provides a clean API for real-time
poker decision recommendations.
"""

import copy
import sys
from pathlib import Path

import joblib
import numpy as np
import preflop_charts
import preflop_equity_lookup
from abstraction import predict_cluster
from equity_service import calculate_equity
from outs_calculator import OutsCalculator
from postflop_holdem import PostflopHoldemHistory, PostflopHoldemInfoSet
from preflop_holdem import PreflopHoldemInfoSet

# Fix joblib unpickling: models were saved when preflop/postflop scripts
# were run as __main__, so pickle looks for classes in __main__.
_main = sys.modules["__main__"]
_main.PreflopHoldemInfoSet = PreflopHoldemInfoSet
_main.PostflopHoldemInfoSet = PostflopHoldemInfoSet

PREFLOP_DISCRETE = {"k", "bMIN", "bMID", "bMAX", "c", "f"}
POSTFLOP_DISCRETE = {"k", "bMIN", "bMID", "bMAX", "c", "f"}


def _safe_int(value, default: int = 0) -> int:
    """Conversione difensiva per sizing/limiti (input da config/stato)."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return default


def _get_action(strategy: dict) -> str:
    """Sample an action from a strategy distribution."""
    actions = list(strategy.keys())
    probs = np.array(list(strategy.values()), dtype=float)
    probs /= probs.sum()
    return np.random.choice(actions, p=probs)


class AssistantEngine:
    """
    Clean wrapper around Gongsta's pre-trained CFR models.

    Usage:
        engine = AssistantEngine()
        rec = engine.recommend(
            hole=["As", "Kh"],
            board=["Qd", "Jh", "2c"],
            history=["b10"],
            pot=120,
            stack=980,
            big_blind=2,
            is_dealer=False,
        )
    """

    def __init__(self, models_dir: "str | Path | None" = None):
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        self.models_dir = Path(models_dir)

        self.preflop_infosets = joblib.load(
            self.models_dir / "preflop_infoSets_batch_19.joblib"
        )
        self.postflop_infosets = joblib.load(
            self.models_dir / "postflop_infoSets_batch_19.joblib"
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        hole: list[str],
        board: list[str],
        history: list[str],
        pot: float,
        stack: float,
        big_blind: float = 2.0,
        is_dealer: bool = False,
        to_call: float = 0.0,
        position: str | None = None,
        effective_stack: float | None = None,
        num_active: int = 2,
    ) -> dict:
        """
        Return a recommendation for the current situation.

        Args:
            hole: list of 2 card strings, e.g. ["As", "Kh"]
            board: list of 0-5 card strings
            history: list of action strings so far in this street,
                     e.g. ["b10"] or ["k"]
            pot: current total pot size
            stack: your remaining stack (hero stack)
            big_blind: big blind size (default 2)
            is_dealer: True if you are the button (dealer)
            to_call: amount to call
            position: optional explicit position (e.g. "CO", "UTG+1").
                      If provided, overrides is_dealer.
            effective_stack: optional effective stack (min stack at table).
                             Used for bet sizing when available.
            num_active: number of active players (2..9). Used to tighten/loosen
                        preflop ranges slightly.

        Returns:
            {
                "action": str,          # e.g. "b40", "k", "c", "f"
                "strategy": dict,       # probability distribution
                "infoset_key": str,
                "stage": str,           # "preflop" | "postflop"
            }
        """
        hole_str = "".join(hole)
        opp_placeholder = "XX"

        # Determina posizione precisa
        actual_position = (
            position.upper() if position else ("BTN" if is_dealer else "BB")
        )

        # Stack effettivo: se non fornito, usa quello dell'eroe
        eff_stack = (
            effective_stack
            if effective_stack is not None and effective_stack > 0
            else stack
        )

        if len(board) == 0:
            result = self._recommend_preflop(
                hole_str,
                opp_placeholder,
                history,
                pot,
                stack,
                big_blind,
                actual_position,
                eff_stack,
                num_active,
            )
        else:
            board_strs = ["".join(board)]
            equity = calculate_equity(hole, board, n=1000)
            result = self._recommend_postflop(
                hole_str,
                opp_placeholder,
                history,
                pot,
                stack,
                big_blind,
                board_strs,
                equity,
                eff_stack,
            )
            # Enrich postflop result with draw info (outs + type)
            outs_calc = OutsCalculator(list(hole), list(board))
            result["draw_outs"] = outs_calc.get_total_outs()
            result["draw_type"] = outs_calc.get_draw_type()

        # Value-bet override: on a late street (turn/river) with no bet to call
        # and high equity, the engine should raise for value, not check/call/fold.
        # This catches cases where the CFR infoset was not found in the
        # pre-trained model (e.g. no action history) and the engine fell back
        # to a uniform random strategy.
        #
        # QUICK WIN #2: Conservative override based on board texture.
        # High-danger boards (flush/straight possible) require higher equity.
        board_danger = self._board_danger_level(board)
        required_equity = {"high": 0.85, "medium": 0.78, "low": 0.72}.get(
            board_danger, 0.75
        )
        if (
            result.get("stage") == "postflop"
            and len(board) >= 4
            and not any(h.startswith("b") for h in (history or []))
        ):
            equity_val = result.get("equity")
            if (
                equity_val is not None
                and equity_val >= required_equity
                and result.get("action") in ("k", "c", "f")
            ):
                bet_size = max(_safe_int(0.5 * pot), 2 * _safe_int(big_blind))
                sizing_stack = effective_stack if effective_stack is not None else stack
                bet_size = min(bet_size, _safe_int(sizing_stack))
                result["action"] = f"b{bet_size}"
                result["value_bet_override"] = True
                result["board_danger"] = board_danger

        # QUICK WIN #3: Cap bet size at 30% of effective stack on postflop to avoid spew.
        if result.get("action", "").startswith("b") and len(board) > 0 and stack > 0:
            try:
                bet_size = int(result["action"][1:])
                sizing_stack = effective_stack if effective_stack is not None else stack
                cap = max(_safe_int(0.30 * sizing_stack), 2 * _safe_int(big_blind))
                if bet_size > cap:
                    result["action"] = f"b{cap}"
                    result["bet_capped_for_stack"] = True
                    print(
                        f"[engine] Bet capped {bet_size} -> {cap} "
                        f"(30% of effective stack, board danger={board_danger})"
                    )
            except (ValueError, IndexError):
                pass

        # QUICK WIN #1: Pot odds check on calls.
        # A call has positive EV only if our equity > pot_odds ratio.
        # If not, fold even with seemingly high equity.
        if result.get("action") == "c" and to_call > 0:
            equity_val = result.get("equity", 0.5)
            pot_after = pot + to_call
            equity_needed = to_call / pot_after if pot_after > 0 else 0
            # 10% margin for safety (also accounts for implied odds being uncertain)
            if equity_val < equity_needed * 1.10:
                result["action"] = "f"
                result["fold_reason"] = (
                    f"pot_odds: need {equity_needed:.1%} equity, have {equity_val:.1%}"
                )
                print(
                    f"[engine] Fold: {result['fold_reason']} "
                    f"(pot={pot}, to_call={to_call})"
                )

        # SAFETY: If to_call > 0 but engine said "k" (check), we can't check.
        # Re-evaluate based on pot odds: call if equity is enough, else fold.
        if to_call > 0 and result.get("action") == "k":
            equity_val = result.get("equity", 0.5)
            pot_after = pot + to_call
            equity_needed = to_call / pot_after if pot_after > 0 else 0
            if equity_val >= equity_needed * 1.10:
                result["action"] = "c"
                print(
                    f"[engine] Converted k -> c (to_call={to_call}, "
                    f"equity {equity_val:.1%} >= needed {equity_needed:.1%})"
                )
            else:
                result["action"] = "f"
                result["fold_reason"] = (
                    f"forced_action: can't check with to_call={to_call}, "
                    f"equity {equity_val:.1%} < needed {equity_needed:.1%}"
                )
                print(
                    f"[engine] Converted k -> f (to_call={to_call}, "
                    f"equity {equity_val:.1%} < needed {equity_needed:.1%})"
                )

        return result

    @staticmethod
    def _board_danger_level(board: list[str]) -> str:
        """
        Classify board texture for value-bet sizing decisions.
        - 'high': flush or straight is very likely (3+ same suit, or 4 connected)
        - 'medium': paired board (full house possible)
        - 'low': dry, uncoordinated board
        """
        if not board or len(board) < 3:
            return "low"
        suits = [c[1] for c in board if len(c) == 2]
        ranks = [c[0] for c in board if len(c) == 2]
        suit_counts = {s: suits.count(s) for s in set(suits)}
        if max(suit_counts.values()) >= 3:
            return "high"
        # Paired board
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        if max(rank_counts.values()) >= 2:
            return "medium"
        # Connected board (e.g., 5-6-7 or T-J-Q)
        rank_order = "23456789TJQKA"
        if len(set(ranks)) >= 3:
            sorted_ranks = sorted(set(ranks), key=lambda r: rank_order.index(r))
            for i in range(len(sorted_ranks) - 2):
                a, b, c = (rank_order.index(r) for r in sorted_ranks[i : i + 3])
                if b - a == 1 and c - b == 1:
                    return "high"
                if b - a == 2 and c - b == 2:
                    return "medium"
        return "low"

    # ------------------------------------------------------------------ #
    #  Preflop
    # ------------------------------------------------------------------ #

    def _recommend_preflop(
        self,
        hole_str,
        opp_str,
        history,
        pot,
        stack,
        big_blind,
        position,
        effective_stack,
        num_active,
    ):
        # Use GTO chart lookup for preflop (far more accurate than under-trained CFR)
        rec = preflop_charts.lookup(
            hole=[hole_str[:2], hole_str[2:4]],
            position=position,
            history=list(history),
            pot=pot,
            stack=stack,
            big_blind=big_blind,
        )

        # Aggiustamento base per numero di giocatori attivi:
        # più siamo in tanti, più stringiamo (specialmente early).
        if (
            num_active > 6
            and position in ("UTG", "UTG+1", "UTG+2", "LJ", "MP")
            and rec["action"] == "bMIN"
            and rec["strategy"].get("bMIN", 0) < 1.0
        ):
            # Marginali: da raise a fold in tavoli pieni early
            rec = {
                "action": "f",
                "strategy": {"f": 1.0},
                "infoset_key": rec["infoset_key"],
                "stage": "preflop",
            }

        abstract_action = rec["action"]
        final_action = self._translate_preflop_action(
            abstract_action, pot, effective_stack, big_blind, history
        )
        return {
            "action": final_action,
            "strategy": rec["strategy"],
            "infoset_key": rec["infoset_key"],
            "stage": "preflop",
            "preflop_equity": preflop_equity_lookup.get_preflop_equity(
                [hole_str[:2], hole_str[2:4]]
            ),
        }

    @staticmethod
    def _abstract_preflop_history(history, big_blind=2):
        stage = copy.deepcopy(history)
        abstracted = stage[:2]

        if len(stage) >= 6 and stage[3] != "c":
            if len(stage) % 2 == 0:
                abstracted += ["bMAX"]
            else:
                abstracted += ["bMIN", "bMAX"]
            return abstracted

        bet_size = big_blind
        pot_total = big_blind + _safe_int(big_blind / 2)

        for action in stage[2:]:
            if action.startswith("b"):
                try:
                    bet_size = _safe_int(action[1:], 0)
                except ValueError:
                    bet_size = big_blind

                last_abs = abstracted[-1]
                if last_abs == "bMIN":
                    abstracted += ["bMID" if bet_size <= 2 * pot_total else "bMAX"]
                elif last_abs == "bMID":
                    abstracted += ["bMAX"]
                elif last_abs == "bMAX":
                    if abstracted[-2] == "bMID":
                        abstracted[-2] = "bMIN"
                    abstracted[-1] = "bMID"
                    abstracted += ["bMAX"]
                else:
                    if bet_size <= pot_total:
                        abstracted += ["bMIN"]
                    elif bet_size <= 2 * pot_total:
                        abstracted += ["bMID"]
                    else:
                        abstracted += ["bMAX"]

                pot_total += bet_size
            elif action == "c":
                pot_total = 2 * bet_size
                abstracted += ["c"]
            else:
                abstracted += [action]

        return abstracted

    @staticmethod
    def _translate_preflop_action(abstract_action, pot, stack, big_blind, history=None):
        if abstract_action == "bMIN":
            # Standard open raise sizing
            return f"b{_safe_int(2.5 * big_blind)}"
        elif abstract_action == "bMID":
            return f"b{max(big_blind, 2 * _safe_int(pot))}"
        elif abstract_action == "bMAX":
            # 3bet / 4bet sizing — roughly 3x the last bet
            last_bet = big_blind
            if history:
                for h in reversed(history):
                    if h.startswith("b"):
                        try:
                            last_bet = int(h[1:])
                        except ValueError:
                            pass
                        break
            sizing = _safe_int(3 * last_bet)
            if sizing >= stack:
                sizing = _safe_int(stack)
            return f"b{sizing}"
        else:
            return abstract_action

    # ------------------------------------------------------------------ #
    #  Postflop
    # ------------------------------------------------------------------ #

    def _recommend_postflop(
        self,
        hole_str,
        opp_str,
        history,
        pot,
        stack,
        big_blind,
        board_strs,
        equity: float | None = None,
        effective_stack: float | None = None,
    ):
        raw_history = [hole_str, opp_str, "/"] + board_strs + list(history)
        abstracted = self._abstract_postflop_history(raw_history, big_blind)

        # Build key manually replicating get_infoSet_key_online logic
        key = self._build_postflop_key(abstracted, hole_str)

        # Use PostflopHoldemHistory only for valid-actions logic
        pph = PostflopHoldemHistory(abstracted)
        infoset = self.postflop_infosets.get(key)
        if infoset is None:
            strategy = {a: 1.0 / len(pph.actions()) for a in pph.actions()}
        else:
            strategy = infoset.get_average_strategy()

        if equity is not None:
            strategy = self._blend_strategy_with_equity(strategy, equity)

        abstract_action = _get_action(strategy)
        final_action = self._translate_postflop_action(
            abstract_action, pot, stack, big_blind
        )

        return {
            "action": final_action,
            "strategy": strategy,
            "infoset_key": key,
            "stage": "postflop",
            "equity": equity,
        }

    @staticmethod
    def _blend_strategy_with_equity(strategy: dict, equity: float) -> dict:
        """
        Mix a GTO strategy with equity-based adjustments.

        High equity (-> 0.5) shifts weight toward betting/raising
        and away from folding. Low equity does the opposite.
        """
        blended = {}
        for action, prob in strategy.items():
            if action in ("bMIN", "bMAX"):
                # More bets when equity is high
                factor = 0.5 + equity
            elif action == "f":
                # More folds when equity is low
                factor = 1.5 - equity
            else:
                factor = 1.0
            blended[action] = max(0.0, prob * factor)

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    @staticmethod
    def _build_postflop_key(abstracted, hole_str):
        """Replicate get_infoSet_key_online without player() dependency."""

        def _norm_card(c):
            if c and len(c) == 2:
                return c[0].upper() + c[1].lower()
            return None

        hand = [_norm_card(hole_str[:2]), _norm_card(hole_str[2:4])]
        community_cards = []
        stage_i = 0
        infoset = []

        for action in abstracted:
            if action not in POSTFLOP_DISCRETE:
                if action == "/":
                    stage_i += 1
                    continue
                if stage_i != 0:
                    chunks = [action[j : j + 2] for j in range(0, len(action), 2)]
                    for c in chunks:
                        nc = _norm_card(c)
                        if nc:
                            community_cards.append(nc)
                if stage_i == 1 or stage_i == 2 or stage_i == 3:
                    infoset.append(str(predict_cluster(hand + community_cards)))
            else:
                infoset.append(action)

        return "".join(infoset)

    @staticmethod
    def _abstract_postflop_history(history, big_blind=2):
        history = copy.deepcopy(history)
        pot_total = big_blind * 2

        flop_start = history.index("/")
        for action in history[:flop_start]:
            if action.startswith("b"):
                bet_size = _safe_int(action[1:], 0)
                pot_total = 2 * bet_size

        abstracted = history[:2]
        stage_start = flop_start
        stage = _get_stage(history[stage_start + 1 :])
        latest_bet = 0

        while True:
            abstracted += ["/"]

            if len(stage) >= 4 and stage[3] != "c":
                abstracted += [stage[0]]
                if stage[-1] == "c":
                    if len(stage) % 2 == 1:
                        abstracted += ["bMAX", "c"]
                    else:
                        if stage[0] == "k":
                            abstracted += ["k", "bMAX", "c"]
                        else:
                            abstracted += ["bMIN", "bMAX", "c"]
                else:
                    if len(stage) % 2 == 0:
                        abstracted += ["bMAX"]
                    else:
                        abstracted += ["bMIN", "bMAX"]
            else:
                for action in stage:
                    if action.startswith("b"):
                        bet_size = _safe_int(action[1:], 0)
                        latest_bet = bet_size

                        if abstracted[-1] == "bMIN":
                            abstracted += ["bMAX"]
                        elif abstracted[-1] == "bMAX":
                            abstracted[-1] = "bMIN"
                            abstracted += ["bMAX"]
                        else:
                            abstracted += ["bMAX" if bet_size >= pot_total else "bMIN"]

                        pot_total += bet_size
                    elif action == "c":
                        pot_total += latest_bet
                        abstracted += ["c"]
                    else:
                        abstracted += [action]

            remaining = history[stage_start + 1 :]
            if "/" not in remaining:
                break
            stage_start = remaining.index("/") + (stage_start + 1)
            stage = _get_stage(history[stage_start + 1 :])

        return abstracted

    @staticmethod
    def _translate_postflop_action(abstract_action, pot, stack, big_blind):
        smallest_bet = _safe_int(big_blind / 2, 1)
        if abstract_action == "bMIN":
            size = max(big_blind, _safe_int(1 / 3 * pot / smallest_bet) * smallest_bet)
            return f"b{size}"
        elif abstract_action == "bMAX":
            size = min(_safe_int(pot), _safe_int(stack))
            return f"b{size}"
        else:
            return abstract_action


def _get_stage(history):
    if "/" in history:
        return history[: history.index("/")]
    return history


# -------------------------------------------------------------------- #
#  Quick manual test
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    engine = AssistantEngine()

    print("--- Preflop test ---")
    rec = engine.recommend(
        hole=["As", "Kh"],
        board=[],
        history=["b10"],
        pot=24,
        stack=988,
        big_blind=2,
    )
    print(rec)

    print("\n--- Postflop test ---")
    rec = engine.recommend(
        hole=["As", "Kh"],
        board=["Qd", "Jh", "2c"],
        history=["b20"],
        pot=64,
        stack=968,
        big_blind=2,
    )
    print(rec)
