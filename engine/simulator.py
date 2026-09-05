"""
Heads-Up Simulator for the Poker AI Assistant.

Plays N hands of heads-up Texas Hold'em No-Limit against a simple opponent
and reports BB/100 performance.

Usage:
    python engine/simulator.py --hands 500 --opponent calling_station
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from assistant_engine import AssistantEngine
from fast_evaluator import Deck, get_player_score


class HeadsUpSimulator:
    """
    Fast heads-up simulator.

    Simplifications:
      - The bot is always the active bettor (villain only reacts).
      - Turn/river betting is skipped; showdown uses the full 5-card board.
      - Villain never leads (no donk bets).
    """

    def __init__(
        self,
        engine: AssistantEngine,
        hands: int = 500,
        bb: int = 2,
        opponent: str = "calling_station",
        stack_bb: int = 100,
    ):
        self.engine = engine
        self.hands = hands
        self.bb = bb
        self.opponent = opponent
        self.stack_bb = stack_bb
        self.results: list[float] = []

    def run(self) -> float:
        for i in range(self.hands):
            hero_is_dealer = i % 2 == 0
            profit = self._play_hand(hero_is_dealer)
            self.results.append(profit)
        return self._report()

    def _play_hand(self, hero_is_dealer: bool) -> float:
        deck = Deck(excluded_cards=[])
        hero_hole = deck[:2]
        villain_hole = deck[2:4]
        board = deck[4:9]

        pot = 0.0
        hero_invested = 0.0

        # ---- Blinds ----
        if hero_is_dealer:
            pot += self.bb * 1.5
            hero_invested += self.bb / 2
        else:
            pot += self.bb * 1.5
            hero_invested += self.bb

        # ---- Preflop ----
        if hero_is_dealer:
            rec = self.engine.recommend(
                hole=hero_hole,
                board=[],
                history=[],
                pot=pot,
                stack=self.stack_bb * self.bb,
                big_blind=self.bb,
                is_dealer=True,
            )
            action = rec["action"]
            if action == "f":
                return -hero_invested
            if action.startswith("b"):
                bet = int(action[1:])
                pot += bet
                hero_invested += bet
                if not self._villain_calls(bet, pot):
                    return pot - hero_invested
                pot += bet
            elif action == "c":
                pot += self.bb / 2
                hero_invested += self.bb / 2
        else:
            # Hero is BB; villain limps (simplified)
            rec = self.engine.recommend(
                hole=hero_hole,
                board=[],
                history=[],
                pot=pot,
                stack=self.stack_bb * self.bb,
                big_blind=self.bb,
                is_dealer=False,
            )
            action = rec["action"]
            if action == "f":
                return -hero_invested
            if action.startswith("b"):
                bet = int(action[1:])
                pot += bet
                hero_invested += bet
                if not self._villain_calls(bet, pot):
                    return pot - hero_invested
                pot += bet
            # else check: proceed to flop

        # ---- Flop ----
        flop = board[:3]
        rec = self.engine.recommend(
            hole=hero_hole,
            board=flop,
            history=[],
            pot=pot,
            stack=self.stack_bb * self.bb - hero_invested,
            big_blind=self.bb,
            is_dealer=hero_is_dealer,
        )
        action = rec["action"]
        if action == "f":
            return -hero_invested
        if action.startswith("b"):
            bet = int(action[1:])
            pot += bet
            hero_invested += bet
            if not self._villain_calls(bet, pot):
                return pot - hero_invested
            pot += bet
        # else check: proceed to showdown

        # ---- Showdown (full board) ----
        hero_score = get_player_score(hero_hole, board)
        villain_score = get_player_score(villain_hole, board)
        if hero_score < villain_score:
            return pot - hero_invested
        if hero_score == villain_score:
            return (pot / 2) - hero_invested
        return -hero_invested

    def _villain_calls(self, facing_bet: float, pot: float) -> bool:
        if self.opponent == "calling_station":
            return True
        if self.opponent == "tight":
            return facing_bet <= pot
        # random
        return random.random() > 0.5

    def _report(self) -> float:
        total = sum(self.results)
        bb_per_hand = total / (self.hands * self.bb)
        bb_per_100 = bb_per_hand * 100
        wins = sum(1 for r in self.results if r > 0)
        print("=" * 50)
        print(f"Heads-Up Simulator Results ({self.hands} hands)")
        print("=" * 50)
        print(f"Opponent model : {self.opponent}")
        print(f"Total profit   : {total:+.0f} chips")
        print(f"BB/100         : {bb_per_100:+.2f}")
        print(f"Win rate       : {wins / self.hands:.1%}")
        print("=" * 50)
        return bb_per_100


def main():
    parser = argparse.ArgumentParser(description="Heads-Up Poker Simulator")
    parser.add_argument("--hands", type=int, default=500, help="Number of hands")
    parser.add_argument(
        "--opponent",
        type=str,
        default="calling_station",
        choices=["calling_station", "tight", "random"],
        help="Opponent strategy",
    )
    parser.add_argument("--bb", type=int, default=2, help="Big blind size")
    parser.add_argument("--stack", type=int, default=100, help="Starting stack in BB")
    args = parser.parse_args()

    engine = AssistantEngine()
    sim = HeadsUpSimulator(
        engine=engine,
        hands=args.hands,
        bb=args.bb,
        opponent=args.opponent,
        stack_bb=args.stack,
    )
    sim.run()


if __name__ == "__main__":
    main()
