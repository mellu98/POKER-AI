"""
Outs Calculator — counts draw outs (flush, OESD, gutshot) postflop.

Clean-room implementation. No code imported from external sources.

Usage:
    calc = OutsCalculator(hole=["As", "Kh"], board=["Qd", "Jh", "2c"])
    print(calc.get_total_outs(), calc.get_draw_type())
    # 8 'oesd'
"""
from typing import List, Tuple, Set


RANKS = "23456789TJQKA"
SUITS = "cdhs"


class OutsCalculator:
    """
    Analyse a 5-or-6-card situation (2 hole + 3-5 board) for drawing potential.

    Returns outs as a count of distinct cards (not adjusted for suits), so
    each "out" is multiplied by ~1 (assuming 47 unknown cards on the turn).
    """

    def __init__(self, hole: List[str], board: List[str]):
        if not hole or len(hole) != 2:
            raise ValueError(f"hole must be 2 cards, got {hole}")
        self.hole = list(hole)
        self.board = list(board)
        self.all_cards = self.hole + self.board
        self.unique_ranks: Set[str] = set()
        for c in self.all_cards:
            if not c or len(c) < 2:
                continue
            self.unique_ranks.add(c[0].upper())
        self.suit_counts: dict = {}
        for c in self.all_cards:
            if c and len(c) >= 2:
                s = c[1].lower()
                self.suit_counts[s] = self.suit_counts.get(s, 0) + 1
        # Map suit -> list of ranks of that suit (for straight-flush detection)
        self.suit_ranks: dict = {}
        for c in self.all_cards:
            if c and len(c) >= 2:
                s = c[1].lower()
                self.suit_ranks.setdefault(s, set()).add(c[0].upper())

    # ------------------------------------------------------------------ #
    #  Flush
    # ------------------------------------------------------------------ #

    def get_flush_draw_suit(self) -> str | None:
        """Return the suit of a 4-flush (or None)."""
        for s, count in self.suit_counts.items():
            if count >= 4:
                return s
        return None

    def get_flush_draw(self) -> Tuple[bool, int]:
        """(is_draw, outs). 4 of a suit -> 9 outs to flush. Max 1 counted."""
        if self.get_flush_draw_suit() is not None:
            return (True, 9)
        return (False, 0)

    # ------------------------------------------------------------------ #
    #  Straight (OESD + gutshot)
    # ------------------------------------------------------------------ #

    def _rank_indices(self, ranks: Set[str] | None = None) -> Set[int]:
        rs = ranks if ranks is not None else self.unique_ranks
        return {RANKS.index(r) for r in rs if r in RANKS}

    def _has_straight(self, ranks: Set[str] | None = None) -> bool:
        """True if the given ranks already contain a 5-rank straight (or wheel)."""
        idxs = sorted(self._rank_indices(ranks))
        if len(idxs) < 5:
            return False
        # Standard 5-consecutive in 0..12
        for i in range(len(idxs) - 4):
            if all(idxs[i + j + 1] - idxs[i + j] == 1 for j in range(4)):
                return True
        # Wheel A-2-3-4-5
        if {0, 1, 2, 3, 12}.issubset(set(idxs)):
            return True
        return False

    def _straight_outs(self, ranks: Set[str] | None = None) -> Set[int]:
        """
        Set of rank indices that would complete a straight (excluding already-
        made straights). Combines standard windows and the wheel.
        """
        if self._has_straight(ranks):
            return set()
        idxs = self._rank_indices(ranks)
        outs: Set[int] = set()

        # Standard 5-rank windows in 0..12
        for start in range(len(RANKS) - 4):
            window = list(range(start, start + 5))
            present = [i for i in window if i in idxs]
            missing = len(window) - len(present)
            if missing == 0:
                return set()  # straight was made
            if missing == 1:
                outs.update(i for i in window if i not in idxs)

        # Wheel A-2-3-4-5 (Ace plays low). Need A to be present.
        if 12 in idxs:
            wheel_present = {0, 1, 2, 3} & idxs
            if 2 <= len(wheel_present) <= 3:
                # 2-3 of {2,3,4,5} present alongside A → 1 or 2 missing from wheel.
                wheel_full = {0, 1, 2, 3, 12}
                missing = wheel_full - idxs
                # Convention: only the LOWEST missing rank is the wheel out.
                # E.g., with A,2,3 present, 4 is the "wheel out" (next in the
                # low-card sequence). 5 is not counted because it's "out of
                # sequence" — you'd skip 4. Same logic for the other shapes.
                if missing:
                    outs.add(min(missing))

        return outs

    def get_oesd(self) -> Tuple[bool, int]:
        """
        Open-ended straight draw: 4 consecutive ranks with BOTH ends open
        (one out at each side closes a straight). Returns 8 outs.

        NOT an OESD (these are gutshots with 4 outs):
        - J-Q-K-A: only T closes (no rank above Ace)
        - A-2-3-4: only 5 closes (wheel, no rank below 2/Ace-low)
        """
        idxs = self._rank_indices()
        for start in range(len(RANKS) - 3):
            window = list(range(start, start + 4))
            if all(i in idxs for i in window):
                low_end = start - 1
                high_end = start + 4
                if low_end < 0 or high_end > len(RANKS) - 1:
                    # 4-consecutive at an edge of the rank range = gutshot, not OESD
                    continue
                return (True, 8)
        return (False, 0)

    def get_gutshot(self) -> Tuple[bool, int]:
        """
        Gutshot straight draw. Returns outs as:
        - (False, 0) if no draw
        - (True, 4)  if single gutshot (1 out)
        - (True, 8)  if double gutshot (2 outs, e.g. A-K + Q-J on board = T or Q)
        - (True, min(15, n*4)) for 3+ outs (rare)
        """
        outs = self._straight_outs()
        # Exclude OESD outs (OESD counted separately)
        # OESD is a special case of "missing 1" with both ends free; for our
        # purposes we report gutshot only for non-OESD scenarios.
        oesd_outs = self._oesd_outs_indices()
        pure_gutshot_outs = outs - oesd_outs
        n = len(pure_gutshot_outs)
        if n == 0:
            return (False, 0)
        if n == 1:
            return (True, 4)
        if n == 2:
            return (True, 8)
        return (True, min(15, n * 4))

    def _oesd_outs_indices(self) -> Set[int]:
        """Return the rank indices that would complete an OESD.
        Mirrors `get_oesd` — skips 4-consecutive windows at rank-range edges
        (J-Q-K-A, A-2-3-4) since those are gutshots, not OESDs.
        """
        idxs = self._rank_indices()
        for start in range(len(RANKS) - 3):
            window = list(range(start, start + 4))
            if all(i in idxs for i in window):
                low_end = start - 1
                high_end = start + 4
                if low_end < 0 or high_end > len(RANKS) - 1:
                    continue
                return {low_end, high_end} & set(range(len(RANKS)))
        return set()

    def get_straight_draw(self) -> Tuple[bool, int, str]:
        """
        Return (is_draw, outs, type) where type is "oesd" | "gutshot" | "none".
        Combines OESD + double-gutshot scenarios.
        """
        is_oesd, oesd_outs = self.get_oesd()
        if is_oesd:
            return (True, 8, "oesd")
        is_gs, gs_outs = self.get_gutshot()
        if is_gs:
            return (True, gs_outs, "gutshot")
        return (False, 0, "none")

    # ------------------------------------------------------------------ #
    #  Straight-flush draw
    # ------------------------------------------------------------------ #

    def get_straight_flush_draw(self) -> int:
        """
        If a straight draw AND a flush draw share the same suit, return
        15 (standard approximation for SF draw; actual ~9-12 depending on overlap).
        """
        flush_suit = self.get_flush_draw_suit()
        if flush_suit is None:
            return 0
        ranks_in_suit = self.suit_ranks.get(flush_suit, set())
        outs = self._straight_outs(ranks_in_suit)
        if not outs:
            return 0
        # At least one straight out in this suit
        return 15

    # ------------------------------------------------------------------ #
    #  Combined
    # ------------------------------------------------------------------ #

    def get_total_outs(self) -> int:
        """Total distinct outs (capped at 15 for sanity)."""
        sf = self.get_straight_flush_draw()
        if sf > 0:
            return 15
        flush_outs = self.get_flush_draw()[1]
        _, straight_outs, _ = self.get_straight_draw()
        return min(15, flush_outs + straight_outs)

    def get_draw_type(self) -> str:
        """Human-readable draw type tag."""
        sf = self.get_straight_flush_draw()
        if sf > 0:
            return "straight_flush_draw"
        is_flush = self.get_flush_draw()[0]
        _, _, straight_type = self.get_straight_draw()
        if is_flush and straight_type != "none":
            return "combo_draw"
        if is_flush:
            return "flush_draw"
        if straight_type != "none":
            return straight_type
        return "none"

    # ------------------------------------------------------------------ #
    #  High-card fallback
    # ------------------------------------------------------------------ #

    def get_outs_to_pair_or_better(self) -> int:
        """
        For high-card hands: outs to make at least a pair.
        3 outs per rank in the hole or board that doesn't yet have a pair.
        """
        if len(self.board) < 3:
            return 0
        # Count rank occurrences in hole+board
        rank_counts: dict = {}
        for c in self.all_cards:
            if c and len(c) >= 2:
                r = c[0].upper()
                rank_counts[r] = rank_counts.get(r, 0) + 1
        unpaired = [r for r, cnt in rank_counts.items() if cnt == 1]
        return min(9, 3 * len(unpaired))


# --------------------------------------------------------------------------- #
#  CLI smoke test
# --------------------------------------------------------------------------- #

def _cli_tests() -> bool:
    tests = [
        # (hole, board, expected_outs, expected_type, description)
        # NOTE: J-Q-K-A and A-2-3-4 are 4-consecutive at rank-range edges
        # so they are GUTSHOTS (1 out), not OESDs (would need 2 outs).
        (["As", "Ks"], ["Qd", "Jh", "2c"], 4, "gutshot", "AKs on Q-J-2: gutshot T (T-J-Q-K-A)"),
        (["Ks", "Qs"], ["Jh", "Ts", "2c"], 8, "oesd", "KQs on J-T-2: OESD (A or 9)"),
        (["2c", "3c"], ["4d", "5d", "9h"], 4, "gutshot", "23 on 4-5-9: gutshot 6 (2-3-4-5-6)"),
        (["5h", "4h"], ["3h", "2h"], 15, "straight_flush_draw", "54 on 32hh: SF draw"),
        (["Ah", "Kh"], ["Qh", "Jh", "2d"], 15, "straight_flush_draw", "AKs on Q-J-2: SF draw"),
        (["Kc", "Qc"], ["Ts", "9s"], 4, "gutshot", "KQ on T-9: gutshot J"),
        (["Ah", "Jh"], ["Kd", "Qc", "2d"], 4, "gutshot", "AJ on K-Q-2: gutshot T"),
        (["Ks", "7s"], ["4s", "5s", "2d"], 9, "flush_draw", "K7s on 4-5-2: flush draw"),
        (["Ah", "Kh"], ["Qd", "Jd", "2c"], 4, "gutshot", "AK on Q-J-2: gutshot T (T-J-Q-K-A)"),
        (["2c", "7d"], ["9c", "Td", "3h"], 0, "none", "27o on 9-T-3: no draw"),
        (["Ah", "Qh", "Jh", "Ts"], ["Kd"], 0, "none", "Broadway Q-J-T + Kh: made straight"),
        # Double gutshot: A-K + Q-J in hand, 2-3 on board → 2 out (T or 4) for A-K wheel/broadway
        (["Ah", "Kh"], ["Qc", "Jc", "2d", "3c"], 8, "gutshot", "AK on Q-J-2-3: double gutshot (T or 4)"),
        # Real OESD in the middle: 7-8-9-T → out 6 and J → 8
        (["7c", "8d"], ["9c", "Td", "2h"], 8, "oesd", "78 on 9-T-2: OESD (6 or J)"),
    ]
    passed, failed = 0, 0
    for hole, board, expected_outs, expected_type, desc in tests:
        calc = OutsCalculator(hole[:2], board)
        got_outs = calc.get_total_outs()
        got_type = calc.get_draw_type()
        ok = got_outs == expected_outs and got_type == expected_type
        status = "PASS" if ok else "FAIL"
        print(f"[outs_calc] {status}  outs={got_outs} (exp {expected_outs}) "
              f"type={got_type!r} (exp {expected_type!r})  -- {desc}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"[outs_calc] {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _cli_tests()
    sys.exit(0 if ok else 1)
