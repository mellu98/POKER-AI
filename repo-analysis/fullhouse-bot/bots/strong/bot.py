"""Solver-aligned heads-up benchmark bot.

The single job of this bot is to be a non-trivial gating opponent for the rest
of the project. It plays the same line every hand (no opponent classification,
no exploit layer) using:

  * HU-specific preflop ranges (the 6-max chart we ship for the champion is
    far too tight for SB-vs-BB heads-up). SB opens ~80%, BB defends ~60%+,
    3-bet/4-bet trees are polarised with the right blockers.
  * Range-conditioned Monte Carlo equity (same trick as the champion's
    `equity_vs_range`) so we don't bluff-catch as if villain held random cards.
  * Pot-odds + minimum-defence-frequency continuing ranges so we are not
    exploitable by overbets.
  * Polar river sizing tied to a fixed 1:2 bluff:value frequency on dry
    boards (closer to 1:1.5 with busted-draw blockers on wet runouts), and
    33%/66%/100% c-bet sizing on flop keyed to board texture.

There is *no* opponent model, *no* per-hand RNG state that bleeds across
hands (RNG is reseeded per decision from a hash of hole+board+street), and
the only state we keep is a small lru_cache of equity estimates. The bot
should look identical to a population profiler regardless of how many hands
we've played.
"""

from __future__ import annotations

import functools
import hashlib
import random
import time

import eval7

BOT_NAME = "fullhouse-strong"
BOT_AVATAR = "robot_2"

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000

_RANK_ORDER = "23456789TJQKA"
_RANK_VAL = {r: i for i, r in enumerate(_RANK_ORDER)}


# Hand classification
def _hand_class(cards: list[str]) -> str:
    r0, s0 = cards[0][0], cards[0][1]
    r1, s1 = cards[1][0], cards[1][1]
    if _RANK_VAL[r0] < _RANK_VAL[r1]:
        r0, r1 = r1, r0
        s0, s1 = s1, s0
    if r0 == r1:
        return r0 + r1
    return r0 + r1 + ("s" if s0 == s1 else "o")


# Approximate equity-vs-random ordering of the 169 starting hands. Used only
# to pick "top X%" ranges for villain conditioning. Order does not need to be
# pixel-perfect; we lean on natural breakpoints (premiums / broadways / pocket
# pairs / suited connectors).
_HAND_RANK: list[str] = [
    "AA", "KK", "QQ", "JJ", "TT", "99",
    "AKs", "AKo", "AQs", "AQo", "AJs", "ATs",
    "KQs", "KJs", "KTs", "QJs", "QTs", "JTs",
    "88", "77", "66", "55", "44", "33", "22",
    "AJo", "ATo", "KQo", "KJo", "QJo",
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s",
    "J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s",
    "T9s", "T8s", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s",
    "98s", "97s", "96s", "95s", "94s", "93s", "92s",
    "87s", "86s", "85s", "84s", "83s", "82s",
    "76s", "75s", "74s", "73s", "72s",
    "65s", "64s", "63s", "62s",
    "54s", "53s", "52s",
    "43s", "42s",
    "32s",
    "KTo", "K9o", "QTo", "Q9o", "JTo", "J9o",
    "T9o", "T8o", "98o", "97o", "87o", "86o", "76o", "65o", "54o",
    "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
    "K8o", "K7o", "K6o", "K5o", "K4o", "K3o", "K2o",
    "Q8o", "Q7o", "Q6o", "Q5o", "Q4o", "Q3o", "Q2o",
    "J8o", "J7o", "J6o", "J5o", "J4o", "J3o", "J2o",
    "T7o", "T6o", "T5o", "T4o", "T3o", "T2o",
    "96o", "95o", "94o", "93o", "92o",
    "85o", "84o", "83o", "82o",
    "75o", "74o", "73o", "72o",
    "64o", "63o", "62o",
    "53o", "52o",
    "43o", "42o",
    "32o",
]
_seen: set[str] = set()
_HAND_RANK = [h for h in _HAND_RANK if not (h in _seen or _seen.add(h))]
_HAND_RANK_IDX = {h: i for i, h in enumerate(_HAND_RANK)}


def _top_pct(pct: float) -> frozenset[str]:
    pct = max(0.01, min(1.0, pct))
    n = max(1, int(round(len(_HAND_RANK) * pct)))
    return frozenset(_HAND_RANK[:n])


@functools.lru_cache(maxsize=128)
def _top_pct_cached(pct_int: int) -> frozenset[str]:
    return _top_pct(pct_int / 100.0)


# Heads-up preflop ranges (SB = dealer, opens first preflop)
#
# These are simplified solver-aligned HU 100bb mins. SB raises ~85% open
# (limp the rest), BB 3-bets ~16% polar (premiums + Axs + suited connector
# bluffs), defends ~60% total. Numbers chosen so:
#   * "play this hand from SB" = SB_OPEN.get(hclass, False)
#   * "3-bet vs SB open from BB" = BB_3BET.get(hclass, 0.0) > threshold
#   * "BB defend vs SB open (call+3bet)" = BB_DEFEND.get(hclass, 0.0) > 0.5
#   * "SB 4-bet vs BB 3-bet" = SB_4BET.get(hclass, False)
#   * "SB continue vs BB 3-bet (call)" = SB_CALL_3BET.get(hclass, 0.0) > 0.5

# SB open-raise range (~85% of hands, drop the absolute trash). Mixed with
# limps for the rest, but for benchmark simplicity we always raise the open
# range and fold/limp everything else.
_SB_OPEN: frozenset[str] = frozenset({
    # All pairs.
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
    # All suited aces, kings, queens, jacks.
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s",
    "JTs", "J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s",
    "T9s", "T8s", "T7s", "T6s", "T5s", "T4s",
    "98s", "97s", "96s", "95s", "94s",
    "87s", "86s", "85s", "84s",
    "76s", "75s", "74s",
    "65s", "64s", "63s",
    "54s", "53s",
    "43s",
    # Offsuit: all broadway + medium offsuit aces + connected gappers.
    "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o", "A4o", "A3o", "A2o",
    "KQo", "KJo", "KTo", "K9o", "K8o", "K7o", "K6o",
    "QJo", "QTo", "Q9o", "Q8o", "Q7o",
    "JTo", "J9o", "J8o",
    "T9o", "T8o", "T7o",
    "98o", "97o",
    "87o", "86o",
    "76o", "75o",
    "65o",
    "54o",
})

# BB 3-bet range vs SB min-raise/open. Polar: top-of-range + suited-ace
# bluffs + a few suited connectors. ~16% of hands.
_BB_3BET: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "JJ", "TT",
    "AKs", "AKo", "AQs", "AQo", "AJs", "ATs",
    "KQs", "KJs",
    "A5s", "A4s", "A3s", "A2s",
    "76s", "65s", "54s",
})

# BB call (flat) range vs SB open. Together with _BB_3BET this defends
# ~60% of hands. Hands not in 3-bet and not in flat are folded.
_BB_FLAT: frozenset[str] = frozenset({
    "99", "88", "77", "66", "55", "44", "33", "22",
    "AJo", "ATo", "A9s", "A8s", "A7s", "A6s",
    "A9o", "A8o", "A7o", "A6o", "A5o",
    "KQo", "KJo", "KTo", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s",
    "K9o", "K8o",
    "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s",
    "QJo", "QTo", "Q9o",
    "JTs", "J9s", "J8s", "J7s",
    "JTo", "J9o",
    "T9s", "T8s", "T7s", "T6s",
    "T9o",
    "98s", "97s", "96s",
    "98o",
    "87s", "86s", "85s",
    "87o",
    "75s", "74s",
    "64s",
    "53s",
    "43s",
})

# SB 4-bet vs BB 3-bet. Tight: premiums for value + a couple suited-A
# bluffs that retain blockers.
_SB_4BET: frozenset[str] = frozenset({
    "AA", "KK", "QQ", "AKs", "AKo",
    "A5s", "A4s",  # blocker bluffs
})

# SB call (flat) vs BB 3-bet. Realisable hands: pocket pairs + broadway
# suited + suited connectors. Folds the rest.
_SB_CALL_3BET: frozenset[str] = frozenset({
    "JJ", "TT", "99", "88", "77", "66", "55",
    "AQs", "AQo", "AJs", "ATs",
    "KQs", "KJs", "KTs",
    "QJs", "QTs", "JTs", "T9s",
    "98s", "87s", "76s", "65s",
})


# Decision RNG: hash-driven so the same situation gets the same answer,
# and the answer does not depend on call ordering across hands. This means
# any opponent-classifier sees a fixed marginal distribution over our hand
# classes -> the bot can't be labeled "nit" or "lag" over a small sample.

def _det_random(*parts) -> float:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / (1 << 64)


# Heads-up position detection
#
# In HU the dealer is SB and acts first preflop, BB acts first postflop.

def _hu_position(state: dict) -> str:
    """Return "SB" or "BB" for the seat to act. HU only (assumes 2 players)."""
    seat = state["seat_to_act"]
    log = state.get("action_log") or []
    for entry in log[:4]:
        if entry.get("action") == "small_blind":
            return "SB" if entry.get("seat") == seat else "BB"
        if entry.get("action") == "big_blind":
            return "BB" if entry.get("seat") == seat else "SB"
    # Fallback: in HU the player whose bet_this_street is SMALL_BLIND posted
    # SB. If we can't tell, assume seat 0 is dealer.
    players = state.get("players") or []
    for p in players:
        if p.get("seat") == seat:
            if p.get("bet_this_street") == SMALL_BLIND:
                return "SB"
            if p.get("bet_this_street") == BIG_BLIND:
                return "BB"
    return "SB" if seat == 0 else "BB"


def _is_heads_up(state: dict) -> bool:
    return len(state.get("players") or []) == 2


# Preflop raise/aggression counting
def _preflop_context(state: dict) -> dict:
    log = state.get("action_log") or []
    raises = 0
    callers = 0
    last_raiser_seat = None
    for entry in log:
        a = entry.get("action")
        if a == "raise" or a == "all_in":
            raises += 1
            last_raiser_seat = entry.get("seat")
            callers = 0
        elif a == "call":
            callers += 1
    return {
        "raises": raises,
        "callers": callers,
        "last_raiser_seat": last_raiser_seat,
        "current_bet": state.get("current_bet", BIG_BLIND),
    }


# Monte Carlo equity (cached, range-conditioned)
_DECK52: list[eval7.Card] = [eval7.Card(r + s) for r in _RANK_ORDER for s in "shdc"]

_CLASS_COMBOS: dict[str, list[tuple[eval7.Card, eval7.Card]]] = {}


def _build_class_combos() -> None:
    for hc in _HAND_RANK:
        combos: list[tuple[eval7.Card, eval7.Card]] = []
        if len(hc) == 2:
            r = hc[0]
            for i, s1 in enumerate("shdc"):
                for s2 in "shdc"[i + 1:]:
                    combos.append((eval7.Card(r + s1), eval7.Card(r + s2)))
        else:
            r1, r2, suited = hc[0], hc[1], hc[2] == "s"
            if suited:
                for s in "shdc":
                    combos.append((eval7.Card(r1 + s), eval7.Card(r2 + s)))
            else:
                for s1 in "shdc":
                    for s2 in "shdc":
                        if s1 == s2:
                            continue
                        combos.append((eval7.Card(r1 + s1), eval7.Card(r2 + s2)))
        _CLASS_COMBOS[hc] = combos


_build_class_combos()


def _equity_vs_range_raw(
    hole: tuple[str, ...],
    board: tuple[str, ...],
    rng_classes: tuple[str, ...],
    trials: int,
    seed: int,
) -> float:
    if not rng_classes:
        rng_classes = tuple(_HAND_RANK[:30])
    rng = random.Random(seed)
    hole_cards = [eval7.Card(c) for c in hole]
    board_cards = [eval7.Card(c) for c in board]
    blocked = set(hole_cards) | set(board_cards)
    avail = [c for c in _DECK52 if c not in blocked]
    needed = 5 - len(board_cards)

    wins = 0.0
    n = 0
    classes = list(rng_classes)
    choice = rng.choice
    sample = rng.sample

    for _ in range(trials):
        opp_cards = None
        for _attempt in range(4):
            cls = choice(classes)
            combos = _CLASS_COMBOS.get(cls)
            if not combos:
                continue
            c0, c1 = choice(combos)
            if c0 in blocked or c1 in blocked:
                continue
            opp_cards = [c0, c1]
            break
        if opp_cards is None:
            opp_cards = sample(avail, 2)
        opp_blocked = {opp_cards[0], opp_cards[1]}
        pool = [c for c in avail if c not in opp_blocked]
        if needed > 0:
            rest = sample(pool, needed)
        else:
            rest = []
        full_board = board_cards + rest
        s_me = eval7.evaluate(hole_cards + full_board)
        s_op = eval7.evaluate(opp_cards + full_board)
        if s_me > s_op:
            wins += 1.0
        elif s_me == s_op:
            wins += 0.5
        n += 1
    return wins / n if n else 0.5


@functools.lru_cache(maxsize=4096)
def _cached_equity(
    hole: tuple[str, ...],
    board: tuple[str, ...],
    rng_classes: tuple[str, ...],
    trials_bucket: int,
) -> float:
    seed = hash((hole, board, rng_classes, trials_bucket)) & 0xFFFF_FFFF
    return _equity_vs_range_raw(hole, board, rng_classes, trials_bucket, seed)


def _equity(
    hole: list[str], board: list[str], rng_classes, trials: int = 200
) -> float:
    h = tuple(sorted(hole))
    b = tuple(sorted(board))
    rc = tuple(sorted(rng_classes))
    return _cached_equity(h, b, rc, trials)


# Villain range estimation for HU
#
# We start from a base "what fraction of hands would villain take this line
# with" and then shrink it by each subsequent raise.

def _villain_range(state: dict) -> frozenset[str]:
    log = state.get("action_log") or []
    raises = 0
    for entry in log:
        a = entry.get("action")
        if a in ("raise", "all_in"):
            raises += 1

    if raises == 0:
        # Limp/check pot. Villain has the wide passive part of their range.
        base_pct = 0.55
    elif raises == 1:
        # Vs a single open. In HU SB opens ~80% so this is wide.
        base_pct = 0.70
    elif raises == 2:
        # 3-bet pot. ~16%.
        base_pct = 0.18
    elif raises == 3:
        # 4-bet pot. ~5%.
        base_pct = 0.06
    else:
        base_pct = 0.03

    # Each postflop barrel from villain narrows further.
    me_seat = state.get("seat_to_act")
    barrels = 0
    for p in state.get("players") or []:
        if p.get("seat") == me_seat:
            continue
        if (p.get("bet_this_street", 0) or 0) > 0 and not p.get("is_folded"):
            barrels += 1
    if state.get("street") != "preflop" and state.get("amount_owed", 0) > 0:
        for _ in range(min(barrels, 3)):
            base_pct *= 0.78

    base_pct = max(0.03, min(0.90, base_pct))
    return _top_pct_cached(int(round(base_pct * 100)))


# Board texture
def _board_features(board: list[str]) -> dict:
    if not board:
        return {"flush": 0, "str_draws": 0, "paired": False, "high": 0, "broadway": 0}
    suits = [c[1] for c in board]
    ranks = sorted({_RANK_VAL[c[0]] for c in board})
    counts: dict[str, int] = {}
    for c in board:
        counts[c[0]] = counts.get(c[0], 0) + 1

    flush_count = max(suits.count(s) for s in set(suits))
    flush = 2 if flush_count >= 3 else (1 if flush_count == 2 else 0)

    str_draws = 0
    for lo in range(0, 9):
        window = set(range(lo, lo + 5))
        if len(window & set(ranks)) >= 2:
            str_draws += 1
    if {0, 1, 2, 3, 12} & set(ranks):
        if len({0, 1, 2, 3, 12} & set(ranks)) >= 2:
            str_draws += 1

    paired = any(v >= 2 for v in counts.values())
    high = max(_RANK_VAL[c[0]] for c in board)
    broadway = sum(1 for c in board if c[0] in "TJQKA")
    return {
        "flush": flush,
        "str_draws": str_draws,
        "paired": paired,
        "high": high,
        "broadway": broadway,
    }


def _texture_label(board: list[str]) -> str:
    if len(board) < 3:
        return "dry"
    tx = _board_features(board)
    coord = tx["str_draws"] + (2 if tx["flush"] == 1 else 0) + (4 if tx["flush"] == 2 else 0)
    if tx["paired"]:
        return "dry"
    if coord <= 2 and tx["broadway"] <= 1:
        return "dry"
    if coord >= 5 or tx["flush"] >= 2:
        return "wet"
    return "semi"


# Hand category on a board
def _hand_category(hole: list[str], board: list[str]) -> str:
    """Return one of: nuts, strong, medium, weak, air."""
    if not board:
        return "preflop"
    score = eval7.evaluate(
        [eval7.Card(c) for c in hole] + [eval7.Card(c) for c in board]
    )
    htype = str(eval7.handtype(score))
    if htype not in ("High Card", "Pair"):
        # Two pair plus is "nuts" tier in this taxonomy. Distinguish trips+
        # from two-pair so we can size bigger.
        if htype in ("Two Pair",):
            return "strong" if _is_top_two_pair(hole, board) else "medium"
        return "nuts"

    h_ranks = [_RANK_VAL[c[0]] for c in hole]
    b_ranks = sorted({_RANK_VAL[c[0]] for c in board})
    top = b_ranks[-1] if b_ranks else 0

    if htype == "Pair":
        if h_ranks[0] == h_ranks[1] and h_ranks[0] > top:
            return "strong"  # overpair
        if top in h_ranks:
            kicker = max(h for h in h_ranks if h != top) if h_ranks[0] != h_ranks[1] else h_ranks[0]
            return "strong" if kicker >= 10 else "medium"
        if h_ranks[0] == h_ranks[1]:
            return "medium"  # under/middle pocket
        return "medium"  # paired non-top board card

    # High card: count outs.
    suits = [c[1] for c in hole]
    b_suits = [c[1] for c in board]
    has_flush_draw = suits[0] == suits[1] and b_suits.count(suits[0]) >= 2
    all_ranks = sorted(set(h_ranks) | set(_RANK_VAL[c[0]] for c in board))
    has_oesd = False
    has_gutshot = False
    for lo in range(0, 11):
        window = set(range(lo, lo + 5))
        hits = len(window & set(all_ranks))
        if hits >= 4:
            has_oesd = True
            break
        if hits == 3 and len(window & set(h_ranks)) >= 1:
            has_gutshot = True

    if has_flush_draw and has_oesd:
        return "strong"  # combo draw plays as value
    if has_flush_draw or has_oesd:
        return "weak"  # strong draw
    if has_gutshot and max(h_ranks) >= 11:
        return "weak"
    if max(h_ranks) >= 12:
        return "weak"  # ace high
    return "air"


def _is_top_two_pair(hole: list[str], board: list[str]) -> bool:
    b_ranks = sorted({_RANK_VAL[c[0]] for c in board}, reverse=True)
    h_ranks = {_RANK_VAL[c[0]] for c in hole}
    if len(b_ranks) < 2:
        return False
    return b_ranks[0] in h_ranks and b_ranks[1] in h_ranks


# Action helpers
def _cap_raise(state: dict, raise_to: int) -> int:
    my_stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    min_to = state["min_raise_to"]
    max_to = my_stack + my_bet
    return max(min_to, min(raise_to, max_to))


def _raise_to(state: dict, target: int) -> dict:
    target = _cap_raise(state, target)
    my_stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    if target >= my_stack + my_bet:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}


def _call(state: dict) -> dict:
    return {"action": "call"}


def _check_or_fold(state: dict) -> dict:
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "fold"}


def _spr(state: dict) -> float:
    pot = max(1, state["pot"])
    stack = state["your_stack"]
    return stack / pot


# Preflop
#
# Strategy is hardcoded per "raises in front of us" branch. Mixed frequencies
# are resolved by hashing hole cards so a given concrete hand always picks
# the same branch.

def _decide_preflop(state: dict) -> dict:
    hole = state["your_cards"]
    hclass = _hand_class(hole)
    ctx = _preflop_context(state)
    raises = ctx["raises"]
    pos = _hu_position(state) if _is_heads_up(state) else "BTN"
    cur_bet = ctx["current_bet"]
    stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    min_to = state["min_raise_to"]
    owed = state["amount_owed"]
    can_check = state.get("can_check", False)
    bb = BIG_BLIND

    # ----------------------- 6+ player / non-HU table -----------------------
    if not _is_heads_up(state):
        return _decide_preflop_multiway(state, hclass, ctx, pos)

    # ----------------------- Heads-up branch --------------------------------

    # SB open (raises == 0, can't check because we owe the SB-to-BB top-up).
    if raises == 0:
        if pos == "SB":
            if hclass in _SB_OPEN:
                # Min-raise to 2.2x to keep SPR reasonable.
                target = int(2.2 * bb)
                return _raise_to(state, max(target, min_to))
            # Out-of-open range from SB: limp (call SB) rather than fold —
            # we can realise equity cheaply with everything.
            return _call(state)
        # BB and nobody raised: option closed if SB limped, just check.
        if can_check:
            return {"action": "check"}
        # Otherwise call (covers oddball state where current_bet == BB).
        return _call(state)

    # Facing a single raise.
    if raises == 1:
        if pos == "BB":
            # vs SB open: 3-bet polar, flat, or fold.
            if hclass in _BB_3BET:
                # 3-bet to 3.3x SB's raise (so ~7.3bb vs 2.2bb open).
                size = int(cur_bet * 3.3)
                return _raise_to(state, max(size, min_to))
            if hclass in _BB_FLAT:
                # Call the open.
                if owed > stack:
                    return {"action": "all_in"}
                return _call(state)
            # Fold.
            return _check_or_fold(state)

        # We are SB and BB limp-checked? Not possible (preflop after BB check
        # is closed). If somehow we are SB facing a raise (BB jammed over our
        # limp), behave as 3-bet defence:
        if hclass in _SB_4BET:
            size = int(cur_bet * 2.4)
            return _raise_to(state, max(size, min_to))
        if hclass in _SB_CALL_3BET:
            return _call(state)
        return _check_or_fold(state)

    # 3-bet pot.
    if raises == 2:
        if pos == "SB":
            # We are SB facing BB 3-bet.
            if hclass in {"AA", "KK"}:
                # All-in for value.
                return {"action": "all_in"}
            if hclass in _SB_4BET:
                size = int(cur_bet * 2.3)
                return _raise_to(state, max(size, min_to))
            if hclass in _SB_CALL_3BET:
                if owed >= stack * 0.85:
                    # Hands that can't get a flop should fold to a shove.
                    if hclass in {"AQs", "AQo", "JJ", "TT"}:
                        return {"action": "all_in"}
                    return _check_or_fold(state)
                return _call(state)
            return _check_or_fold(state)

        # BB facing 4-bet (SB 4-bet our 3-bet).
        if hclass in {"AA", "KK", "AKs"}:
            return {"action": "all_in"}
        if hclass in {"QQ", "AKo"}:
            # SPR-tuned: if 4-bet is small, flat; if large, jam.
            if owed >= stack * 0.5:
                return {"action": "all_in"}
            return _call(state)
        return _check_or_fold(state)

    # 4-bet+ pot (raises >= 3): only premiums continue.
    if hclass in {"AA", "KK", "AKs"}:
        return {"action": "all_in"}
    return _check_or_fold(state)


def _decide_preflop_multiway(
    state: dict, hclass: str, ctx: dict, pos: str
) -> dict:
    """Fallback for 3+ player tables. Use a tight TAG range — we are mainly
    being run heads-up but the validator may invoke a multi-way state."""
    raises = ctx["raises"]
    can_check = state.get("can_check", False)
    cur_bet = ctx["current_bet"]
    min_to = state["min_raise_to"]
    stack = state["your_stack"]

    # Tight open range for non-HU.
    open_range = {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AKo", "AQs", "AQo", "AJs", "ATs",
        "KQs", "KJs", "QJs", "JTs", "T9s", "98s",
        "AJo", "KQo",
    }
    if raises == 0:
        if hclass in open_range:
            target = int(2.5 * BIG_BLIND)
            return _raise_to(state, max(target, min_to))
        return _check_or_fold(state)
    if raises == 1:
        if hclass in {"AA", "KK", "QQ", "AKs"}:
            return _raise_to(state, max(int(cur_bet * 3.0), min_to))
        if hclass in {"JJ", "TT", "AQs", "AJs", "KQs", "AKo"}:
            return _call(state)
        return _check_or_fold(state)
    if raises >= 2:
        if hclass in {"AA", "KK", "AKs"}:
            return {"action": "all_in"}
        if hclass in {"QQ", "AKo"}:
            return _call(state)
        return _check_or_fold(state)
    return _check_or_fold(state)


# Postflop
#
# Strategy is "GTO-flavoured":
#   - Sizing chosen by texture (33/66/100% pot) for value, polar for bluff.
#   - Bluff frequency tied to a fixed schedule (45% c-bet flop, 35% turn bluff
#     if value, ~25% river bluff with blockers) but we *never* turn a hand
#     into a bluff that has showdown value.
#   - Continue (defend) decisions use pot odds + a 12% threshold pad to keep
#     us slightly above the minimum-defence frequency vs typical bluff freqs.

def _was_preflop_aggressor(state: dict) -> bool:
    log = state.get("action_log") or []
    me_seat = state.get("seat_to_act")
    last_pf_raiser = None
    for entry in log:
        if entry.get("action") in ("small_blind", "big_blind"):
            continue
        # Postflop entries have a `street` only if recorded — here action_log
        # is the flat list, so we need to detect when preflop ended. We look
        # at all raise entries that come before any action on the flop. The
        # engine resets bet_this_street, but action_log preserves chronological
        # order. We treat anything before the first non-blind/non-preflop-bet
        # signal as preflop. Cheap proxy: just take the last raise/all_in
        # before community_cards length grew. Since the flat log doesn't
        # carry street, we approximate by taking the *first* raise (which in
        # HU is the SB open if any) -- in HU only one preflop raise typically
        # happens per hand for our profile.
        a = entry.get("action")
        if a in ("raise", "all_in"):
            last_pf_raiser = entry.get("seat")
    return last_pf_raiser == me_seat


def _street_raise_count(state: dict) -> int:
    """How many raises have occurred on the current street (postflop only)."""
    log = state.get("action_log") or []
    street = state.get("street", "preflop")
    if street == "preflop":
        return 0
    # We don't know the precise street boundary from the flat log. Use the
    # current `current_bet` and the players' `bet_this_street` to detect open
    # aggression: if any player has bet_this_street > 0, someone has bet.
    bets_this_st = sum(
        1 for p in (state.get("players") or [])
        if (p.get("bet_this_street", 0) or 0) > 0
    )
    return min(bets_this_st, 3)


def _cbet_sizing_frac(board: list[str], polar: bool) -> float:
    """Return c-bet sizing as a fraction of pot."""
    tx = _board_features(board)
    paired = tx["paired"]
    coord = tx["str_draws"] + (2 if tx["flush"] == 1 else 0) + (4 if tx["flush"] == 2 else 0)
    broad = tx["broadway"]
    # Dry, paired, or one-broadway boards: small. Wet/two-broadway: bigger.
    if paired and coord <= 2:
        return 0.33
    if coord <= 2 and broad <= 1:
        return 0.33
    if coord >= 5 or tx["flush"] >= 2:
        return 0.80 if polar else 0.66
    return 0.55


def _decide_postflop(state: dict, deadline: float) -> dict:
    hole = state["your_cards"]
    board = state["community_cards"]
    pot = max(1, state["pot"])
    stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    owed = state["amount_owed"]
    can_check = state.get("can_check", False)
    street = state["street"]

    spr = _spr(state)
    cat = _hand_category(hole, board)
    rng = _villain_range(state)

    # Adaptive trial count: budget ~120-200 simulations.
    trials = 180 if time.perf_counter() < deadline - 0.5 else 80
    equity = _equity(hole, board, rng, trials=trials)
    eff_eq = max(0.0, equity - 0.02)

    # 3-bet-pot low-SPR shove-or-fold tree.
    raises_pre = _count_preflop_raises(state)
    if spr < 4.0 and raises_pre >= 2:
        if cat in ("nuts", "strong"):
            if can_check:
                return _raise_to(state, my_bet + pot)
            return {"action": "all_in"}
        if not can_check:
            call_eq = owed / (pot + owed) if pot + owed else 1.0
            if eff_eq >= call_eq + 0.06:
                return _call(state)
            return {"action": "fold"}
        return {"action": "check"}

    # Facing a bet.
    if not can_check and owed > 0:
        call_eq = owed / (pot + owed) if pot + owed else 1.0
        pad = 0.05  # base pad — slightly looser than pot-odds to fight bluffs

        bet_to_pot = owed / pot
        if bet_to_pot >= 0.7:
            pad += 0.03
        if street in ("turn", "river") and bet_to_pot >= 0.7:
            pad += 0.02

        # Multi-street barrel pressure (rough).
        if street == "river" and bet_to_pot >= 0.9:
            pad += 0.02

        if eff_eq >= call_eq + pad:
            # Raise for value with strong hands when not facing a re-raise.
            rcount = _street_raise_count(state)
            if (
                cat in ("nuts",)
                and rcount <= 1
                and eff_eq >= 0.78
            ):
                frac = 0.85
                target = my_bet + int((pot + owed) * frac) + owed
                return _raise_to(state, target)
            return _call(state)
        return {"action": "fold"}

    # Checked to us / first to act on a new street.
    rcount = _street_raise_count(state)
    pfa = _was_preflop_aggressor(state)

    # Value-bet thresholds by street.
    value_th = {"flop": 0.55, "turn": 0.58, "river": 0.60}.get(street, 0.6)

    # Hash-derived per-hand mixed frequency for bluff selection.
    bluff_roll = _det_random(
        tuple(hole), tuple(board), street, "bluff",
    )

    # Value bet.
    if eff_eq >= value_th or cat in ("nuts", "strong"):
        polar = cat == "nuts" and street == "river" and _texture_label(board) != "dry"
        frac = _cbet_sizing_frac(board, polar=polar)
        if street == "river" and cat == "nuts" and eff_eq >= 0.82:
            # Overbet for thin value with the nuts on safe runouts.
            frac = max(frac, 1.0)
        target = my_bet + int(pot * frac)
        return _raise_to(state, target)

    # As preflop aggressor on flop: cbet with balanced freq.
    if street == "flop" and pfa and rcount == 0:
        tx_label = _texture_label(board)
        # On dry boards, cbet very wide (~70%). On wet, cbet narrower (~50%)
        # but bigger.
        cbet_freq = 0.70 if tx_label == "dry" else 0.45
        if eff_eq >= 0.25 and bluff_roll < cbet_freq:
            frac = _cbet_sizing_frac(board, polar=False)
            target = my_bet + int(pot * frac)
            return _raise_to(state, target)

    # Turn barrel as preflop aggressor with good equity or strong blockers.
    if street == "turn" and pfa and rcount == 0:
        tx_label = _texture_label(board)
        if eff_eq >= 0.45 and bluff_roll < 0.55:
            frac = _cbet_sizing_frac(board, polar=False)
            target = my_bet + int(pot * (frac + 0.10))
            return _raise_to(state, target)
        # Pure bluff: high-card blockers with strong equity-denial vs flop
        # cbet-call range.
        if (
            cat == "weak"
            and tx_label != "dry"
            and bluff_roll < 0.30
            and max(_RANK_VAL[c[0]] for c in hole) >= 12
        ):
            target = my_bet + int(pot * 0.55)
            return _raise_to(state, target)

    # River bluff as preflop aggressor on missed draws / blocker hands.
    if street == "river" and pfa and rcount == 0:
        if cat == "weak" and bluff_roll < 0.28:
            # Bluff with hands that block villain calling range. Use ace-high
            # or king-high blockers.
            if max(_RANK_VAL[c[0]] for c in hole) >= 11:
                target = my_bet + int(pot * 0.75)
                return _raise_to(state, target)

    # Out-of-position float / probe: as the non-aggressor on the flop, lead
    # tiny with strong hands to deny equity, but never bluff-lead.
    if street == "flop" and not pfa and rcount == 0:
        if cat in ("nuts", "strong") and bluff_roll < 0.50:
            target = my_bet + int(pot * 0.40)
            return _raise_to(state, target)

    return {"action": "check"}


def _count_preflop_raises(state: dict) -> int:
    log = state.get("action_log") or []
    if state.get("street") == "preflop":
        n = 0
        for entry in log:
            a = entry.get("action")
            if a in ("raise", "all_in"):
                n += 1
        return n
    # Postflop: count raises before community changed. Since action_log is
    # flat and doesn't carry street, we approximate by counting all raises
    # then subtracting the postflop ones (raises with `bet_this_street > 0`
    # are this street). The simplest proxy: total raises minus this-street
    # raises.
    total_raises = 0
    for entry in log:
        a = entry.get("action")
        if a in ("raise", "all_in"):
            total_raises += 1
    this_street_raises = _street_raise_count(state)
    return max(0, total_raises - this_street_raises)


# Warmup
_WARMED = False


def _warmup() -> None:
    global _WARMED
    if _WARMED:
        return
    eval7.evaluate(
        [eval7.Card("As"), eval7.Card("Kh"), eval7.Card("Qd"),
         eval7.Card("Jc"), eval7.Card("Tc")]
    )
    sample_range = _top_pct_cached(30)
    _equity(["As", "Kh"], [], sample_range, trials=40)
    _equity(["7h", "2c"], ["As", "Kd", "5c"], sample_range, trials=40)
    _equity(["Qs", "Js"], ["Tc", "9d", "2h"], sample_range, trials=40)
    _WARMED = True


# Entrypoint
def decide(state: dict) -> dict:
    if state.get("type") == "warmup":
        _warmup()
        return {"ok": True}
    if not _WARMED:
        _warmup()

    deadline = time.perf_counter() + 1.5
    try:
        if state.get("street") == "preflop":
            return _decide_preflop(state)
        return _decide_postflop(state, deadline)
    except Exception:
        if state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}
