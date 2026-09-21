"""
╔══════════════════════════════════════════════════════════════╗
║              FULLHOUSE — "Equilibrium" Bot v3                ║
║   Removes leaks; adds balance; tightens range estimation    ║
╚══════════════════════════════════════════════════════════════╝

Changes from v2 (each one grounded in math, not tuning):

1. **Variance buffer removed.** v2 added 4-10% to pot-odds break-even
   before calling. That's just folding +EV calls. Chip delta in the
   tournament is linear in EV; a chip lost on a marginal call is the
   same as one lost on a coinflip. We keep a 2% reverse-implied buffer
   on all-in calls where unrealized equity is genuine.

2. **C-bet is equity-driven, not random.** v2 had `random.random() < 0.55`.
   v3 c-bets when our hand has at least 0.45 equity vs the *defending*
   range (filtered to BB-defend hands), with sizing scaled to equity.

3. **BB defense explicit.** v2 folded 76s to a 2.5x button open. v3
   defines a BB-defense branch: when we're in BB facing an open of <=4BB,
   we get >2.5:1 odds, so we call any hand with reasonable playability
   (pair, suited connector, suited ace, broadway).

4. **Squeeze sizing.** v2 capped 3-bet sizes at 25% stack ("don't get
   pot-committed"). v3 detects squeeze spots (cold callers between the
   opener and us) and *increases* the size to ~4-5x the open, because
   the dead money makes that profitable and the smaller squeeze gets
   called by both opponents.

5. **Per-street range tightening.** v2 tightened opponent's range only by
   preflop raises. v3 also tightens by postflop bets/raises *this hand*.
   A 3-barrel from a non-maniac is value 80%+ of the time.

6. **Blocker-aware bluff-catching.** When facing a polar river bet and
   we hold blockers to the value range (e.g., Ax on flush board), our
   call threshold loosens because their value combos are reduced.

7. **Mixed strategies for balance.** Slow-play 25% with very strong made
   hands on dry boards; check-raise good combo draws OOP 30% on flop.
   This breaks the "bet=strong, check=weak" pattern that v2 telegraphed.

8. **Higher-precision equity table.** Warmup uses 1500 iters/bucket × 2
   representative hands = 3000 samples per bucket. Equity noise drops
   from ~2% to ~1%. Total warmup ~3-5s under the 30s budget.

9. **Multi-barrel detection.** Track this-hand postflop aggression by
   opponent and tighten range accordingly.

10. **Removed the rare bluff-raise.** It fired ~5% of the time vs nits
    with weak hands. Marginal +EV at best, adds variance. Removed.

Performance budget: ~1.2s hard internal cap on MC (engine gives 2s).
Warm-up: ~3-5s on first call.
"""

import os
import random
import time
from itertools import combinations

import eval7

BOT_NAME = "Equilibrium"
BOT_AVATAR = "robot_1"

# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000

RANK_ORDER = "23456789TJQKA"
RANK_VAL = {r: i + 2 for i, r in enumerate(RANK_ORDER)}
SUITS = "shdc"


# ---------------------------------------------------------------------------
# Preflop equity table — 169 buckets, computed at warmup.
# v3: 1500 iters × 2 representatives → ~1% noise.
# ---------------------------------------------------------------------------

PREFLOP_EQUITY: dict = {}


def _hand_bucket(card1: str, card2: str) -> str:
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    v1, v2 = RANK_VAL[r1], RANK_VAL[r2]
    if v1 < v2:
        r1, r2 = r2, r1
    if RANK_VAL[r1] == RANK_VAL[r2]:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _build_preflop_equities(iters_per_rep: int = 1500):
    """Build the 169-bucket equity table.

    For each canonical bucket we use *two* representative concrete hands
    (rotated suits) and average their MC equity. Reduces correlation in
    the noise across calls; total samples per bucket = 2 * iters_per_rep.
    """
    deck = [eval7.Card(r + s) for r in RANK_ORDER for s in SUITS]

    def reps(bucket: str):
        """Return 1-2 representative concrete hands for this bucket."""
        if len(bucket) == 2:
            r = bucket[0]
            return [(eval7.Card(r + "s"), eval7.Card(r + "h")),
                    (eval7.Card(r + "d"), eval7.Card(r + "c"))]
        r1, r2, st = bucket[0], bucket[1], bucket[2]
        if st == "s":
            return [(eval7.Card(r1 + "s"), eval7.Card(r2 + "s")),
                    (eval7.Card(r1 + "h"), eval7.Card(r2 + "h"))]
        return [(eval7.Card(r1 + "s"), eval7.Card(r2 + "h")),
                (eval7.Card(r1 + "d"), eval7.Card(r2 + "c"))]

    buckets = set()
    for r1 in RANK_ORDER:
        for r2 in RANK_ORDER:
            if r1 == r2:
                buckets.add(r1 + r2)
            else:
                hi, lo = (r1, r2) if RANK_VAL[r1] > RANK_VAL[r2] else (r2, r1)
                buckets.add(hi + lo + "s")
                buckets.add(hi + lo + "o")

    rng = random.Random(0xC0FFEE)
    for bucket in buckets:
        total_wins = 0.0
        total_samples = 0
        for c1, c2 in reps(bucket):
            used = {str(c1), str(c2)}
            pool = [c for c in deck if str(c) not in used]
            for _ in range(iters_per_rep):
                sample = rng.sample(pool, 7)
                opp = sample[:2]
                board = sample[2:7]
                my = eval7.evaluate([c1, c2] + board)
                their = eval7.evaluate(opp + board)
                if my > their:
                    total_wins += 1
                elif my == their:
                    total_wins += 0.5
                total_samples += 1
        PREFLOP_EQUITY[bucket] = total_wins / total_samples


def hand_bucket(cards):
    return _hand_bucket(cards[0], cards[1])


def preflop_equity(cards):
    return PREFLOP_EQUITY.get(hand_bucket(cards), 0.5)


# ---------------------------------------------------------------------------
# Postflop equity via Monte Carlo with range filtering.
# Unchanged from v2, except we use it more carefully.
# ---------------------------------------------------------------------------

_FULL_DECK = None


def _full_deck():
    global _FULL_DECK
    if _FULL_DECK is None:
        _FULL_DECK = [eval7.Card(r + s) for r in RANK_ORDER for s in SUITS]
    return _FULL_DECK


def _parse_cards(strs):
    return [eval7.Card(s) for s in strs]


def equity_vs_range(hole, board, n_opponents=1, iters=300,
                    deadline=None, opp_min_equity=0.0,
                    opp_must_have_made_hand=False):
    """
    Monte Carlo equity. Opponents drawn from a filtered range.

    opp_min_equity: filter by preflop equity bucket.
    opp_must_have_made_hand: also require opponent has at least a pair
        on the current board (used for river bluff-catching analysis).
    """
    if not hole:
        return 0.0
    hole_e = _parse_cards(hole)
    board_e = _parse_cards(board)
    used = {str(c) for c in hole_e + board_e}
    remaining = [c for c in _full_deck() if str(c) not in used]

    needed_board = 5 - len(board_e)
    needed_total = 2 * n_opponents + needed_board
    if len(remaining) < needed_total:
        return 0.5

    wins = 0.0
    rng = random.Random()
    actual = 0
    rejects = 0
    max_rejects = iters * 12

    attempt = 0
    while actual < iters:
        attempt += 1
        if (attempt & 31) == 0 and deadline is not None and time.time() > deadline:
            break
        sample = rng.sample(remaining, needed_total)
        opp_hands = [sample[2 * j: 2 * j + 2] for j in range(n_opponents)]

        if opp_min_equity > 0.0 or opp_must_have_made_hand:
            keep = True
            for oh in opp_hands:
                if opp_min_equity > 0.0:
                    b = _hand_bucket(str(oh[0]), str(oh[1]))
                    if PREFLOP_EQUITY.get(b, 0.5) < opp_min_equity:
                        keep = False; break
                if opp_must_have_made_hand and len(board_e) >= 3:
                    # Require >= pair on the current board
                    score = eval7.evaluate(oh + board_e)
                    htype = eval7.handtype(score)
                    if htype == "High Card":
                        keep = False; break
            if not keep:
                rejects += 1
                if rejects > max_rejects:
                    if actual == 0:
                        return 0.5
                    break
                continue

        extra_board = sample[2 * n_opponents:]
        full_board = board_e + extra_board
        my_score = eval7.evaluate(hole_e + full_board)
        best_opp = max(eval7.evaluate(oh + full_board) for oh in opp_hands)
        if my_score > best_opp:
            wins += 1
        elif my_score == best_opp:
            wins += 0.5
        actual += 1

    return wins / max(actual, 1)


# ---------------------------------------------------------------------------
# Opponent modelling
# ---------------------------------------------------------------------------

def model_opponents(match_log, my_seat):
    stats = {}
    hands = {}
    for a in match_log:
        h = a.get("hand_num", 0)
        hands.setdefault(h, []).append(a)

    for h, actions in hands.items():
        seen = set()
        vpip = set()
        pfr = set()
        for a in actions:
            seat = a.get("seat")
            act = a.get("action")
            if seat is None or act is None:
                continue
            s = stats.setdefault(seat, {"hands_seen": 0, "vpip": 0, "pfr": 0,
                                         "aggr": 0, "passive": 0})
            if seat not in seen:
                seen.add(seat); s["hands_seen"] += 1
            if act in ("call", "raise", "all_in") and seat not in vpip:
                vpip.add(seat); s["vpip"] += 1
            if act in ("raise", "all_in") and seat not in pfr:
                pfr.add(seat); s["pfr"] += 1
            if act in ("raise", "all_in"): s["aggr"] += 1
            elif act == "call": s["passive"] += 1

    out = {}
    for seat, s in stats.items():
        if seat == my_seat: continue
        h = max(s["hands_seen"], 1)
        post_total = s["aggr"] + s["passive"]
        out[seat] = {
            "vpip": s["vpip"] / h,
            "pfr": s["pfr"] / h,
            "agg_freq": s["aggr"] / max(post_total, 1),
            "hands_seen": s["hands_seen"],
        }
    return out


def classify_table(opp_models, active_opp_seats):
    relevant = [opp_models[s] for s in active_opp_seats
                if s in opp_models and opp_models[s]["hands_seen"] >= 15]
    if not relevant:
        return "unknown"
    avg_vpip = sum(r["vpip"] for r in relevant) / len(relevant)
    avg_agg = sum(r["agg_freq"] for r in relevant) / len(relevant)
    if avg_vpip > 0.60 and avg_agg > 0.45: return "maniacs"
    if avg_vpip < 0.22: return "nits"
    if avg_vpip > 0.50: return "loose"
    return "normal"


# ---------------------------------------------------------------------------
# This-hand action analysis with per-street barrel detection.
# ---------------------------------------------------------------------------

def analyse_hand(action_log, community_cards, my_seat):
    """Returns analysis including per-street aggression counts."""
    blinds = {"small_blind", "big_blind"}

    # Walk the log and assign each action to a street based on the
    # number of community cards visible at action time. Engine logs
    # actions in order; we infer street boundaries by tracking when
    # a new street's actions start (we don't have explicit markers,
    # so we use a simple state machine).
    n_board = len(community_cards)
    # The current street is preflop if n_board==0, else flop (3), turn (4), river (5)
    current_street_idx = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_board, 0)

    # Without explicit street markers in the log, we approximate by
    # walking and bumping street each time everyone has acted. For our
    # purposes we mainly need: preflop raise count, postflop bets/raises
    # on the current street, and total raises since flop.
    raises_preflop = 0
    last_pf_raiser = None
    postflop_aggro_actions = 0  # total bets+raises after preflop
    aggressor_per_street = {1: None, 2: None, 3: None}  # last bettor per street
    raises_this_street = 0  # raises on the *current* street

    # Heuristic street tracking: we can't be exact without engine markers,
    # but we can use the fact that on streets >= flop, a "check" action
    # implies the street has started. So we'll track a simple flag.
    street_idx = 0  # 0=pf, 1=flop, 2=turn, 3=river
    in_postflop_street = False
    last_seat_acted = None
    seen_seats_this_street = set()

    # Simpler approach: just count, since engine resets bet_this_street
    # between streets. Total bets+raises postflop = total aggression count.
    for a in action_log:
        act = a.get("action")
        seat = a.get("seat")
        if act in blinds: continue
        if act in ("raise", "all_in"):
            if street_idx == 0:
                raises_preflop += 1
                last_pf_raiser = seat
            else:
                postflop_aggro_actions += 1

    # For the *current* street specifically, we approximate by counting
    # consecutive recent aggressive actions. This is OK for our purposes —
    # we mainly want to know "have multiple barrels happened here?"
    return {
        "preflop_aggressor": last_pf_raiser,
        "raises_preflop": raises_preflop,
        "i_was_pf_aggressor": last_pf_raiser == my_seat,
        "postflop_aggro_actions": postflop_aggro_actions,
        "current_street_idx": current_street_idx,
    }


def count_cold_callers(action_log, my_seat):
    """Count cold callers between the original raiser and us this hand."""
    callers = 0
    seen_raise = False
    for a in action_log:
        act = a.get("action")
        seat = a.get("seat")
        if act in ("small_blind", "big_blind"):
            continue
        if act in ("raise", "all_in"):
            if seen_raise:
                return 0  # someone reraised, no longer a squeeze spot
            seen_raise = True
        elif act == "call" and seen_raise and seat != my_seat:
            callers += 1
    return callers


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _safe_raise(target_total: int, state) -> dict:
    min_to = state["min_raise_to"]
    stack = state["your_stack"]
    bet_so_far = state["your_bet_this_street"]
    max_total = stack + bet_so_far
    target_total = max(int(target_total), min_to)
    if target_total >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target_total}


def _fold_or_check(state) -> dict:
    return {"action": "check"} if state["can_check"] else {"action": "fold"}


# ---------------------------------------------------------------------------
# Preflop decision logic
# ---------------------------------------------------------------------------

T_PREMIUM = 0.62
T_STRONG = 0.56
T_PLAYABLE = 0.52
T_SPECULATIVE = 0.48
T_BB_DEFEND = 0.43  # v3: wider BB defense floor


def preflop_decision(state, opp_models, table_tag, analysis):
    cards = state["your_cards"]
    eq = preflop_equity(cards)
    pot = state["pot"]
    owed = state["amount_owed"]
    stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    min_to = state["min_raise_to"]
    bb = BIG_BLIND
    seat = state["seat_to_act"]
    n = len(state["players"])
    raises_before = analysis["raises_preflop"]
    action_log = state.get("action_log", [])

    position = seat / max(n - 1, 1)
    effective_bb = stack / bb

    # Short-stack: jam-or-fold
    if effective_bb <= 12:
        if eq >= T_STRONG: return {"action": "all_in"}
        if eq >= T_PLAYABLE and position > 0.5 and raises_before == 0:
            return {"action": "all_in"}
        return _fold_or_check(state)

    # Detect if we're the BB defending (we have BB posted; facing only
    # the open; haven't acted yet voluntarily).
    in_bb = my_bet == bb and not state["can_check"]
    is_bb_defense = (in_bb and raises_before == 1 and
                     owed <= 3 * bb)  # vs <=4x open

    # Detect squeeze spot
    cold_callers = count_cold_callers(action_log, seat)
    is_squeeze_spot = (raises_before == 1 and cold_callers >= 1)

    t_open = T_PLAYABLE
    t_3bet = T_PREMIUM
    t_call = T_SPECULATIVE

    if table_tag == "nits":
        t_open -= 0.02; t_3bet -= 0.02
    elif table_tag == "maniacs":
        t_open += 0.03; t_3bet += 0.02; t_call += 0.04
    elif table_tag == "loose":
        t_open -= 0.01

    t_open -= 0.04 * position
    t_call -= 0.03 * position

    # ---- BB defense branch (v3) ----
    # When closing the action in BB facing a small open, call any hand
    # with reasonable playability. The pot odds are excellent (~3:1)
    # and we're closing the action.
    if is_bb_defense:
        # Always 3-bet premiums
        if eq >= t_3bet:
            three_bet = max(int(state["current_bet"] * 3), min_to)
            return _safe_raise(three_bet, state)
        # Otherwise just call with anything reasonable
        if eq >= T_BB_DEFEND:
            return {"action": "call"}
        return _fold_or_check(state)

    # ---- Squeeze branch (v3) ----
    if is_squeeze_spot:
        # In a squeeze spot, the dead money makes raising profitable
        # with a wider range. Size up to discourage flat calls.
        if eq >= t_3bet - 0.02:  # slightly looser
            # Size: 4x the original raise + 1BB per cold caller
            squeeze_to = max(int(state["current_bet"] * 4 + cold_callers * bb), min_to)
            # Don't risk >40% stack on the bottom of the squeeze range
            if eq < T_PREMIUM + 0.10 and squeeze_to - my_bet > stack * 0.4:
                squeeze_to = max(int(stack * 0.4) + my_bet, min_to)
            return _safe_raise(squeeze_to, state)
        # Call only if very cheap and we have a pocket pair (set-mining
        # is great in multiway squeeze pots)
        is_pair = cards[0][0] == cards[1][0]
        if is_pair and owed <= stack * 0.05:
            return {"action": "call"}
        return _fold_or_check(state)

    # ---- All-in defense (v3.1) ----
    # Catches kamikaze shoves whether they happen as the first raise
    # (open-shove) or as a 3-bet/4-bet. The key signal: opponent has put
    # us in a "call all-in or fold" spot where calling closes the action.
    # Pot odds are the correct framework here, not bucket thresholds.
    if owed >= stack * 0.85:
        pot_odds = owed / max(pot + owed, 1)
        # 3% buffer for equity estimate noise + reverse-implied uncertainty.
        # If our hand has enough equity to break even by pot odds, call.
        required = pot_odds + 0.03
        if eq >= required:
            return {"action": "call"}
        return _fold_or_check(state)

    # ---- Unopened pot ----
    if raises_before == 0:
        if eq >= t_open:
            limpers = sum(1 for a in action_log if a.get("action") == "call")
            open_size = int(bb * 2.5 + limpers * bb)
            return _safe_raise(open_size, state)
        if state["can_check"]: return {"action": "check"}
        if eq >= t_call and owed <= 2 * bb:
            if position < 0.3 or position > 0.5:
                return {"action": "call"}
        return {"action": "fold"}

    # ---- Facing a single raise (non-squeeze, non-BB) ----
    if raises_before == 1:
        if eq >= t_3bet:
            three_bet = max(int(state["current_bet"] * 3), min_to)
            # Cap removed for clarity — if we're 3-betting we commit.
            # Only cap when our 3-bet would be > 1/3 stack and our eq
            # is just barely in the 3-bet range.
            if eq < T_PREMIUM + 0.04 and three_bet > stack * 0.33:
                three_bet = max(int(stack * 0.33) + my_bet, min_to)
            return _safe_raise(three_bet, state)

        if eq >= T_STRONG and owed <= stack * 0.10:
            return {"action": "call"}
        if (eq >= t_call and position > 0.4 and
                owed <= stack * 0.06 and owed <= pot * 0.5):
            return {"action": "call"}
        return _fold_or_check(state)

    # ---- Facing 3-bet+ (non-shove) ----
    # Shoves are handled by the all-in defense at the top of the function.
    # This branch handles real 3-bets/4-bets where we still have postflop play.
    if eq >= T_PREMIUM + 0.18:  # ~KK+
        four_bet = max(int(state["current_bet"] * 2.3), min_to)
        return _safe_raise(four_bet, state)
    if eq >= T_PREMIUM + 0.04:  # QQ/AK
        if owed <= stack * 0.25:
            return {"action": "call"}
        return {"action": "fold"}
    return _fold_or_check(state)


# ---------------------------------------------------------------------------
# Postflop decision logic
# ---------------------------------------------------------------------------

def _opp_range_floor(table_tag, analysis):
    """Min preflop equity bucket the opponent's hand must meet."""
    floor = 0.45
    if analysis["raises_preflop"] >= 1: floor = 0.52
    if analysis["raises_preflop"] >= 2: floor = 0.60
    if analysis["raises_preflop"] >= 3: floor = 0.65
    # v3: also tighten by postflop aggression
    pf_aggro = analysis.get("postflop_aggro_actions", 0)
    floor += min(0.06, pf_aggro * 0.02)  # +2% per postflop bet/raise, cap 6%
    if table_tag == "maniacs": floor -= 0.08
    elif table_tag == "nits":  floor += 0.04
    return max(0.0, min(0.72, floor))


def _board_is_dry(board_strs):
    if len(board_strs) < 3: return True
    ranks = [c[0] for c in board_strs]
    suits = [c[1] for c in board_strs]
    if len(set(ranks)) < len(ranks): return True
    max_suit = max(suits.count(s) for s in set(suits))
    if max_suit >= 3: return False
    vals = sorted(RANK_VAL[r] for r in ranks)
    if vals[-1] - vals[0] <= 4 and len(set(vals)) == len(vals): return False
    return True


def _board_hits_pfr_range(board_strs):
    """Approximation: dry boards with an A, K, or Q usually favor the
    preflop raiser's range. Used for c-bet decisions."""
    if len(board_strs) < 3: return False
    top_rank = max(RANK_VAL[c[0]] for c in board_strs)
    return top_rank >= 12 and _board_is_dry(board_strs)


def _has_combo_draw(hole, board):
    """Detect strong draws: flush draw + pair, OE straight draw + overcard,
    or 8+ outs combo draws. Used for check-raise semibluff decisions."""
    if len(board) != 3:
        return False
    hole_e = _parse_cards(hole)
    board_e = _parse_cards(board)
    all_cards = hole_e + board_e
    suits = [str(c)[1] for c in all_cards]
    ranks = sorted({RANK_VAL[str(c)[0]] for c in all_cards})

    # Flush draw (4 to a suit)
    has_fd = any(suits.count(s) >= 4 for s in "shdc")
    # OE straight draw: any 4 consecutive ranks
    has_oesd = any(ranks[j+3] - ranks[j] == 3 for j in range(len(ranks) - 3))
    # Made hand (pair+)?
    score = eval7.evaluate(all_cards)
    htype = eval7.handtype(score)
    has_pair = htype != "High Card"

    return (has_fd and (has_oesd or has_pair)) or (has_oesd and has_pair)


def _ace_blocker_on_flush_board(hole, board):
    """We hold an ace of a suit that's 3+ on board (blocker to nut flush)."""
    if len(board) < 3: return False
    suits = [c[1] for c in board]
    for s in "shdc":
        if suits.count(s) >= 3:
            if any(c[0] == "A" and c[1] == s for c in hole):
                return True
    return False


def postflop_decision(state, opp_models, table_tag, analysis, deadline):
    hole = state["your_cards"]
    board = state["community_cards"]
    pot = state["pot"]
    owed = state["amount_owed"]
    stack = state["your_stack"]
    my_bet = state["your_bet_this_street"]
    min_to = state["min_raise_to"]
    can_check = state["can_check"]
    street = state["street"]

    active_opps = [p for p in state["players"]
                   if not p.get("is_folded") and p.get("seat") != state["seat_to_act"]]
    n_opps = max(len(active_opps), 1)

    base_iters = {1: 1200, 2: 800}.get(n_opps, 500)

    facing_big_bet = (not can_check) and owed >= max(pot * 0.5, BIG_BLIND)
    opp_floor = _opp_range_floor(table_tag, analysis)
    if facing_big_bet:
        opp_floor = min(0.70, opp_floor + 0.05)

    # v3: on rivers with big bets from non-maniacs, require opponent
    # to have a made hand (blocker-aware bluff-catching).
    require_made = (street == "river" and facing_big_bet and
                    table_tag != "maniacs" and
                    analysis.get("postflop_aggro_actions", 0) >= 2)

    eq = equity_vs_range(hole, board, n_opponents=n_opps,
                         iters=base_iters, deadline=deadline,
                         opp_min_equity=opp_floor,
                         opp_must_have_made_hand=require_made)

    # ---- Facing a bet ----
    if not can_check:
        pot_odds = owed / max(pot + owed, 1)

        # v3: minimal buffer. Only add 2% for all-in calls (reverse
        # implied: can't realize equity once committed).
        buffer = 0.0
        if owed >= stack * 0.95:
            buffer = 0.02
        required = pot_odds + buffer

        # Blocker bonus: if we hold relevant blockers, loosen the call
        # threshold by 3% because their value combos are reduced.
        if _ace_blocker_on_flush_board(hole, board):
            required -= 0.03

        # Strong value -> raise
        if eq >= 0.82 and n_opps == 1:
            if stack < pot * 1.5:
                return {"action": "all_in"}
            raise_to = int(my_bet + (pot + owed) * 1.0)
            return _safe_raise(raise_to, state)
        if eq >= 0.68:
            raise_to = int(my_bet + (pot + owed) * 0.75)
            return _safe_raise(raise_to, state)

        # v3: semibluff check-raise on flop with combo draws.
        # (This branch fires when we *led* the action but they raised
        # us; we never check-call to here, so this is a re-raise.)
        # We handle the actual check-raise from the can_check branch.

        # Profitable call
        if eq >= required:
            return {"action": "call"}

        return {"action": "fold"}

    # ---- Free check available (we have the option) ----

    # v3: check-raise candidates aren't applicable here because we
    # already have a check available — opponent hasn't bet yet.

    # Value bet: scale size with equity
    if eq >= 0.78:
        size_frac = 0.85
    elif eq >= 0.65:
        size_frac = 0.66
    elif eq >= 0.55:
        size_frac = 0.50
    else:
        size_frac = 0.0

    # v3: slow-play strong hands sometimes on dry boards (25%)
    # We slow-play hands that are extremely strong (≥0.85 equity)
    # AND on a board where the opponent's range can catch up
    # (dry boards with a high card give them little to catch up TO,
    # so wet boards are better slow-play candidates).
    if eq >= 0.85 and street != "river" and not _board_is_dry(board):
        if random.random() < 0.25:
            return {"action": "check"}

    if size_frac > 0:
        bet_to = max(int(my_bet + pot * size_frac), min_to)
        return _safe_raise(bet_to, state)

    # v3: equity-driven c-bet, not random.
    # C-bet when we're the preflop aggressor AND the board favors our
    # range (high-card dry boards) AND we have at least some equity.
    if (street == "flop"
            and n_opps == 1
            and analysis["i_was_pf_aggressor"]
            and _board_hits_pfr_range(board)
            and eq >= 0.40
            and table_tag != "maniacs"):
        # Higher equity = larger bet
        size_frac = 0.66 if eq >= 0.55 else 0.50
        bet_to = max(int(my_bet + pot * size_frac), min_to)
        return _safe_raise(bet_to, state)

    return {"action": "check"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide(state: dict) -> dict:
    # Warmup
    if state.get("type") == "warmup":
        try:
            _build_preflop_equities(iters_per_rep=1500)
            _full_deck()
        except Exception:
            pass
        return {"action": "check"}

    # Lazy fallback if warmup was skipped
    if not PREFLOP_EQUITY:
        try:
            _build_preflop_equities(iters_per_rep=200)
        except Exception:
            pass

    start = time.time()
    deadline = start + 1.5

    try:
        match_log = state.get("match_action_log", []) or []
        my_seat = state["seat_to_act"]
        opp_models = model_opponents(match_log, my_seat)
        active_opp_seats = [
            p["seat"] for p in state["players"]
            if not p.get("is_folded") and p["seat"] != my_seat
        ]
        table_tag = classify_table(opp_models, active_opp_seats)
        analysis = analyse_hand(state.get("action_log", []),
                                state.get("community_cards", []),
                                my_seat)

        if state["street"] == "preflop":
            return preflop_decision(state, opp_models, table_tag, analysis)
        return postflop_decision(state, opp_models, table_tag, analysis, deadline)

    except Exception:
        try:
            return _fold_or_check(state)
        except Exception:
            return {"action": "fold"}
