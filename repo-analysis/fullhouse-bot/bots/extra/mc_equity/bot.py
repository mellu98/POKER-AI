"""Monte-Carlo preflop equity bot with per-table-type range floors. No data files."""

import os
import random
import time
from itertools import combinations

import eval7

BOT_NAME = "MC Equity"
BOT_AVATAR = "robot_1"

# ---------------------------------------------------------------------------
# Engine constants & Globals
# ---------------------------------------------------------------------------

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000

RANK_ORDER = "23456789TJQKA"
RANK_VAL = {r: i + 2 for i, r in enumerate(RANK_ORDER)}
SUITS = "shdc"

PREFLOP_EQUITY: dict = {}
HAND_TRACKER: dict = {}

# ---------------------------------------------------------------------------
# Hand Tracking (Safe State Management)
# ---------------------------------------------------------------------------

def track_hand(state):
    hand_id = state["hand_id"]
    if hand_id not in HAND_TRACKER:
        HAND_TRACKER[hand_id] = {
            "raises_preflop": 0,
            "postflop_aggro_actions": 0,
            "i_was_pf_aggressor": False,
            "preflop_aggressor": None
        }
        if len(HAND_TRACKER) > 20:
            oldest = list(HAND_TRACKER.keys())[0]
            del HAND_TRACKER[oldest]
            
    t = HAND_TRACKER[hand_id]
    street = state["street"]
    my_seat = state["seat_to_act"]
    
    total_raises = 0
    last_raiser = None
    for a in state.get("action_log", []):
        if a.get("action") in ("raise", "all_in"):
            total_raises += 1
            last_raiser = a.get("seat")
            
    if street == "preflop":
        t["raises_preflop"] = total_raises
        t["preflop_aggressor"] = last_raiser
        t["i_was_pf_aggressor"] = (last_raiser == my_seat)
    else:
        t["postflop_aggro_actions"] = max(0, total_raises - t["raises_preflop"])
        
    return t

# ---------------------------------------------------------------------------
# Preflop equity table
# ---------------------------------------------------------------------------

def _hand_bucket(card1: str, card2: str) -> str:
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    v1, v2 = RANK_VAL[r1], RANK_VAL[r2]
    if v1 < v2:
        r1, r2 = r2, r1
    if RANK_VAL[r1] == RANK_VAL[r2]:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")

def _build_preflop_equities(iters_per_rep: int = 1000):
    deck = [eval7.Card(r + s) for r in RANK_ORDER for s in SUITS]

    def reps(bucket: str):
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
                if my > their: total_wins += 1
                elif my == their: total_wins += 0.5
                total_samples += 1
        PREFLOP_EQUITY[bucket] = total_wins / total_samples

def hand_bucket(cards):
    return _hand_bucket(cards[0], cards[1])

def preflop_equity(cards):
    return PREFLOP_EQUITY.get(hand_bucket(cards), 0.5)

# ---------------------------------------------------------------------------
# Postflop equity via Monte Carlo
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
                    deadline=None, opp_min_equity=0.0):
    
    if not hole: return 0.0
    hole_e = _parse_cards(hole)
    board_e = _parse_cards(board)
    used = {str(c) for c in hole_e + board_e}
    remaining = [c for c in _full_deck() if str(c) not in used]

    needed_board = 5 - len(board_e)
    needed_total = 2 * n_opponents + needed_board
    if len(remaining) < needed_total: return 0.5

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

        if opp_min_equity > 0.0:
            keep = True
            for oh in opp_hands:
                b = _hand_bucket(str(oh[0]), str(oh[1]))
                if PREFLOP_EQUITY.get(b, 0.5) < opp_min_equity:
                    keep = False; break
            if not keep:
                rejects += 1
                if rejects > max_rejects:
                    if actual == 0: return 0.5
                    break
                continue

        extra_board = sample[2 * n_opponents:]
        full_board = board_e + extra_board
        my_score = eval7.evaluate(hole_e + full_board)
        best_opp = max(eval7.evaluate(oh + full_board) for oh in opp_hands)
        if my_score > best_opp: wins += 1
        elif my_score == best_opp: wins += 0.5
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
            if seat is None or act is None: continue
            s = stats.setdefault(seat, {"hands_seen": 0, "vpip": 0, "pfr": 0, "aggr": 0, "passive": 0})
            if seat not in seen: seen.add(seat); s["hands_seen"] += 1
            if act in ("call", "raise", "all_in") and seat not in vpip: vpip.add(seat); s["vpip"] += 1
            if act in ("raise", "all_in") and seat not in pfr: pfr.add(seat); s["pfr"] += 1
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
    relevant = [opp_models[s] for s in active_opp_seats if s in opp_models and opp_models[s]["hands_seen"] >= 15]
    if not relevant: return "unknown"
    avg_vpip = sum(r["vpip"] for r in relevant) / len(relevant)
    avg_agg = sum(r["agg_freq"] for r in relevant) / len(relevant)
    if avg_vpip > 0.60 and avg_agg > 0.45: return "maniacs"
    if avg_vpip < 0.22: return "nits"
    if avg_vpip > 0.50: return "loose"
    return "normal"

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
T_BB_DEFEND = 0.45

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

    if effective_bb <= 12:
        if eq >= T_STRONG: return {"action": "all_in"}
        if eq >= T_PLAYABLE and position > 0.5 and raises_before == 0:
            return {"action": "all_in"}
        return _fold_or_check(state)

    # ALL-IN DEFENSE - HEAVY SURVIVAL TAX
    if owed >= stack * 0.85:
        pot_odds = owed / max(pot + owed, 1)
        # We demand a massive +12% premium to call a preflop shove. 
        # Keeps us out of coinflips against kamikazes.
        required = pot_odds + 0.12  
        if eq >= required: return {"action": "call"}
        return _fold_or_check(state)

    in_bb = my_bet == bb and not state["can_check"]
    is_bb_defense = (in_bb and raises_before == 1 and owed <= 3 * bb) 

    t_open = T_PLAYABLE
    t_3bet = T_PREMIUM
    t_call = T_SPECULATIVE

    if table_tag == "nits":
        t_open -= 0.02; t_3bet -= 0.02
    elif table_tag == "maniacs":
        t_open += 0.03; t_3bet += 0.02; t_call += 0.04

    t_open -= 0.04 * position
    t_call -= 0.03 * position

    if is_bb_defense:
        if eq >= t_3bet:
            return _safe_raise(max(int(state["current_bet"] * 3), min_to), state)
        if eq >= T_BB_DEFEND: return {"action": "call"}
        return _fold_or_check(state)

    if raises_before == 0:
        if eq >= t_open:
            limpers = sum(1 for a in action_log if a.get("action") == "call")
            return _safe_raise(int(bb * 2.5 + limpers * bb), state)
        if state["can_check"]: return {"action": "check"}
        if eq >= t_call and owed <= 2 * bb and (position < 0.3 or position > 0.5):
            return {"action": "call"}
        return {"action": "fold"}

    if raises_before == 1:
        if eq >= t_3bet:
            three_bet = max(int(state["current_bet"] * 3), min_to)
            if eq < T_PREMIUM + 0.04 and three_bet > stack * 0.33:
                three_bet = max(int(stack * 0.33) + my_bet, min_to)
            return _safe_raise(three_bet, state)
        if eq >= T_STRONG and owed <= stack * 0.10: return {"action": "call"}
        if eq >= t_call and position > 0.4 and owed <= stack * 0.06 and owed <= pot * 0.5:
            return {"action": "call"}
        return _fold_or_check(state)

    if eq >= T_PREMIUM + 0.18: 
        return _safe_raise(max(int(state["current_bet"] * 2.3), min_to), state)
    if eq >= T_PREMIUM + 0.04 and owed <= stack * 0.25: 
        return {"action": "call"}
    
    return _fold_or_check(state)

# ---------------------------------------------------------------------------
# Postflop decision logic
# ---------------------------------------------------------------------------

def _opp_range_floor(table_tag, analysis, facing_big_bet=False, facing_shove=False):
    """
    Highly accurate range narrowing. Dampens aggression metrics 
    if the table profile is verified as hyper-aggressive bluffs.
    """
    floor = 0.45
    if analysis["raises_preflop"] >= 1: floor = 0.52
    if analysis["raises_preflop"] >= 2: floor = 0.60
    if analysis["raises_preflop"] >= 3: floor = 0.65
    
    # Track postflop barrels
    pf_aggro = analysis.get("postflop_aggro_actions", 0)
    
    # MANIAC ADJUSTMENT: If facing verified maniacs, drastically lower the floor
    if table_tag == "maniacs":
        floor += (pf_aggro * 0.02)  # Maniacs barrel air; drop scaling from 0.06 to 0.02
        if facing_big_bet: floor += 0.01
        if facing_shove: floor += 0.02
        floor -= 0.12               # Loosen baseline target against wide ranges
    else:
        floor += (pf_aggro * 0.06)  # Strict GTO narrowing for normal/tight players
        if facing_big_bet: floor += 0.05
        if facing_shove: floor += 0.10
        if table_tag == "nits":  floor += 0.06
    
    return max(0.0, min(0.78, floor))

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

    active_opps = [p for p in state["players"] if not p.get("is_folded") and p.get("seat") != state["seat_to_act"]]
    n_opps = max(len(active_opps), 1)
    base_iters = {1: 1200, 2: 800}.get(n_opps, 500)

    facing_big_bet = (not can_check) and owed >= max(pot * 0.5, BIG_BLIND)
    facing_shove = (not can_check) and owed >= stack * 0.85
    
    opp_floor = _opp_range_floor(table_tag, analysis, facing_big_bet, facing_shove)
    eq = equity_vs_range(hole, board, n_opponents=n_opps, iters=base_iters, deadline=deadline, opp_min_equity=opp_floor)

    if not can_check:
        pot_odds = owed / max(pot + owed, 1)

        # POSTFLOP SURVIVAL TAX
        buffer = 0.04
        if owed >= stack * 0.4: buffer = 0.08
        if owed >= stack * 0.75: buffer = 0.12
        required = pot_odds + buffer

        if eq >= 0.82 and n_opps == 1:
            if stack < pot * 1.5: return {"action": "all_in"}
            return _safe_raise(int(my_bet + (pot + owed) * 1.0), state)
        if eq >= 0.68:
            return _safe_raise(int(my_bet + (pot + owed) * 0.75), state)

        if eq >= required: return {"action": "call"}
        return {"action": "fold"}

    # Value bet scaling
    if eq >= 0.78: size_frac = 0.85
    elif eq >= 0.65: size_frac = 0.66
    elif eq >= 0.55: size_frac = 0.50
    else: size_frac = 0.0

    if size_frac > 0:
        return _safe_raise(max(int(my_bet + pot * size_frac), min_to), state)

    return {"action": "check"}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide(state: dict) -> dict:
    if state.get("type") == "warmup":
        try:
            _build_preflop_equities(iters_per_rep=1000)
            _full_deck()
        except Exception: pass
        return {"action": "check"}

    if not PREFLOP_EQUITY:
        try: _build_preflop_equities(iters_per_rep=200)
        except Exception: pass

    start = time.time()
    deadline = start + 1.5

    try:
        match_log = state.get("match_action_log", []) or []
        my_seat = state["seat_to_act"]
        opp_models = model_opponents(match_log, my_seat)
        active_opp_seats = [p["seat"] for p in state["players"] if not p.get("is_folded") and p["seat"] != my_seat]
        table_tag = classify_table(opp_models, active_opp_seats)
        
        analysis = track_hand(state)

        if state["street"] == "preflop":
            return preflop_decision(state, opp_models, table_tag, analysis)
        return postflop_decision(state, opp_models, table_tag, analysis, deadline)

    except Exception:
        try: return _fold_or_check(state)
        except Exception: return {"action": "fold"}