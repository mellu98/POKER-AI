"""
╔══════════════════════════════════════════════════════════════╗
║              FULLHOUSE — "Equilibrium Opus" v4.2             ║
║   The Hybrid: Clean global tracking + MC Table Lookups       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import random
import time
from itertools import combinations

import eval7
import numpy as np

BOT_NAME = "Equilibrium Opus v2"
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
GLOBAL_MATCH_TRACKER: dict = {}

# Precompute the 169 Hand Buckets for lookup indexing
_buckets = []
for r1 in RANK_ORDER:
    for r2 in RANK_ORDER:
        if r1 == r2:
            _buckets.append(r1 + r2)
        else:
            hi, lo = (r1, r2) if RANK_VAL[r1] > RANK_VAL[r2] else (r2, r1)
            _buckets.append(hi + lo + "s")
            _buckets.append(hi + lo + "o")
_buckets = sorted(list(set(_buckets)))
HAND_TO_IDX = {b: i for i, b in enumerate(_buckets)}

# Load Precomputed Equity Table
EQUITY_TABLE = None
try:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    table_path = os.path.join(data_dir, "equity_table.npy")
    if os.path.exists(table_path):
        EQUITY_TABLE = np.load(table_path)
except Exception:
    pass

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
# Board Texture Classification
# ---------------------------------------------------------------------------

def _board_texture_bin(board_strs):
    """Maps board to 5 bins to match the (169, 5, 3, 3) lookup table."""
    if len(board_strs) < 3: return 0
    
    ranks = [RANK_VAL[c[0]] for c in board_strs]
    suits = [c[1] for c in board_strs]
    
    paired = len(set(ranks)) < len(ranks)
    flushy = max(suits.count(s) for s in set(suits)) >= 3
    
    unique_ranks = sorted(set(ranks))
    straighty = False
    if len(unique_ranks) >= 3:
        for i in range(len(unique_ranks) - 2):
            if unique_ranks[i+2] - unique_ranks[i] <= 4:
                straighty = True
                break
        if {14, 2, 3}.issubset(set(unique_ranks)): straighty = True # Wheel exception
        
    if paired: return 1
    if flushy and straighty: return 4
    if straighty: return 3
    if flushy: return 2
    return 0 # Dry

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

    rng = random.Random(0xC0FFEE)
    for bucket in _buckets:
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
# Postflop equity
# ---------------------------------------------------------------------------

_FULL_DECK = None

def _full_deck():
    global _FULL_DECK
    if _FULL_DECK is None:
        _FULL_DECK = [eval7.Card(r + s) for r in RANK_ORDER for s in SUITS]
    return _FULL_DECK

def _parse_cards(strs):
    return [eval7.Card(s) for s in strs]

def equity_vs_range(hole, board, n_opponents=1, iters=300, deadline=None, opp_min_equity=0.0):
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

def _lookup_equity(hole, board, n_opps, street, base_iters, deadline, opp_floor):
    if opp_floor > 0.48:
        return equity_vs_range(hole, board, n_opponents=n_opps,
                               iters=base_iters, deadline=deadline,
                               opp_min_equity=opp_floor)
    
    if EQUITY_TABLE is not None:
        try:
            h_idx = HAND_TO_IDX.get(hand_bucket(hole))
            t_idx = _board_texture_bin(board)
            s_idx = {"flop": 0, "turn": 1, "river": 2}.get(street, 0)
            o_idx = min(max(1, n_opps) - 1, 2)
            if h_idx is not None:
                return float(EQUITY_TABLE[h_idx, t_idx, s_idx, o_idx])
        except Exception:
            pass
    
    return equity_vs_range(hole, board, n_opponents=n_opps,
                           iters=base_iters, deadline=deadline,
                           opp_min_equity=opp_floor)

# ---------------------------------------------------------------------------
# Opponent modelling (Fixed Global State)
# ---------------------------------------------------------------------------

def model_opponents(state):
    hand_id = state["hand_id"]
    
    # Update our global match log with the longest action log seen for this hand
    current_log = state.get("action_log", [])
    if hand_id not in GLOBAL_MATCH_TRACKER or len(current_log) > len(GLOBAL_MATCH_TRACKER[hand_id]):
        GLOBAL_MATCH_TRACKER[hand_id] = list(current_log)
        
    # Map seats to bot_ids (seat positions are static per match)
    seat_to_bot = {p["seat"]: p.get("bot_id", str(p["seat"])) for p in state["players"]}
    my_seat = state["seat_to_act"]
    my_bot_id = seat_to_bot.get(my_seat)
    
    stats = {}
    for hid, actions in GLOBAL_MATCH_TRACKER.items():
        seen_in_hand = set()
        vpip_in_hand = set()
        pfr_in_hand = set()
        
        for a in actions:
            seat = a.get("seat")
            act = a.get("action")
            if seat is None or act is None or act in ("small_blind", "big_blind"):
                continue
                
            bot_id = seat_to_bot.get(seat, str(seat))
            s = stats.setdefault(bot_id, {"hands_seen": 0, "vpip": 0, "pfr": 0, "aggr": 0, "passive": 0})
            
            if bot_id not in seen_in_hand:
                seen_in_hand.add(bot_id)
                s["hands_seen"] += 1
                
            if act in ("call", "raise", "all_in") and bot_id not in vpip_in_hand:
                vpip_in_hand.add(bot_id)
                s["vpip"] += 1
                
            if act in ("raise", "all_in") and bot_id not in pfr_in_hand:
                pfr_in_hand.add(bot_id)
                s["pfr"] += 1
                
            if act in ("raise", "all_in"): s["aggr"] += 1
            elif act == "call": s["passive"] += 1
            
    out = {}
    for bot_id, s in stats.items():
        if bot_id == my_bot_id: continue
        h = max(s["hands_seen"], 1)
        post_total = s["aggr"] + s["passive"]
        out[bot_id] = {
            "vpip": s["vpip"] / h,
            "pfr": s["pfr"] / h,
            "agg_freq": s["aggr"] / max(post_total, 1),
            "hands_seen": s["hands_seen"],
        }
    return out, seat_to_bot

def classify_table(opp_models, active_opp_bot_ids):
    # Reduced hands_seen requirement from 15 to 8 so it adapts much faster
    relevant = [opp_models[b] for b in active_opp_bot_ids if b in opp_models and opp_models[b]["hands_seen"] >= 8]
    if not relevant: return "unknown"
    
    avg_vpip = sum(r["vpip"] for r in relevant) / len(relevant)
    avg_agg = sum(r["agg_freq"] for r in relevant) / len(relevant)
    
    # NEW: Realistic 6-max averages
    if avg_vpip > 0.40 and avg_agg > 0.35: return "maniacs"
    if avg_vpip < 0.18: return "nits"
    if avg_vpip > 0.32: return "loose"
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

    if owed >= stack * 0.85:
        pot_odds = owed / max(pot + owed, 1)
        if table_tag == "maniacs": required = pot_odds + 0.08
        elif table_tag == "nits": required = pot_odds + 0.04
        else: required = pot_odds + 0.06
        
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

def _opp_range_floor(opp_models, active_opp_bot_ids, table_tag, analysis, facing_big_bet=False, facing_shove=False):
    floor = 0.45
    if analysis["raises_preflop"] >= 1: floor = 0.52
    if analysis["raises_preflop"] >= 2: floor = 0.60
    if analysis["raises_preflop"] >= 3: floor = 0.65
    
    pf_aggro = analysis.get("postflop_aggro_actions", 0)
    floor += (pf_aggro * 0.06) 
    
    if facing_big_bet: floor += 0.05
    if facing_shove: floor += 0.10

    # HEADS-UP TARGETING FIX
    if len(active_opp_bot_ids) == 1:
        villain = opp_models.get(active_opp_bot_ids[0], {})
        # Note: Changed from 15 to 8 to match our new fast-adapting tracker
        if villain.get("hands_seen", 0) >= 8:
            v_vpip = villain.get("vpip", 0.5)
            v_agg = villain.get("agg_freq", 0.5)
            
            if v_vpip > 0.60 and v_agg > 0.45: floor -= 0.12 # Target Maniac
            elif v_vpip < 0.22: floor += 0.08                # Target Nit
            elif v_vpip > 0.50: floor -= 0.05                # Target Loose
            return max(0.0, min(0.78, floor))

    # Fallback to general table vibe for multi-way pots
    if table_tag == "maniacs": floor -= 0.10
    elif table_tag == "nits":  floor += 0.06
    
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
    
    # NEW: Extract bot_ids and pass them to the updated range floor function
    active_opp_bot_ids = [p.get("bot_id", str(p["seat"])) for p in active_opps]
    opp_floor = _opp_range_floor(opp_models, active_opp_bot_ids, table_tag, analysis, facing_big_bet, facing_shove)
    
    eq = _lookup_equity(hole, board, n_opps, street, base_iters, deadline, opp_floor)

    if not can_check:
        pot_odds = owed / max(pot + owed, 1)

        buffer = 0.04 + (n_opps - 1) * 0.015
        if owed >= stack * 0.4: buffer += 0.04
        if owed >= stack * 0.75: buffer += 0.08
        required = pot_odds + buffer

        if eq >= 0.82 and n_opps == 1:
            if stack < pot * 1.5: return {"action": "all_in"}
            return _safe_raise(int(my_bet + (pot + owed) * 1.0), state)
        if eq >= 0.68:
            return _safe_raise(int(my_bet + (pot + owed) * 0.75), state)

        if eq >= required: return {"action": "call"}
        return {"action": "fold"}

    val_thresh = 0.72 if n_opps >= 3 else 0.65
    size_frac = 0.0
    
    if eq >= 0.78: 
        if street == "flop" and random.random() < 0.20:
            return {"action": "check"}
        size_frac = 0.85 + random.uniform(-0.12, 0.12)
    elif eq >= val_thresh: 
        size_frac = 0.66 + random.uniform(-0.12, 0.12)
    elif eq >= 0.55: 
        size_frac = 0.50 + random.uniform(-0.12, 0.12)

    if size_frac > 0:
        return _safe_raise(max(int(my_bet + pot * size_frac), min_to), state)

    if can_check and size_frac == 0.0:
        if street == "flop" and analysis.get("i_was_pf_aggressor") and n_opps <= 2 and eq >= 0.28 and table_tag != "maniacs":
            if _board_texture_bin(board) == 0: 
                if random.random() < 0.55:
                    cbet_frac = random.uniform(0.45, 0.55)
                    return _safe_raise(max(int(my_bet + pot * cbet_frac), min_to), state)
                    
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
        opp_models, seat_to_bot = model_opponents(state)
        my_seat = state["seat_to_act"]
        
        # Track by bot_id, not seat, so we can isolate specific players correctly
        active_opp_bot_ids = [p.get("bot_id", str(p["seat"])) for p in state["players"] if not p.get("is_folded") and p["seat"] != my_seat]
        
        table_tag = classify_table(opp_models, active_opp_bot_ids)
        analysis = track_hand(state)

        if state["street"] == "preflop":
            return preflop_decision(state, opp_models, table_tag, analysis)
        
        return postflop_decision(state, opp_models, table_tag, analysis, deadline)

    except Exception:
        try: return _fold_or_check(state)
        except Exception: return {"action": "fold"}