"""
╔══════════════════════════════════════════════════════════════════╗
║              "Hydra" — Adaptive Frequency Exploiter              ║
║         (Flawless Edition: Multi-way safe, Loop-safe)            ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
  1. Preflop: Analytic hand-strength function (instant calculation).
  2. Postflop: Fast Monte Carlo equity (strict 1,200 iteration cap).
  3. Profiling: Tracks Opponent Fold-to-Bet Rate and Aggression.
  4. Exploitation: Widens value/bluff thresholds if the active 
     opponents are over-folding or under-bluffing.
"""

import time
import random
from collections import defaultdict

import eval7

BOT_NAME = "Hydra"
BOT_AVATAR = "robot_2"

# ── Constants & Precomputations ──────────────────────────────────────────────
SB, BB = 50, 100
RANK_VAL = {r: i for i, r in enumerate("23456789TJQKA", 2)}
SUITS = "shdc"

_DECK_STRS = [r + s for r in "23456789TJQKA" for s in SUITS]
_FULL_DECK = [eval7.Card(c) for c in _DECK_STRS]

# ── Global Opponent Tracking ─────────────────────────────────────────────────
# Tracks raw frequencies rather than complex range modeling
_OPP_STATS = defaultdict(lambda: {"actions": 0, "calls": 0, "raises": 0, "folds": 0})
_HAND_COUNT = defaultdict(int)

def _ingest_log(log, my_bid):
    """Safely parse the sliding match_action_log to build frequency stats."""
    seen = defaultdict(int)
    for e in log:
        hn = e.get("hand_num")
        if hn is None: 
            continue
        
        seen[hn] += 1
        # Deduplication: Skip if we've already parsed this action for this hand
        if seen[hn] - 1 < _HAND_COUNT[hn]: 
            continue
        _HAND_COUNT[hn] = seen[hn]

        bid, act = e.get("bot_id"), e.get("action")
        if bid is None or bid == my_bid or act is None: 
            continue
        if act in ("small_blind", "big_blind"): 
            continue

        s = _OPP_STATS[bid]
        s["actions"] += 1
        if act == "call": 
            s["calls"] += 1
        elif act in ("raise", "all_in"): 
            s["raises"] += 1
        elif act == "fold": 
            s["folds"] += 1


# ── Core Math Engines ────────────────────────────────────────────────────────
def _preflop_equity(cards):
    """Fast, analytic preflop equity vs random hand. (Sklansky-Malmuth based)"""
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    v1, v2 = RANK_VAL[r1], RANK_VAL[r2]
    hi, lo = max(v1, v2), min(v1, v2)
    suited = (s1 == s2)
    pair = (v1 == v2)

    if pair:
        return 0.50 + (hi - 2) * 0.029

    eq = 0.28 + hi * 0.022 + lo * 0.011
    if suited: 
        eq += 0.035
    
    gap = hi - lo - 1
    if gap == 0:     eq += 0.018
    elif gap == 1:   eq += 0.006
    elif gap >= 4:   eq -= 0.012
    
    if hi >= 10 and lo >= 10: 
        eq += 0.015

    return max(0.30, min(0.87, eq))

def _mc_equity(hole_strs, board_strs, n_opps, deadline):
    """Lightning-fast Monte Carlo capped strictly at 1,200 iterations."""
    hole = [eval7.Card(c) for c in hole_strs]
    board = [eval7.Card(c) for c in board_strs]
    used = set(hole_strs) | set(board_strs)
    deck = [c for c in _FULL_DECK if str(c) not in used]

    need_board = 5 - len(board)
    need_opp = 2 * n_opps
    if need_board + need_opp > len(deck):
        return 0.5

    wins, iters = 0.0, 0
    shuffle = random.shuffle
    evaluate = eval7.evaluate
    now = time.perf_counter

    # SAFETY RAILS: Fast execution to prevent hanging
    max_iters = 300
    min_iters = 50

    while iters < max_iters:
        if now() >= deadline:
            break
            
        shuffle(deck)
        full_board = board + deck[need_opp:need_opp + need_board]
        my_score = evaluate(hole + full_board)
        
        best_opp = -1
        for i in range(0, need_opp, 2):
            s = evaluate([deck[i], deck[i + 1]] + full_board)
            if s > best_opp:
                best_opp = s
                
        if my_score > best_opp:
            wins += 1.0
        elif my_score == best_opp:
            wins += 0.5
        iters += 1

    return wins / max(iters, 1)


# ── Board & Hand Analysis Helpers ────────────────────────────────────────────
def _board_texture(board):
    if len(board) < 3:
        return {"dry": True, "high": 14}
    ranks = [RANK_VAL[c[0]] for c in board]
    suits = [c[1] for c in board]
    
    paired = len(set(ranks)) < len(ranks)
    max_suit = max({s: suits.count(s) for s in set(suits)}.values())
    flush_possible = max_suit >= 3

    sorted_ranks = sorted(set(ranks))
    straight_possible = any(sorted_ranks[i+2] - sorted_ranks[i] <= 4 for i in range(len(sorted_ranks)-2))

    dry = not flush_possible and not straight_possible and not paired
    return {"dry": dry, "high": max(ranks)}

def _has_draw(hole, board):
    if len(board) < 3: return False
    all_suits = [c[1] for c in hole] + [c[1] for c in board]
    if any(all_suits.count(s) >= 4 for s in SUITS): return True
    all_ranks = sorted(set(RANK_VAL[c[0]] for c in hole + board))
    return any(all_ranks[i+3] - all_ranks[i] <= 4 for i in range(len(all_ranks)-3))

def _have_top_pair_plus(hole, board):
    if len(board) < 3: return False
    board_ranks = [RANK_VAL[c[0]] for c in board]
    hole_ranks = [RANK_VAL[c[0]] for c in hole]
    top_board = max(board_ranks)
    
    # Overpair
    if hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > top_board: return True
    # Top pair
    if top_board in hole_ranks: return True
    # Two pair
    return sum(1 for r in hole_ranks if r in board_ranks) >= 2


# ── Engine Action Builders ───────────────────────────────────────────────────
def _do_raise(state, target_total):
    min_to = state["min_raise_to"]
    stack = state["your_stack"]
    max_total = stack + state["your_bet_this_street"]
    target_total = max(int(target_total), min_to)
    
    if target_total >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target_total}

def _fold_or_check(state):
    return {"action": "check"} if state["can_check"] else {"action": "fold"}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DECISION LOGIC
# ═════════════════════════════════════════════════════════════════════════════
def decide(state: dict) -> dict:
    if state.get("type") == "warmup":
        return {"action": "check"}

    try:
        t0 = time.perf_counter()
        my_seat = state["seat_to_act"]
        my_bid = state["players"][my_seat]["bot_id"]

        # 1. Update frequency stats from the match log
        _ingest_log(state.get("match_action_log") or [], my_bid)

        cards = state["your_cards"]
        board = state["community_cards"]
        pot = state["pot"]
        owed = state["amount_owed"]
        stack = state["your_stack"]
        my_bet = state["your_bet_this_street"]
        min_to = state["min_raise_to"]
        can_check = state["can_check"]
        street = state["street"]

        # 2. Extract Active Opponent Averages
        active_bids = [p["bot_id"] for p in state["players"] if not p["is_folded"] and p["seat"] != my_seat]
        n_opps = max(1, len(active_bids))

        frs, ags = [], []
        for bid in active_bids:
            s = _OPP_STATS[bid]
            facings = s["folds"] + s["calls"] + s["raises"]
            frs.append(s["folds"] / max(1, facings) if facings > 0 else 0.40)
            ags.append(s["raises"] / max(1, s["calls"] + s["raises"]) if s["calls"] + s["raises"] > 0 else 0.40)

        opp_fold_rate = sum(frs) / len(frs) if frs else 0.40
        opp_aggro = sum(ags) / len(ags) if ags else 0.40
        has_reads = any(_OPP_STATS[b]["actions"] >= 5 for b in active_bids)

        # ── PREFLOP ──
        if street == "preflop":
            eq = _preflop_equity(cards)
            raises = sum(1 for a in state.get("action_log", []) if a.get("action") in ("raise", "all_in"))
            in_bb = any(a.get("action") == "big_blind" and a.get("seat") == my_seat for a in state.get("action_log", []))

            # Short stack push/fold
            if stack / BB <= 10:
                return {"action": "all_in"} if eq >= 0.54 else _fold_or_check(state)

            # All-in defense logic
            if owed >= stack * 0.80:
                return {"action": "call"} if eq >= (owed / max(pot + owed, 1)) + 0.02 else _fold_or_check(state)

            # Unopened pot: Positional logic
            if raises == 0:
                if in_bb and owed == 0:
                    return {"action": "check"}

                # Open wider if the table over-folds
                open_thresh = 0.46
                if has_reads and opp_fold_rate > 0.50:
                    open_thresh = 0.42

                if eq >= open_thresh:
                    open_size = int(BB * 2.5) if eq < 0.70 else int(BB * 3)
                    return _do_raise(state, open_size)
                return _fold_or_check(state)

            # Facing 1 raise
            if raises == 1:
                if eq >= 0.64:
                    return _do_raise(state, max(int(state["current_bet"] * 3.2), min_to))
                # Bluff 3-bet against high-folders
                if has_reads and opp_fold_rate > 0.55 and eq >= 0.45 and random.random() < 0.25 and owed <= stack * 0.06:
                    return _do_raise(state, max(int(state["current_bet"] * 3), min_to))
                # Standard Defense
                if owed <= BB * 3 and eq >= 0.42: return {"action": "call"}
                if owed <= BB * 6 and eq >= 0.48: return {"action": "call"}
                if eq >= 0.53: return {"action": "call"}
                return _fold_or_check(state)

            # Facing multiple raises (4-bet+)
            if eq >= 0.72: return {"action": "all_in"}
            if eq >= 0.62 and owed <= stack * 0.25: return {"action": "call"}
            return _fold_or_check(state)


        # ── POSTFLOP ──
        deadline = t0 + 0.9
        eq = _mc_equity(cards, board, n_opps, deadline)
        
        tex = _board_texture(board)
        draw = _has_draw(cards, board)
        tp_plus = _have_top_pair_plus(cards, board)
        
        last_raiser = next((a.get("seat") for a in reversed(state.get("action_log", [])) if a.get("action") in ("raise", "all_in")), None)
        i_raised_pf = (last_raiser == my_seat)

        # ── FACING A BET ──
        if not can_check:
            pot_odds = owed / max(pot + owed, 1)

            if eq >= 0.80:
                return _do_raise(state, my_bet + (pot + owed) * 0.90)
            if eq >= 0.70:
                return _do_raise(state, my_bet + (pot + owed) * 0.75) if street == "river" else {"action": "call"}
            if street == "flop" and draw and eq >= 0.38 and random.random() < 0.30:
                return _do_raise(state, my_bet + (pot + owed) * 0.80)
            
            if eq >= pot_odds: return {"action": "call"}
            if draw and street != "river" and eq >= pot_odds * 0.80: return {"action": "call"}
            
            # Hero call adjustment against hyper-aggressive players
            if has_reads and opp_aggro > 0.55 and eq >= pot_odds * 0.85 and tp_plus:
                return {"action": "call"}
            
            return {"action": "fold"}


        # ── WE CAN CHECK (Opportunity to Bet) ──
        if eq >= 0.78:   size = 0.80
        elif eq >= 0.65: size = 0.65
        elif eq >= 0.55: size = 0.60 if (has_reads and opp_fold_rate < 0.25) else 0.40 # Thin value / block
        else:            size = 0.0

        # Exploitative C-bet logic
        if size == 0 and street == "flop" and i_raised_pf:
            should_cbet = False
            if eq >= 0.45: should_cbet = True
            elif tex["dry"] and tex["high"] >= 11: should_cbet = True
            elif has_reads and opp_fold_rate > 0.45: should_cbet = random.random() < 0.70
            else: should_cbet = random.random() < 0.40

            if should_cbet:
                size = 0.40 if tex["dry"] else 0.60

        # Exploitative Barrel logic
        if size == 0 and street in ("turn", "river") and i_raised_pf:
            if eq >= 0.50: size = 0.55
            elif has_reads and opp_fold_rate > 0.50 and random.random() < 0.35: size = 0.55

        # Semi-bluffing draws
        if size == 0 and draw and street != "river":
            if eq >= 0.35: size = 0.55
            elif random.random() < 0.30: size = 0.45

        # Deceptive Check
        if eq >= 0.88 and street != "river" and tex["dry"] and random.random() < 0.30:
            return {"action": "check"}

        if size > 0:
            return _do_raise(state, my_bet + pot * size)

        return {"action": "check"}

    except Exception:
        # Fallback safety net to never crash the engine
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}