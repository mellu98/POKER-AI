#!/usr/bin/env python3
"""
Stake Texas Hold'em Poker Assistant — Pro Mode v1.3
Usage:
  python assistant.py --overlay              # manual input via HUD
  python assistant.py --overlay --api-key KEY  # auto-detect + manual fallback
  python assistant.py --test                 # test screen capture only
"""

import os
import sys
import random
import argparse
import base64
import json
import re
import itertools
import tkinter as tk
from tkinter import messagebox
from collections import Counter

# ─────────────────────────────────────────────
# CARD UTILITIES
# ─────────────────────────────────────────────

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]
FULL_DECK = [r + s for r in RANKS for s in SUITS]
RANK_MAP = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def normalize_card(c):
    c = c.strip().upper()
    sm = {"S": "S", "H": "H", "D": "D", "C": "C"}
    if len(c) == 2:
        r, s = c[0], sm.get(c[1], c[1])
        if r in [x.upper() for x in RANKS] and s in ["S", "H", "D", "C"]:
            return r + s.lower()
    elif len(c) == 3 and c[:2] == "10":
        return "T" + sm.get(c[2], c[2]).lower()
    return None


def card_rank(c):
    return RANK_MAP.get(c[0].upper(), 0)


def card_suit(c):
    return c[1].lower()


def hi_rank_name(r):
    return {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T"}.get(r, str(r))


# ─────────────────────────────────────────────
# HAND EVALUATOR
# ─────────────────────────────────────────────


def hand_rank_5(cards):
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]
    is_flush = len(set(suits)) == 1
    is_straight = len(set(ranks)) == 5 and ranks[0] - ranks[4] == 4
    if set(ranks) == {14, 2, 3, 4, 5}:
        ranks = [5, 4, 3, 2, 1]
        is_straight = True
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freq = [g[1] for g in groups]
    vals = [g[0] for g in groups]
    if is_straight and is_flush:
        return (8, ranks[0])
    if freq[0] == 4:
        return (7, vals[0], vals[1])
    if freq[:2] == [3, 2]:
        return (6, vals[0], vals[1])
    if is_flush:
        return (5,) + tuple(ranks)
    if is_straight:
        return (4, ranks[0])
    if freq[0] == 3:
        return (3, vals[0]) + tuple(vals[1:])
    if freq[:2] == [2, 2]:
        return (2, max(vals[0], vals[1]), min(vals[0], vals[1]), vals[2])
    if freq[0] == 2:
        return (1, vals[0]) + tuple(vals[1:])
    return (0,) + tuple(ranks)


def best_holdem_hand(hole, board):
    all_cards = hole + board
    best = None
    for five in itertools.combinations(all_cards, 5):
        rank = hand_rank_5(list(five))
        if best is None or rank > best:
            best = rank
    return best


def hand_category(rt):
    return {
        8: "Straight Flush",
        7: "Four of a Kind",
        6: "Full House",
        5: "Flush",
        4: "Straight",
        3: "Three of a Kind",
        2: "Two Pair",
        1: "One Pair",
        0: "High Card",
    }.get(rt[0], "Unknown")


# ─────────────────────────────────────────────
# MONTE CARLO ENGINE
# ─────────────────────────────────────────────


def monte_carlo_holdem(hole, board, num_opponents=1, simulations=2000):
    if num_opponents <= 0:
        return {
            "win": 100.0,
            "tie": 0.0,
            "lose": 0.0,
            "hand_distribution": {},
            "simulations": 0,
        }
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in known]
    need = 5 - len(board)
    wins = ties = losses = 0
    my_hands = {}
    for _ in range(simulations):
        s = random.sample(deck, need + num_opponents * 2)
        sim_board = board + s[:need]
        my_rank = best_holdem_hand(hole, sim_board)
        my_hands[my_rank[0]] = my_hands.get(my_rank[0], 0) + 1
        opp_ranks = [
            best_holdem_hand(s[need + i * 2 : need + i * 2 + 2], sim_board)
            for i in range(num_opponents)
        ]
        bo = max(opp_ranks)
        if my_rank > bo:
            wins += 1
        elif my_rank == bo:
            ties += 1
        else:
            losses += 1
    t = simulations
    hd = {
        hand_category((k,)): round(v / t * 100, 1)
        for k, v in sorted(my_hands.items(), key=lambda x: -x[1])[:5]
    }
    return {
        "win": round(wins / t * 100, 1),
        "tie": round(ties / t * 100, 1),
        "lose": round(losses / t * 100, 1),
        "hand_distribution": hd,
        "simulations": t,
    }


# ─────────────────────────────────────────────
# SCREEN CAPTURE
# ─────────────────────────────────────────────


def capture_screen():
    out = os.path.join(os.getcwd(), "stake_screen.jpg")
    try:
        from PIL import ImageGrab, Image

        img = ImageGrab.grab()
        if max(img.size) > 1280:
            scale = 1280 / max(img.size)
            img = img.resize(
                (int(img.size[0] * scale), int(img.size[1] * scale)),
                Image.Resampling.LANCZOS,
            )
        img.convert("RGB").save(out, "JPEG", quality=80, optimize=True)
        print(f"  [i] Screenshot: {out} ({os.path.getsize(out)//1024}KB)")
        return out
    except Exception as e:
        raise RuntimeError(f"Capture failed: {e}. Install pillow: pip install pillow")


# ─────────────────────────────────────────────
# GEMINI VISION
# ─────────────────────────────────────────────

VISION_PROMPT = """Analyze this Stake.com Texas Hold'em screenshot.
STAKE.COM UI HINTS:
- HOLE CARDS: Associate the 2 cards next to 'Zackbabyjrjr' as hole cards.
- BOARD CARDS: Face-up community cards in the center slots only.
- DEALER BUTTON: Look for a small white button marked 'D'. 

DETERMINE POSITION of 'Zackbabyjrjr' relative to 'D':
1. If 'D' is at Zackbabyjrjr's seat -> Zackbabyjrjr is BTN.
2. If 'D' is 1 seat counter-clockwise to Zackbabyjrjr -> Zackbabyjrjr is SB.
3. If 'D' is 2 seats counter-clockwise to Zackbabyjrjr -> Zackbabyjrjr is BB.
4. Rotate as expected for other seats (CO, HJ, MP, UTG).
(Positions: BTN, SB, BB, UTG, MP, HJ, CO)

SUIT IDENTIFICATION:
- RED = HEARTS (rounded h) or DIAMONDS (angular d).
- BLACK = SPADES (stem s) or CLUBS (clover c).
Look at the LARGE icon in the center of the card!

Return STRICT JSON:
{
  "hole_cards": [], 
  "board_cards": [],
  "pot_size": 0.0,
  "bet_facing": 0.0,
  "stack_size": 0.0,
  "num_opponents": 0,
  "position": "BTN",
  "confidence": "high"
}
(No markdown. Only detect what is visible.)
"""


def detect_with_vision(image_path, api_key):
    import urllib.request
    import urllib.error

    ext = os.path.splitext(image_path)[1].lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        ext, "image/png"
    )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = json.dumps(
        {
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/google/agentic-coding",
            "X-Title": "Stake Holdem Assistant",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not data.get("choices"):
            err = data.get("error", {}).get("message", "Unknown error")
            raise ValueError(f"OpenRouter Error: {err}")

        raw = data["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)

    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        if e.code == 429:
            raise RuntimeError(
                "OpenRouter Rate Limit (429). Please wait a few seconds."
            )
        raise RuntimeError(f"OpenRouter API {e.code}: {err_body[:200]}")
    except Exception as ex:
        raise RuntimeError(f"AI parsing error: {ex}")


# ─────────────────────────────────────────────
# VALIDATE CARDS
# ─────────────────────────────────────────────


def validate_cards(detected):
    errors = []
    raw_h = detected.get("hole_cards", [])
    raw_b = detected.get("board_cards", [])
    for r in raw_h:
        if normalize_card(r) is None:
            errors.append(f"Invalid hole card: '{r}'")
    for r in raw_b:
        if normalize_card(r) is None:
            errors.append(f"Invalid board card: '{r}'")
    hole = [c for c in [normalize_card(x) for x in raw_h] if c]
    board = [c for c in [normalize_card(x) for x in raw_b] if c]
    if len(hole) != 2:
        errors.append(f"Need 2 hole cards, got {len(hole)}")
    if len(board) not in [0, 3, 4, 5]:
        errors.append(f"Board needs 0/3/4/5 cards, got {len(board)}")
    if len(hole + board) != len(set(hole + board)):
        errors.append("Duplicate cards")
    return hole, board, errors


# ─────────────────────────────────────────────
# BOARD TEXTURE CLASSIFIER
# ─────────────────────────────────────────────


def board_texture(board):
    """
    Classifies the board as dry / semi-wet / wet / monotone.
    Wet boards = many draws possible → charge draws, don't bluff often.
    Dry boards = few draws → bluff more, value bet thinner.
    """
    if len(board) < 3:
        return {
            "type": "unknown",
            "wet_score": 0,
            "paired": False,
            "monotone": False,
            "two_tone": False,
            "rainbow": False,
            "connected": False,
            "high_card": False,
        }

    ranks = sorted([card_rank(c) for c in board], reverse=True)
    suits = [card_suit(c) for c in board]
    suit_c = Counter(suits)

    max_suited = max(suit_c.values())
    monotone = max_suited >= 3
    two_tone = max_suited == 2
    rainbow = max_suited == 1

    rank_c = Counter(ranks)
    paired = max(rank_c.values()) >= 2

    uniq = sorted(set(ranks))
    gaps = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    connected = any(g <= 2 for g in gaps)

    high_card = ranks[0] >= 10

    wet = 0
    if monotone:
        wet += 2
    elif two_tone:
        wet += 1
    if connected:
        wet += 1
    if not paired and connected and not rainbow:
        wet += 1

    texture = (
        "monotone"
        if monotone
        else "wet" if wet >= 2 else "semi-wet" if wet == 1 else "dry"
    )

    return {
        "type": texture,
        "wet_score": wet,
        "paired": paired,
        "monotone": monotone,
        "two_tone": two_tone,
        "rainbow": rainbow,
        "connected": connected,
        "high_card": high_card,
    }


# ─────────────────────────────────────────────
# POSITION-AWARE PREFLOP RANGES
# ─────────────────────────────────────────────

# Minimum position rank required to open-raise each hand category.
# POSITION_RANK: BTN=6, CO=5, HJ=4, MP=3, EP=2, BB=1, SB=0
OPEN_THRESHOLDS = {
    "premium_pair": 0,  # AA-TT: raise from anywhere
    "mid_pair": 2,  # 77-99: EP+ fine
    "small_pair": 3,  # 22-66: MP+ for set mining
    "broadway": 0,  # AK, AQ, KQ offsuit
    "strong_ace": 0,  # AK, AQ suited
    "suited_ace_hi": 3,  # AJs-ATs: MP+
    "suited_ace_lo": 4,  # A9s-A2s: HJ+ (BTN steal fine)
    "suited_broadway": 2,  # KQs, KJs, QJs: EP+ ok
    "suited_connector": 4,  # JTs, T9s, 98s: HJ+
    "weak_suited": 5,  # 87s, 76s: CO+ only
    "offsuit_connector": 6,  # JTo, T9o: BTN only
}


def preflop_category(hole, suited, hi, lo):
    paired = hi == lo
    if paired:
        if hi >= 10:
            return "premium_pair"
        if hi >= 7:
            return "mid_pair"
        return "small_pair"
    if hi == 14:
        if lo >= 11:
            return "strong_ace"  # AK, AQ, AJ
        if suited and lo >= 10:
            return "suited_ace_hi"  # ATs
        if suited:
            return "suited_ace_lo"  # A9s and below
        if lo >= 10:
            return "broadway"  # ATo offsuit
        return None  # A9o and worse — fold EP/MP
    gap = hi - lo
    if hi >= 11 and lo >= 10:
        return "broadway"  # KQ, KJ, QJ, JT offsuit
    if hi >= 11 and suited:
        return "suited_broadway"
    if suited and gap <= 1 and lo >= 9:
        return "suited_connector"
    if suited and gap <= 1:
        return "weak_suited"
    if gap <= 1 and lo >= 9:
        return "offsuit_connector"
    return None


# ─────────────────────────────────────────────
# OPPONENT COUNT EQUITY SCALING
# ─────────────────────────────────────────────


def equity_threshold_adjust(base_threshold, num_opponents):
    """
    Scales call/raise equity thresholds upward with more opponents.
    1 opp → +0%  |  2 opps → +3%  |  3 opps → +6%  |  4+ → +9%
    """
    adjustment = min(num_opponents - 1, 3) * 0.03
    return base_threshold + adjustment


# ─────────────────────────────────────────────
# PRO RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

POSITION_RANK = {"BTN": 6, "CO": 5, "HJ": 4, "MP": 3, "EP": 2, "BB": 1, "SB": 0}


def get_recommendation(hole, board, equity, context=None):
    rec = get_recommendation_inner(hole, board, equity, context)

    if rec["action"] == "RAISE":
        stage = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
            len(board), "river"
        )
        ctx = context or {}
        pos = ctx.get("position", "MP").upper()
        pot = float(ctx.get("pot_size", 0))
        num_opponents = int(ctx.get("num_opponents", equity.get("opponents", 1)))

        tx = board_texture(board) if len(board) >= 3 else {"type": "unknown"}
        tex_type = tx.get("type", "unknown")

        if stage == "preflop":
            if pos in ("BTN", "CO"):
                sz = "2.5x"
            elif num_opponents > 2:
                sz = "4x (+1x per limper)"
            else:
                sz = "3x"
            rec["reasoning"] = str(rec.get("reasoning", "")) + f"  [suggest {sz} raise]"

        elif stage == "flop":
            pct = (
                0.33 if tex_type == "dry" else 0.50 if tex_type == "semi-wet" else 0.75
            )
            if pot > 0:
                rec["reasoning"] = (
                    str(rec.get("reasoning", ""))
                    + f"  [suggest ~{round(pot * pct, 1)} chips ({int(pct*100)}% pot)]"
                )
            else:
                rec["reasoning"] = (
                    str(rec.get("reasoning", "")) + f"  [suggest {int(pct*100)}% pot]"
                )

        elif stage == "turn":
            pct = 0.75 if tex_type == "wet" else 0.55
            if pot > 0:
                rec["reasoning"] = (
                    str(rec.get("reasoning", ""))
                    + f"  [suggest ~{round(pot * pct, 1)} chips ({int(pct*100)}% pot)]"
                )
            else:
                rec["reasoning"] = (
                    str(rec.get("reasoning", "")) + f"  [suggest {int(pct*100)}% pot]"
                )

    return rec


def get_recommendation_inner(hole, board, equity, context=None):
    if len(hole) < 2:
        return {
            "action": "FOLD",
            "confidence": "LOW",
            "reasoning": "Incomplete card data.",
            "details": {},
        }

    win = equity["win"] / 100
    stage = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(len(board), "river")
    ctx = context or {}
    pot = float(ctx.get("pot_size", 0))
    bet = float(ctx.get("bet_facing", 0))
    stack = float(ctx.get("stack_size", 0))
    pos = ctx.get("position", "MP")
    num_opponents = int(ctx.get("num_opponents", equity.get("opponents", 1)))

    pos_rank = POSITION_RANK.get(pos.upper(), 3)
    in_pos = pos_rank >= 4  # HJ, CO, BTN

    # Pot odds
    if bet > 0 and pot > 0:
        por = bet / (pot + bet)
        has_po = win >= por
        por_pct = round(por * 100, 1)
    else:
        por = 0
        has_po = True
        por_pct = 0

    # SPR
    spr = (stack / pot) if pot > 0 and stack > 0 else 10

    # Bet sizing read
    if pot > 0 and bet > 0:
        br = bet / pot
        bet_read = (
            "small probe/blocker"
            if br <= 0.33
            else (
                "medium value/draw"
                if br <= 0.66
                else "large value/protection" if br <= 1.0 else "overbet (polarised)"
            )
        )
    else:
        br = 0
        bet_read = ""

    # Board texture
    tx = board_texture(board) if len(board) >= 3 else {}

    # Draw detection + rule of 2/4
    def detect_draws():
        if not board:
            return [], False, False, 0
        combined = hole + board
        notes = []
        sc = Counter(card_suit(c) for c in combined)
        fd = max(sc.values()) >= 4
        if fd:
            notes.append("flush draw (~9 outs)")
        rs = sorted(set(card_rank(c) for c in combined))
        oesd = any(rs[i + 3] - rs[i] == 3 for i in range(len(rs) - 3))
        gs = any(rs[i + 3] - rs[i] <= 4 for i in range(len(rs) - 3))
        if oesd:
            notes.append("OESD (~8 outs)")
        elif gs:
            notes.append("gutshot (~4 outs)")
        outs = (9 if fd else 0) + (8 if oesd else 4 if gs else 0)
        de = (
            min(outs * 4 / 100, 0.54)
            if stage == "flop"
            else min(outs * 2 / 100, 0.30) if stage == "turn" else 0
        )
        return notes, fd, oesd, de

    dn, hf, ho, de = detect_draws()
    has_draw = bool(dn)
    ds = f"  [{', '.join(dn)}]" if dn else ""
    eff = min(win + de * 0.5, 0.99)

    # Made hand strength
    if board and len(board) >= 3:
        br2 = best_holdem_hand(hole, board)
        mh = hand_category(br2)
        ms = {
            "Straight Flush": 8,
            "Four of a Kind": 7,
            "Full House": 6,
            "Flush": 5,
            "Straight": 4,
            "Three of a Kind": 3,
            "Two Pair": 2,
            "One Pair": 1,
            "High Card": 0,
        }.get(mh, 0)
        strong = ms >= 3 and win >= 0.50
    else:
        strong = False
        mh = ""
        ms = 0

    # Opponent-scaled thresholds
    call_thresh = equity_threshold_adjust(0.40, num_opponents)
    raise_thresh = equity_threshold_adjust(0.55, num_opponents)

    if spr < 4:
        raise_thresh -= 0.05
    elif spr > 10:
        raise_thresh += 0.05

    spr_advice = (
        "Commit with top pair or better"
        if spr < 2
        else (
            "Commit with two pair or better"
            if spr < 4
            else (
                "Need strong hand to stack off"
                if spr <= 10
                else "Deep stack — need near-nuts to commit"
            )
        )
    )

    details = {
        "pot_odds_required": por_pct,
        "effective_equity": round(eff * 100, 1),
        "spr": round(spr, 1),
        "spr_advice": spr_advice,
        "position": pos,
        "in_position": in_pos,
        "bet_read": bet_read,
        "draw_notes": dn,
        "board_texture": tx.get("type", "N/A"),
        "num_opponents": num_opponents,
    }

    # ── PREFLOP ──
    if stage == "preflop":
        h1, h2 = card_rank(hole[0]), card_rank(hole[1])
        s1, s2 = card_suit(hole[0]), card_suit(hole[1])
        suited = s1 == s2
        paired = h1 == h2
        hi, lo = max(h1, h2), min(h1, h2)

        hand_cat = preflop_category(hole, suited, hi, lo)
        min_pos = OPEN_THRESHOLDS.get(hand_cat, 99) if hand_cat else 99
        can_open = pos_rank >= min_pos

        # Low SPR shove range: commit with premium hands
        spr_shove = spr < 4 and win >= equity_threshold_adjust(0.52, num_opponents)
        if spr_shove and (paired or (hi >= 13 and lo >= 10)):
            return {
                "action": "RAISE",
                "confidence": "HIGH",
                "reasoning": f"Low SPR ({spr:.1f}) — commit with premium ({hi_rank_name(hi)}{hi_rank_name(lo)})",
                "details": details,
            }

        if can_open:
            label = (
                f"{hi_rank_name(hi)}{hi_rank_name(hi)}"
                if paired
                else f'{hi_rank_name(hi)}{hi_rank_name(lo)}{"s" if suited else "o"}'
            )
            a = "RAISE"
            c = "HIGH" if min_pos <= 2 else "MED"
            rsn = f'{hand_cat.replace("_"," ").title()} ({label}) — open from {pos}'
        elif win >= (bet / (pot + bet) if (pot + bet) > 0 else 0.50) and pos in ("BB",):
            a = "CALL"
            c = "MED"
            rsn = f"BB defend ({win*100:.0f}% equity, {por_pct}% needed)"
        elif win * 100 >= 42 and in_pos:
            a = "CALL"
            c = "MED"
            rsn = f"Marginal hand — call in position ({win*100:.0f}%)"
        else:
            a = "FOLD"
            c = "MED"
            rsn = f"Hand too weak or position too early to open ({pos})"

        if a == "CALL" and bet > 0 and not has_po:
            return {
                "action": "FOLD",
                "confidence": "HIGH",
                "reasoning": f"{rsn} — pot odds insufficient ({win*100:.0f}% vs {por_pct}% needed)",
                "details": details,
            }

        return {"action": a, "confidence": c, "reasoning": rsn, "details": details}

    # ── FLOP / TURN ──
    if stage in ("flop", "turn"):
        tex_type = tx.get("type", "unknown")
        tex_note = f"  [{tex_type} board]" if tex_type != "unknown" else ""

        # Wet boards require stronger hands to raise
        if tex_type == "monotone":
            raise_thresh += 0.08
        elif tex_type == "wet":
            raise_thresh += 0.04

        if bet > 0 and eff < por:
            if has_draw and spr > 4:
                note = (
                    f"Drawing but pot odds tight ({eff*100:.0f}% vs {por_pct}% needed)"
                )
                if in_pos:
                    return {
                        "action": "CALL",
                        "confidence": "LOW",
                        "reasoning": f"{note} — implied odds in position{ds}{tex_note}",
                        "details": details,
                    }
                else:
                    return {
                        "action": "FOLD",
                        "confidence": "MED",
                        "reasoning": f"{note} — fold OOP{ds}{tex_note}",
                        "details": details,
                    }
            return {
                "action": "FOLD",
                "confidence": "HIGH",
                "reasoning": f"Pot odds: need {por_pct}%, have {eff*100:.0f}% — fold{ds}{tex_note}",
                "details": details,
            }

        sn = (
            "  [low SPR: ok to commit]"
            if spr < 4
            else "  [deep: need strong hand]" if spr >= 10 else ""
        )
        pn = f'  [{"in" if in_pos else "out of"} position]'
        bn = f"  [opp bet: {bet_read}]" if bet_read else ""

        if strong or win >= raise_thresh:
            hl = f"{mh}, " if mh else ""
            aa = "RAISE"
            cc = "HIGH" if win >= 0.72 else "MED"
            # Monotone board OOP with non-nut hand → peel instead of raising
            if tex_type == "monotone" and not in_pos and ms < 5:
                aa = "CALL"
                rr = f"Strong ({hl}{win*100:.0f}%) but OOP on monotone — call and reassess{tex_note}"
            else:
                rr = f"Strong hand ({hl}{win*100:.0f}%) — build pot, charge draws{sn}{pn}{bn}{tex_note}"
            return {"action": aa, "confidence": cc, "reasoning": rr, "details": details}

        if win >= call_thresh:
            if in_pos and bet == 0:
                return {
                    "action": "RAISE",
                    "confidence": "MED",
                    "reasoning": f"Good equity ({win*100:.0f}%) — bet in position{sn}{tex_note}",
                    "details": details,
                }
            return {
                "action": "CALL",
                "confidence": "MED",
                "reasoning": f"Decent equity ({win*100:.0f}%) — call{pn}{bn}{tex_note}",
                "details": details,
            }

        if win >= 0.40 and (hf or ho):
            return {
                "action": "CALL",
                "confidence": "MED",
                "reasoning": f'Strong draw ({win*100:.0f}%) — {"semi-bluff" if in_pos else "flat call"}{ds}{tex_note}',
                "details": details,
            }

        if has_draw and has_po:
            return {
                "action": "CALL",
                "confidence": "LOW",
                "reasoning": f"Drawing, pot odds ok ({win*100:.0f}%, need {por_pct}%){ds}{tex_note}",
                "details": details,
            }

        if bet == 0 and in_pos:
            return {
                "action": "CALL",
                "confidence": "LOW",
                "reasoning": f"Weak equity ({win*100:.0f}%) — check back, free card{tex_note}",
                "details": details,
            }

        return {
            "action": "FOLD",
            "confidence": "HIGH",
            "reasoning": f"Insufficient equity ({win*100:.0f}%), no draw — fold{pn}{bn}{tex_note}",
            "details": details,
        }

    # ── RIVER ──
    tex_type = tx.get("type", "unknown")
    tex_note = f"  [{tex_type} board]" if tex_type != "unknown" else ""

    if bet > 0 and not has_po:
        return {
            "action": "FOLD",
            "confidence": "HIGH",
            "reasoning": f"River: need {por_pct}%, have {win*100:.0f}% — fold.{tex_note}",
            "details": details,
        }

    hl = f"{mh}, " if mh else ""

    # Bet sizing calibrated to board texture
    if pot > 0:
        sizing = 0.75 if tex_type == "wet" else 0.50 if tex_type == "dry" else 0.65
        size_note = f"  [suggest ~{round(pot * sizing, 1)} ({int(sizing*100)}% pot)]"
    else:
        size_note = "  [bet 50-75% pot]"

    if strong or win >= 0.70:
        sn = size_note if bet == 0 else f"  [vs {bet_read}]" if bet_read else ""
        return {
            "action": "RAISE",
            "confidence": "HIGH",
            "reasoning": f"Strong river ({hl}{win*100:.0f}%) — bet/raise for value{sn}{tex_note}",
            "details": details,
        }

    if win >= 0.52:
        pn = f"  [thin value: {size_note.strip()}]" if in_pos and bet == 0 else ""
        return {
            "action": "CALL",
            "confidence": "HIGH",
            "reasoning": f"Ahead ({win*100:.0f}%) — call{pn}{tex_note}",
            "details": details,
        }

    if win >= 0.35:
        if br < 0.40 and bet > 0:
            return {
                "action": "CALL",
                "confidence": "MED",
                "reasoning": f"Marginal ({win*100:.0f}%) but small bet — call vs {bet_read}{tex_note}",
                "details": details,
            }
        return {
            "action": "FOLD",
            "confidence": "MED",
            "reasoning": f"Marginal ({win*100:.0f}%) — fold vs medium/large bet{tex_note}",
            "details": details,
        }

    return {
        "action": "FOLD",
        "confidence": "HIGH",
        "reasoning": f"Below break-even ({win*100:.0f}%) — fold.{tex_note}",
        "details": details,
    }


# ─────────────────────────────────────────────
# INPUT DIALOG
# ─────────────────────────────────────────────


class InputDialog(tk.Toplevel):
    POSITIONS = ["BTN", "CO", "HJ", "MP", "EP", "BB", "SB"]

    def __init__(self, parent, defaults):
        super().__init__(parent)
        self.title("Hold'em Input")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)
        self.result = None
        BG = "#1a1a2e"
        DARK = "#0d0d0d"
        GREEN = "#00e676"
        WHITE = "#ffffff"
        DIM = "#9e9e9e"
        es = dict(
            bg=DARK,
            fg=GREEN,
            insertbackground=WHITE,
            font=("Consolas", 11),
            relief="flat",
            highlightthickness=1,
            highlightcolor=GREEN,
            highlightbackground="#2a2a2e",
        )

        def lbl(text, row, col):
            tk.Label(
                self,
                text=text,
                bg=BG,
                fg=WHITE,
                font=("Consolas", 9, "bold"),
                anchor="w",
            ).grid(row=row, column=col, sticky="w", padx=(12, 4), pady=(6, 0))

        def ent(row, col, w=14, default=""):
            e = tk.Entry(self, width=w, justify="center", **es)
            e.grid(row=row, column=col, padx=(4, 12), pady=(0, 2), sticky="ew")
            e.insert(0, str(default))
            return e

        tk.Label(
            self,
            text="♠  HOLD'EM ANALYSER",
            bg=BG,
            fg=GREEN,
            font=("Consolas", 12, "bold"),
        ).grid(row=0, column=0, columnspan=4, pady=(14, 8))

        tk.Label(self, text="CARDS", bg=BG, fg=DIM, font=("Consolas", 8)).grid(
            row=1, column=0, columnspan=4, sticky="w", padx=12
        )
        tk.Frame(self, bg="#2a2a2e", height=1).grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 4)
        )
        lbl("Hole cards", 3, 0)
        self.hole_ent = ent(3, 1, 14, defaults.get("hole", ""))
        lbl("Board cards", 3, 2)
        self.board_ent = ent(3, 3, 18, defaults.get("board", ""))

        tk.Label(self, text="TABLE", bg=BG, fg=DIM, font=("Consolas", 8)).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 0)
        )
        tk.Frame(self, bg="#2a2a2e", height=1).grid(
            row=5, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 4)
        )
        lbl("Opponents", 6, 0)
        self.opp_var = tk.IntVar(value=defaults.get("opponents", 2))
        of = tk.Frame(self, bg=BG)
        of.grid(row=6, column=1, sticky="w", padx=(4, 12))
        tk.Scale(
            of,
            from_=1,
            to=9,
            orient="horizontal",
            variable=self.opp_var,
            bg=BG,
            fg=WHITE,
            troughcolor=DARK,
            highlightthickness=0,
            length=120,
            showvalue=True,
            font=("Consolas", 8),
        ).pack()
        lbl("Position", 6, 2)
        self.pos_var = tk.StringVar(value=defaults.get("position", "MP"))
        pf = tk.Frame(self, bg=BG)
        pf.grid(row=6, column=3, sticky="w", padx=(4, 12))
        for p in self.POSITIONS:
            tk.Radiobutton(
                pf,
                text=p,
                variable=self.pos_var,
                value=p,
                bg=BG,
                fg=WHITE,
                selectcolor="#1565c0",
                activebackground=BG,
                font=("Consolas", 8),
            ).pack(side="left")

        tk.Label(
            self, text="MONEY  (0 = unknown)", bg=BG, fg=DIM, font=("Consolas", 8)
        ).grid(row=7, column=0, columnspan=4, sticky="w", padx=12, pady=(10, 0))
        tk.Frame(self, bg="#2a2a2e", height=1).grid(
            row=8, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 4)
        )
        lbl("Pot size", 9, 0)
        self.pot_ent = ent(9, 1, 10, defaults.get("pot", 0))
        lbl("Bet facing", 9, 2)
        self.bet_ent = ent(9, 3, 10, defaults.get("bet", 0))
        lbl("Your stack", 10, 0)
        self.stack_ent = ent(10, 1, 10, defaults.get("stack", 0))

        tk.Button(
            self,
            text="  ANALYSE  ",
            command=self._submit,
            bg="#1565c0",
            fg=WHITE,
            font=("Consolas", 11, "bold"),
            relief="flat",
            padx=10,
            pady=6,
            activebackground="#1976d2",
            cursor="hand2",
        ).grid(row=11, column=0, columnspan=4, pady=16)

        self.bind("<Return>", lambda e: self._submit())
        self.bind("<Escape>", lambda e: self.destroy())
        self.hole_ent.focus_set()
        self.transient(parent)
        self.grab_set()
        self.update_idletasks()
        px = parent.winfo_x() + parent.winfo_width() // 2 - self.winfo_width() // 2
        py = parent.winfo_y() + parent.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{px}+{py}")
        parent.wait_window(self)

    def _submit(self):
        def sf(w):
            try:
                return max(0.0, float(w.get().strip()))
            except Exception:
                return 0.0

        self.result = {
            "hole_cards": [c.strip() for c in self.hole_ent.get().split() if c.strip()],
            "board_cards": [
                c.strip() for c in self.board_ent.get().split() if c.strip()
            ],
            "opponents": self.opp_var.get(),
            "position": self.pos_var.get(),
            "pot_size": sf(self.pot_ent),
            "bet_facing": sf(self.bet_ent),
            "stack_size": sf(self.stack_ent),
            "raw": {
                "hole": self.hole_ent.get().strip(),
                "board": self.board_ent.get().strip(),
            },
        }
        self.destroy()


def get_input(parent, defaults):
    if parent:
        dlg = InputDialog(parent, defaults)
        if dlg.result:
            return dlg.result
        raise RuntimeError("Input cancelled.")
    print("\n── HOLD'EM INPUT ──")
    hole = input(f"Hole cards [{defaults.get('hole','')}]: ") or defaults.get(
        "hole", ""
    )
    board = input(f"Board cards [{defaults.get('board','')}]: ") or defaults.get(
        "board", ""
    )
    try:
        opps = int(
            input(f"Opponents [{defaults.get('opponents',2)}]: ")
            or defaults.get("opponents", 2)
        )
    except Exception:
        opps = defaults.get("opponents", 2)
    pos = input(
        f"Position BTN/CO/HJ/MP/EP/BB/SB [{defaults.get('position','MP')}]: "
    ).strip().upper() or defaults.get("position", "MP")
    try:
        pot = float(input(f"Pot [{defaults.get('pot',0)}]: ") or 0)
    except Exception:
        pot = 0.0
    try:
        bet = float(input(f"Bet facing [{defaults.get('bet',0)}]: ") or 0)
    except Exception:
        bet = 0.0
    try:
        stack = float(input(f"Stack [{defaults.get('stack',0)}]: ") or 0)
    except Exception:
        stack = 0.0
    return {
        "hole_cards": [c.strip() for c in hole.split() if c.strip()],
        "board_cards": [c.strip() for c in board.split() if c.strip()],
        "opponents": opps,
        "position": pos,
        "pot_size": pot,
        "bet_facing": bet,
        "stack_size": stack,
        "raw": {"hole": hole, "board": board},
    }


# ─────────────────────────────────────────────
# OVERLAY HUD
# ─────────────────────────────────────────────


class OverlayHUD:
    BG = "#0d0d0d"
    HEADER_BG = "#1a1a2e"
    AW = "#00e676"
    AT = "#ffeb3b"
    AL = "#f44336"
    DIM = "#9e9e9e"
    BRIGHT = "#ffffff"
    IC = {
        "STRONG FAVORITE": "#00e676",
        "SLIGHT EDGE": "#c6ff00",
        "UNDERDOG": "#ff9800",
        "BIG UNDERDOG": "#f44336",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hold'em HUD")
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-alpha", 0.92)
        self.root.configure(bg=self.BG)
        self.root.geometry("320x450+30+30")
        self.root.resizable(False, False)
        self._dx = self._dy = 0
        self._scanning = False
        self._cb = None
        self._build()

    def _build(self):
        r = self.root
        BG = self.BG
        hdr = tk.Frame(r, bg=self.HEADER_BG, cursor="fleur")
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="♠  HOLD'EM HUD",
            bg=self.HEADER_BG,
            fg=self.BRIGHT,
            font=("Consolas", 10, "bold"),
        ).pack(side="left", padx=8, pady=4)
        cl = tk.Label(
            hdr,
            text=" ✕ ",
            bg=self.HEADER_BG,
            fg="#ff5252",
            font=("Consolas", 10, "bold"),
            cursor="hand2",
        )
        cl.pack(side="right", padx=4)
        cl.bind("<Button-1>", lambda e: r.destroy())
        hdr.bind(
            "<ButtonPress-1>",
            lambda e: (setattr(self, "_dx", e.x), setattr(self, "_dy", e.y)),
        )
        hdr.bind(
            "<B1-Motion>",
            lambda e: r.geometry(
                f"+{r.winfo_x()+e.x-self._dx}+{r.winfo_y()+e.y-self._dy}"
            ),
        )

        inf = tk.Frame(r, bg=BG)
        inf.pack(fill="x", padx=10, pady=(6, 0))
        self.stage_var = tk.StringVar(value="—")
        self.hand_var = tk.StringVar(value="—")
        tk.Label(
            inf, textvariable=self.stage_var, bg=BG, fg=self.DIM, font=("Consolas", 9)
        ).pack(side="left")
        tk.Label(
            r, textvariable=self.hand_var, bg=BG, fg="#b0bec5", font=("Consolas", 9)
        ).pack(anchor="w", padx=10)
        self.odds_var = tk.StringVar(value="")
        tk.Label(
            r, textvariable=self.odds_var, bg=BG, fg="#78909c", font=("Consolas", 8)
        ).pack(anchor="w", padx=10)
        tk.Frame(r, bg="#2a2a2a", height=1).pack(fill="x", padx=8, pady=3)

        pf = tk.Frame(r, bg=BG)
        pf.pack(fill="x", padx=10)
        self.wv = tk.StringVar(value="--.-")
        self.tv = tk.StringVar(value="--.-")
        self.lv = tk.StringVar(value="--.-")

        def pc(parent_frame, label, var, color):
            f = tk.Frame(parent_frame, bg=BG)
            f.pack(side="left", expand=True)
            tk.Label(f, text=label, bg=BG, fg=color, font=("Consolas", 8, "bold")).pack()
            tk.Label(
                f, textvariable=var, bg=BG, fg=color, font=("Consolas", 22, "bold")
            ).pack()
            tk.Label(f, text="%", bg=BG, fg=color, font=("Consolas", 9)).pack()

        pc(pf, "WIN", self.wv, self.AW)
        pc(pf, "TIE", self.tv, self.AT)
        pc(pf, "LOSE", self.lv, self.AL)

        bf = tk.Frame(r, bg=BG)
        bf.pack(fill="x", padx=10, pady=(6, 2))
        self.canvas = tk.Canvas(bf, height=10, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="x")

        self.iv = tk.StringVar(value="Waiting...")
        self.il = tk.Label(
            r, textvariable=self.iv, bg=BG, fg=self.DIM, font=("Consolas", 10, "bold")
        )
        self.il.pack(pady=(4, 0))
        self.bv = tk.StringVar(value="")
        tk.Label(
            r, textvariable=self.bv, bg=BG, fg="#607d8b", font=("Consolas", 8)
        ).pack(pady=(0, 2))
        tk.Frame(r, bg="#2a2a2a", height=1).pack(fill="x", padx=8, pady=(2, 3))

        br = tk.Frame(r, bg=BG)
        br.pack(fill="x", padx=10, pady=(0, 3))
        self.btn = tk.Label(
            br,
            text=" 📸 SCAN ",
            bg="#1565c0",
            fg="#ffffff",
            font=("Consolas", 10, "bold"),
            cursor="hand2",
            relief="flat",
            padx=6,
            pady=4,
        )
        self.btn.pack(side="left")
        self.btn.bind("<Button-1>", self._on_click)

        self.edit_btn = tk.Label(
            br,
            text=" ✎ EDIT ",
            bg="#455a64",
            fg="#ffffff",
            font=("Consolas", 10, "bold"),
            cursor="hand2",
            relief="flat",
            padx=6,
            pady=4,
        )
        self.edit_btn.pack(side="left", padx=(4, 0))
        self.edit_btn.bind("<Button-1>", self._on_edit)

        self.sv = tk.StringVar(value="Ready")
        self.sl = tk.Label(
            br,
            textvariable=self.sv,
            bg=BG,
            fg="#546e7a",
            font=("Consolas", 8),
            wraplength=100,
            justify="left",
        )
        self.sl.pack(side="left", padx=(8, 0))

        tk.Frame(r, bg="#2a2a2a", height=1).pack(fill="x", padx=8, pady=(2, 3))
        rf = tk.Frame(r, bg="#0a1628")
        rf.pack(fill="x", padx=8, pady=(0, 8))
        self.av = tk.StringVar(value="—  awaiting input —")
        self.al = tk.Label(
            rf,
            textvariable=self.av,
            bg="#0a1628",
            fg=self.DIM,
            font=("Consolas", 13, "bold"),
        )
        self.al.pack(pady=(6, 2))
        self.rv = tk.StringVar(value="")
        tk.Label(
            rf,
            textvariable=self.rv,
            bg="#0a1628",
            fg="#78909c",
            font=("Consolas", 7),
            wraplength=280,
            justify="center",
        ).pack(pady=(0, 6))

    def update(self, detected, equity, context=None):
        ctx = context or {}
        board = detected.get("board_cards", [])
        hole = detected.get("hole_cards", [])
        stage = {0: "Pre-flop", 3: "Flop", 4: "Turn", 5: "River"}.get(len(board), "?")
        win, tie, lose = equity["win"], equity["tie"], equity["lose"]
        if win >= 65:
            it, ik = "★ STRONG FAVORITE", "STRONG FAVORITE"
        elif win >= 45:
            it, ik = "◆ SLIGHT EDGE", "SLIGHT EDGE"
        elif win >= 30:
            it, ik = "▼ UNDERDOG", "UNDERDOG"
        else:
            it, ik = "✖ BIG UNDERDOG", "BIG UNDERDOG"
        best = next(iter(equity.get("hand_distribution", {})), "")
        pot = ctx.get("pot_size", 0)
        bet = ctx.get("bet_facing", 0)
        pos = ctx.get("position", "")
        self.stage_var.set(f"{stage}  •  {equity.get('opponents','?')} opp  •  {pos}")
        self.hand_var.set(
            f"Hand: {' '.join(hole)}   Board: {' '.join(board) or '(none)'}"
        )
        if pot > 0 and bet > 0:
            req = round(bet / (pot + bet) * 100, 1)
            self.odds_var.set(f"Pot: {pot}  Bet: {bet}  →  need {req}% equity to call")
        elif pot > 0:
            self.odds_var.set(f"Pot: {pot}  (no bet facing)")
        else:
            self.odds_var.set("")
        self.wv.set(f"{win:.1f}")
        self.tv.set(f"{tie:.1f}")
        self.lv.set(f"{lose:.1f}")
        self.iv.set(it)
        self.il.config(fg=self.IC.get(ik, self.DIM))
        self.bv.set(f"Most likely: {best}" if best else "")
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()
        self.canvas.delete("all")
        ww = int(win / 100 * w)
        tw = int(tie / 100 * w)
        self.canvas.create_rectangle(0, 0, ww, 10, fill=self.AW, outline="")
        self.canvas.create_rectangle(ww, 0, ww + tw, 10, fill=self.AT, outline="")
        self.canvas.create_rectangle(ww + tw, 0, w, 10, fill=self.AL, outline="")
        rec = get_recommendation(hole, board, equity, ctx)
        d = rec.get("details", {})
        AS = {
            "RAISE": ("⬆ RAISE / BET", "#00e676"),
            "CALL": ("➡ CALL / CHECK", "#ffeb3b"),
            "FOLD": ("✖  FOLD", "#f44336"),
        }
        at, ac = AS.get(rec["action"], (rec["action"], self.DIM))
        cm = {"HIGH": "!!!", "MED": " !!", "LOW": "  !"}.get(rec["confidence"], "")
        rsn_text = str(rec.get("reasoning", ""))
        if (
            isinstance(d, dict)
            and d.get("spr_advice")
            and ctx.get("pot_size", 0) > 0
            and ctx.get("stack_size", 0) > 0
        ):
            rsn_text += f"\nSPR Guidance: {d['spr_advice']}"
        self.av.set(f"{at}  {cm}")
        self.al.config(fg=ac)
        self.rv.set(rsn_text)

    def _on_click(self, event=None):
        if self._scanning or not self._cb:
            return
        self._scanning = True
        self.btn.config(bg="#424242", fg="#9e9e9e", text=" ⌛ WAIT ")
        try:
            self._cb()
            self.btn.config(bg="#1565c0", fg="#ffffff", text=" 📸 SCAN ")
            self.sv.set("Done ✓")
        except Exception:
            self.btn.config(bg="#b71c1c", fg="#ffffff", text=" ✖ ERROR ")
            self.root.after(
                3000,
                lambda: self.btn.config(bg="#1565c0", fg="#ffffff", text=" 📸 SCAN "),
            )
        finally:
            self._scanning = False

    def _on_edit(self, event=None):
        if self._cb:
            self._cb(force_manual=True)

    def set_callback(self, fn):
        self._cb = fn

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────
# DISPLAY RESULTS (terminal)
# ─────────────────────────────────────────────


def display_results(detected, equity, num_opponents, context=None):
    ctx = context or {}
    stage = {0: "Pre-flop", 3: "Flop", 4: "Turn", 5: "River"}.get(
        len(detected["board_cards"]), "Unknown"
    )
    pot = ctx.get("pot_size", 0)
    bet = ctx.get("bet_facing", 0)
    pos = ctx.get("position", "?")
    stack = ctx.get("stack_size", 0)
    print("\n" + "═" * 55)
    print("  ♠  TEXAS HOLD'EM — PRO ANALYSIS")
    print("═" * 55)
    print(f"  Stage    : {stage}")
    print(f"  Hand     : {' '.join(detected['hole_cards'])}")
    print(f"  Board    : {' '.join(detected['board_cards']) or '(none)'}")
    print(f"  Opp/Pos  : {num_opponents} opponents  |  {pos}")
    if pot > 0:
        print(f"  Pot/Bet  : {pot} / {bet}  |  Stack: {stack}")
        if bet > 0:
            print(
                f"  Pot odds : need {round(bet/(pot+bet)*100,1)}% equity to call breakeven"
            )
    tx = (
        board_texture(detected["board_cards"])
        if len(detected["board_cards"]) >= 3
        else {}
    )
    if tx:
        print(f"  Board    : {tx.get('type','?')} texture")
    print("─" * 55)
    win, tie, lose = equity["win"], equity["tie"], equity["lose"]
    bl = 30
    wb = int(win / 100 * bl)
    tb = int(tie / 100 * bl)
    bar = "█" * wb + "░" * tb + "▒" * (bl - wb - tb)
    print(f"\n  [{bar}]")
    print(f"  WIN  {win:>6.1f}%  │  TIE  {tie:>5.1f}%  │  LOSE  {lose:>5.1f}%")
    ind = (
        "🟢 STRONG FAVORITE"
        if win >= 65
        else (
            "🟡 SLIGHT EDGE"
            if win >= 45
            else "🟠 UNDERDOG" if win >= 30 else "🔴 BIG UNDERDOG"
        )
    )
    print(f"\n  {ind}")
    print("\n  Hand distribution:")
    for h, p in equity["hand_distribution"].items():
        print(f"    {h:<18} {p:>5.1f}%  {'▪'*int(p/5)}")
    rec = get_recommendation(
        detected["hole_cards"], detected["board_cards"], equity, ctx
    )
    cm = {"HIGH": "!!!", "MED": " !!", "LOW": "  !"}.get(rec["confidence"], "")
    icon = {"RAISE": "⬆", "CALL": "➡", "FOLD": "✖"}.get(rec["action"], "")
    d = rec.get("details", {})
    print("\n" + "─" * 55)
    print(f"  ACTION  :  {icon} {rec['action']:<5}  {cm}")
    print(f"  REASON  :  {rec['reasoning']}")
    if d.get("pot_odds_required", 0) > 0:
        print(
            f"  POT ODDS:  need {d['pot_odds_required']}%  |  effective equity {d['effective_equity']}%"
        )
    if (
        isinstance(d, dict)
        and d.get("spr", 0) > 0
        and ctx.get("pot_size", 0) > 0
        and ctx.get("stack_size", 0) > 0
    ):
        spr = d["spr"]
        print(f"  SPR     :  {spr}  ({d.get('spr_advice', '')})")
    if d.get("board_texture") and d["board_texture"] != "N/A":
        print(f"  TEXTURE :  {d['board_texture']}")
    print("─" * 55)
    print(f"\n  Simulations: {equity['simulations']:,}")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def clean_num(val, default=0.0):
    if isinstance(val, (int, float)):
        return float(val)
    if not val:
        return default
    try:
        return float(re.sub(r"[^\d.]", "", str(val))) or default
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(description="Stake Texas Hold'em — Pro")
    parser.add_argument(
        "--opponents", type=int, default=2, help="Default opponents (default: 2)"
    )
    parser.add_argument(
        "--sims", type=int, default=2000, help="Monte Carlo sims (default: 2000)"
    )
    parser.add_argument("--overlay", action="store_true", help="Show floating HUD")
    parser.add_argument(
        "--manual", action="store_true", help="Force manual input, skip auto-detect"
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("GEMINI_API_KEY", "")
    )
    parser.add_argument("--image", type=str, help="Analyse existing image file")
    parser.add_argument("--test", action="store_true", help="Test screen capture only")
    args = parser.parse_args()

    print("\n♠  Stake Texas Hold'em Assistant — Pro Mode v1.3")
    print("   ─────────────────────────────────────────\n")

    if args.test:
        print("  [→] Testing screen capture...")
        try:
            path = capture_screen()
            print(f"  [✓] Saved: {path}")
            if sys.platform == "win32":
                os.startfile(path)
        except Exception as e:
            print(f"  [!] Failed: {e}")
        return

    overlay = None
    last = {
        "hole": "",
        "board": "",
        "opponents": args.opponents,
        "position": "MP",
        "pot": 0,
        "bet": 0,
        "stack": 0,
    }

    def run_once(force_manual=False):
        nonlocal last
        parent = overlay.root if overlay else None
        detected = None
        key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key and os.path.exists("api.txt"):
            try:
                with open("api.txt", "r") as f:
                    key = f.read().strip().rstrip(".")
                    if key:
                        print(f"  [✓] Loaded API key from api.txt ({key[:6]}...)")
            except Exception:
                pass

        if not force_manual and not args.manual and key:
            try:
                print("  [→] Starting screen scan...")
                if overlay:
                    overlay.hand_var.set("Scanning new hand...")
                    overlay.sv.set("Capturing screen...")
                    overlay.sl.config(fg="#ffeb3b")
                    overlay.root.update()

                img_path = capture_screen()
                if overlay:
                    overlay.sv.set("AI predicting cards...")
                    overlay.root.update()

                print("  [→] Sending image to AI...")
                detected = detect_with_vision(img_path, key)

                h = detected.get("hole_cards", [])
                b = detected.get("board_cards", [])

                if h:
                    print(f"  [✓] AI detected: {' '.join(h)} | {' '.join(b)}")
                else:
                    print("  [!] AI found no hole cards. Entering manual mode.")
                    last["hole"] = ""
                    last["board"] = " ".join(b)
                    detected = None
            except Exception as e:
                print(f"  [!] Auto-detect failed: {e}")
                detected = None

        if not detected:
            defaults = {
                "hole": last["hole"],
                "board": last["board"],
                "opponents": last["opponents"],
                "position": last["position"],
                "pot": last["pot"],
                "bet": last["bet"],
                "stack": last["stack"],
            }
            try:
                data = get_input(parent, defaults)
                detected = {
                    "hole_cards": data["hole_cards"],
                    "board_cards": data["board_cards"],
                    "pot_size": data["pot_size"],
                    "bet_facing": data["bet_facing"],
                    "stack_size": data["stack_size"],
                    "position": data["position"],
                    "opponents": data["opponents"],
                    "raw": data["raw"],
                    "confidence": "manual",
                }
            except RuntimeError:
                return

        last.update(
            {
                "hole": " ".join(detected.get("hole_cards", [])),
                "board": " ".join(detected.get("board_cards", [])),
                "opponents": int(
                    detected.get(
                        "num_opponents", detected.get("opponents", last["opponents"])
                    )
                ),
                "position": str(detected.get("position", last["position"]))
                .strip()
                .upper(),
                "pot": clean_num(detected.get("pot_size", 0)),
                "bet": clean_num(detected.get("bet_facing", 0)),
                "stack": clean_num(detected.get("stack_size", 0)),
            }
        )

        num_opps = last["opponents"]
        hole_raw = detected.get("hole_cards", [])
        board_raw = detected.get("board_cards", [])

        while True:
            hole, board, errors = validate_cards(
                {"hole_cards": hole_raw, "board_cards": board_raw}
            )
            if not errors:
                break
            emsg = "\n".join(f"• {e}" for e in errors)
            if parent:
                messagebox.showerror(
                    "Invalid Cards", f"Issues:\n{emsg}\nPlease fix.", parent=parent
                )
                try:
                    d2 = get_input(parent, last)
                    hole_raw = d2["hole_cards"]
                    board_raw = d2["board_cards"]
                except RuntimeError:
                    return
            else:
                print(f"  ⚠️  {emsg}\n  Re-enter cards.")
                d2 = get_input(None, last)
                hole_raw = d2["hole_cards"]
                board_raw = d2["board_cards"]

        final = {
            "hole_cards": hole,
            "board_cards": board,
            "confidence": detected.get("confidence", "high"),
        }
        ctx = {
            "pot_size": float(last["pot"]),
            "bet_facing": float(last["bet"]),
            "stack_size": float(last["stack"]),
            "position": last["position"],
            "num_opponents": num_opps,
        }

        print(f"  [→] Running Monte Carlo ({args.sims:,} sims, {num_opps} opp)...")
        equity = monte_carlo_holdem(
            hole, board, num_opponents=num_opps, simulations=args.sims
        )
        display_results(final, equity, num_opps, ctx)

        if overlay:
            equity["opponents"] = num_opps
            overlay.update(final, equity, ctx)

    if args.overlay:
        try:
            overlay = OverlayHUD()
            overlay.set_callback(run_once)
            print(
                "  [✓] Overlay HUD launched — click SCAN to auto-detect or manually update."
            )
            overlay.run()
        except Exception as ex:
            print(f"  [!] Overlay failed: {ex}")
            run_once()
    else:
        run_once()


if __name__ == "__main__":
    main()
