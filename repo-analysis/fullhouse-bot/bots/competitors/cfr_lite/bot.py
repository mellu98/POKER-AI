"""FullHouseCFRLite: Edge plus regret-matching postflop action abstraction."""

import eval7
import json
import os
import random

BOT_NAME = "FullHouseCFRLite"

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
DECK = [eval7.Card(r + s) for r in RANKS for s in "shdc"]
BIG_BLIND = 100

DEFAULT_CONFIG = {
    "max_seen_hands": 80,
    "equity_iterations": {
        "flop": 150,
        "turn": 180,
        "river": 220,
        "heads_up_bonus": 50,
        "multiway_penalty": 55,
        "multiway_floor": 80,
    },
    "river_unknown_large_bet_pot_fraction": 0.32,
    "passive_big_bet_extra_risk": 0.025,
}


def _load_config():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "strategy_config.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception:
        return DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    config["equity_iterations"] = dict(DEFAULT_CONFIG["equity_iterations"], **loaded.get("equity_iterations", {}))
    return config


CONFIG = _load_config()


def _load_preflop_equity():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        with open(os.path.join(data_dir, "preflop_equity.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


PREFLOP_EQUITY = _load_preflop_equity()

_SEEN_ACTIONS = {}
_OPPONENTS = {}
_SEEN_MATCH_ACTIONS = 0


def decide(state: dict) -> dict:
    """Return one legal poker action for the current game state."""
    try:
        _observe_actions(state)
        if state.get("street") == "preflop":
            return _decide_preflop(state)
        return _decide_postflop(state)
    except Exception:
        if state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}


def _observe_actions(state):
    global _SEEN_MATCH_ACTIONS
    match_log = state.get("match_action_log") or []
    if match_log:
        if _SEEN_MATCH_ACTIONS > len(match_log):
            _SEEN_MATCH_ACTIONS = 0
        for entry in match_log[_SEEN_MATCH_ACTIONS:]:
            bot_id = entry.get("bot_id")
            action = entry.get("action")
            if bot_id and action:
                _record_observed_action(bot_id, action)
        _SEEN_MATCH_ACTIONS = len(match_log)
        return

    hand_id = state.get("hand_id", "")
    log = state.get("action_log") or []
    old_len = _SEEN_ACTIONS.get(hand_id, 0)
    if old_len > len(log):
        old_len = 0

    players = state.get("players") or []
    my_seat = state.get("seat_to_act")
    seat_to_id = {}
    for player in players:
        seat_to_id[player.get("seat")] = player.get("bot_id")

    for entry in log[old_len:]:
        action = entry.get("action")
        if action in ("small_blind", "big_blind"):
            continue
        seat = entry.get("seat")
        if seat == my_seat:
            continue
        bot_id = seat_to_id.get(seat)
        if bot_id is None:
            continue
        _record_observed_action(bot_id, action)

    _SEEN_ACTIONS[hand_id] = len(log)
    if len(_SEEN_ACTIONS) > int(CONFIG.get("max_seen_hands", 80)):
        _SEEN_ACTIONS.clear()


def _record_observed_action(bot_id, action):
    stats = _OPPONENTS.get(bot_id)
    if stats is None:
        stats = {"actions": 0, "raises": 0, "calls": 0, "folds": 0, "checks": 0}
        _OPPONENTS[bot_id] = stats
    stats["actions"] += 1
    if action in ("raise", "all_in"):
        stats["raises"] += 1
    elif action == "call":
        stats["calls"] += 1
    elif action == "fold":
        stats["folds"] += 1
    elif action == "check":
        stats["checks"] += 1


def _decide_preflop(state):
    cards = state["your_cards"]
    score = _preflop_score(cards)
    pos = _position_name(state)
    owed = int(state.get("amount_owed", 0))
    pot = max(1, int(state.get("pot", 0)))
    current_bet = int(state.get("current_bet", 0))
    stack = int(state.get("your_stack", 0))
    my_bet = int(state.get("your_bet_this_street", 0))
    raise_count = _raise_count(state)
    limpers = _limper_count(state)
    active_opponents = max(1, _opponents_in_hand(state))
    equity = _preflop_equity(cards, active_opponents)
    profile = _table_profile()
    pressure = profile["raise_rate"]

    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if _is_premium(cards):
        if stack <= pot * 1.8 or raise_count >= 2:
            return {"action": "all_in"}
        target = max(current_bet * 3 + pot // 2, BIG_BLIND * (4 + limpers))
        return _raise_to(state, target)

    if raise_count == 0:
        threshold = _open_threshold(pos, active_opponents)
        open_bar = _open_equity_bar(pos, active_opponents)
        if pos == "bb" and state.get("can_check"):
            if score >= 78 or (score >= 60 and equity >= open_bar + 0.10) or (score >= 64 and equity >= open_bar and random.random() < 0.35):
                return _raise_to(state, BIG_BLIND * (4 + limpers))
            return {"action": "check"}

        if score >= threshold or (score >= threshold - 8 and equity >= open_bar + 0.03):
            size = 2.5 + min(limpers, 4) * 0.9
            if score >= 84 or equity >= open_bar + 0.08:
                size += 0.7
            if pos in ("sb", "bb"):
                size += 0.3
            return _raise_to(state, int(BIG_BLIND * size))

        call_price = owed / max(1, pot + owed)
        if owed > 0 and (
            _can_speculate(cards, score, call_price, active_opponents, pos)
            or (equity >= call_price + 0.09 and score >= 48)
        ):
            return {"action": "call"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    threebet_threshold = 85 + max(0, raise_count - 1) * 5
    continue_threshold = 67 + max(0, raise_count - 1) * 6 + max(0, active_opponents - 2) * 3
    price = owed / max(1, pot + owed)
    if price <= 0.18:
        continue_threshold -= 6
    if pos in ("button", "bb"):
        continue_threshold -= 3
    if profile["actions"] >= 12 and pressure > 0.55:
        continue_threshold -= 3
    if profile["actions"] >= 12 and profile["call_rate"] > 0.52 and active_opponents >= 2:
        continue_threshold += 2

    if score >= threebet_threshold:
        if stack <= pot * 2.2 or score >= 94:
            return {"action": "all_in"} if stack <= pot * 2.2 else _raise_to(state, current_bet * 3 + pot // 2)
        return _raise_to(state, current_bet * 3 + pot // 3)

    if raise_count == 1 and _threebet_bluff_candidate(cards, pos) and random.random() < 0.18:
        return _raise_to(state, current_bet * 3 + pot // 4)

    if score >= continue_threshold or (equity >= price + 0.13 and score >= 58 and raise_count <= 1):
        return {"action": "call"}
    if _can_set_mine_or_draw(cards, score, price, stack, current_bet, active_opponents):
        return {"action": "call"}
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def _decide_postflop(state):
    owed = int(state.get("amount_owed", 0))
    pot = max(1, int(state.get("pot", 0)))
    stack = int(state.get("your_stack", 0))
    n_opp = max(1, _opponents_in_hand(state))
    equity = _estimate_equity(state, n_opp)
    made = _made_rank(state)
    draw = _draw_score(state)
    wet = _board_wetness(state.get("community_cards") or [])
    price = owed / max(1, pot + owed)
    villain = _last_aggressor_profile(state)
    villain_pressure = villain["raise_rate"]
    cfr_action = _cfrlite_postflop_action(state, equity, made, draw, wet, n_opp, villain)
    if cfr_action is not None:
        return cfr_action

    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    multiway_tax = 0.035 * max(0, n_opp - 1)
    needed = price + 0.045 + multiway_tax
    if state.get("street") == "river":
        needed += 0.035
    if villain["actions"] >= 8 and villain_pressure > 0.58:
        needed -= 0.025
    elif villain["actions"] >= 6 and villain_pressure < 0.22:
        needed += 0.065
    elif villain["actions"] < 6:
        needed += 0.018
    if villain["call_rate"] > 0.52 and owed > pot * 0.45:
        needed += float(CONFIG.get("passive_big_bet_extra_risk", 0.025))

    if state.get("can_check"):
        value_bar = 0.58 + multiway_tax
        if equity >= value_bar or made >= 5:
            return _value_bet(state, pot, stack, equity, made, wet, n_opp)
        if state.get("street") != "river" and draw >= 2 and n_opp <= 1 and equity >= 0.38:
            if random.random() < 0.30:
                return _semi_bluff(state, pot, stack, wet)
        if state.get("street") != "river" and draw >= 3 and n_opp == 2 and equity >= 0.46:
            if random.random() < 0.18:
                return _semi_bluff(state, pot, stack, wet)
        if n_opp == 1 and wet <= 1 and 0.34 <= equity <= 0.52 and random.random() < 0.08:
            return _semi_bluff(state, pot, stack, wet)
        return {"action": "check"}

    if (
        state.get("street") == "river"
        and made <= 1
        and owed > pot * float(CONFIG.get("river_unknown_large_bet_pot_fraction", 0.32))
        and (villain["actions"] < 8 or villain_pressure < 0.42)
    ):
        return {"action": "fold"}

    if (equity >= 0.78 and made >= 3 and n_opp <= 2) or made >= 6:
        if stack <= pot * 1.15 and (made >= 5 or equity >= 0.86):
            return {"action": "all_in"}
        if random.random() < (0.62 if n_opp <= 1 else 0.42):
            return _raise_to(state, int(state.get("current_bet", 0)) + int(pot * 0.80))

    draw_bonus = 0.045 * draw if state.get("street") != "river" else 0
    if equity + draw_bonus >= needed:
        if draw >= 3 and n_opp <= 2 and random.random() < 0.18 and stack > pot:
            return _raise_to(state, int(state.get("current_bet", 0)) + int(pot * 0.65))
        return {"action": "call"}

    if state.get("street") != "river" and draw >= 3 and price <= 0.20 and owed <= stack * 0.18:
        return {"action": "call"}
    return {"action": "fold"}


def _raise_to(state, target):
    stack = int(state.get("your_stack", 0))
    my_bet = int(state.get("your_bet_this_street", 0))
    all_in_to = stack + my_bet
    min_to = int(state.get("min_raise_to", 0))
    target = int(max(min_to, target))
    if all_in_to <= min_to or target >= all_in_to:
        return {"action": "all_in"}
    return {"action": "raise", "amount": max(min_to, min(target, all_in_to - 1))}


def _value_bet(state, pot, stack, equity, made, wet, n_opp):
    frac = 0.54
    if made >= 5 or equity >= 0.75:
        frac = 0.72
    if wet >= 3:
        frac += 0.12
    if n_opp >= 3:
        frac += 0.08
    target = int(pot * min(0.95, frac))
    if stack <= pot * 1.15 and (made >= 5 or (n_opp <= 1 and equity >= 0.86)):
        return {"action": "all_in"}
    return _raise_to(state, target)


def _semi_bluff(state, pot, stack, wet):
    frac = 0.43 + 0.08 * min(3, wet)
    if stack <= pot * 0.9:
        return {"action": "all_in"}
    return _raise_to(state, int(pot * frac))


def _cfrlite_postflop_action(state, equity, made, draw, wet, opponents, villain):
    """One-step depth-limited action abstraction with regret-matching.

    This ports the useful idea from the Pokerbots CFR repos, not their game
    tree: compare fold/call-or-check/pot-pressure terminal EVs, then sample
    from positive regret. It only fires in medium-confidence spots and lets the
    base Edge logic handle obvious value, trash, and emergency cases.
    """
    pot = max(1, int(state.get("pot", 0)))
    owed = int(state.get("amount_owed", 0))
    stack = int(state.get("your_stack", 0))
    if stack <= 0 or opponents <= 0:
        return None

    street = state.get("street")
    if street == "river" and made <= 1:
        return None
    if opponents >= 3 and made < 4 and draw < 3:
        return None

    can_check = bool(state.get("can_check"))
    price = owed / max(1, pot + owed)
    # Skip cases where base thresholds are clearer.
    if not can_check and (equity < price - 0.02 or equity > price + 0.30):
        return None
    if can_check and (equity < 0.30 or equity > 0.72 or made >= 5):
        return None

    realization = 0.86 - 0.08 * max(0, opponents - 1)
    if draw >= 3 and street != "river":
        realization += 0.08
    if made >= 2:
        realization += 0.06
    if wet >= 4 and made < 2:
        realization -= 0.05
    realization = max(0.55, min(1.05, realization))

    fold_rate = villain.get("fold_rate", 0.28)
    raise_rate = villain.get("raise_rate", 0.30)
    call_rate = villain.get("call_rate", 0.34)
    actions = villain.get("actions", 0)
    if actions < 8:
        fold_rate = 0.30
        raise_rate = 0.30
        call_rate = 0.34

    pressure_fold = fold_rate + 0.08 * max(0, opponents - 1)
    if raise_rate > 0.55:
        pressure_fold -= 0.05
    if call_rate > 0.52:
        pressure_fold -= 0.08
    if wet >= 4:
        pressure_fold -= 0.04
    pressure_fold = max(0.05, min(0.62, pressure_fold))

    bet_to = _abstract_bet_to(state, pot, stack, made, draw, wet, opponents)
    if bet_to is None:
        return None
    my_bet = int(state.get("your_bet_this_street", 0))
    invest = max(0, bet_to - my_bet)
    if invest <= 0 or invest > stack + my_bet:
        return None

    check_call_ev = equity * realization * (pot + owed) - owed
    fold_ev = 0.0 if not can_check else check_call_ev - 0.01 * pot
    # Villain folds: win current pot. Villain calls: showdown for bigger pot.
    raise_showdown_pot = pot + owed + invest
    raise_ev = pressure_fold * pot + (1.0 - pressure_fold) * (
        equity * realization * raise_showdown_pot - invest
    )
    if opponents >= 2:
        raise_ev -= 0.04 * pot
    if street == "river" and draw > 0 and made < 2:
        raise_ev -= 0.08 * pot

    if can_check:
        values = [check_call_ev, raise_ev]
        actions = ["check", "raise"]
    else:
        values = [fold_ev, check_call_ev, raise_ev]
        actions = ["fold", "call", "raise"]

    baseline = sum(values) / len(values)
    regrets = [max(0.0, value - baseline) for value in values]
    if max(values) < baseline + 0.025 * pot:
        return None
    # Keep this from turning into a pure strategy; CFR-derived strategies mix.
    for i in range(len(regrets)):
        regrets[i] += 0.015 * pot
    total = sum(regrets)
    if total <= 0:
        return None
    pick = random.random() * total
    acc = 0.0
    chosen = actions[0]
    for action, weight in zip(actions, regrets):
        acc += weight
        if pick <= acc:
            chosen = action
            break

    if chosen == "raise":
        return _raise_to(state, bet_to)
    if chosen == "call":
        return {"action": "call"}
    return {"action": "check"} if can_check else {"action": "fold"}


def _abstract_bet_to(state, pot, stack, made, draw, wet, opponents):
    min_to = int(state.get("min_raise_to", 0))
    current_bet = int(state.get("current_bet", 0))
    my_bet = int(state.get("your_bet_this_street", 0))
    all_in_to = stack + my_bet
    if all_in_to <= min_to:
        return all_in_to
    frac = 0.58
    if made >= 3 or draw >= 3:
        frac += 0.12
    if wet >= 4:
        frac += 0.08
    if opponents >= 2:
        frac += 0.08
    target = current_bet + int(pot * min(0.92, frac))
    if all_in_to <= pot * 1.10 and (made >= 4 or draw >= 3):
        return all_in_to
    return max(min_to, min(target, all_in_to - 1))


def _estimate_equity(state, opponents):
    hero = [_card(c) for c in state["your_cards"]]
    board = [_card(c) for c in (state.get("community_cards") or [])]
    known = set(state["your_cards"] + (state.get("community_cards") or []))
    deck = [c for c in DECK if str(c) not in known]
    missing_board = 5 - len(board)
    need = opponents * 2 + missing_board
    if need <= 0 or len(deck) < need:
        return 0.5

    street = state.get("street")
    iteration_cfg = CONFIG.get("equity_iterations", {})
    if street == "flop":
        iterations = int(iteration_cfg.get("flop", 150))
    elif street == "turn":
        iterations = int(iteration_cfg.get("turn", 180))
    else:
        iterations = int(iteration_cfg.get("river", 220))
    if opponents >= 3:
        iterations = max(
            int(iteration_cfg.get("multiway_floor", 80)),
            iterations - int(iteration_cfg.get("multiway_penalty", 55)),
        )
    if opponents == 1:
        iterations += int(iteration_cfg.get("heads_up_bonus", 50))

    wins = 0.0
    for _ in range(iterations):
        sample = random.sample(deck, need)
        runout = sample[opponents * 2:]
        full_board = board + runout
        hero_score = eval7.evaluate(hero + full_board)
        tied = 1
        ahead = True
        for i in range(opponents):
            opp_score = eval7.evaluate(sample[i * 2:i * 2 + 2] + full_board)
            if opp_score > hero_score:
                ahead = False
                break
            if opp_score == hero_score:
                tied += 1
        if ahead:
            wins += 1.0 / tied
    return wins / iterations


def _made_rank(state):
    cards = [_card(c) for c in state["your_cards"] + (state.get("community_cards") or [])]
    if len(cards) < 5:
        return 0
    hand_name = str(eval7.handtype(eval7.evaluate(cards))).lower()
    if "straight flush" in hand_name:
        return 8
    if "four" in hand_name:
        return 7
    if "full house" in hand_name:
        return 6
    if "flush" in hand_name:
        return 5
    if "straight" in hand_name:
        return 4
    if "three" in hand_name:
        return 3
    if "two pair" in hand_name:
        return 2
    if "pair" in hand_name:
        return 1
    return 0


def _draw_score(state):
    if state.get("street") == "river":
        return 0
    cards = state["your_cards"] + (state.get("community_cards") or [])
    suits = {}
    for c in cards:
        suits[c[1]] = suits.get(c[1], 0) + 1
    score = 0
    for suit, count in suits.items():
        if count >= 4 and (state["your_cards"][0][1] == suit or state["your_cards"][1][1] == suit):
            score += 2
    ranks = sorted(set(_rank_value(c[0]) for c in cards))
    if 14 in ranks:
        ranks = [1] + ranks
    for start in range(1, 11):
        have = sum(1 for r in range(start, start + 5) if r in ranks)
        if have >= 4:
            score += 2
            break
    return min(score, 4)


def _board_wetness(board):
    if len(board) < 3:
        return 0
    suits = {}
    ranks = []
    for c in board:
        suits[c[1]] = suits.get(c[1], 0) + 1
        ranks.append(_rank_value(c[0]))
    wet = 0
    if max(suits.values()) >= 3:
        wet += 2
    elif max(suits.values()) == 2:
        wet += 1
    unique = sorted(set(ranks + ([1] if 14 in ranks else [])))
    for start in range(1, 11):
        have = sum(1 for r in range(start, start + 5) if r in unique)
        if have >= 3:
            wet += 1
        if have >= 4:
            wet += 1
            break
    if len(set(ranks)) < len(ranks):
        wet += 1
    return min(wet, 5)


def _preflop_score(cards):
    r1, r2 = _rank_value(cards[0][0]), _rank_value(cards[1][0])
    hi, lo = max(r1, r2), min(r1, r2)
    suited = cards[0][1] == cards[1][1]
    gap = hi - lo
    if hi == lo:
        return min(100, 47 + hi * 3.6)
    score = hi * 4.0 + lo * 2.0
    if suited:
        score += 5.5
    if gap == 1:
        score += 4.0
    elif gap == 2:
        score += 2.0
    elif gap >= 5:
        score -= 4.0
    if hi == 14:
        score += 5.5
    if hi >= 11 and lo >= 10:
        score += 5.0
    if lo <= 5 and hi < 10:
        score -= 3.0
    return max(0, min(100, score))


def _is_premium(cards):
    key = _hand_key(cards)
    return key in ("AA", "KK", "QQ", "AKs", "AKo")


def _threebet_bluff_candidate(cards, pos):
    key = _hand_key(cards)
    if pos not in ("button", "sb", "bb", "cutoff"):
        return False
    return key in ("A5s", "A4s", "A3s", "K9s", "KTs", "QTs", "JTs", "T9s")


def _can_speculate(cards, score, price, opponents, pos):
    key = _hand_key(cards)
    if price > 0.24:
        return False
    if key[0] == key[1] and price <= 0.18:
        return True
    if score >= 50 and pos in ("button", "bb", "sb", "cutoff"):
        return True
    if opponents >= 3 and _is_suited_connector(cards) and price <= 0.20:
        return True
    return False


def _can_set_mine_or_draw(cards, score, price, stack, current_bet, opponents):
    key = _hand_key(cards)
    if price > 0.22:
        return False
    if key[0] == key[1] and stack >= max(BIG_BLIND * 20, current_bet * 10):
        return True
    if opponents >= 2 and _is_suited_connector(cards) and score >= 45:
        return True
    return False


def _is_suited_connector(cards):
    if cards[0][1] != cards[1][1]:
        return False
    gap = abs(_rank_value(cards[0][0]) - _rank_value(cards[1][0]))
    return gap <= 2 and min(_rank_value(cards[0][0]), _rank_value(cards[1][0])) >= 5


def _open_threshold(pos, opponents):
    thresholds = {
        "early": 72,
        "middle": 66,
        "cutoff": 58,
        "button": 49,
        "sb": 53,
        "bb": 52,
    }
    base = thresholds.get(pos, 65)
    if opponents <= 2 and pos in ("button", "sb"):
        base -= 5
    return base


def _open_equity_bar(pos, opponents):
    base = 0.55 - 0.055 * max(0, opponents - 1)
    if pos in ("button", "cutoff"):
        base -= 0.045
    elif pos in ("sb", "bb"):
        base -= 0.025
    elif pos == "early":
        base += 0.025
    return max(0.255, min(0.57, base))


def _preflop_equity(cards, opponents):
    table = PREFLOP_EQUITY.get(_hand_key(cards))
    if not table:
        return _preflop_score(cards) / 100.0
    key = str(max(1, min(6, int(opponents))))
    return float(table.get(key, _preflop_score(cards) / 100.0))


def _position_name(state):
    players = state.get("players") or []
    n = max(2, len(players))
    seat = int(state.get("seat_to_act", 0))
    sb, bb = _blind_seats(state)
    if n == 2:
        if seat == sb:
            return "button" if state.get("street") != "preflop" else "sb"
        return "bb"
    if bb is None:
        return "middle"
    dist = (seat - bb) % n
    if dist == 0:
        return "bb"
    if dist == n - 1:
        return "sb"
    if dist == n - 2:
        return "button"
    if dist == n - 3:
        return "cutoff"
    if dist <= 1:
        return "early"
    return "middle"


def _blind_seats(state):
    sb = None
    bb = None
    for entry in state.get("action_log") or []:
        if entry.get("action") == "small_blind":
            sb = entry.get("seat")
        elif entry.get("action") == "big_blind":
            bb = entry.get("seat")
        if sb is not None and bb is not None:
            return sb, bb
    return sb, bb


def _raise_count(state):
    count = 0
    for entry in state.get("action_log") or []:
        if entry.get("action") in ("raise", "all_in"):
            count += 1
    return count


def _limper_count(state):
    count = 0
    for entry in state.get("action_log") or []:
        action = entry.get("action")
        if action in ("raise", "all_in"):
            break
        if action == "call":
            count += 1
    return count


def _opponents_in_hand(state):
    my_seat = state.get("seat_to_act")
    count = 0
    for player in state.get("players") or []:
        if player.get("seat") == my_seat:
            continue
        if not player.get("is_folded") and player.get("state") != "busted":
            count += 1
    return count


def _table_profile():
    raises = 0
    calls = 0
    folds = 0
    passive = 0
    for stats in _OPPONENTS.values():
        raises += stats.get("raises", 0)
        calls += stats.get("calls", 0)
        folds += stats.get("folds", 0)
        passive += stats.get("calls", 0) + stats.get("checks", 0)
    actions = raises + passive + folds
    return {
        "actions": actions,
        "raise_rate": raises / max(1, raises + passive),
        "call_rate": calls / max(1, actions),
        "fold_rate": folds / max(1, actions),
    }


def _last_aggressor_profile(state):
    players = state.get("players") or []
    seat_to_id = {}
    for player in players:
        seat_to_id[player.get("seat")] = player.get("bot_id")
    for entry in reversed(state.get("action_log") or []):
        if entry.get("action") in ("raise", "all_in"):
            bot_id = seat_to_id.get(entry.get("seat"))
            stats = _OPPONENTS.get(bot_id, {})
            actions = stats.get("actions", 0)
            calls = stats.get("calls", 0)
            return {
                "actions": actions,
                "raise_rate": stats.get("raises", 0) / max(1, actions),
                "call_rate": calls / max(1, actions),
                "fold_rate": stats.get("folds", 0) / max(1, actions),
            }
    return _table_profile()


def _hand_key(cards):
    r1, r2 = cards[0][0], cards[1][0]
    v1, v2 = _rank_value(r1), _rank_value(r2)
    if v1 == v2:
        return r1 + r2
    if v1 > v2:
        hi, lo = r1, r2
    else:
        hi, lo = r2, r1
    return hi + lo + ("s" if cards[0][1] == cards[1][1] else "o")


def _rank_value(rank):
    return RANK_VALUE.get(rank, 0)


def _card(text):
    return eval7.Card(text)
