"""Pure-Python 6-max NLHE game engine for Deep CFR traversal.

Mirrors the Numba kernel game logic but callable from Python (no JIT).
State is a numpy int32[59] array.
"""

from __future__ import annotations

import numpy as np

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000
N_ACTIONS = 5
SPR_ALLIN_THRESHOLD = 4

FOLD, CHECK_CALL, BET_50, BET_POT, ALL_IN = range(5)
BET_FRACS = (0.0, 0.0, 0.50, 1.0, 0.0)

S_STACKS = 0
S_BETS = 6
S_INVESTED = 12
S_FOLDED = 18
S_ALLIN = 24
S_PENDING = 30
S_POT = 36
S_CURBET = 37
S_NRAISES = 38
S_LASTRAISER = 39
S_TOACT = 40
S_STREET = 41
S_DECKIDX = 42
S_DEALER = 43
S_NPLAYERS = 44
S_TERMINAL = 45
S_PAYOFFS = 46
S_LAST_FULL_RAISE = 52
S_INITIAL_STACKS = 53
STATE_SIZE = 59


def new_hand(
    n_players: int,
    dealer: int,
    rng: np.random.Generator,
    starting_stacks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deck = rng.permutation(52).astype(np.int8)
    hole_cards = np.empty((n_players, 2), dtype=np.int8)
    for i in range(n_players):
        hole_cards[i, 0] = deck[2 * i]
        hole_cards[i, 1] = deck[2 * i + 1]

    state = np.zeros(STATE_SIZE, dtype=np.int32)
    state[S_NPLAYERS] = n_players
    state[S_DEALER] = dealer
    for i in range(n_players):
        stack = (
            STARTING_STACK
            if starting_stacks is None
            else max(0, int(starting_stacks[i]))
        )
        state[S_STACKS + i] = stack
        state[S_INITIAL_STACKS + i] = stack
    state[S_DECKIDX] = 2 * n_players
    state[S_LASTRAISER] = -1
    state[S_LAST_FULL_RAISE] = BIG_BLIND

    is_hu = n_players == 2
    sb = dealer if is_hu else (dealer + 1) % n_players
    bb = (dealer + 1) % n_players if is_hu else (dealer + 2) % n_players

    _put_in(state, sb, min(SMALL_BLIND, state[S_STACKS + sb]))
    _put_in(state, bb, min(BIG_BLIND, state[S_STACKS + bb]))
    state[S_CURBET] = max(state[S_BETS + sb], state[S_BETS + bb])

    state[S_TOACT] = sb if is_hu else (bb + 1) % n_players
    for i in range(n_players):
        if state[S_FOLDED + i] == 0 and state[S_ALLIN + i] == 0:
            state[S_PENDING + i] = 1

    return state, deck, hole_cards


def _put_in(state: np.ndarray, seat: int, amount: int) -> None:
    amount = max(0, min(amount, state[S_STACKS + seat]))
    state[S_STACKS + seat] -= amount
    state[S_BETS + seat] += amount
    state[S_INVESTED + seat] += amount
    state[S_POT] += amount
    if state[S_STACKS + seat] == 0:
        state[S_ALLIN + seat] = 1


def legal_actions(state: np.ndarray, seat: int) -> np.ndarray:
    """Returns bool[5] legal action mask."""
    legal = np.zeros(N_ACTIONS, dtype=np.bool_)
    pot = state[S_POT]
    my_bet = state[S_BETS + seat]
    cur_bet = state[S_CURBET]
    to_call = max(0, cur_bet - my_bet)
    my_stack = state[S_STACKS + seat]
    n_players = state[S_NPLAYERS]
    last_full_raise = max(int(state[S_LAST_FULL_RAISE]), BIG_BLIND)

    opp_stack = 0
    for i in range(n_players):
        if i != seat and state[S_FOLDED + i] == 0:
            opp_stack = max(opp_stack, state[S_STACKS + i])

    if to_call > 0:
        legal[FOLD] = True
    legal[CHECK_CALL] = True

    can_raise = my_stack > to_call
    if can_raise:
        pot_after_call = pot + to_call
        bet_room = my_stack - to_call
        bet_50 = round(0.50 * pot_after_call)
        if bet_50 >= last_full_raise and bet_room > bet_50:
            legal[BET_50] = True
        if pot_after_call >= last_full_raise and bet_room > pot_after_call:
            legal[BET_POT] = True
        eff_stack = min(my_stack, opp_stack) if opp_stack > 0 else my_stack
        spr = eff_stack / max(pot, 1)
        if spr < SPR_ALLIN_THRESHOLD:
            legal[ALL_IN] = True

    return legal


def _bet_chip_amount(action_id: int, pot: int, to_call: int, my_stack: int) -> int:
    if action_id == FOLD:
        return 0
    if action_id == CHECK_CALL:
        return min(to_call, my_stack)
    if action_id == ALL_IN:
        return my_stack
    frac = BET_FRACS[action_id]
    pot_after_call = pot + to_call
    bet = round(frac * pot_after_call)
    return min(to_call + bet, my_stack)


def step(
    state: np.ndarray, deck: np.ndarray, hole_cards: np.ndarray, action_id: int
) -> None:
    """Apply action for current actor. Mutates state in place."""
    seat = state[S_TOACT]
    n = state[S_NPLAYERS]
    pot = state[S_POT]
    my_bet = state[S_BETS + seat]
    cur_bet = state[S_CURBET]
    to_call = max(0, cur_bet - my_bet)
    my_stack = state[S_STACKS + seat]

    state[S_PENDING + seat] = 0

    if action_id == FOLD:
        state[S_FOLDED + seat] = 1
    elif action_id == CHECK_CALL:
        _put_in(state, seat, min(to_call, my_stack))
    else:
        prev_bet = state[S_CURBET]
        chips = _bet_chip_amount(action_id, pot, to_call, my_stack)
        _put_in(state, seat, chips)
        new_bet = state[S_BETS + seat]
        state[S_CURBET] = max(state[S_CURBET], new_bet)
        raise_size = max(0, state[S_CURBET] - prev_bet)
        if raise_size >= max(state[S_LAST_FULL_RAISE], BIG_BLIND):
            state[S_LAST_FULL_RAISE] = raise_size
            state[S_NRAISES] += 1
            state[S_LASTRAISER] = seat
            for i in range(n):
                if i != seat and state[S_FOLDED + i] == 0 and state[S_ALLIN + i] == 0:
                    state[S_PENDING + i] = 1

    # Check uncontested
    in_hand = 0
    winner = -1
    for i in range(n):
        if state[S_FOLDED + i] == 0:
            in_hand += 1
            winner = i
    if in_hand == 1:
        _resolve_uncontested(state, winner)
        return

    # Advance actor or street
    nxt = _next_actor(state, seat)
    if nxt < 0:
        _advance_street(state, deck, hole_cards)
    else:
        state[S_TOACT] = nxt


def _next_actor(state: np.ndarray, last: int) -> int:
    n = state[S_NPLAYERS]
    for offset in range(1, n + 1):
        cand = (last + offset) % n
        if (
            state[S_PENDING + cand] == 1
            and state[S_FOLDED + cand] == 0
            and state[S_ALLIN + cand] == 0
        ):
            return cand
    return -1


def _resolve_uncontested(state: np.ndarray, winner: int) -> None:
    n = state[S_NPLAYERS]
    state[S_STACKS + winner] += state[S_POT]
    for i in range(n):
        state[S_PAYOFFS + i] = state[S_STACKS + i] - state[S_INITIAL_STACKS + i]
    state[S_POT] = 0
    state[S_TERMINAL] = 1


def _resolve_showdown(
    state: np.ndarray, deck: np.ndarray, hole_cards: np.ndarray
) -> None:
    """Simplified showdown using eval7."""
    import eval7

    n = state[S_NPLAYERS]
    board_idx = 2 * n
    _CARDS = [
        eval7.Card(s)
        for s in [
            "2c",
            "3c",
            "4c",
            "5c",
            "6c",
            "7c",
            "8c",
            "9c",
            "Tc",
            "Jc",
            "Qc",
            "Kc",
            "Ac",
            "2d",
            "3d",
            "4d",
            "5d",
            "6d",
            "7d",
            "8d",
            "9d",
            "Td",
            "Jd",
            "Qd",
            "Kd",
            "Ad",
            "2h",
            "3h",
            "4h",
            "5h",
            "6h",
            "7h",
            "8h",
            "9h",
            "Th",
            "Jh",
            "Qh",
            "Kh",
            "Ah",
            "2s",
            "3s",
            "4s",
            "5s",
            "6s",
            "7s",
            "8s",
            "9s",
            "Ts",
            "Js",
            "Qs",
            "Ks",
            "As",
        ]
    ]

    board = [_CARDS[deck[board_idx + b]] for b in range(5)]
    scores = np.zeros(n, dtype=np.int32)
    in_hand = []
    for i in range(n):
        if state[S_FOLDED + i] == 0:
            in_hand.append(i)
            hand = [_CARDS[hole_cards[i, 0]], _CARDS[hole_cards[i, 1]]] + board
            scores[i] = eval7.evaluate(hand)

    if len(in_hand) <= 1:
        if in_hand:
            _resolve_uncontested(state, in_hand[0])
        return

    # Side-pot resolution
    levels = sorted(
        set(state[S_INVESTED + i] for i in range(n) if state[S_INVESTED + i] > 0)
    )
    awarded = np.zeros(n, dtype=np.int32)
    prev = 0
    for lvl in levels:
        per = lvl - prev
        contributors = [i for i in range(n) if state[S_INVESTED + i] >= lvl]
        amount = per * len(contributors)
        eligible = [i for i in in_hand if state[S_INVESTED + i] >= lvl]
        if not eligible:
            prev = lvl
            continue
        best = max(scores[i] for i in eligible)
        winners = [i for i in eligible if scores[i] == best]
        split = amount // len(winners)
        rem = amount % len(winners)
        for idx, w in enumerate(winners):
            awarded[w] += split + (rem if idx == 0 else 0)
        prev = lvl

    for i in range(n):
        state[S_STACKS + i] += awarded[i]
        state[S_PAYOFFS + i] = state[S_STACKS + i] - state[S_INITIAL_STACKS + i]
    state[S_POT] = 0
    state[S_TERMINAL] = 1


def _advance_street(
    state: np.ndarray, deck: np.ndarray, hole_cards: np.ndarray
) -> None:
    n = state[S_NPLAYERS]
    for i in range(n):
        state[S_BETS + i] = 0
    state[S_CURBET] = 0
    state[S_NRAISES] = 0
    state[S_LASTRAISER] = -1
    state[S_LAST_FULL_RAISE] = BIG_BLIND

    street = state[S_STREET]
    if street == 3:
        _resolve_showdown(state, deck, hole_cards)
        return

    new_street = street + 1
    board_idx = 2 * n
    if new_street == 1:
        state[S_DECKIDX] = board_idx + 3
    elif new_street == 2:
        state[S_DECKIDX] = board_idx + 4
    elif new_street == 3:
        state[S_DECKIDX] = board_idx + 5
    state[S_STREET] = new_street

    actors = sum(
        1
        for i in range(n)
        if state[S_FOLDED + i] == 0
        and state[S_ALLIN + i] == 0
        and state[S_STACKS + i] > 0
    )
    if actors == 0:
        while state[S_STREET] < 3:
            state[S_STREET] += 1
        _resolve_showdown(state, deck, hole_cards)
        return

    dealer = state[S_DEALER]
    if n == 2:
        bb_seat = (dealer + 1) % 2
        sb_seat = dealer
        state[S_TOACT] = (
            bb_seat
            if (state[S_FOLDED + bb_seat] == 0 and state[S_ALLIN + bb_seat] == 0)
            else sb_seat
        )
    else:
        for offset in range(1, n + 1):
            cand = (dealer + offset) % n
            if state[S_FOLDED + cand] == 0 and state[S_ALLIN + cand] == 0:
                state[S_TOACT] = cand
                break

    for i in range(n):
        if state[S_FOLDED + i] == 0 and state[S_ALLIN + i] == 0:
            state[S_PENDING + i] = 1
        else:
            state[S_PENDING + i] = 0
