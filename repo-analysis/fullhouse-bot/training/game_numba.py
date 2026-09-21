"""Numba-JIT compiled 6-max NLHE game engine for Deep CFR traversal.

State is np.int32[59], cards are np.int8[52] deck.
All functions are pure numeric - no Python objects, no dicts, no classes.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from ._hand_eval_lut import evaluate_lut

# Constants (module-level, accessible from other Numba modules)
SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000
N_ACTIONS = 5
SPR_ALLIN_THRESHOLD = 4

FOLD = 0
CHECK_CALL = 1
BET_50 = 2
BET_POT = 3
ALL_IN = 4

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

# Bet fractions indexed by action id
_BET_FRACS = np.array([0.0, 0.0, 0.50, 1.0, 0.0], dtype=np.float64)

# Board cards dealt per street: preflop=0, flop=3, turn=4, river=5
_BOARD_COUNTS = np.array([0, 3, 4, 5], dtype=np.int32)


# Card conversion
@nb.njit(cache=True)
def _game_to_eval_card(card):
    """Convert game encoding (suit*13+rank) to evaluator encoding (rank*4+suit)."""
    rank = card % nb.int8(13)
    suit = card // nb.int8(13)
    return nb.int8(rank * nb.int8(4) + suit)


# Deck
@nb.njit(cache=True)
def _shuffle_deck(deck):
    """Fisher-Yates shuffle in-place using Numba-compatible RNG."""
    n = len(deck)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp


# Hand setup
@nb.njit(cache=True)
def new_hand(n_players, dealer):
    """Create and deal a new hand. Returns (state, deck, hole_cards)."""
    starting_stacks = np.empty(n_players, dtype=np.int32)
    for i in range(n_players):
        starting_stacks[i] = STARTING_STACK
    return new_hand_with_stacks(n_players, dealer, starting_stacks)


@nb.njit(cache=True)
def new_hand_with_stacks(n_players, dealer, starting_stacks):
    """Create and deal a new hand with explicit per-seat starting stacks."""
    # Build and shuffle deck
    deck = np.empty(52, dtype=np.int8)
    for i in range(52):
        deck[i] = nb.int8(i)
    _shuffle_deck(deck)

    # Deal hole cards
    hole_cards = np.empty((n_players, 2), dtype=np.int8)
    for i in range(n_players):
        hole_cards[i, 0] = deck[2 * i]
        hole_cards[i, 1] = deck[2 * i + 1]

    # Initialize state
    state = np.zeros(STATE_SIZE, dtype=np.int32)
    state[S_NPLAYERS] = n_players
    state[S_DEALER] = dealer
    for i in range(n_players):
        stack = starting_stacks[i]
        if stack < 0:
            stack = np.int32(0)
        state[S_STACKS + i] = stack
        state[S_INITIAL_STACKS + i] = stack
    state[S_DECKIDX] = 2 * n_players
    state[S_LASTRAISER] = -1
    state[S_LAST_FULL_RAISE] = BIG_BLIND

    # Post blinds: heads-up special case
    is_hu = n_players == 2
    if is_hu:
        sb = dealer
        bb = (dealer + 1) % 2
    else:
        sb = (dealer + 1) % n_players
        bb = (dealer + 2) % n_players

    sb_amount = SMALL_BLIND
    if state[S_STACKS + sb] < sb_amount:
        sb_amount = state[S_STACKS + sb]
    _put_in(state, sb, sb_amount)

    bb_amount = BIG_BLIND
    if state[S_STACKS + bb] < bb_amount:
        bb_amount = state[S_STACKS + bb]
    _put_in(state, bb, bb_amount)

    if state[S_BETS + sb] > state[S_BETS + bb]:
        state[S_CURBET] = state[S_BETS + sb]
    else:
        state[S_CURBET] = state[S_BETS + bb]

    # First to act: heads-up = SB, 6-max = left of BB
    if is_hu:
        state[S_TOACT] = sb
    else:
        state[S_TOACT] = (bb + 1) % n_players

    # Mark all non-folded, non-allin players as pending
    for i in range(n_players):
        if state[S_FOLDED + i] == 0 and state[S_ALLIN + i] == 0:
            state[S_PENDING + i] = 1

    return state, deck, hole_cards


# Chip movement
@nb.njit(cache=True)
def _put_in(state, seat, amount):
    """Transfer chips from stack to pot, mark all-in if stack hits zero."""
    if amount < 0:
        amount = 0
    stack = state[S_STACKS + seat]
    if amount > stack:
        amount = stack
    state[S_STACKS + seat] -= amount
    state[S_BETS + seat] += amount
    state[S_INVESTED + seat] += amount
    state[S_POT] += amount
    if state[S_STACKS + seat] == 0:
        state[S_ALLIN + seat] = 1


# Legal actions
@nb.njit(cache=True)
def legal_actions(state, seat):
    """Compute bool[5] legal action mask."""
    legal = np.zeros(N_ACTIONS, dtype=nb.boolean)
    pot = state[S_POT]
    my_bet = state[S_BETS + seat]
    cur_bet = state[S_CURBET]
    to_call = cur_bet - my_bet
    if to_call < 0:
        to_call = 0
    my_stack = state[S_STACKS + seat]
    n_players = state[S_NPLAYERS]
    last_full_raise = state[S_LAST_FULL_RAISE]
    if last_full_raise < BIG_BLIND:
        last_full_raise = BIG_BLIND

    # Max opponent stack (among non-folded)
    opp_stack = np.int32(0)
    for i in range(n_players):
        if i != seat and state[S_FOLDED + i] == 0:
            if state[S_STACKS + i] > opp_stack:
                opp_stack = state[S_STACKS + i]

    if to_call > 0:
        legal[FOLD] = True
    legal[CHECK_CALL] = True

    can_raise = my_stack > to_call
    if can_raise:
        pot_after_call = pot + to_call
        bet_room = my_stack - to_call
        # Use Python-compatible rounding: round(x) = int(x + 0.5) for positive x
        bet_50 = np.int32(np.float64(0.50) * np.float64(pot_after_call) + 0.5)
        if bet_50 >= last_full_raise and bet_room > bet_50:
            legal[BET_50] = True
        if pot_after_call >= last_full_raise and bet_room > pot_after_call:
            legal[BET_POT] = True
        eff_stack = my_stack
        if opp_stack > 0 and opp_stack < eff_stack:
            eff_stack = opp_stack
        spr = np.float64(eff_stack) / np.float64(max(pot, np.int32(1)))
        if spr < np.float64(SPR_ALLIN_THRESHOLD):
            legal[ALL_IN] = True

    return legal


# Bet sizing
@nb.njit(cache=True)
def _bet_chip_amount(action_id, pot, to_call, my_stack):
    """Compute chip amount for a given action."""
    if action_id == FOLD:
        return np.int32(0)
    if action_id == CHECK_CALL:
        if to_call < my_stack:
            return np.int32(to_call)
        return np.int32(my_stack)
    if action_id == ALL_IN:
        return np.int32(my_stack)
    frac = _BET_FRACS[action_id]
    pot_after_call = np.float64(pot + to_call)
    bet = np.int32(frac * pot_after_call + 0.5)  # round
    total = to_call + bet
    if total < my_stack:
        return np.int32(total)
    return np.int32(my_stack)


# Next actor
@nb.njit(cache=True)
def _next_actor(state, last):
    """Find next pending, non-folded, non-allin player after 'last'."""
    n = state[S_NPLAYERS]
    for offset in range(1, n + 1):
        cand = (last + offset) % n
        if (
            state[S_PENDING + cand] == 1
            and state[S_FOLDED + cand] == 0
            and state[S_ALLIN + cand] == 0
        ):
            return np.int32(cand)
    return np.int32(-1)


# Resolution
@nb.njit(cache=True)
def _resolve_uncontested(state, winner):
    """Award entire pot to the last player standing."""
    n = state[S_NPLAYERS]
    state[S_STACKS + winner] += state[S_POT]
    for i in range(n):
        state[S_PAYOFFS + i] = state[S_STACKS + i] - state[S_INITIAL_STACKS + i]
    state[S_POT] = 0
    state[S_TERMINAL] = 1


@nb.njit(cache=True)
def _resolve_showdown(state, deck, hole_cards, flush_lut, nf_keys, nf_vals):
    """Full side-pot showdown resolution using LUT hand evaluator."""
    n = state[S_NPLAYERS]
    board_start = 2 * n

    # Evaluate hand strength for each non-folded player
    scores = np.zeros(n, dtype=np.int32)
    hand_buf = np.empty(7, dtype=np.int8)
    in_hand_count = 0
    last_in = np.int32(-1)

    for i in range(n):
        if state[S_FOLDED + i] == 0:
            in_hand_count += 1
            last_in = np.int32(i)
            # Build 7-card hand: 2 hole + 5 board, converted to eval format
            hand_buf[0] = _game_to_eval_card(hole_cards[i, 0])
            hand_buf[1] = _game_to_eval_card(hole_cards[i, 1])
            for b in range(5):
                hand_buf[2 + b] = _game_to_eval_card(deck[board_start + b])
            scores[i] = evaluate_lut(hand_buf, 7, flush_lut, nf_keys, nf_vals)

    if in_hand_count <= 1:
        if in_hand_count == 1:
            _resolve_uncontested(state, last_in)
        return

    # Collect unique investment levels (sorted)
    inv_levels = np.empty(n, dtype=np.int32)
    n_levels = 0
    for i in range(n):
        if state[S_INVESTED + i] > 0:
            val = state[S_INVESTED + i]
            # Insert into sorted array if not duplicate
            found = False
            for k in range(n_levels):
                if inv_levels[k] == val:
                    found = True
                    break
            if not found:
                # Insertion sort
                pos = n_levels
                for k in range(n_levels):
                    if inv_levels[k] > val:
                        pos = k
                        break
                # Shift right
                for k in range(n_levels, pos, -1):
                    inv_levels[k] = inv_levels[k - 1]
                inv_levels[pos] = val
                n_levels += 1

    # Side-pot resolution
    awarded = np.zeros(n, dtype=np.int32)
    prev = np.int32(0)
    for lvl_idx in range(n_levels):
        lvl = inv_levels[lvl_idx]
        per = lvl - prev

        # Count contributors (anyone who invested >= lvl)
        n_contributors = 0
        for i in range(n):
            if state[S_INVESTED + i] >= lvl:
                n_contributors += 1
        amount = per * n_contributors

        # Find best score among eligible (non-folded, invested >= lvl)
        best_score = np.int32(-1)
        for i in range(n):
            if state[S_FOLDED + i] == 0 and state[S_INVESTED + i] >= lvl:
                if scores[i] > best_score:
                    best_score = scores[i]

        if best_score < 0:
            prev = lvl
            continue

        # Count winners (tied for best)
        n_winners = 0
        for i in range(n):
            if (
                state[S_FOLDED + i] == 0
                and state[S_INVESTED + i] >= lvl
                and scores[i] == best_score
            ):
                n_winners += 1

        if n_winners == 0:
            prev = lvl
            continue

        split = amount // n_winners
        rem = amount % n_winners
        winner_idx = 0
        for i in range(n):
            if (
                state[S_FOLDED + i] == 0
                and state[S_INVESTED + i] >= lvl
                and scores[i] == best_score
            ):
                awarded[i] += split
                if winner_idx == 0:
                    awarded[i] += rem
                winner_idx += 1

        prev = lvl

    # Apply awards and compute payoffs
    for i in range(n):
        state[S_STACKS + i] += awarded[i]
        state[S_PAYOFFS + i] = state[S_STACKS + i] - state[S_INITIAL_STACKS + i]
    state[S_POT] = 0
    state[S_TERMINAL] = 1


# Street advancement
@nb.njit(cache=True)
def _advance_street(state, deck, hole_cards, flush_lut, nf_keys, nf_vals):
    """Reset bets, advance street counter, handle showdown if needed."""
    n = state[S_NPLAYERS]

    # Reset per-street state
    for i in range(n):
        state[S_BETS + i] = 0
    state[S_CURBET] = 0
    state[S_NRAISES] = 0
    state[S_LASTRAISER] = -1
    state[S_LAST_FULL_RAISE] = BIG_BLIND

    street = state[S_STREET]

    # River already played - go to showdown
    if street == 3:
        _resolve_showdown(state, deck, hole_cards, flush_lut, nf_keys, nf_vals)
        return

    new_street = street + 1
    board_start = 2 * n
    if new_street == 1:
        state[S_DECKIDX] = board_start + 3
    elif new_street == 2:
        state[S_DECKIDX] = board_start + 4
    elif new_street == 3:
        state[S_DECKIDX] = board_start + 5
    state[S_STREET] = new_street

    # Count players who can still act (not folded, not all-in, has chips)
    actors = 0
    for i in range(n):
        if (
            state[S_FOLDED + i] == 0
            and state[S_ALLIN + i] == 0
            and state[S_STACKS + i] > 0
        ):
            actors += 1

    # Only run it out when nobody can act.
    if actors == 0:
        while state[S_STREET] < 3:
            state[S_STREET] += 1
        _resolve_showdown(state, deck, hole_cards, flush_lut, nf_keys, nf_vals)
        return

    # Set next actor: first non-folded, non-allin player after dealer
    dealer = state[S_DEALER]
    if n == 2:
        bb_seat = (dealer + 1) % 2
        sb_seat = dealer
        if state[S_FOLDED + bb_seat] == 0 and state[S_ALLIN + bb_seat] == 0:
            state[S_TOACT] = bb_seat
        else:
            state[S_TOACT] = sb_seat
    else:
        for offset in range(1, n + 1):
            cand = (dealer + offset) % n
            if state[S_FOLDED + cand] == 0 and state[S_ALLIN + cand] == 0:
                state[S_TOACT] = cand
                break

    # Mark pending
    for i in range(n):
        if state[S_FOLDED + i] == 0 and state[S_ALLIN + i] == 0:
            state[S_PENDING + i] = 1
        else:
            state[S_PENDING + i] = 0


# Step (main action application)
@nb.njit(cache=True)
def step(state, deck, hole_cards, action_id, flush_lut, nf_keys, nf_vals):
    """Apply action for current actor. Mutates state in place."""
    seat = state[S_TOACT]
    n = state[S_NPLAYERS]
    pot = state[S_POT]
    my_bet = state[S_BETS + seat]
    cur_bet = state[S_CURBET]
    to_call = cur_bet - my_bet
    if to_call < 0:
        to_call = 0
    my_stack = state[S_STACKS + seat]

    state[S_PENDING + seat] = 0

    if action_id == FOLD:
        state[S_FOLDED + seat] = 1
    elif action_id == CHECK_CALL:
        call_amt = to_call
        if call_amt > my_stack:
            call_amt = my_stack
        _put_in(state, seat, call_amt)
    else:
        prev_bet = state[S_CURBET]
        chips = _bet_chip_amount(action_id, pot, to_call, my_stack)
        _put_in(state, seat, chips)
        if state[S_BETS + seat] > state[S_CURBET]:
            state[S_CURBET] = state[S_BETS + seat]
        raise_size = state[S_CURBET] - prev_bet
        if raise_size < 0:
            raise_size = np.int32(0)
        last_full_raise = state[S_LAST_FULL_RAISE]
        if last_full_raise < BIG_BLIND:
            last_full_raise = BIG_BLIND
        if raise_size >= last_full_raise:
            state[S_LAST_FULL_RAISE] = raise_size
            state[S_NRAISES] += 1
            state[S_LASTRAISER] = seat
            for i in range(n):
                if i != seat and state[S_FOLDED + i] == 0 and state[S_ALLIN + i] == 0:
                    state[S_PENDING + i] = 1

    # Check uncontested
    in_hand = 0
    winner = np.int32(-1)
    for i in range(n):
        if state[S_FOLDED + i] == 0:
            in_hand += 1
            winner = np.int32(i)
    if in_hand == 1:
        _resolve_uncontested(state, winner)
        return

    # Advance actor or street
    nxt = _next_actor(state, seat)
    if nxt < 0:
        _advance_street(state, deck, hole_cards, flush_lut, nf_keys, nf_vals)
    else:
        state[S_TOACT] = nxt
