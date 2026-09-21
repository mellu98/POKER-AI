"""Numba-JIT Deep CFR traversal with external sampling.

All hot-path functions are @nb.njit(cache=True). The only non-JIT function
is worker_run_jit(), which serves as the multiprocessing entry point.
"""

from __future__ import annotations


import numba as nb
import numpy as np

from .game_numba import (
    N_ACTIONS,
    STARTING_STACK,
    BIG_BLIND,
    S_TERMINAL,
    S_PAYOFFS,
    S_TOACT,
    S_NPLAYERS,
    S_STREET,
    S_DEALER,
    S_FOLDED,
    S_STACKS,
    S_POT,
    S_NRAISES,
    S_LASTRAISER,
    S_INVESTED,
    S_CURBET,
    S_BETS,
    new_hand,
    new_hand_with_stacks,
    legal_actions,
    step,
    _game_to_eval_card,
)
from ._hand_eval_lut import evaluate_lut

FEATURE_DIM = 51
_BOARD_COUNTS = np.array([0, 3, 4, 5], dtype=np.int32)


# Feature encoding (51-dim dense)


@nb.njit(cache=True)
def _encode_fast(
    state,
    hole_cards,
    deck,
    seat,
    legal,
    flush_lut,
    nf_keys,
    nf_vals,
    eq_paired,
    eq_suited,
    eq_offsuit,
):
    """Encode game state into float32[51] dense feature vector."""
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    n = state[S_NPLAYERS]
    street = state[S_STREET]

    c0 = np.int32(hole_cards[seat, 0])
    c1 = np.int32(hole_cards[seat, 1])
    r0 = c0 % np.int32(13)
    r1 = c1 % np.int32(13)
    s0 = c0 // np.int32(13)
    s1 = c1 // np.int32(13)
    hi = r0 if r0 > r1 else r1
    lo = r1 if r0 > r1 else r0

    board_start = 2 * n
    n_board = _BOARD_COUNTS[street]

    # --- CARD FEATURES [0-16] ---

    # 0: raw_hand_quality (MC preflop equity)
    if r0 == r1:
        features[0] = eq_paired[r0]
    elif s0 == s1:
        features[0] = eq_suited[hi, lo]
    else:
        features[0] = eq_offsuit[hi, lo]

    # 1: hand_strength (postflop eval7 LUT)
    if street > 0:
        hand = np.empty(2 + n_board, dtype=np.int8)
        hand[0] = _game_to_eval_card(hole_cards[seat, 0])
        hand[1] = _game_to_eval_card(hole_cards[seat, 1])
        for b in range(n_board):
            hand[2 + b] = _game_to_eval_card(deck[board_start + b])
        rank = evaluate_lut(hand, np.int32(2 + n_board), flush_lut, nf_keys, nf_vals)
        features[1] = np.float32(rank) / np.float32(135004160.0)

    # Draw features (only on flop/turn, not river or preflop)
    if street > 0 and street < 3:
        # Flush draw
        suit_counts = np.zeros(4, dtype=np.int32)
        suit_counts[s0] += 1
        suit_counts[s1] += 1
        for b in range(n_board):
            suit_counts[np.int32(deck[board_start + b]) // 13] += 1

        max_suit = np.int32(0)
        flush_suit = np.int32(0)
        for i in range(4):
            if suit_counts[i] > max_suit:
                max_suit = suit_counts[i]
                flush_suit = i
        flush_draw = max_suit == np.int32(4)
        features[2] = np.float32(1.0) if flush_draw else np.float32(0.0)

        # Nut flush draw
        if flush_draw:
            hero_flush_ranks = np.int32(-1)
            if s0 == flush_suit and (hero_flush_ranks < r0):
                hero_flush_ranks = r0
            if s1 == flush_suit and (hero_flush_ranks < r1):
                hero_flush_ranks = r1
            max_flush_rank = hero_flush_ranks
            for b in range(n_board):
                bc = np.int32(deck[board_start + b])
                if bc // 13 == flush_suit and bc % 13 > max_flush_rank:
                    max_flush_rank = bc % 13
            features[6] = (
                np.float32(1.0)
                if hero_flush_ranks == max_flush_rank
                else np.float32(0.0)
            )

        # Straight draw: count completing ranks
        rank_present = np.zeros(13, dtype=np.int32)
        rank_present[r0] = 1
        rank_present[r1] = 1
        for b in range(n_board):
            rank_present[np.int32(deck[board_start + b]) % 13] = 1

        out_rank_present = np.zeros(13, dtype=np.int32)
        for start in range(9):
            cnt = np.int32(0)
            for r in range(start, start + 5):
                cnt += rank_present[r]
            if cnt == 4:
                for r in range(start, start + 5):
                    if rank_present[r] == 0:
                        out_rank_present[r] = 1

        # Wheel: A-2-3-4-5
        wcnt = (
            rank_present[12]
            + rank_present[0]
            + rank_present[1]
            + rank_present[2]
            + rank_present[3]
        )
        if wcnt == 4:
            for r_idx in range(5):
                wr = np.int32(12) if r_idx == 0 else np.int32(r_idx - 1)
                if rank_present[wr] == 0:
                    out_rank_present[wr] = 1

        out_ranks = np.int32(0)
        for r in range(13):
            out_ranks += out_rank_present[r]

        oesd = out_ranks >= 2
        gutshot = out_ranks == 1
        features[3] = np.float32(1.0) if oesd else np.float32(0.0)
        features[4] = np.float32(1.0) if gutshot else np.float32(0.0)

        flush_outs = np.int32(9) if flush_draw else np.int32(0)
        straight_outs = out_ranks * np.int32(4)
        overlap = np.int32(0)
        if flush_draw and out_ranks > 0:
            overlap = np.int32(2) if out_ranks > np.int32(2) else out_ranks
        total_outs = flush_outs + straight_outs - overlap
        features[5] = np.float32(total_outs) / np.float32(20.0)

    # 7-11: hole card properties
    features[7] = np.float32(hi) / np.float32(12.0)
    features[8] = np.float32(lo) / np.float32(12.0)
    features[9] = np.float32(1.0) if s0 == s1 else np.float32(0.0)
    features[10] = np.float32(1.0) if r0 == r1 else np.float32(0.0)
    if hi != lo:
        features[11] = np.float32(hi - lo - 1) / np.float32(12.0)

    # 12-16: board texture
    if n_board > 0:
        board_ranks = np.empty(n_board, dtype=np.int32)
        board_suits = np.empty(n_board, dtype=np.int32)
        for b in range(n_board):
            bc = np.int32(deck[board_start + b])
            board_ranks[b] = bc % 13
            board_suits[b] = bc // 13

        # Board paired
        bp = False
        for i in range(n_board):
            for j in range(i + 1, n_board):
                if board_ranks[i] == board_ranks[j]:
                    bp = True
                    break
            if bp:
                break
        features[12] = np.float32(1.0) if bp else np.float32(0.0)

        # Board monotone
        bm = True
        for i in range(1, n_board):
            if board_suits[i] != board_suits[0]:
                bm = False
                break
        features[13] = np.float32(1.0) if bm else np.float32(0.0)

        # Overcards
        oc = np.int32(0)
        for b in range(n_board):
            if board_ranks[b] > hi:
                oc += 1
        features[14] = np.float32(oc) / np.float32(5.0)

        # Board wet: fraction of board pairs within 2 ranks
        if n_board >= 2:
            close = np.int32(0)
            total_pairs = np.int32(0)
            for i in range(n_board):
                for j in range(i + 1, n_board):
                    total_pairs += 1
                    diff = board_ranks[i] - board_ranks[j]
                    if diff < 0:
                        diff = -diff
                    if diff <= 2:
                        close += 1
            features[15] = np.float32(close) / np.float32(total_pairs)

        # Straight possible: 3+ board cards in one 5-rank window
        b_rank_present = np.zeros(13, dtype=np.int32)
        for b in range(n_board):
            b_rank_present[board_ranks[b]] = 1
        sp = False
        for start in range(9):
            cnt = np.int32(0)
            for r in range(start, start + 5):
                cnt += b_rank_present[r]
            if cnt >= 3:
                sp = True
                break
        if not sp:
            wcnt2 = (
                b_rank_present[12]
                + b_rank_present[0]
                + b_rank_present[1]
                + b_rank_present[2]
                + b_rank_present[3]
            )
            if wcnt2 >= 3:
                sp = True
        features[16] = np.float32(1.0) if sp else np.float32(0.0)

    # --- POSITION/STREET [17-26] ---
    features[17 + street] = np.float32(1.0)
    pos = (seat - state[S_DEALER]) % n
    if pos > 5:
        pos = 5
    features[21 + pos] = np.float32(1.0)

    # --- STACK/POT [27-35] ---
    pot = state[S_POT]
    my_stack = state[S_STACKS + seat]
    features[27] = np.float32(pot) / np.float32(STARTING_STACK)
    features[28] = np.float32(my_stack) / np.float32(STARTING_STACK)

    min_stack = np.int32(2147483647)
    for i in range(n):
        if state[S_FOLDED + i] == 0 and state[S_STACKS + i] < min_stack:
            min_stack = state[S_STACKS + i]
    if min_stack == np.int32(2147483647):
        min_stack = np.int32(0)
    pot_denom = pot if pot > 0 else np.int32(1)
    spr_val = np.float32(min_stack) / np.float32(pot_denom)
    if spr_val > np.float32(10.0):
        spr_val = np.float32(10.0)
    features[29] = spr_val / np.float32(10.0)

    to_call = state[S_CURBET] - state[S_BETS + seat]
    if to_call < 0:
        to_call = 0
    denom = pot + to_call
    if denom > 0:
        features[30] = np.float32(to_call) / np.float32(denom)

    # Opponent stacks (non-folded, sorted desc)
    opp_stacks = np.empty(5, dtype=np.int32)
    opp_count = np.int32(0)
    for i in range(n):
        if i != seat and state[S_FOLDED + i] == 0 and opp_count < 5:
            opp_stacks[opp_count] = state[S_STACKS + i]
            opp_count += 1
    # Simple sort desc
    for i in range(opp_count):
        for j in range(i + 1, opp_count):
            if opp_stacks[j] > opp_stacks[i]:
                tmp = opp_stacks[i]
                opp_stacks[i] = opp_stacks[j]
                opp_stacks[j] = tmp
    for i in range(opp_count):
        features[31 + i] = np.float32(opp_stacks[i]) / np.float32(STARTING_STACK)

    # --- BETTING [36-45] ---
    nr = state[S_NRAISES]
    if nr > 4:
        nr = 4
    features[36] = np.float32(nr) / np.float32(4.0)
    features[37] = np.float32(1.0) if state[S_LASTRAISER] >= 0 else np.float32(0.0)
    features[38] = np.float32(1.0) if state[S_LASTRAISER] == seat else np.float32(0.0)
    features[39] = np.float32(state[S_BETS + seat]) / np.float32(STARTING_STACK)
    features[40] = np.float32(state[S_INVESTED + seat]) / np.float32(STARTING_STACK)
    features[41] = np.float32(state[S_CURBET]) / np.float32(pot_denom)
    my_stack_f = np.float32(my_stack) if my_stack > 0 else np.float32(1.0)
    features[42] = np.float32(to_call) / my_stack_f

    max_vill_bets = np.int32(0)
    max_vill_invested = np.int32(0)
    for i in range(n):
        if i != seat and state[S_FOLDED + i] == 0:
            if state[S_BETS + i] > max_vill_bets:
                max_vill_bets = state[S_BETS + i]
            if state[S_INVESTED + i] > max_vill_invested:
                max_vill_invested = state[S_INVESTED + i]
    features[43] = np.float32(max_vill_bets) / np.float32(STARTING_STACK)
    features[44] = np.float32(max_vill_invested) / np.float32(STARTING_STACK)

    n_active = np.int32(0)
    for i in range(n):
        if state[S_FOLDED + i] == 0:
            n_active += 1
    features[45] = np.float32(n_active) / np.float32(6.0)

    # --- LEGAL MASK [46-50] ---
    for a in range(N_ACTIONS):
        features[46 + a] = np.float32(1.0) if legal[a] else np.float32(0.0)

    return features


# Dueling forward pass with LayerNorm


@nb.njit(cache=True)
def _layer_norm(x, gamma, beta):
    mu = np.float32(0.0)
    n = len(x)
    for i in range(n):
        mu += x[i]
    mu /= np.float32(n)
    var = np.float32(0.0)
    for i in range(n):
        d = x[i] - mu
        var += d * d
    var /= np.float32(n)
    inv_std = np.float32(1.0) / np.sqrt(var + np.float32(1e-5))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = gamma[i] * (x[i] - mu) * inv_std + beta[i]
    return out


@nb.njit(cache=True)
def _leaky_relu_inplace(x):
    for i in range(len(x)):
        if x[i] < np.float32(0.0):
            x[i] = np.float32(0.01) * x[i]


@nb.njit(cache=True)
def _forward_dueling(
    x,
    tw0,
    tb0,
    tln0g,
    tln0b,
    tw1,
    tb1,
    tln1g,
    tln1b,
    vw0,
    vb0,
    vw1,
    vb1,
    aw0,
    ab0,
    aw1,
    ab1,
    legal_f,
):
    """Dueling forward pass: trunk → value + advantage heads → combination."""
    # Trunk layer 0
    h = np.dot(tw0, x) + tb0
    h = _layer_norm(h, tln0g, tln0b)
    _leaky_relu_inplace(h)

    # Trunk layer 1
    h = np.dot(tw1, h) + tb1
    h = _layer_norm(h, tln1g, tln1b)
    _leaky_relu_inplace(h)

    # Value head
    v = np.dot(vw0, h) + vb0
    _leaky_relu_inplace(v)
    V = np.dot(vw1, v) + vb1  # shape (1,)

    # Advantage head
    a = np.dot(aw0, h) + ab0
    _leaky_relu_inplace(a)
    A = np.dot(aw1, a) + ab1  # shape (5,)

    # Dueling combination
    a_masked = A * legal_f
    n_legal = np.float32(0.0)
    a_sum = np.float32(0.0)
    for i in range(N_ACTIONS):
        n_legal += legal_f[i]
        a_sum += a_masked[i]
    a_mean = a_sum / max(n_legal, np.float32(1.0))

    out = np.zeros(N_ACTIONS, dtype=np.float32)
    for i in range(N_ACTIONS):
        out[i] = (V[0] + a_masked[i] - a_mean * legal_f[i]) * legal_f[i]
    return out


@nb.njit(cache=True)
def _strategy(
    features,
    legal_f,
    tw0,
    tb0,
    tln0g,
    tln0b,
    tw1,
    tb1,
    tln1g,
    tln1b,
    vw0,
    vb0,
    vw1,
    vb1,
    aw0,
    ab0,
    aw1,
    ab1,
):
    """Forward pass + regret matching to get action probabilities."""
    raw = _forward_dueling(
        features,
        tw0,
        tb0,
        tln0g,
        tln0b,
        tw1,
        tb1,
        tln1g,
        tln1b,
        vw0,
        vb0,
        vw1,
        vb1,
        aw0,
        ab0,
        aw1,
        ab1,
        legal_f,
    )

    # Clamp to >= 0 (regret matching)
    for i in range(N_ACTIONS):
        if raw[i] < np.float32(0.0):
            raw[i] = np.float32(0.0)
    for i in range(N_ACTIONS):
        raw[i] *= legal_f[i]

    total = np.float32(0.0)
    for i in range(N_ACTIONS):
        total += raw[i]

    if total > np.float32(0.0):
        for i in range(N_ACTIONS):
            raw[i] /= total
        return raw

    n_legal = np.float32(0.0)
    for i in range(N_ACTIONS):
        n_legal += legal_f[i]
    if n_legal <= np.float32(0.0):
        raise ValueError("legal mask has no legal actions")
    for i in range(N_ACTIONS):
        raw[i] = legal_f[i] / n_legal
    return raw


@nb.njit(cache=True)
def _sample_match_stacks(n_players):
    total = STARTING_STACK * n_players
    min_stack = BIG_BLIND
    stacks = np.empty(n_players, dtype=np.int32)
    if total <= min_stack * n_players:
        for i in range(n_players):
            stacks[i] = STARTING_STACK
        return stacks

    weights = np.empty(n_players, dtype=np.float64)
    weight_sum = np.float64(0.0)
    for i in range(n_players):
        u = np.random.random()
        if u < 1e-6:
            u = 1e-6
        w = -np.log(u)
        if w < 1e-6:
            w = 1e-6
        weights[i] = w
        weight_sum += w

    remaining = total - min_stack * n_players
    assigned = np.int32(0)
    for i in range(n_players):
        extra = np.int32(np.floor(np.float64(remaining) * weights[i] / weight_sum))
        stacks[i] = min_stack + extra
        assigned += stacks[i]
    stacks[n_players - 1] += total - assigned

    for i in range(n_players - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = stacks[i]
        stacks[i] = stacks[j]
        stacks[j] = tmp
    return stacks


@nb.njit(cache=True)
def _renormalize_strategy(sigma, legal_f):
    total = np.float32(0.0)
    n_legal = np.float32(0.0)
    for a in range(N_ACTIONS):
        if legal_f[a] > np.float32(0.0) and sigma[a] > np.float32(0.0):
            total += sigma[a]
        if legal_f[a] > np.float32(0.0):
            n_legal += np.float32(1.0)
    if total > np.float32(0.0):
        for a in range(N_ACTIONS):
            sigma[a] = (
                sigma[a] / total
                if legal_f[a] > np.float32(0.0) and sigma[a] > np.float32(0.0)
                else np.float32(0.0)
            )
        return
    if n_legal <= np.float32(0.0):
        raise ValueError("legal mask has no legal actions")
    for a in range(N_ACTIONS):
        sigma[a] = legal_f[a] / n_legal


@nb.njit(cache=True)
def _sample_action(sigma, legal):
    r = np.float32(np.random.random())
    cumsum = np.float32(0.0)
    last_legal = 0
    for a in range(N_ACTIONS):
        if legal[a]:
            last_legal = a
        cumsum += sigma[a]
        if r < cumsum:
            return a
    if cumsum > np.float32(0.0):
        return last_legal
    raise ValueError("sampled strategy had no cumulative mass")


# External sampling traversal


@nb.njit(cache=True)
def _traverse_external(
    state,
    deck,
    hole_cards,
    traverser,
    tw0,
    tb0,
    tln0g,
    tln0b,
    tw1,
    tb1,
    tln1g,
    tln1b,
    vw0,
    vb0,
    vw1,
    vb1,
    aw0,
    ab0,
    aw1,
    ab1,
    out_features,
    out_advantages,
    out_count,
    out_strat_features,
    out_strategies,
    out_strat_weights,
    out_strat_count,
    flush_lut,
    nf_keys,
    nf_vals,
    eq_paired,
    eq_suited,
    eq_offsuit,
    opponent_exploration,
    traverser_reach,
):
    if state[S_TERMINAL] == 1:
        return np.float32(state[S_PAYOFFS + traverser])

    seat = state[S_TOACT]
    legal = legal_actions(state, seat)

    n_legal = 0
    for a in range(N_ACTIONS):
        if legal[a]:
            n_legal += 1
    if n_legal == 0:
        raise ValueError("game engine returned no legal actions")

    features = _encode_fast(
        state,
        hole_cards,
        deck,
        seat,
        legal,
        flush_lut,
        nf_keys,
        nf_vals,
        eq_paired,
        eq_suited,
        eq_offsuit,
    )

    legal_f = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            legal_f[a] = np.float32(1.0)

    sigma = _strategy(
        features,
        legal_f,
        tw0,
        tb0,
        tln0g,
        tln0b,
        tw1,
        tb1,
        tln1g,
        tln1b,
        vw0,
        vb0,
        vw1,
        vb1,
        aw0,
        ab0,
        aw1,
        ab1,
    )
    _renormalize_strategy(sigma, legal_f)

    if seat != traverser:
        if opponent_exploration > np.float32(0.0):
            eps = opponent_exploration
            if eps > np.float32(1.0):
                eps = np.float32(1.0)
            n_legal_f = np.float32(0.0)
            for a in range(N_ACTIONS):
                n_legal_f += legal_f[a]
            if n_legal_f > np.float32(0.0):
                for a in range(N_ACTIONS):
                    sigma[a] = (np.float32(1.0) - eps) * sigma[a] + eps * legal_f[
                        a
                    ] / n_legal_f
            _renormalize_strategy(sigma, legal_f)
        action = _sample_action(sigma, legal)

        child = state.copy()
        step(child, deck, hole_cards, action, flush_lut, nf_keys, nf_vals)
        return _traverse_external(
            child,
            deck,
            hole_cards,
            traverser,
            tw0,
            tb0,
            tln0g,
            tln0b,
            tw1,
            tb1,
            tln1g,
            tln1b,
            vw0,
            vb0,
            vw1,
            vb1,
            aw0,
            ab0,
            aw1,
            ab1,
            out_features,
            out_advantages,
            out_count,
            out_strat_features,
            out_strategies,
            out_strat_weights,
            out_strat_count,
            flush_lut,
            nf_keys,
            nf_vals,
            eq_paired,
            eq_suited,
            eq_offsuit,
            opponent_exploration,
            traverser_reach,
        )

    sidx = out_strat_count[0]
    if sidx < out_strat_features.shape[0]:
        for f in range(FEATURE_DIM):
            out_strat_features[sidx, f] = features[f]
        for a in range(N_ACTIONS):
            out_strategies[sidx, a] = sigma[a]
        out_strat_weights[sidx] = traverser_reach
        out_strat_count[0] = sidx + 1

    child_values = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            child = state.copy()
            step(child, deck, hole_cards, a, flush_lut, nf_keys, nf_vals)
            child_values[a] = _traverse_external(
                child,
                deck,
                hole_cards,
                traverser,
                tw0,
                tb0,
                tln0g,
                tln0b,
                tw1,
                tb1,
                tln1g,
                tln1b,
                vw0,
                vb0,
                vw1,
                vb1,
                aw0,
                ab0,
                aw1,
                ab1,
                out_features,
                out_advantages,
                out_count,
                out_strat_features,
                out_strategies,
                out_strat_weights,
                out_strat_count,
                flush_lut,
                nf_keys,
                nf_vals,
                eq_paired,
                eq_suited,
                eq_offsuit,
                opponent_exploration,
                traverser_reach * sigma[a],
            )

    value = np.float32(0.0)
    for a in range(N_ACTIONS):
        value += sigma[a] * child_values[a]

    advantages = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            advantages[a] = (child_values[a] - value) / np.float32(100.0)

    idx = out_count[0]
    if idx < out_features.shape[0]:
        for f in range(FEATURE_DIM):
            out_features[idx, f] = features[f]
        for a in range(N_ACTIONS):
            out_advantages[idx, a] = advantages[a]
        out_count[0] = idx + 1

    return value


# Batch traversal runner


@nb.njit(cache=True)
def run_traversals(
    n_traversals,
    n_players,
    traverser,
    tw0,
    tb0,
    tln0g,
    tln0b,
    tw1,
    tb1,
    tln1g,
    tln1b,
    vw0,
    vb0,
    vw1,
    vb1,
    aw0,
    ab0,
    aw1,
    ab1,
    seed,
    flush_lut,
    nf_keys,
    nf_vals,
    eq_paired,
    eq_suited,
    eq_offsuit,
    opponent_exploration,
    randomize_stacks,
):
    """Run n_traversals, return collected (features, advantages)."""
    np.random.seed(seed)

    max_samples = n_traversals * 60
    out_features = np.zeros((max_samples, FEATURE_DIM), dtype=np.float32)
    out_advantages = np.zeros((max_samples, N_ACTIONS), dtype=np.float32)
    out_count = np.zeros(1, dtype=np.int64)
    out_strat_features = np.zeros((max_samples, FEATURE_DIM), dtype=np.float32)
    out_strategies = np.zeros((max_samples, N_ACTIONS), dtype=np.float32)
    out_strat_weights = np.zeros(max_samples, dtype=np.float32)
    out_strat_count = np.zeros(1, dtype=np.int64)

    for _ in range(n_traversals):
        dealer = np.random.randint(0, n_players)
        if randomize_stacks:
            starting_stacks = _sample_match_stacks(n_players)
            state, deck, hole_cards = new_hand_with_stacks(
                n_players, dealer, starting_stacks
            )
        else:
            state, deck, hole_cards = new_hand(n_players, dealer)
        _traverse_external(
            state,
            deck,
            hole_cards,
            traverser,
            tw0,
            tb0,
            tln0g,
            tln0b,
            tw1,
            tb1,
            tln1g,
            tln1b,
            vw0,
            vb0,
            vw1,
            vb1,
            aw0,
            ab0,
            aw1,
            ab1,
            out_features,
            out_advantages,
            out_count,
            out_strat_features,
            out_strategies,
            out_strat_weights,
            out_strat_count,
            flush_lut,
            nf_keys,
            nf_vals,
            eq_paired,
            eq_suited,
            eq_offsuit,
            np.float32(opponent_exploration),
            np.float32(1.0),
        )

    n_samples = out_count[0]
    n_strat_samples = out_strat_count[0]
    return (
        out_features[:n_samples].copy(),
        out_advantages[:n_samples].copy(),
        out_strat_features[:n_strat_samples].copy(),
        out_strategies[:n_strat_samples].copy(),
        out_strat_weights[:n_strat_samples].copy(),
    )


# Multiprocessing entry point


def worker_run_jit(args):
    """Non-JIT wrapper for multiprocessing Pool.map.

    Expects the mapping shape produced by ``training.train``.
    """
    seed = int(args["seed"])
    n_traversals = int(args["n_traversals"])
    traverser = int(args["traverser"])
    n_players = int(args["n_players"])
    weights_list = args["weights"]
    biases_list = args["biases"]
    flush_lut = args["flush_lut"]
    nf_keys = args["nf_keys"]
    nf_vals = args["nf_vals"]
    eq_paired = args["eq_paired"]
    eq_suited = args["eq_suited"]
    eq_offsuit = args["eq_offsuit"]
    opponent_exploration = float(args["opponent_exploration"])
    randomize_stacks = bool(args.get("randomize_stacks", True))

    # Unpack dueling weights
    tw0 = weights_list[0].astype(np.float32)
    tb0 = biases_list[0].astype(np.float32)
    tln0g = weights_list[1].astype(np.float32)
    tln0b = biases_list[1].astype(np.float32)
    tw1 = weights_list[2].astype(np.float32)
    tb1 = biases_list[2].astype(np.float32)
    tln1g = weights_list[3].astype(np.float32)
    tln1b = biases_list[3].astype(np.float32)
    vw0 = weights_list[4].astype(np.float32)
    vb0 = biases_list[4].astype(np.float32)
    vw1 = weights_list[5].astype(np.float32)
    vb1 = biases_list[5].astype(np.float32)
    aw0 = weights_list[6].astype(np.float32)
    ab0 = biases_list[6].astype(np.float32)
    aw1 = weights_list[7].astype(np.float32)
    ab1 = biases_list[7].astype(np.float32)

    result = run_traversals(
        n_traversals,
        n_players,
        traverser,
        tw0,
        tb0,
        tln0g,
        tln0b,
        tw1,
        tb1,
        tln1g,
        tln1b,
        vw0,
        vb0,
        vw1,
        vb1,
        aw0,
        ab0,
        aw1,
        ab1,
        seed,
        flush_lut,
        nf_keys,
        nf_vals,
        eq_paired,
        eq_suited,
        eq_offsuit,
        np.float32(opponent_exploration),
        randomize_stacks,
    )
    max_samples = n_traversals * 60
    n_adv = result[0].shape[0]
    n_strat = result[2].shape[0]
    if n_adv >= max_samples or n_strat >= max_samples:
        import logging

        logging.getLogger(__name__).warning(
            "traversal buffer full: %d/%d advantage, %d/%d strategy samples "
            "(increase multiplier from 60)",
            n_adv,
            max_samples,
            n_strat,
            max_samples,
        )
    return result
