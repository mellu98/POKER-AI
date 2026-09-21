"""Lookup-table poker hand evaluator for fast MCCFR training.

Prime-product hashing for non-flush hands, bitmask lookup for flushes.
Tables are generated once from the reference evaluator and saved to disk.
~73K entries, <4 MB total.
"""

from __future__ import annotations

import time
from pathlib import Path

import numba as nb
import numpy as np

from ._hand_eval import _evaluate_n

_i8 = nb.int8
_i32 = nb.int32
_i64 = nb.int64

_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int64)
_HT_SIZE = 1 << 18  # 262144 slots - 28% load factor for ~73K entries


# Numba fast path
@nb.njit(_i32(_i64[:], _i32[:], _i64), cache=True)
def _nf_lookup(ht_keys, ht_vals, product):
    """Hash-table lookup for non-flush hands by prime product."""
    mask = _i32(len(ht_keys) - 1)
    h = product ^ (product >> _i64(16))
    slot = _i32(h) & mask
    for _ in range(len(ht_keys)):
        k = ht_keys[slot]
        if k == product:
            return ht_vals[slot]
        if k == _i64(0):
            return _i32(0)
        slot = (slot + _i32(1)) & mask
    return _i32(0)


@nb.njit(cache=True)
def evaluate_lut(hand, n, flush_lut, nf_ht_keys, nf_ht_vals):
    """Evaluate best 5-card poker hand from n (5-7) cards via lookup tables."""
    sc0 = _i32(0)
    sc1 = _i32(0)
    sc2 = _i32(0)
    sc3 = _i32(0)
    sm0 = _i32(0)
    sm1 = _i32(0)
    sm2 = _i32(0)
    sm3 = _i32(0)
    product = _i64(1)

    for i in range(n):
        card = hand[i]
        rank = _i32(card >> _i8(2))
        suit = card & _i8(3)
        product *= _PRIMES[rank]
        bit = _i32(1) << rank
        if suit == _i8(0):
            sc0 += _i32(1)
            sm0 |= bit
        elif suit == _i8(1):
            sc1 += _i32(1)
            sm1 |= bit
        elif suit == _i8(2):
            sc2 += _i32(1)
            sm2 |= bit
        else:
            sc3 += _i32(1)
            sm3 |= bit

    if sc0 >= _i32(5):
        return flush_lut[sm0]
    if sc1 >= _i32(5):
        return flush_lut[sm1]
    if sc2 >= _i32(5):
        return flush_lut[sm2]
    if sc3 >= _i32(5):
        return flush_lut[sm3]

    return _nf_lookup(nf_ht_keys, nf_ht_vals, product)


# Table generation (one-time, uses the reference evaluator)
def _gen_rank_distributions(n_cards: int, n_ranks: int = 13, max_per: int = 4):
    """Yield all valid rank count tuples summing to n_cards."""

    def _recurse(rank, remaining):
        if rank == n_ranks:
            if remaining == 0:
                yield ()
            return
        for count in range(min(max_per, remaining) + 1):
            for rest in _recurse(rank + 1, remaining - count):
                yield (count,) + rest

    yield from _recurse(0, n_cards)


def generate_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build flush + non-flush lookup tables."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

    # -- Flush table: bitmask → hand rank --
    flush_lut = np.zeros(8192, dtype=np.int32)
    for mask in range(8192):
        pc = bin(mask).count("1")
        if pc < 5 or pc > 7:
            continue
        hand = np.array(
            [np.int8(r * 4) for r in range(13) if mask & (1 << r)], dtype=np.int8
        )
        flush_lut[mask] = _evaluate_n(hand, np.int32(len(hand)))

    # -- Non-flush hash table: prime product → hand rank --
    products: list[int] = []
    values: list[int] = []

    for n_cards in (5, 6, 7):
        for dist in _gen_rank_distributions(n_cards):
            product = 1
            for rank, count in enumerate(dist):
                product *= primes[rank] ** count
            cards = []
            suit_idx = 0
            for rank, count in enumerate(dist):
                for _ in range(count):
                    cards.append(np.int8(rank * 4 + (suit_idx % 4)))
                    suit_idx += 1
            hand = np.array(cards, dtype=np.int8)
            score = int(_evaluate_n(hand, np.int32(n_cards)))
            products.append(product)
            values.append(score)

    ht_keys = np.zeros(_HT_SIZE, dtype=np.int64)
    ht_vals = np.zeros(_HT_SIZE, dtype=np.int32)
    mask = _HT_SIZE - 1

    for i in range(len(products)):
        p = np.int64(products[i])
        h = int(p ^ (p >> np.int64(16)))
        slot = h & mask
        while ht_keys[slot] != 0:
            slot = (slot + 1) & mask
        ht_keys[slot] = p
        ht_vals[slot] = np.int32(values[i])

    return flush_lut, ht_keys, ht_vals


def load_or_generate(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LUT from .npz or generate, save, and return."""
    path = Path(path)
    if path.exists():
        data = np.load(path)
        return data["flush_lut"], data["nf_ht_keys"], data["nf_ht_vals"]

    print("[hand_eval_lut] generating lookup tables...")
    t0 = time.time()
    flush_lut, nf_ht_keys, nf_ht_vals = generate_tables()
    elapsed = time.time() - t0
    n_entries = int(np.count_nonzero(nf_ht_keys))
    n_flush = int(np.count_nonzero(flush_lut))
    print(
        f"[hand_eval_lut] {n_entries} non-flush + {n_flush} flush entries "
        f"in {elapsed:.1f}s"
    )

    np.savez(path, flush_lut=flush_lut, nf_ht_keys=nf_ht_keys, nf_ht_vals=nf_ht_vals)
    print(f"[hand_eval_lut] saved to {path}")
    return flush_lut, nf_ht_keys, nf_ht_vals
