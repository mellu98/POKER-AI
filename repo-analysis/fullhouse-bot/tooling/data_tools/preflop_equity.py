"""Generate the 169-hand preflop equity table."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import eval7
import numpy as np

N_SIMS_DEFAULT = 2_000_000


def _make_deck() -> list[eval7.Card]:
    return [eval7.Card(rank + suit) for suit in "cdhs" for rank in "23456789TJQKA"]


def compute_equities(
    n_sims: int = N_SIMS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paired = np.zeros(13, dtype=np.float64)
    suited = np.zeros((13, 13), dtype=np.float64)
    offsuit = np.zeros((13, 13), dtype=np.float64)
    deck = _make_deck()

    for r0 in range(13):
        for r1 in range(r0 + 1):
            if r0 == r1:
                combos = [(r0, r0, 0, 1)]
            else:
                combos = [(r0, r1, 0, 0, True), (r0, r1, 0, 1, False)]

            for combo in combos:
                if len(combo) == 4:
                    hi_r, lo_r, hi_s, lo_s = combo
                    is_pair = True
                    is_suited_hand = False
                else:
                    hi_r, lo_r, hi_s, lo_s, is_suited_hand = combo
                    is_pair = False

                hero_idx0 = hi_s * 13 + hi_r
                hero_idx1 = lo_s * 13 + lo_r
                hero = [deck[hero_idx0], deck[hero_idx1]]
                remaining_indices = [
                    idx for idx in range(52) if idx not in (hero_idx0, hero_idx1)
                ]
                remaining = [deck[idx] for idx in remaining_indices]

                wins = 0
                ties = 0
                rng = np.random.default_rng(hi_r * 1000 + lo_r * 10 + hi_s + lo_s)
                for _ in range(n_sims):
                    perm = rng.permutation(len(remaining))
                    opp = [remaining[perm[0]], remaining[perm[1]]]
                    board = [
                        remaining[perm[2]],
                        remaining[perm[3]],
                        remaining[perm[4]],
                        remaining[perm[5]],
                        remaining[perm[6]],
                    ]

                    hero_score = eval7.evaluate(hero + board)
                    opp_score = eval7.evaluate(opp + board)
                    if hero_score > opp_score:
                        wins += 1
                    elif hero_score == opp_score:
                        ties += 1

                equity = (wins + ties * 0.5) / n_sims
                if is_pair:
                    paired[hi_r] = equity
                elif is_suited_hand:
                    suited[hi_r, lo_r] = equity
                else:
                    offsuit[hi_r, lo_r] = equity

            if r0 % 3 == 0 and r1 == 0:
                print(f"  Progress: rank {r0}/12 ...")

    return (
        paired.astype(np.float32),
        suited.astype(np.float32),
        offsuit.astype(np.float32),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate preflop equity table")
    parser.add_argument(
        "--sims",
        type=int,
        default=N_SIMS_DEFAULT,
        help=f"Simulations per hand class (default {N_SIMS_DEFAULT})",
    )
    args = parser.parse_args(argv)

    out_path = Path(__file__).resolve().parents[2] / "data" / "preflop_equity.npz"
    print(f"Generating 169-hand preflop equity table ({args.sims:,} sims/class)...")
    start = time.time()
    paired, suited, offsuit = compute_equities(args.sims)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")
    print("\nSanity checks:")
    print(f"  AA:  {paired[12]:.4f} (expect ~0.852)")
    print(f"  KK:  {paired[11]:.4f} (expect ~0.824)")
    print(f"  22:  {paired[0]:.4f}  (expect ~0.501)")
    print(f"  55:  {paired[3]:.4f}  (expect ~0.602)")
    print(f"  AKs: {suited[12, 11]:.4f} (expect ~0.670)")
    print(f"  AKo: {offsuit[12, 11]:.4f} (expect ~0.653)")
    print(f"  72o: {offsuit[5, 0]:.4f}  (expect ~0.345)")
    print(f"  72s: {suited[5, 0]:.4f}  (expect ~0.382)")
    np.savez(str(out_path), paired=paired, suited=suited, offsuit=offsuit)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
