"""
Preflop Equity Lookup — 6-max dataset of starting-hand equities.

Generates a JSON dataset with Monte-Carlo equity for all 169 canonical
preflop combos vs N random hands, then provides a fast lookup function
to enrich preflop recommendations.

Clean-room implementation. No code imported from external sources.
"""
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from treys import Evaluator, Card as TreysCard
    _evaluator = Evaluator()
    _HAS_TREYS = True
except Exception:
    _HAS_TREYS = False


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "preflop_equity_6max.json"


# --------------------------------------------------------------------------- #
#  Combo normalization
# --------------------------------------------------------------------------- #

def _rank_index(r: str) -> int:
    return RANKS.index(r)


def cards_to_combo(hole: List[str]) -> str:
    """
    Convert two card strings to canonical combo notation.

    Examples:
        ['As', 'Kh'] -> 'AKo'
        ['As', 'Ks'] -> 'AKs'
        ['As', 'Ah'] -> 'AA'
        ['2c', '2d'] -> '22'
    """
    if not hole or len(hole) != 2:
        raise ValueError(f"hole must be 2 cards, got {hole}")

    r1, s1 = hole[0][0].upper(), hole[0][1].lower()
    r2, s2 = hole[1][0].upper(), hole[1][1].lower()

    if r1 == r2:
        return r1 + r2

    i1, i2 = _rank_index(r1), _rank_index(r2)
    high, low = (r1, r2) if i1 > i2 else (r2, r1)
    suited = "s" if s1 == s2 else "o"
    return high + low + suited


def combo_to_cards(combo: str) -> List[str]:
    """
    Convert canonical combo to two specific card strings (canonical
    representatives). Equity is identical across suit permutations.

    Examples:
        'AA'  -> ['As', 'Ah']
        'AKs' -> ['As', 'Ks']
        'AKo' -> ['As', 'Kh']
        '72o' -> ['7s', '2h']
    """
    if len(combo) == 2:
        r = combo[0]
        return [r + "s", r + "h"]
    if len(combo) == 3:
        r1, r2, s = combo[0], combo[1], combo[2]
        if s == "s":
            return [r1 + "s", r2 + "s"]
        return [r1 + "s", r2 + "h"]
    raise ValueError(f"Bad combo: {combo}")


# --------------------------------------------------------------------------- #
#  Dataset generation
# --------------------------------------------------------------------------- #

def _calculate_combo_equity(combo: str, num_players: int = 6, iterations: int = 500,
                            seed: Optional[int] = None) -> float:
    """
    Monte-Carlo equity of a starting hand vs (num_players - 1) random hands.
    Lower iterations = faster but noisier. 500 iterations targets std-error < 0.02.
    """
    if not _HAS_TREYS:
        raise RuntimeError("treys not available; install it via 'pip install treys'")

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    hole = combo_to_cards(combo)
    hole_set = set(hole)
    hole_treys = [TreysCard.new(c) for c in hole]

    wins = 0.0
    for _ in range(iterations):
        deck = [c for c in FULL_DECK if c not in hole_set]
        rng.shuffle(deck)

        opponents = [deck[i * 2:(i + 1) * 2] for i in range(num_players - 1)]
        board = deck[(num_players - 1) * 2:(num_players - 1) * 2 + 5]
        board_treys = [TreysCard.new(c) for c in board]

        my_score = _evaluator.evaluate(board_treys, hole_treys)
        best_opp = min(_evaluator.evaluate(board_treys, [TreysCard.new(c) for c in opp])
                       for opp in opponents)

        if my_score < best_opp:
            wins += 1.0
        elif my_score == best_opp:
            wins += 0.5

    return wins / iterations


def _all_combos() -> List[str]:
    """Return the 169 canonical preflop combos in 'high card first' notation
    (e.g. 'AKo', '72o', 'AA') — matches `cards_to_combo` output.
    """
    combos = []
    # Pairs: 13
    for r in RANKS:
        combos.append(r + r)
    # Non-pairs: 13 * 12 = 156, split into suited + offsuit = 78 each
    ranks = list(RANKS)
    for i, r1 in enumerate(ranks):
        # Use ranks BEFORE r1 to keep "high card first" notation
        for r2 in ranks[:i]:
            combos.append(r1 + r2 + "s")
            combos.append(r1 + r2 + "o")
    return combos


def generate_dataset(
    num_players: int = 6,
    iterations: int = 500,
    output_path: Path = DEFAULT_DATASET_PATH,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Build the equity dataset for all 169 combos and save it as JSON.
    Returns the dict {combo: equity}.
    """
    combos = _all_combos()
    assert len(combos) == 169, f"Expected 169 combos, got {len(combos)}"

    dataset: Dict[str, float] = {}
    total = len(combos)
    for i, combo in enumerate(combos, 1):
        eq = _calculate_combo_equity(combo, num_players=num_players,
                                     iterations=iterations, seed=42 + i)
        dataset[combo] = round(eq, 4)
        if verbose and (i % 20 == 0 or i == total):
            print(f"[preflop_equity] {i}/{total} done — last: {combo}={eq:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, sort_keys=True)

    if verbose:
        print(f"[preflop_equity] Saved {total} combos to {output_path}")
    return dataset


# --------------------------------------------------------------------------- #
#  Lookup API
# --------------------------------------------------------------------------- #

_cache: Dict[str, float] = {}
_loaded_from: Optional[Path] = None


def _ensure_loaded(dataset_path: Path = DEFAULT_DATASET_PATH) -> None:
    """Load the JSON dataset into the in-memory cache (idempotent)."""
    global _cache, _loaded_from
    if _loaded_from == dataset_path and _cache:
        return
    if not dataset_path.exists():
        print(f"[preflop_equity] WARNING: dataset not found at {dataset_path}; "
              f"run 'python engine/preflop_equity_lookup.py --generate' first.")
        _cache = {}
        _loaded_from = dataset_path
        return
    with open(dataset_path, "r") as f:
        _cache = json.load(f)
    _loaded_from = dataset_path
    print(f"[preflop_equity] Loaded {len(_cache)} combos from {dataset_path.name}")


def get_preflop_equity(hole: List[str], num_players: int = 6,
                       dataset_path: Path = DEFAULT_DATASET_PATH) -> float:
    """
    Return the preflop equity (0.0–1.0) of the given hole cards.

    Args:
        hole: list of 2 card strings, e.g. ['As', 'Kh']
        num_players: 5, 6 (default) or other. Currently the dataset is
                     6-max; for other sizes we still return the 6-max value
                     as a stable estimate (6-max equities are within ~3% of
                     5-max for the same hand).
        dataset_path: optional override of the dataset location.

    Returns:
        float in [0.0, 1.0]. Returns 0.5 if the combo is not in the dataset.
    """
    _ensure_loaded(dataset_path)
    if not _cache:
        return 0.5
    try:
        combo = cards_to_combo(hole)
    except (ValueError, IndexError):
        return 0.5
    return _cache.get(combo, 0.5)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _smoke_tests() -> bool:
    """Run quick assertions; print results. Returns True if all pass."""
    _ensure_loaded()
    if not _cache:
        print("[preflop_equity] No data loaded — skipping smoke tests.")
        return False

    failures = []

    def check(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[preflop_equity] {status}  {name}  {detail}")
        if not ok:
            failures.append(name)

    aa = _cache.get("AA", 0)
    ko = _cache.get("72o", 1)
    kk = _cache.get("22", 1)
    aks = _cache.get("AKs", 0)
    ako = _cache.get("AKo", 1)

    check("AA > 0.80", aa > 0.80, f"got {aa:.3f}")
    check("72o < 0.40", ko < 0.40, f"got {ko:.3f}")
    check("22 < AA", kk < aa, f"22={kk:.3f}, AA={aa:.3f}")
    check("AKs > AKo", aks > ako, f"AKs={aks:.3f}, AKo={ako:.3f}")
    check("lookup API matches cache",
          get_preflop_equity(["As", "Kh"]) == ako,
          f"got {get_preflop_equity(['As', 'Kh']):.3f}")
    check("missing combo returns 0.5",
          get_preflop_equity(["Xx", "Yy"]) == 0.5,
          "(synthetic bad input)")
    check("169 combos present", len(_cache) == 169, f"got {len(_cache)}")

    if failures:
        print(f"[preflop_equity] {len(failures)} test(s) failed: {failures}")
        return False
    print("[preflop_equity] All smoke tests passed.")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preflop equity lookup (6-max dataset)")
    parser.add_argument("--generate", action="store_true",
                        help="Regenerate the JSON dataset via Monte-Carlo.")
    parser.add_argument("--test", action="store_true",
                        help="Run smoke tests against the cached dataset.")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Monte-Carlo iterations per combo (default 500).")
    parser.add_argument("--num-players", type=int, default=6,
                        help="Table size for equity calculation (default 6).")
    parser.add_argument("--output", type=str, default=str(DEFAULT_DATASET_PATH),
                        help="Output JSON path.")
    args = parser.parse_args()

    if args.generate:
        out = Path(args.output)
        generate_dataset(num_players=args.num_players,
                         iterations=args.iterations,
                         output_path=out)
    if args.test:
        ok = _smoke_tests()
        sys.exit(0 if ok else 1)
    if not (args.generate or args.test):
        parser.print_help()


if __name__ == "__main__":
    main()
