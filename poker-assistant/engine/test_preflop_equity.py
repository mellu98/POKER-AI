"""
Tests for engine/preflop_equity_lookup.py

Run with:  python engine/test_preflop_equity.py

Pattern: function-based, executed via `if __name__ == "__main__"`.
"""
import sys
from pathlib import Path

# Make preflop_equity_lookup importable when run from project root.
sys.path.insert(0, str(Path(__file__).parent))

import preflop_equity_lookup as pel


def test_cards_to_combo_basic():
    assert pel.cards_to_combo(["As", "Kh"]) == "AKo", "AKo case failed"
    assert pel.cards_to_combo(["As", "Ks"]) == "AKs", "AKs case failed"
    assert pel.cards_to_combo(["As", "Ah"]) == "AA", "AA pair case failed"
    assert pel.cards_to_combo(["2c", "2d"]) == "22", "22 pair case failed"
    assert pel.cards_to_combo(["Kc", "As"]) == "AKo", "reversed order case failed"
    assert pel.cards_to_combo(["Ts", "9h"]) == "T9o", "T9o case failed"
    print("[test] PASS  cards_to_combo basic cases")


def test_combo_to_cards_basic():
    assert pel.combo_to_cards("AA") == ["As", "Ah"]
    assert pel.combo_to_cards("AKs") == ["As", "Ks"]
    assert pel.combo_to_cards("AKo") == ["As", "Kh"]
    assert pel.combo_to_cards("72o") == ["7s", "2h"]
    assert pel.combo_to_cards("22") == ["2s", "2h"]
    print("[test] PASS  combo_to_cards basic cases")


def test_canonical_form_is_invariant():
    """Suited/offsuit distinction preserved, but order doesn't matter."""
    assert pel.cards_to_combo(["As", "Kh"]) == pel.cards_to_combo(["Kc", "Ad"])
    assert pel.cards_to_combo(["As", "Ks"]) == pel.cards_to_combo(["Kc", "Ac"])
    assert pel.cards_to_combo(["As", "Ah"]) == pel.cards_to_combo(["Ac", "Ad"])
    print("[test] PASS  canonical form is order-invariant")


def test_all_combos_count():
    """Standard 169 preflop combos = 13 pairs + 12*13 non-pairs (s+o)."""
    combos = pel._all_combos()
    assert len(combos) == 169, f"Expected 169 combos, got {len(combos)}"
    pairs = [c for c in combos if len(c) == 2]
    suited = [c for c in combos if c.endswith("s") and len(c) == 3]
    offsuit = [c for c in combos if c.endswith("o") and len(c) == 3]
    assert len(pairs) == 13, f"Expected 13 pairs, got {len(pairs)}"
    assert len(suited) == 78, f"Expected 78 suited, got {len(suited)}"
    assert len(offsuit) == 78, f"Expected 78 offsuit, got {len(offsuit)}"
    print(f"[test] PASS  169 combos ({len(pairs)} pairs, "
          f"{len(suited)} suited, {len(offsuit)} offsuit)")


def test_dataset_exists_and_loaded():
    """Dataset must exist; cache must contain 169 entries after load."""
    if not pel.DEFAULT_DATASET_PATH.exists():
        print(f"[test] SKIP  dataset not found at {pel.DEFAULT_DATASET_PATH} "
              f"(run --generate first)")
        return
    pel._ensure_loaded()
    assert len(pel._cache) == 169, f"Expected 169 in cache, got {len(pel._cache)}"
    print(f"[test] PASS  dataset has 169 entries")


def test_equity_ordering_invariant():
    """Sanity-check well-known equity ordering relationships (6-max vs 5 random)."""
    if not pel.DEFAULT_DATASET_PATH.exists():
        print("[test] SKIP  equity_ordering (no dataset)")
        return
    pel._ensure_loaded()
    aa = pel._cache.get("AA", 0)
    kk = pel._cache.get("KK", 0)
    qq = pel._cache.get("QQ", 0)
    aks = pel._cache.get("AKs", 0)
    ako = pel._cache.get("AKo", 0)
    aqs = pel._cache.get("AQs", 0)
    aq2 = pel._cache.get("AQo", 0)
    weak = pel._cache.get("72o", 0)

    assert aa > kk > qq, f"Pocket pair ordering broken: AA={aa}, KK={kk}, QQ={qq}"
    assert aks > ako, f"Suited > offsuit broken: AKs={aks}, AKo={ako}"
    assert aqs > aq2, f"Suited > offsuit broken: AQs={aqs}, AQo={aq2}"
    assert weak < 0.35, f"72o should be weak (<0.35 in 6-max), got {weak}"
    assert aa > 0.40, f"AA should be > 0.40 in 6-max, got {aa}"
    print(f"[test] PASS  equity ordering (6-max): AA={aa:.3f} > KK={kk:.3f} > QQ={qq:.3f}; "
          f"AKs={aks:.3f} > AKo={ako:.3f}; 72o={weak:.3f}")


def test_get_preflop_equity_api():
    """get_preflop_equity must return values consistent with the cache (6-max)."""
    if not pel.DEFAULT_DATASET_PATH.exists():
        print("[test] SKIP  get_preflop_equity_api (no dataset)")
        return
    eq_ak = pel.get_preflop_equity(["As", "Kh"])
    eq_aa = pel.get_preflop_equity(["As", "Ah"])
    eq_72 = pel.get_preflop_equity(["7c", "2d"])
    eq_bad = pel.get_preflop_equity(["Xx", "Yy"])

    assert 0.20 < eq_ak < 0.35, f"AKo should be ~0.27 in 6-max, got {eq_ak}"
    assert eq_aa > 0.40, f"AA should be > 0.40 in 6-max, got {eq_aa}"
    assert eq_72 < 0.35, f"72o should be < 0.35, got {eq_72}"
    assert eq_bad == 0.5, f"Bad input should return 0.5, got {eq_bad}"
    print(f"[test] PASS  get_preflop_equity API (6-max): AKo={eq_ak:.3f}, "
          f"AA={eq_aa:.3f}, 72o={eq_72:.3f}, bad=0.500")


def test_cache_consistency():
    """Two consecutive lookups for the same hand must return the same value."""
    if not pel.DEFAULT_DATASET_PATH.exists():
        print("[test] SKIP  cache_consistency (no dataset)")
        return
    a = pel.get_preflop_equity(["As", "Kh"])
    b = pel.get_preflop_equity(["As", "Kh"])
    assert a == b, f"Cache inconsistent: {a} != {b}"
    print(f"[test] PASS  cache consistency: {a:.3f} == {b:.3f}")


def main():
    tests = [
        test_cards_to_combo_basic,
        test_combo_to_cards_basic,
        test_canonical_form_is_invariant,
        test_all_combos_count,
        test_dataset_exists_and_loaded,
        test_equity_ordering_invariant,
        test_get_preflop_equity_api,
        test_cache_consistency,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[test] FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[test] ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n[test] Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
