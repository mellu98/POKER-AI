"""
GTO Preflop Range Generator

Generates realistic GTO-approximation preflop ranges for 6-max 100BB cash games.
Based on equilibrium principles and standard solver outputs.

Usage:
    python tools/generate_gto_ranges.py

Output:
    - database/preflop/open_ranges.json
    - database/preflop/3bet_ranges.json
    - database/preflop/4bet_ranges.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


# All 169 unique starting hands
RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']


@dataclass
class HandStrength:
    """Represents intrinsic hand strength for range generation."""
    hand: str
    score: float  # 0.0 to 1.0
    category: str  # premium, strong, medium, speculative, weak


def get_all_hands() -> List[str]:
    """Generate all 169 unique starting hands."""
    hands = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                # Pocket pair
                hands.append(f"{r1}{r2}")
            elif i < j:
                # Suited (higher rank first)
                hands.append(f"{r1}{r2}s")
                # Offsuit
                hands.append(f"{r1}{r2}o")
    return hands


def calculate_hand_strength(hand: str) -> HandStrength:
    """
    Calculate intrinsic hand strength score.

    Factors:
    - High card value
    - Pair vs suited vs offsuit
    - Connectivity (gap between cards)
    """
    if len(hand) == 2:
        # Pocket pair
        r1 = hand[0]
        rank_val = RANKS.index(r1)
        # AA = 1.0, KK = 0.95, ..., 22 = 0.35
        score = 1.0 - (rank_val * 0.05)
        if score > 0.85:
            category = "premium"
        elif score > 0.6:
            category = "strong"
        else:
            category = "medium"
    else:
        r1, r2 = hand[0], hand[1]
        suited = hand.endswith('s')

        rank1 = RANKS.index(r1)
        rank2 = RANKS.index(r2)
        gap = rank2 - rank1 - 1  # Gap between cards

        # Base score from high card
        high_card_score = (12 - rank1) / 12  # A = 1.0, 2 = 0.0

        # Second card contribution
        second_card_score = (12 - rank2) / 12 * 0.5

        # Suited bonus
        suited_bonus = 0.08 if suited else 0

        # Connectivity bonus (0 gap = best)
        connectivity_bonus = max(0, (4 - gap) * 0.02) if gap >= 0 else 0

        # Broadway bonus (both cards T+)
        broadway_bonus = 0.05 if rank1 <= 4 and rank2 <= 4 else 0

        score = min(0.95, high_card_score * 0.5 + second_card_score +
                   suited_bonus + connectivity_bonus + broadway_bonus)

        # Categorize
        if score > 0.75:
            category = "premium"
        elif score > 0.55:
            category = "strong"
        elif score > 0.40:
            category = "medium"
        elif score > 0.25:
            category = "speculative"
        else:
            category = "weak"

    return HandStrength(hand=hand, score=score, category=category)


def generate_open_ranges() -> Dict:
    """
    Generate opening ranges for each position.

    Range sizes (approximate):
    - UTG: 15% (tight)
    - MP: 18%
    - CO: 27%
    - BTN: 45% (widest)
    - SB: 40% (3-bet or fold, some limp)
    """
    # Position thresholds (hands with score >= threshold are opened)
    position_thresholds = {
        "UTG": 0.60,  # ~15% of hands
        "MP": 0.55,   # ~18% of hands
        "CO": 0.45,   # ~27% of hands
        "BTN": 0.30,  # ~45% of hands
        "SB": 0.35,   # ~40% of hands (3-bet focused)
    }

    # Position sizing
    position_sizing = {
        "UTG": 2.5,
        "MP": 2.5,
        "CO": 2.5,
        "BTN": 2.5,
        "SB": 3.0,  # Larger from SB
    }

    all_hands = get_all_hands()
    ranges = {
        "_metadata": {
            "description": "6-max 100BB cash game opening ranges",
            "source": "GTO equilibrium approximation",
            "version": "2.0.0"
        }
    }

    for position, threshold in position_thresholds.items():
        ranges[position] = {
            "_description": f"Opening range for {position}"
        }

        for hand in all_hands:
            strength = calculate_hand_strength(hand)

            if strength.score >= threshold:
                # Calculate frequency based on how close to threshold
                margin = strength.score - threshold
                if margin > 0.15:
                    frequency = 1.0
                elif margin > 0.05:
                    frequency = 0.85
                else:
                    frequency = 0.5 + (margin / 0.1) * 0.35

                ranges[position][hand] = {
                    "action": "raise",
                    "sizing": position_sizing[position],
                    "frequency": round(frequency, 2)
                }

    return ranges


def generate_3bet_ranges() -> Dict:
    """
    Generate 3-bet and call ranges vs opens from different positions.

    Key principles:
    - 3-bet tighter vs earlier position opens
    - Call wider in position
    - 3-bet or fold from SB (no flatting)
    """
    ranges = {
        "_metadata": {
            "description": "3-bet and call ranges vs opens",
            "format": "POSITION_vs_RAISER: {3bet: {...}, call: {...}}",
            "sizing": "3x open + 1BB per caller",
            "version": "2.0.0"
        }
    }

    # Define 3-bet and call thresholds for each matchup
    # Format: (3bet_threshold, call_threshold)
    matchups = {
        # From MP
        "MP_vs_UTG": (0.80, 0.65),
        # From CO
        "CO_vs_UTG": (0.75, 0.60),
        "CO_vs_MP": (0.70, 0.55),
        # From BTN
        "BTN_vs_UTG": (0.70, 0.55),
        "BTN_vs_MP": (0.65, 0.50),
        "BTN_vs_CO": (0.60, 0.45),
        # From SB (3-bet or fold, no calling)
        "SB_vs_UTG": (0.75, None),
        "SB_vs_MP": (0.70, None),
        "SB_vs_CO": (0.65, None),
        "SB_vs_BTN": (0.55, None),
        # From BB
        "BB_vs_UTG": (0.80, 0.50),
        "BB_vs_MP": (0.75, 0.45),
        "BB_vs_CO": (0.70, 0.40),
        "BB_vs_BTN": (0.60, 0.35),
        "BB_vs_SB": (0.55, 0.40),
    }

    all_hands = get_all_hands()

    for matchup, (three_bet_thresh, call_thresh) in matchups.items():
        ranges[matchup] = {"3bet": {}, "call": {}}

        for hand in all_hands:
            strength = calculate_hand_strength(hand)

            # 3-bet range
            if strength.score >= three_bet_thresh:
                margin = strength.score - three_bet_thresh
                frequency = min(1.0, 0.6 + margin * 2)
                ranges[matchup]["3bet"][hand] = {
                    "sizing_multiplier": 3.0,
                    "frequency": round(frequency, 2)
                }

            # Call range (if applicable and not in 3-bet range)
            elif call_thresh and strength.score >= call_thresh:
                margin = strength.score - call_thresh
                frequency = min(1.0, 0.5 + margin * 2)
                ranges[matchup]["call"][hand] = {
                    "frequency": round(frequency, 2)
                }

    return ranges


def generate_4bet_ranges() -> Dict:
    """
    Generate 4-bet and call ranges when facing a 3-bet.

    Key principles:
    - Very tight 4-bet range (value heavy)
    - Some 4-bet bluffs with blockers (Ax suited)
    - Call with implied odds hands
    """
    ranges = {
        "_metadata": {
            "description": "4-bet and call ranges vs 3-bet",
            "sizing": "2.2-2.5x 3-bet size",
            "version": "2.0.0"
        }
    }

    # 4-bet is very tight - only premium hands
    # Format: (4bet_threshold, call_threshold)
    matchups = {
        # Opened from EP, facing 3-bet
        "UTG_vs_3bet": (0.92, 0.75),
        "MP_vs_3bet": (0.90, 0.70),
        # Opened from LP, facing 3-bet
        "CO_vs_3bet": (0.85, 0.65),
        "BTN_vs_3bet": (0.80, 0.60),
        "SB_vs_3bet": (0.85, 0.70),
    }

    # Special 4-bet bluff hands (Ax suited for blocker value)
    bluff_hands = {'A5s', 'A4s', 'A3s', 'A2s'}

    all_hands = get_all_hands()

    for matchup, (four_bet_thresh, call_thresh) in matchups.items():
        ranges[matchup] = {"4bet": {}, "call": {}}

        for hand in all_hands:
            strength = calculate_hand_strength(hand)

            # 4-bet value range
            if strength.score >= four_bet_thresh:
                ranges[matchup]["4bet"][hand] = {
                    "sizing_multiplier": 2.3,
                    "frequency": 1.0
                }
            # 4-bet bluff with Ax suited blockers
            elif hand in bluff_hands:
                # Lower frequency bluff based on position
                freq = 0.3 if "UTG" in matchup or "MP" in matchup else 0.5
                ranges[matchup]["4bet"][hand] = {
                    "sizing_multiplier": 2.3,
                    "frequency": freq
                }
            # Call range
            elif strength.score >= call_thresh:
                margin = strength.score - call_thresh
                frequency = min(1.0, 0.6 + margin)
                ranges[matchup]["call"][hand] = {
                    "frequency": round(frequency, 2)
                }

    return ranges


def save_ranges(ranges: Dict, filename: str, output_dir: Path):
    """Save ranges to JSON file."""
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(ranges, f, indent=2)
    print(f"Saved: {output_path}")

    # Count hands in range
    total_hands = sum(
        len([k for k in v.keys() if not k.startswith('_')])
        for k, v in ranges.items()
        if isinstance(v, dict) and not k.startswith('_')
    )
    print(f"  Total entries: {total_hands}")


def main():
    """Generate all preflop ranges."""
    print("GTO Preflop Range Generator")
    print("=" * 50)

    # Find output directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "database" / "preflop"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Generate and save open ranges
    print("Generating opening ranges...")
    open_ranges = generate_open_ranges()
    save_ranges(open_ranges, "open_ranges.json", output_dir)

    # Generate and save 3-bet ranges
    print("\nGenerating 3-bet ranges...")
    three_bet_ranges = generate_3bet_ranges()
    save_ranges(three_bet_ranges, "3bet_ranges.json", output_dir)

    # Generate and save 4-bet ranges
    print("\nGenerating 4-bet ranges...")
    four_bet_ranges = generate_4bet_ranges()
    save_ranges(four_bet_ranges, "4bet_ranges.json", output_dir)

    print("\n" + "=" * 50)
    print("Range generation complete!")
    print("\nTo verify, run:")
    print("  python -c \"from src.strategy.preflop_strategy import PreflopStrategy; ps = PreflopStrategy(); print('Loaded:', len(ps.open_ranges), 'positions')\"")


if __name__ == "__main__":
    main()
