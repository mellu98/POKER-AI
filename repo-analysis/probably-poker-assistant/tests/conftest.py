"""
Shared pytest fixtures for Poker Assistant tests.

This module provides common test fixtures, mock objects, and test data
used across all test modules.
"""
import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.detection.game_state import GameState, BettingRound


# =============================================================================
# CARD DATA FIXTURES
# =============================================================================

@pytest.fixture
def all_ranks():
    """All card ranks."""
    return ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


@pytest.fixture
def all_suits():
    """All card suits."""
    return ['h', 'd', 'c', 's']


@pytest.fixture
def sample_hole_cards():
    """Common hole card combinations for testing."""
    return {
        'pocket_aces': ['Ah', 'As'],
        'pocket_kings': ['Kh', 'Ks'],
        'ace_king_suited': ['Ah', 'Kh'],
        'ace_king_offsuit': ['Ah', 'Kd'],
        'pocket_twos': ['2h', '2s'],
        'seven_two_offsuit': ['7h', '2d'],
        'suited_connectors': ['9h', '8h'],
        'medium_pair': ['8h', '8s'],
    }


@pytest.fixture
def sample_community_cards():
    """Sample community card boards for testing."""
    return {
        'dry_board': ['Ah', '7c', '2d'],
        'wet_board': ['Jh', 'Th', '9h'],
        'paired_board': ['Kh', 'Kd', '5c'],
        'straight_board': ['9h', '8d', '7c'],
        'flush_board': ['Ah', 'Kh', '7h', '3h', '2h'],
        'full_house_board': ['Ah', 'Ad', 'Ac', 'Kh', 'Kd'],
        'empty': [],
        'flop_only': ['Ah', 'Kd', '7c'],
        'turn': ['Ah', 'Kd', '7c', '2s'],
        'river': ['Ah', 'Kd', '7c', '2s', '9h'],
    }


@pytest.fixture
def known_hands():
    """Known hand evaluations for testing."""
    return [
        # (hole_cards, community_cards, expected_hand_type, description)
        (['Ah', 'Kh'], ['Qh', 'Jh', 'Th'], 'ROYAL_FLUSH', 'Royal Flush'),
        (['9h', '8h'], ['7h', '6h', '5h'], 'STRAIGHT_FLUSH', 'Straight Flush'),
        (['Ah', 'Ad'], ['Ac', 'As', 'Kh'], 'FOUR_OF_A_KIND', 'Four of a Kind'),
        (['Kh', 'Kd'], ['Ks', '2h', '2d'], 'FULL_HOUSE', 'Full House'),
        (['7h', '2h'], ['Ah', 'Kh', '9h'], 'FLUSH', 'Flush'),
        (['9c', '8d'], ['7h', '6s', '5c'], 'STRAIGHT', 'Straight'),
        (['Qs', 'Qh'], ['Qd', '5c', '3h'], 'THREE_OF_A_KIND', 'Three of a Kind'),
        (['Jh', 'Jd'], ['9h', '9c', '2s'], 'TWO_PAIR', 'Two Pair'),
        (['Ah', 'Kd'], ['Ac', '7h', '3c'], 'PAIR', 'Pair'),
        (['Ah', 'Kd'], ['Qc', '7h', '3c'], 'HIGH_CARD', 'High Card'),
    ]


# =============================================================================
# GAME STATE FIXTURES
# =============================================================================

@pytest.fixture
def preflop_game_state():
    """Game state for preflop scenario."""
    return GameState(
        hole_cards=['Ah', 'Kh'],
        community_cards=[],
        pot_size=15,
        stack_size=1000,
        current_bet=10,
        betting_round=BettingRound.PREFLOP,
        num_opponents=5
    )


@pytest.fixture
def flop_game_state():
    """Game state for flop scenario."""
    return GameState(
        hole_cards=['Ah', 'Kh'],
        community_cards=['Qh', '7d', '2c'],
        pot_size=50,
        stack_size=950,
        current_bet=0,
        betting_round=BettingRound.FLOP,
        num_opponents=2
    )


@pytest.fixture
def turn_game_state():
    """Game state for turn scenario."""
    return GameState(
        hole_cards=['Ah', 'Kh'],
        community_cards=['Qh', '7d', '2c', 'Jh'],
        pot_size=100,
        stack_size=900,
        current_bet=25,
        betting_round=BettingRound.TURN,
        num_opponents=1
    )


@pytest.fixture
def river_game_state():
    """Game state for river scenario."""
    return GameState(
        hole_cards=['Ah', 'Kh'],
        community_cards=['Qh', '7d', '2c', 'Jh', 'Th'],
        pot_size=200,
        stack_size=800,
        current_bet=50,
        betting_round=BettingRound.RIVER,
        num_opponents=1
    )


@pytest.fixture
def various_positions():
    """All poker positions."""
    return ['UTG', 'UTG+1', 'MP', 'CO', 'BTN', 'SB', 'BB']


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_window_handle():
    """Mock Windows window handle."""
    return 12345


@pytest.fixture
def mock_screen_image():
    """Mock screen capture as numpy array."""
    # Create a 1920x1080 RGB image
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def mock_card_image():
    """Mock card region image."""
    # Create a small card-sized image
    return np.zeros((100, 70, 3), dtype=np.uint8)


@pytest.fixture
def mock_text_region():
    """Mock text region for OCR testing."""
    # Create a small text region
    return np.ones((30, 100, 3), dtype=np.uint8) * 255


@pytest.fixture
def mock_win32gui():
    """Mock win32gui module."""
    with patch.dict('sys.modules', {'win32gui': MagicMock()}):
        mock = MagicMock()
        mock.FindWindow.return_value = 12345
        mock.GetWindowRect.return_value = (0, 0, 1920, 1080)
        mock.GetWindowText.return_value = "Poker just got started"
        mock.EnumWindows = MagicMock()
        yield mock


@pytest.fixture
def mock_mss():
    """Mock mss screen capture."""
    with patch('mss.mss') as mock:
        mock_instance = MagicMock()
        mock_instance.grab.return_value = MagicMock(
            rgb=b'\x00' * (1920 * 1080 * 3),
            width=1920,
            height=1080
        )
        mock.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_config_dir(tmp_path):
    """Temporary config directory for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_settings(test_config_dir):
    """Sample settings.json for testing."""
    settings = {
        "capture": {
            "window_title": "Poker just got",
            "interval_seconds": 1.0
        },
        "detection": {
            "card_confidence": 0.85
        },
        "strategy": {
            "style": "tight_aggressive",
            "aggression_factor": 1.2
        },
        "overlay": {
            "width": 320,
            "height": 240,
            "opacity": 0.9
        }
    }
    settings_file = test_config_dir / "settings.json"
    settings_file.write_text(json.dumps(settings, indent=2))
    return settings_file


@pytest.fixture
def sample_anchor_config(test_config_dir):
    """Sample anchor_config.json for testing."""
    anchor_config = {
        "active_anchor": "ignition_logo",
        "anchors": {
            "ignition_logo": {
                "template": "models/anchors/ignition_logo.png",
                "threshold": 0.8
            }
        },
        "regions": {
            "hole_cards": {"x": 100, "y": 200, "w": 150, "h": 100},
            "community_cards": {"x": 400, "y": 300, "w": 350, "h": 100},
            "pot_amount": {"x": 450, "y": 200, "w": 100, "h": 30},
            "player_stack": {"x": 100, "y": 350, "w": 100, "h": 30},
            "current_bet": {"x": 200, "y": 350, "w": 100, "h": 30}
        }
    }
    config_file = test_config_dir / "anchor_config.json"
    config_file.write_text(json.dumps(anchor_config, indent=2))
    return config_file


# =============================================================================
# EQUITY CALCULATION FIXTURES
# =============================================================================

@pytest.fixture
def equity_scenarios():
    """Known equity scenarios for testing."""
    return [
        # (hole_cards, opponent_range, board, expected_equity_range)
        (['Ah', 'As'], 'random', [], (80, 90)),  # AA vs random ~85%
        (['Ah', 'Kh'], 'random', [], (63, 70)),  # AKs vs random ~65%
        (['7h', '2d'], 'random', [], (30, 40)),  # 72o vs random ~35%
        (['Ah', 'As'], ['Kh', 'Ks'], [], (80, 85)),  # AA vs KK ~82%
    ]


@pytest.fixture
def pot_odds_scenarios():
    """Known pot odds scenarios for testing."""
    return [
        # (pot_size, bet_to_call, expected_odds, expected_required_equity)
        (100, 50, 3.0, 25.0),  # 3:1 odds, need 25% equity
        (100, 100, 2.0, 33.3),  # 2:1 odds, need 33.3% equity
        (100, 25, 5.0, 16.7),  # 5:1 odds, need 16.7% equity
        (50, 50, 2.0, 33.3),  # 2:1 odds, need 33.3% equity
    ]


# =============================================================================
# PYQT5 FIXTURES (for UI tests)
# =============================================================================

@pytest.fixture
def qapp(request):
    """Create QApplication for UI tests."""
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
    except ImportError:
        pytest.skip("PyQt5 not available")


@pytest.fixture
def mock_display_manager():
    """Mock DisplayManager for testing."""
    mock = MagicMock()
    mock.overlay = MagicMock()
    mock.update_overlay = MagicMock()
    return mock


# =============================================================================
# FILE SYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def temp_database_dir(tmp_path):
    """Temporary database directory for testing."""
    db_dir = tmp_path / "database"
    db_dir.mkdir()

    # Create preflop subdirectory
    preflop_dir = db_dir / "preflop"
    preflop_dir.mkdir()

    # Create learning subdirectory
    learning_dir = db_dir / "learning"
    learning_dir.mkdir()

    return db_dir


@pytest.fixture
def sample_preflop_ranges(temp_database_dir):
    """Sample preflop ranges for testing."""
    open_ranges = {
        "UTG": ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo"],
        "MP": ["AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs"],
        "CO": ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "AKs", "AKo", "AQs", "AQo"],
        "BTN": ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "AKs", "AKo", "AQs", "AQo", "AJs"],
        "SB": ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo"],
        "BB": []  # BB defends, doesn't open
    }

    ranges_file = temp_database_dir / "preflop" / "open_ranges.json"
    ranges_file.write_text(json.dumps(open_ranges, indent=2))
    return ranges_file


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_game_state(
    hole_cards: List[str],
    community_cards: List[str] = None,
    pot_size: float = 100,
    stack_size: float = 1000,
    current_bet: float = 0,
    num_opponents: int = 5
) -> GameState:
    """Helper function to create GameState objects."""
    if community_cards is None:
        community_cards = []

    # Determine betting round
    if len(community_cards) == 0:
        betting_round = BettingRound.PREFLOP
    elif len(community_cards) == 3:
        betting_round = BettingRound.FLOP
    elif len(community_cards) == 4:
        betting_round = BettingRound.TURN
    else:
        betting_round = BettingRound.RIVER

    return GameState(
        hole_cards=hole_cards,
        community_cards=community_cards,
        pot_size=pot_size,
        stack_size=stack_size,
        current_bet=current_bet,
        betting_round=betting_round,
        num_opponents=num_opponents
    )
