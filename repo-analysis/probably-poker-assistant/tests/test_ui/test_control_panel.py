"""
Comprehensive tests for UI components.

Tests widget rendering, functionality, and signal emissions.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if PyQt5 is available
try:
    from PyQt5.QtWidgets import QApplication
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Skip UI tests if PyQt5 is not available or on non-Windows
pytestmark = pytest.mark.skipif(
    not PYQT5_AVAILABLE or sys.platform != 'win32',
    reason="UI tests require PyQt5 and Windows display"
)


class TestCardSelectorWidget:
    """Test suite for CardSelectorWidget."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def card_selector(self, app):
        """Create CardSelectorWidget instance."""
        from src.ui.control_panel.widgets.card_selector import CardSelectorWidget
        return CardSelectorWidget()

    @pytest.mark.unit
    def test_widget_creation(self, card_selector):
        """Test widget initializes correctly."""
        assert card_selector is not None
        assert card_selector.rank_combo is not None
        assert card_selector.suit_combo is not None

    @pytest.mark.unit
    def test_initial_state(self, card_selector):
        """Test initial state is empty."""
        assert card_selector.get_card() == ''
        assert not card_selector.is_valid()

    @pytest.mark.unit
    def test_set_card_ace_hearts(self, card_selector):
        """Test setting a card."""
        card_selector.set_card('Ah')
        assert card_selector.get_card() == 'Ah'
        assert card_selector.is_valid()

    @pytest.mark.unit
    def test_set_card_king_spades(self, card_selector):
        """Test setting king of spades."""
        card_selector.set_card('Ks')
        assert card_selector.get_card() == 'Ks'

    @pytest.mark.unit
    def test_set_card_lowercase(self, card_selector):
        """Test setting card with lowercase input."""
        card_selector.set_card('qd')
        assert card_selector.get_card() == 'Qd'

    @pytest.mark.unit
    def test_clear_card(self, card_selector):
        """Test clearing card selection."""
        card_selector.set_card('Ah')
        card_selector.clear()
        assert card_selector.get_card() == ''

    @pytest.mark.unit
    def test_signal_emission(self, card_selector):
        """Test signal is emitted on card change."""
        signal_received = []
        card_selector.card_changed.connect(lambda x: signal_received.append(x))

        card_selector.set_card('Th')
        # Signal should be emitted
        assert len(signal_received) >= 1

    @pytest.mark.unit
    def test_all_ranks(self, card_selector):
        """Test all rank options exist."""
        expected_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', '-']
        assert card_selector.RANKS == expected_ranks

    @pytest.mark.unit
    def test_all_suits(self, card_selector):
        """Test all suit options exist."""
        # Hearts, Diamonds, Clubs, Spades, None
        assert len(card_selector.SUITS) == 5


class TestHandSelectorWidget:
    """Test suite for HandSelectorWidget (hole cards)."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def hand_selector(self, app):
        """Create HandSelectorWidget instance."""
        from src.ui.control_panel.widgets.card_selector import HandSelectorWidget
        return HandSelectorWidget("Test Hand")

    @pytest.mark.unit
    def test_widget_creation(self, hand_selector):
        """Test widget initializes correctly."""
        assert hand_selector is not None
        assert hand_selector.card1 is not None
        assert hand_selector.card2 is not None

    @pytest.mark.unit
    def test_initial_auto_detect(self, hand_selector):
        """Test auto-detect is enabled by default."""
        assert hand_selector.is_auto_detect()

    @pytest.mark.unit
    def test_set_hand(self, hand_selector):
        """Test setting a hand."""
        hand_selector.set_auto_detect(False)
        hand_selector.set_hand(['Ah', 'Ks'])
        assert hand_selector.get_hand() == ['Ah', 'Ks']

    @pytest.mark.unit
    def test_set_partial_hand(self, hand_selector):
        """Test setting only one card."""
        hand_selector.set_auto_detect(False)
        hand_selector.set_hand(['Ah'])
        hand = hand_selector.get_hand()
        assert 'Ah' in hand

    @pytest.mark.unit
    def test_toggle_auto_detect(self, hand_selector):
        """Test toggling auto-detect mode."""
        hand_selector.set_auto_detect(False)
        assert not hand_selector.is_auto_detect()
        hand_selector.set_auto_detect(True)
        assert hand_selector.is_auto_detect()

    @pytest.mark.unit
    def test_cards_disabled_in_auto_mode(self, hand_selector):
        """Test cards are disabled when auto-detect is on."""
        # Need to toggle off first, then on to trigger the disable
        hand_selector.set_auto_detect(False)  # Enable cards first
        hand_selector.set_auto_detect(True)   # Then disable
        # Card selectors should be disabled
        assert not hand_selector.card1.isEnabled()
        assert not hand_selector.card2.isEnabled()

    @pytest.mark.unit
    def test_cards_enabled_in_manual_mode(self, hand_selector):
        """Test cards are enabled when auto-detect is off."""
        hand_selector.set_auto_detect(False)
        # Card selectors should be enabled
        assert hand_selector.card1.isEnabled()
        assert hand_selector.card2.isEnabled()


class TestCommunitySelectorWidget:
    """Test suite for CommunitySelectorWidget."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def community_selector(self, app):
        """Create CommunitySelectorWidget instance."""
        from src.ui.control_panel.widgets.card_selector import CommunitySelectorWidget
        return CommunitySelectorWidget("Test Community")

    @pytest.mark.unit
    def test_widget_creation(self, community_selector):
        """Test widget initializes correctly."""
        assert community_selector is not None
        assert community_selector.flop1 is not None
        assert community_selector.flop2 is not None
        assert community_selector.flop3 is not None
        assert community_selector.turn is not None
        assert community_selector.river is not None

    @pytest.mark.unit
    def test_initial_auto_detect(self, community_selector):
        """Test auto-detect is enabled by default."""
        assert community_selector.is_auto_detect()

    @pytest.mark.unit
    def test_set_flop(self, community_selector):
        """Test setting flop cards."""
        community_selector.set_auto_detect(False)
        community_selector.set_community(['Ah', 'Ks', 'Qd'])
        cards = community_selector.get_community()
        assert len(cards) == 3
        assert 'Ah' in cards
        assert 'Ks' in cards
        assert 'Qd' in cards

    @pytest.mark.unit
    def test_set_turn(self, community_selector):
        """Test setting turn card."""
        community_selector.set_auto_detect(False)
        community_selector.set_community(['Ah', 'Ks', 'Qd', '7c'])
        cards = community_selector.get_community()
        assert len(cards) == 4

    @pytest.mark.unit
    def test_set_river(self, community_selector):
        """Test setting river card."""
        community_selector.set_auto_detect(False)
        community_selector.set_community(['Ah', 'Ks', 'Qd', '7c', '2s'])
        cards = community_selector.get_community()
        assert len(cards) == 5


class TestControlPanelWindow:
    """Test suite for ControlPanelWindow."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def control_panel(self, app):
        """Create ControlPanelWindow instance."""
        from src.ui.control_panel.main_panel import ControlPanelWindow
        return ControlPanelWindow()

    @pytest.mark.unit
    def test_window_creation(self, control_panel):
        """Test window initializes correctly."""
        assert control_panel is not None
        assert control_panel.windowTitle() == "POKER ASSISTANT CONTROL PANEL"

    @pytest.mark.unit
    def test_initial_state_not_running(self, control_panel):
        """Test initial state is not running."""
        assert not control_panel.is_running()

    @pytest.mark.unit
    def test_set_running(self, control_panel):
        """Test setting running state."""
        control_panel.set_running(True)
        assert control_panel.is_running()
        control_panel.set_running(False)
        assert not control_panel.is_running()

    @pytest.mark.unit
    def test_has_action_display(self, control_panel):
        """Test action display widget exists."""
        assert control_panel.action_display is not None

    @pytest.mark.unit
    def test_has_hand_selector(self, control_panel):
        """Test hand selector widget exists."""
        assert control_panel.hand_selector is not None

    @pytest.mark.unit
    def test_has_community_selector(self, control_panel):
        """Test community selector widget exists."""
        assert control_panel.community_selector is not None

    @pytest.mark.unit
    def test_has_statistics_panel(self, control_panel):
        """Test statistics panel widget exists."""
        assert control_panel.statistics_panel is not None

    @pytest.mark.unit
    def test_has_calibration_buttons(self, control_panel):
        """Test calibration buttons widget exists."""
        assert control_panel.calibration_buttons is not None

    @pytest.mark.unit
    def test_has_start_stop_button(self, control_panel):
        """Test start/stop button exists."""
        assert control_panel.start_stop_btn is not None

    @pytest.mark.unit
    def test_start_stop_signal(self, control_panel):
        """Test start/stop signal emission."""
        signals_received = []
        control_panel.start_stop_toggled.connect(lambda x: signals_received.append(x))

        control_panel.start_stop_btn.click()
        assert len(signals_received) == 1
        assert signals_received[0] == True  # First click should start

    @pytest.mark.unit
    def test_calibration_signal(self, control_panel):
        """Test calibration request signal."""
        signals_received = []
        control_panel.calibration_requested.connect(lambda x: signals_received.append(x))

        # Click calibration buttons
        control_panel.calibration_buttons.auto_resize_clicked.emit()
        control_panel.calibration_buttons.set_anchor_clicked.emit()
        control_panel.calibration_buttons.reset_clicked.emit()

        assert 'auto_resize' in signals_received
        assert 'set_anchor' in signals_received
        assert 'reset' in signals_received


class TestCalibrationButtonsWidget:
    """Test suite for CalibrationButtonsWidget."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def calibration_buttons(self, app):
        """Create CalibrationButtonsWidget instance."""
        from src.ui.control_panel.widgets.calibration_buttons import CalibrationButtonsWidget
        return CalibrationButtonsWidget("Test Calibration")

    @pytest.mark.unit
    def test_widget_creation(self, calibration_buttons):
        """Test widget initializes correctly."""
        assert calibration_buttons is not None

    @pytest.mark.unit
    def test_has_auto_resize_signal(self, calibration_buttons):
        """Test auto resize signal exists."""
        assert hasattr(calibration_buttons, 'auto_resize_clicked')

    @pytest.mark.unit
    def test_has_set_anchor_signal(self, calibration_buttons):
        """Test set anchor signal exists."""
        assert hasattr(calibration_buttons, 'set_anchor_clicked')

    @pytest.mark.unit
    def test_has_config_regions_signal(self, calibration_buttons):
        """Test config regions signal exists."""
        assert hasattr(calibration_buttons, 'config_regions_clicked')

    @pytest.mark.unit
    def test_has_reset_signal(self, calibration_buttons):
        """Test reset signal exists."""
        assert hasattr(calibration_buttons, 'reset_clicked')


class TestStatisticsPanel:
    """Test suite for StatisticsPanel."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def statistics_panel(self, app):
        """Create StatisticsPanel instance."""
        from src.ui.control_panel.widgets.statistics_panel import StatisticsPanel
        return StatisticsPanel("Test Statistics")

    @pytest.mark.unit
    def test_widget_creation(self, statistics_panel):
        """Test widget initializes correctly."""
        assert statistics_panel is not None

    @pytest.mark.unit
    def test_update_from_decision(self, statistics_panel):
        """Test updating from decision object."""
        # Create mock decision with all required attributes
        mock_decision = MagicMock()
        mock_decision.equity = 65.5
        mock_decision.confidence = 80
        mock_decision.reasoning = ["Strong hand", "Good position"]
        mock_decision.pot_odds = 3.0  # Must be a number for comparison
        mock_decision.hand_evaluation = MagicMock()
        mock_decision.hand_evaluation.description = "Pair of Aces"

        mock_game_state = MagicMock()
        mock_game_state.pot_size = 100
        mock_game_state.current_bet = 20

        # Should not raise
        statistics_panel.update_from_decision(mock_decision, mock_game_state)


class TestActionDisplayWidget:
    """Test suite for ActionDisplayWidget."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def action_display(self, app):
        """Create ActionDisplayWidget instance."""
        from src.ui.control_panel.widgets.action_display import ActionDisplayWidget
        return ActionDisplayWidget()

    @pytest.mark.unit
    def test_widget_creation(self, action_display):
        """Test widget initializes correctly."""
        assert action_display is not None

    @pytest.mark.unit
    def test_update_from_decision(self, action_display):
        """Test updating from decision object."""
        mock_decision = MagicMock()
        mock_decision.action = "raise"
        mock_decision.amount_bb = 3.0
        mock_decision.confidence = 85

        # Should not raise
        action_display.update_from_decision(mock_decision)


class TestFirstRunWizard:
    """Test suite for FirstRunWizard."""

    @pytest.fixture
    def app(self):
        """Create QApplication for testing."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def wizard(self, app):
        """Create FirstRunWizard instance."""
        from src.ui.control_panel.first_run_wizard import FirstRunWizard
        return FirstRunWizard()

    @pytest.mark.unit
    def test_wizard_creation(self, wizard):
        """Test wizard initializes correctly."""
        assert wizard is not None
        assert wizard.windowTitle() == "Welcome to Poker Assistant"

    @pytest.mark.unit
    def test_wizard_has_options(self, wizard):
        """Test wizard has all option buttons."""
        assert wizard.live_radio is not None
        assert wizard.test_radio is not None
        assert wizard.skip_radio is not None

    @pytest.mark.unit
    def test_live_radio_default_selected(self, wizard):
        """Test live calibration is default selection."""
        assert wizard.live_radio.isChecked()

    @pytest.mark.unit
    def test_wizard_signals_exist(self, wizard):
        """Test wizard has required signals."""
        assert hasattr(wizard, 'calibration_requested')
        assert hasattr(wizard, 'test_mode_selected')
        assert hasattr(wizard, 'setup_skipped')


class TestNeedsFirstRunSetup:
    """Test suite for needs_first_run_setup function."""

    @pytest.mark.unit
    def test_needs_setup_no_config(self, tmp_path):
        """Test returns True when no config exists."""
        from src.ui.control_panel.first_run_wizard import needs_first_run_setup
        assert needs_first_run_setup(tmp_path) == True

    @pytest.mark.unit
    def test_needs_setup_empty_config(self, tmp_path):
        """Test returns True when config has no active anchor."""
        from src.ui.control_panel.first_run_wizard import needs_first_run_setup
        import json

        config_file = tmp_path / 'anchor_config.json'
        config_file.write_text(json.dumps({"active_anchor": None, "regions": {}}))

        assert needs_first_run_setup(tmp_path) == True

    @pytest.mark.unit
    def test_no_setup_needed_with_config(self, tmp_path):
        """Test returns False when config is complete."""
        from src.ui.control_panel.first_run_wizard import needs_first_run_setup
        import json

        config_file = tmp_path / 'anchor_config.json'
        config_file.write_text(json.dumps({
            "active_anchor": "test_anchor",
            "regions": {"hole_cards": {"off_x": 0, "off_y": 0, "w": 100, "h": 50}}
        }))

        assert needs_first_run_setup(tmp_path) == False
