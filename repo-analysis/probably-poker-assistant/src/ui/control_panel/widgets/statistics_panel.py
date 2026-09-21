"""
Statistics panel widget for displaying poker statistics.

Shows win probability, equity, pot odds, required equity, EV, and hand strength.
"""
from typing import Optional
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ..styles import COLORS, FONTS, get_label_style


class StatRow(QWidget):
    """A single statistic row with label and value."""

    def __init__(self, label_text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui(label_text)

    def _setup_ui(self, label_text: str):
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        # Label
        self.label = QLabel(label_text + ":")
        self.label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_normal']}px;")
        self.label.setMinimumWidth(120)

        # Value
        self.value = QLabel("-")
        self.value.setStyleSheet(get_label_style('stats'))
        self.value.setAlignment(Qt.AlignRight)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.value)

    def set_value(self, text: str, color: Optional[str] = None):
        """Set the value text and optionally override color."""
        self.value.setText(text)
        if color:
            self.value.setStyleSheet(f"color: {color}; font-size: {FONTS['size_large']}px; font-weight: bold;")
        else:
            self.value.setStyleSheet(get_label_style('stats'))


class StatisticsPanel(QGroupBox):
    """
    Panel displaying poker statistics.

    Fields:
    - Win Probability (equity)
    - Pot Odds
    - Required Equity
    - Expected Value (EV)
    - Hand Strength
    """

    def __init__(self, title: str = "STATISTICS", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Create stat rows
        self.win_prob = StatRow("Win Probability")
        self.equity = StatRow("Equity")
        self.pot_odds = StatRow("Pot Odds")
        self.required_equity = StatRow("Required Equity")
        self.ev = StatRow("Expected Value")
        self.hand_strength = StatRow("Hand Strength")

        # Add all rows
        layout.addWidget(self.win_prob)
        layout.addWidget(self.equity)
        layout.addWidget(self.pot_odds)
        layout.addWidget(self.required_equity)
        layout.addWidget(self.ev)
        layout.addWidget(self.hand_strength)

    def update_stats(self,
                     win_probability: Optional[float] = None,
                     equity: Optional[float] = None,
                     pot_odds: Optional[float] = None,
                     ev: Optional[float] = None,
                     hand_strength: Optional[str] = None):
        """
        Update all statistics.

        Args:
            win_probability: Win probability percentage (0-100)
            equity: Equity percentage (0-100)
            pot_odds: Pot odds ratio (e.g., 2.5 for 2.5:1)
            ev: Expected value in chips
            hand_strength: Hand strength description string
        """
        # Win probability
        if win_probability is not None:
            self.win_prob.set_value(f"{win_probability:.1f}%", COLORS['text_stats'])
        else:
            self.win_prob.set_value("-")

        # Equity
        if equity is not None:
            self.equity.set_value(f"{equity:.1f}%", COLORS['text_stats'])
        else:
            self.equity.set_value("-")

        # Pot odds
        if pot_odds is not None and pot_odds > 0:
            self.pot_odds.set_value(f"{pot_odds:.1f}:1", COLORS['text_stats'])
            # Calculate required equity
            required = 100 / (pot_odds + 1)
            self.required_equity.set_value(f"{required:.1f}%", COLORS['text_stats'])
        else:
            self.pot_odds.set_value("-")
            self.required_equity.set_value("-")

        # Expected value
        if ev is not None:
            if ev > 0:
                color = COLORS['accent_positive']
                ev_text = f"+{ev:.2f}"
            elif ev < 0:
                color = COLORS['accent_negative']
                ev_text = f"{ev:.2f}"
            else:
                color = COLORS['accent_warning']
                ev_text = "0.00"
            self.ev.set_value(f"{ev_text} chips", color)
        else:
            self.ev.set_value("-")

        # Hand strength
        if hand_strength:
            self.hand_strength.set_value(hand_strength, COLORS['text_stats'])
        else:
            self.hand_strength.set_value("-")

    def clear_stats(self):
        """Clear all statistics to default state."""
        self.win_prob.set_value("-")
        self.equity.set_value("-")
        self.pot_odds.set_value("-")
        self.required_equity.set_value("-")
        self.ev.set_value("-")
        self.hand_strength.set_value("-")

    def update_from_decision(self, decision, game_state=None):
        """
        Update stats from Decision and GameState objects.

        Args:
            decision: Decision dataclass from decision_engine
            game_state: Optional GameState for EV calculation
        """
        if decision is None:
            self.clear_stats()
            return

        # Calculate EV if we have game state
        ev = None
        if game_state and decision.pot_odds and decision.pot_odds > 0:
            try:
                pot = game_state.pot_size or 0
                bet = game_state.current_bet or 0
                equity_decimal = decision.equity / 100

                if bet > 0:
                    win_amount = pot + bet
                    lose_amount = bet
                    ev = (win_amount * equity_decimal) - (lose_amount * (1 - equity_decimal))
            except Exception:
                ev = None

        self.update_stats(
            win_probability=decision.equity,
            equity=decision.equity,
            pot_odds=decision.pot_odds,
            ev=ev,
            hand_strength=decision.hand_evaluation.description if decision.hand_evaluation else None
        )
