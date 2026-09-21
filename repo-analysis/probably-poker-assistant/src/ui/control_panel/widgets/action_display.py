"""
Action display widget for showing suggested poker actions.

Large, color-coded display of the recommended action (fold/call/raise).
"""
from typing import Optional
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ..styles import COLORS, FONTS, get_action_frame_style


class ActionDisplayWidget(QFrame):
    """
    Large action suggestion display with color coding.

    Color scheme:
    - FOLD: Red background
    - CALL/CHECK: Green background
    - RAISE/BET: Blue background
    """

    ACTION_COLORS = {
        'fold': COLORS['accent_negative'],
        'call': COLORS['accent_positive'],
        'check': COLORS['accent_positive'],
        'raise': COLORS['accent_neutral'],
        'bet': COLORS['accent_neutral'],
        'all-in': COLORS['accent_warning'],
    }

    def __init__(self, parent: Optional[QFrame] = None):
        super().__init__(parent)
        self._action = ""
        self._amount = None
        self._confidence = 0.0
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI components."""
        self.setMinimumHeight(80)
        self._update_style('neutral')

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(15, 10, 15, 10)

        # Main action label
        self.action_label = QLabel("WAITING...")
        self.action_label.setAlignment(Qt.AlignCenter)
        action_font = QFont(FONTS['family'], FONTS['size_action'], QFont.Bold)
        self.action_label.setFont(action_font)
        self.action_label.setStyleSheet(f"color: {COLORS['text_primary']}; background: transparent;")

        # Confidence label (smaller, below action)
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        conf_font = QFont(FONTS['family'], FONTS['size_small'])
        self.confidence_label.setFont(conf_font)
        self.confidence_label.setStyleSheet(f"color: {COLORS['text_secondary']}; background: transparent;")

        layout.addWidget(self.action_label)
        layout.addWidget(self.confidence_label)

    def _update_style(self, action_type: str):
        """Update frame style based on action type."""
        self.setStyleSheet(get_action_frame_style(action_type))

    def set_action(self, action: str, amount_bb: Optional[float] = None, confidence: float = 0.0):
        """
        Set the displayed action.

        Args:
            action: Action type (fold, call, check, raise, bet)
            amount_bb: Optional amount in big blinds
            confidence: Confidence score 0-1
        """
        self._action = action.lower()
        self._amount = amount_bb
        self._confidence = confidence

        # Update style
        self._update_style(self._action)

        # Build action text
        action_text = action.upper()
        if amount_bb is not None and amount_bb > 0:
            action_text += f" {amount_bb:.1f} BB"

        self.action_label.setText(action_text)

        # Update confidence
        if confidence > 0:
            conf_percent = confidence * 100
            self.confidence_label.setText(f"{conf_percent:.0f}% confidence")
        else:
            self.confidence_label.setText("")

    def set_waiting(self):
        """Set display to waiting state."""
        self._action = ""
        self._amount = None
        self._confidence = 0.0

        self._update_style('neutral')
        self.action_label.setText("WAITING...")
        self.confidence_label.setText("")

    def set_error(self, message: str = "ERROR"):
        """Set display to error state."""
        self._action = "error"
        self._amount = None
        self._confidence = 0.0

        self._update_style('fold')  # Red for error
        self.action_label.setText(message.upper())
        self.confidence_label.setText("")

    def get_action(self) -> str:
        """Get current action string."""
        return self._action

    def get_amount(self) -> Optional[float]:
        """Get current amount in BB."""
        return self._amount

    def get_confidence(self) -> float:
        """Get current confidence."""
        return self._confidence

    def update_from_decision(self, decision):
        """
        Update display from a Decision object.

        Args:
            decision: Decision dataclass from decision_engine
        """
        if decision is None:
            self.set_waiting()
            return

        self.set_action(
            action=decision.action,
            amount_bb=decision.amount_bb,
            confidence=decision.confidence
        )
