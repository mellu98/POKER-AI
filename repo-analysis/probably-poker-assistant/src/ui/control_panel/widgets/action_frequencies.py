"""
Action frequencies widget for displaying GTO action percentages.

Shows fold/call/raise percentages with visual bars.
"""
from typing import Optional, Dict
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QProgressBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ..styles import COLORS, FONTS


class ActionBar(QWidget):
    """A single action with label and progress bar."""

    ACTION_COLORS = {
        'fold': '#E74C3C',      # Red
        'check': '#27AE60',     # Green
        'call': '#27AE60',      # Green
        'raise': '#3498DB',     # Blue
        'bet': '#3498DB',       # Blue
    }

    def __init__(self, action_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.action_name = action_name
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        # Action label
        self.label = QLabel(self.action_name.upper())
        self.label.setMinimumWidth(50)
        self.label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: {FONTS['size_small']}px;
            font-weight: bold;
        """)

        # Progress bar
        self.bar = QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setMaximum(100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.bar.setFormat("%v%")
        self.bar.setMinimumWidth(100)
        self.bar.setMaximumHeight(18)

        color = self.ACTION_COLORS.get(self.action_name.lower(), COLORS['accent_neutral'])
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background-color: {COLORS['bg_secondary']};
                text-align: center;
                font-size: {FONTS['size_small']}px;
                color: {COLORS['text_primary']};
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)

        # Percentage label
        self.pct_label = QLabel("0%")
        self.pct_label.setMinimumWidth(40)
        self.pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.pct_label.setStyleSheet(f"""
            color: {COLORS['text_stats']};
            font-size: {FONTS['size_normal']}px;
            font-weight: bold;
        """)

        layout.addWidget(self.label)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.pct_label)

    def set_value(self, percentage: float):
        """Set the percentage value (0-100)."""
        pct = max(0, min(100, int(percentage)))
        self.bar.setValue(pct)
        self.pct_label.setText(f"{pct}%")


class ActionFrequenciesWidget(QGroupBox):
    """
    Widget showing GTO action frequencies.

    Displays fold/check/call/raise percentages with visual bars.
    """

    def __init__(self, title: str = "GTO FREQUENCIES", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 15, 10, 10)

        # Create action bars
        self.fold_bar = ActionBar("Fold")
        self.check_bar = ActionBar("Check")
        self.call_bar = ActionBar("Call")
        self.raise_bar = ActionBar("Raise")
        self.bet_bar = ActionBar("Bet")

        layout.addWidget(self.fold_bar)
        layout.addWidget(self.check_bar)
        layout.addWidget(self.call_bar)
        layout.addWidget(self.raise_bar)
        layout.addWidget(self.bet_bar)

        # Additional info row
        self.info_layout = QHBoxLayout()
        self.spr_label = QLabel("SPR: -")
        self.spr_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: {FONTS['size_small']}px;
        """)
        self.position_label = QLabel("Position: -")
        self.position_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: {FONTS['size_small']}px;
        """)
        self.info_layout.addWidget(self.spr_label)
        self.info_layout.addStretch()
        self.info_layout.addWidget(self.position_label)

        layout.addLayout(self.info_layout)

    def update_frequencies(self,
                           fold: float = 0,
                           check: float = 0,
                           call: float = 0,
                           raise_pct: float = 0,
                           bet: float = 0,
                           spr: Optional[float] = None,
                           has_position: bool = True):
        """
        Update all action frequencies.

        Args:
            fold: Fold percentage (0-100)
            check: Check percentage (0-100)
            call: Call percentage (0-100)
            raise_pct: Raise percentage (0-100)
            bet: Bet percentage (0-100)
            spr: Stack-to-pot ratio
            has_position: Whether we have position
        """
        self.fold_bar.set_value(fold)
        self.check_bar.set_value(check)
        self.call_bar.set_value(call)
        self.raise_bar.set_value(raise_pct)
        self.bet_bar.set_value(bet)

        # Update SPR
        if spr is not None:
            if spr > 10:
                spr_text = "SPR: Deep (>10)"
            elif spr > 4:
                spr_text = f"SPR: {spr:.1f} (Medium)"
            else:
                spr_text = f"SPR: {spr:.1f} (Short)"
            self.spr_label.setText(spr_text)
        else:
            self.spr_label.setText("SPR: -")

        # Update position
        pos_text = "Position: IP" if has_position else "Position: OOP"
        self.position_label.setText(pos_text)

    def update_from_decision(self, decision):
        """
        Update from a Decision object.

        Args:
            decision: Decision dataclass from decision_engine
        """
        if decision is None or decision.action_frequencies is None:
            self.clear()
            return

        freqs = decision.action_frequencies
        self.update_frequencies(
            fold=freqs.fold,
            check=freqs.check,
            call=freqs.call,
            raise_pct=freqs.raise_,
            bet=freqs.bet,
            spr=decision.spr,
            has_position=decision.position_advantage
        )

    def clear(self):
        """Clear all frequencies to default state."""
        self.fold_bar.set_value(0)
        self.check_bar.set_value(0)
        self.call_bar.set_value(0)
        self.raise_bar.set_value(0)
        self.bet_bar.set_value(0)
        self.spr_label.setText("SPR: -")
        self.position_label.setText("Position: -")
