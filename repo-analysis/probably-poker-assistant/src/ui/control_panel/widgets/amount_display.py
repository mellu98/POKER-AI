"""
Amount display widget for pot, stack, and blind levels.

Provides input fields for manually entering or displaying detected amounts.
"""
from typing import Optional
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QWidget
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator

from ..styles import COLORS, FONTS, get_line_edit_style


class AmountField(QWidget):
    """A single amount input field with label."""
    value_changed = pyqtSignal(float)

    def __init__(self, label_text: str, placeholder: str = "0", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui(label_text, placeholder)
        self._connect_signals()

    def _setup_ui(self, label_text: str, placeholder: str):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Label
        self.label = QLabel(label_text)
        self.label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_small']}px;")

        # Input field
        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.setStyleSheet(get_line_edit_style())
        self.input.setMaximumWidth(100)
        self.input.setAlignment(Qt.AlignRight)

        # Validator for numeric input
        validator = QDoubleValidator(0.0, 1000000.0, 2)
        self.input.setValidator(validator)

        layout.addWidget(self.label)
        layout.addWidget(self.input)

    def _connect_signals(self):
        """Connect internal signals."""
        self.input.textChanged.connect(self._on_text_changed)

    def _on_text_changed(self, text: str):
        """Handle text change."""
        try:
            value = float(text) if text else 0.0
            self.value_changed.emit(value)
        except ValueError:
            pass

    def get_value(self) -> float:
        """Get current value."""
        try:
            return float(self.input.text()) if self.input.text() else 0.0
        except ValueError:
            return 0.0

    def set_value(self, value: Optional[float]):
        """Set the value."""
        if value is not None:
            self.input.setText(f"{value:.2f}")
        else:
            self.input.clear()

    def set_enabled(self, enabled: bool):
        """Enable or disable the field."""
        self.input.setEnabled(enabled)


class BlindField(QWidget):
    """Blind level input showing SB/BB."""
    blinds_changed = pyqtSignal(float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Label
        self.label = QLabel("BLINDS")
        self.label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_small']}px;")

        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)

        self.sb_input = QLineEdit()
        self.sb_input.setPlaceholderText("SB")
        self.sb_input.setStyleSheet(get_line_edit_style())
        self.sb_input.setMaximumWidth(50)
        self.sb_input.setAlignment(Qt.AlignRight)
        validator = QDoubleValidator(0.0, 10000.0, 2)
        self.sb_input.setValidator(validator)

        separator = QLabel("/")
        separator.setStyleSheet(f"color: {COLORS['text_primary']};")

        self.bb_input = QLineEdit()
        self.bb_input.setPlaceholderText("BB")
        self.bb_input.setStyleSheet(get_line_edit_style())
        self.bb_input.setMaximumWidth(50)
        self.bb_input.setAlignment(Qt.AlignRight)
        self.bb_input.setValidator(validator)

        input_layout.addWidget(self.sb_input)
        input_layout.addWidget(separator)
        input_layout.addWidget(self.bb_input)

        layout.addWidget(self.label)
        layout.addLayout(input_layout)

    def _connect_signals(self):
        """Connect internal signals."""
        self.sb_input.textChanged.connect(self._on_changed)
        self.bb_input.textChanged.connect(self._on_changed)

    def _on_changed(self):
        """Handle change."""
        sb = self.get_sb()
        bb = self.get_bb()
        self.blinds_changed.emit(sb, bb)

    def get_sb(self) -> float:
        """Get small blind value."""
        try:
            return float(self.sb_input.text()) if self.sb_input.text() else 0.0
        except ValueError:
            return 0.0

    def get_bb(self) -> float:
        """Get big blind value."""
        try:
            return float(self.bb_input.text()) if self.bb_input.text() else 0.0
        except ValueError:
            return 0.0

    def set_blinds(self, sb: Optional[float], bb: Optional[float]):
        """Set blind values."""
        if sb is not None:
            self.sb_input.setText(f"{sb:.2f}")
        else:
            self.sb_input.clear()

        if bb is not None:
            self.bb_input.setText(f"{bb:.2f}")
        else:
            self.bb_input.clear()

    def set_enabled(self, enabled: bool):
        """Enable or disable the fields."""
        self.sb_input.setEnabled(enabled)
        self.bb_input.setEnabled(enabled)


class AmountDisplayWidget(QGroupBox):
    """
    Widget for displaying/editing pot, stack, and blind amounts.
    """
    amounts_changed = pyqtSignal(dict)  # Emits dict with pot, stack, sb, bb

    def __init__(self, title: str = "AMOUNTS", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout(self)
        layout.setSpacing(15)

        # Pot field
        self.pot_field = AmountField("POT")
        self.pot_field.set_value(0)

        # Stack field
        self.stack_field = AmountField("STACK")
        self.stack_field.set_value(0)

        # Blinds field
        self.blinds_field = BlindField()
        self.blinds_field.set_blinds(0.5, 1.0)

        layout.addWidget(self.pot_field)
        layout.addWidget(self.stack_field)
        layout.addWidget(self.blinds_field)
        layout.addStretch()

    def _connect_signals(self):
        """Connect internal signals."""
        self.pot_field.value_changed.connect(self._on_changed)
        self.stack_field.value_changed.connect(self._on_changed)
        self.blinds_field.blinds_changed.connect(self._on_changed)

    def _on_changed(self, *args):
        """Handle any value change."""
        self.amounts_changed.emit(self.get_amounts())

    def get_amounts(self) -> dict:
        """Get all amounts as dictionary."""
        return {
            'pot': self.pot_field.get_value(),
            'stack': self.stack_field.get_value(),
            'sb': self.blinds_field.get_sb(),
            'bb': self.blinds_field.get_bb(),
        }

    def set_amounts(self,
                    pot: Optional[float] = None,
                    stack: Optional[float] = None,
                    sb: Optional[float] = None,
                    bb: Optional[float] = None):
        """Set amounts."""
        if pot is not None:
            self.pot_field.set_value(pot)
        if stack is not None:
            self.stack_field.set_value(stack)
        if sb is not None or bb is not None:
            self.blinds_field.set_blinds(sb, bb)

    def update_from_game_state(self, game_state):
        """
        Update from GameState object.

        Args:
            game_state: GameState dataclass from game_state module
        """
        if game_state is None:
            return

        self.pot_field.set_value(game_state.pot_size)
        self.stack_field.set_value(game_state.stack_size)

    def set_enabled(self, enabled: bool):
        """Enable or disable all fields."""
        self.pot_field.set_enabled(enabled)
        self.stack_field.set_enabled(enabled)
        self.blinds_field.set_enabled(enabled)
