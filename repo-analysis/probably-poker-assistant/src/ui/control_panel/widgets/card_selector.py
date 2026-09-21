"""
Card selector widgets for manual card input.

Provides dropdown-based card selection with rank and suit selectors.
"""
from typing import Optional, List
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QLabel,
    QGroupBox, QCheckBox, QGridLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

from ..styles import COLORS, FONTS, get_combo_box_style, get_checkbox_style


class CardSelectorWidget(QWidget):
    """
    Single card selector with rank and suit dropdowns.

    Emits card_changed signal with card string like "Ah", "Ks", etc.
    """
    card_changed = pyqtSignal(str)  # Emits card notation like "Ah"

    RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', '-']
    SUITS = ['\u2665', '\u2666', '\u2663', '\u2660', '-']  # Hearts, Diamonds, Clubs, Spades
    SUIT_MAP = {'\u2665': 'h', '\u2666': 'd', '\u2663': 'c', '\u2660': 's', '-': ''}

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # Rank selector
        self.rank_combo = QComboBox()
        self.rank_combo.addItems(self.RANKS)
        self.rank_combo.setCurrentText('-')
        self.rank_combo.setStyleSheet(get_combo_box_style())
        self.rank_combo.setFixedWidth(55)

        # Suit selector
        self.suit_combo = QComboBox()
        self.suit_combo.addItems(self.SUITS)
        self.suit_combo.setCurrentText('-')
        self.suit_combo.setStyleSheet(get_combo_box_style())
        self.suit_combo.setFixedWidth(55)

        # Update suit colors
        self._update_suit_colors()

        layout.addWidget(self.rank_combo)
        layout.addWidget(self.suit_combo)

    def _connect_signals(self):
        """Connect internal signals."""
        self.rank_combo.currentTextChanged.connect(self._on_selection_changed)
        self.suit_combo.currentTextChanged.connect(self._on_selection_changed)
        self.suit_combo.currentTextChanged.connect(self._update_suit_colors)

    def _update_suit_colors(self):
        """Update suit dropdown text color based on selected suit."""
        suit = self.suit_combo.currentText()
        if suit in ['\u2665', '\u2666']:  # Hearts, Diamonds
            color = COLORS['card_red']
        else:
            color = COLORS['text_primary']

        style = get_combo_box_style()
        # Override just the text color
        self.suit_combo.setStyleSheet(style + f"QComboBox {{ color: {color}; }}")

    def _on_selection_changed(self):
        """Handle selection change and emit signal."""
        card = self.get_card()
        self.card_changed.emit(card)

    def get_card(self) -> str:
        """
        Get current card as notation string.

        Returns:
            Card notation like "Ah", "Ks", or "" if not fully selected
        """
        rank = self.rank_combo.currentText()
        suit = self.suit_combo.currentText()

        if rank == '-' or suit == '-':
            return ''

        suit_char = self.SUIT_MAP.get(suit, '')
        return f"{rank}{suit_char}"

    def set_card(self, card: str):
        """
        Set card from notation string.

        Args:
            card: Card notation like "Ah", "Ks", or "" to clear
        """
        if not card or len(card) < 2:
            self.rank_combo.setCurrentText('-')
            self.suit_combo.setCurrentText('-')
            return

        rank = card[0].upper()
        suit_char = card[1].lower()

        # Map suit character to symbol
        suit_reverse_map = {'h': '\u2665', 'd': '\u2666', 'c': '\u2663', 's': '\u2660'}
        suit = suit_reverse_map.get(suit_char, '-')

        if rank in self.RANKS:
            self.rank_combo.setCurrentText(rank)
        if suit in self.SUITS:
            self.suit_combo.setCurrentText(suit)

    def clear(self):
        """Clear the selection."""
        self.rank_combo.setCurrentText('-')
        self.suit_combo.setCurrentText('-')

    def is_valid(self) -> bool:
        """Check if a valid card is selected."""
        return self.get_card() != ''


class HandSelectorWidget(QGroupBox):
    """
    Two-card hand selector for hole cards.

    Provides two CardSelectorWidget instances with auto-detect checkbox.
    """
    hand_changed = pyqtSignal(list)  # Emits list of card strings

    def __init__(self, title: str = "YOUR HAND", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._auto_detect = True
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Card selectors row
        cards_layout = QHBoxLayout()
        self.card1 = CardSelectorWidget()
        self.card2 = CardSelectorWidget()
        cards_layout.addWidget(self.card1)
        cards_layout.addWidget(self.card2)
        cards_layout.addStretch()

        # Auto-detect checkbox
        self.auto_check = QCheckBox("Auto Detect")
        self.auto_check.setChecked(True)
        self.auto_check.setStyleSheet(get_checkbox_style())

        layout.addLayout(cards_layout)
        layout.addWidget(self.auto_check)

    def _connect_signals(self):
        """Connect internal signals."""
        self.card1.card_changed.connect(self._on_card_changed)
        self.card2.card_changed.connect(self._on_card_changed)
        self.auto_check.toggled.connect(self._on_auto_toggled)

    def _on_card_changed(self):
        """Handle card selection change."""
        if not self._auto_detect:
            self.hand_changed.emit(self.get_hand())

    def _on_auto_toggled(self, checked: bool):
        """Handle auto-detect toggle."""
        self._auto_detect = checked
        self.card1.setEnabled(not checked)
        self.card2.setEnabled(not checked)
        if not checked:
            self.hand_changed.emit(self.get_hand())

    def get_hand(self) -> List[str]:
        """
        Get current hand as list of card strings.

        Returns:
            List of card notations, e.g., ["Ah", "Ks"]
        """
        cards = []
        c1 = self.card1.get_card()
        c2 = self.card2.get_card()
        if c1:
            cards.append(c1)
        if c2:
            cards.append(c2)
        return cards

    def set_hand(self, cards: List[str]):
        """
        Set hand from list of card strings.

        Args:
            cards: List of card notations
        """
        if len(cards) >= 1:
            self.card1.set_card(cards[0])
        else:
            self.card1.clear()

        if len(cards) >= 2:
            self.card2.set_card(cards[1])
        else:
            self.card2.clear()

    def is_auto_detect(self) -> bool:
        """Check if auto-detect is enabled."""
        return self._auto_detect

    def set_auto_detect(self, enabled: bool):
        """Set auto-detect mode."""
        self.auto_check.setChecked(enabled)


class CommunitySelectorWidget(QGroupBox):
    """
    Community cards selector (flop, turn, river).

    Provides five CardSelectorWidget instances organized as 3+1+1.
    """
    community_changed = pyqtSignal(list)  # Emits list of card strings

    def __init__(self, title: str = "COMMUNITY", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._auto_detect = True
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Labels row
        labels_layout = QHBoxLayout()
        flop_label = QLabel("FLOP")
        flop_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_small']}px;")
        turn_label = QLabel("TURN")
        turn_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_small']}px;")
        river_label = QLabel("RIVER")
        river_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {FONTS['size_small']}px;")

        labels_layout.addWidget(flop_label)
        labels_layout.addSpacing(90)  # Align with card selectors
        labels_layout.addWidget(turn_label)
        labels_layout.addSpacing(30)
        labels_layout.addWidget(river_label)
        labels_layout.addStretch()

        # Cards row
        cards_layout = QHBoxLayout()
        self.flop1 = CardSelectorWidget()
        self.flop2 = CardSelectorWidget()
        self.flop3 = CardSelectorWidget()
        self.turn = CardSelectorWidget()
        self.river = CardSelectorWidget()

        cards_layout.addWidget(self.flop1)
        cards_layout.addWidget(self.flop2)
        cards_layout.addWidget(self.flop3)
        cards_layout.addSpacing(10)
        cards_layout.addWidget(self.turn)
        cards_layout.addSpacing(10)
        cards_layout.addWidget(self.river)
        cards_layout.addStretch()

        # Auto-detect checkbox
        self.auto_check = QCheckBox("Auto Detect")
        self.auto_check.setChecked(True)
        self.auto_check.setStyleSheet(get_checkbox_style())

        layout.addLayout(labels_layout)
        layout.addLayout(cards_layout)
        layout.addWidget(self.auto_check)

    def _connect_signals(self):
        """Connect internal signals."""
        for card_widget in [self.flop1, self.flop2, self.flop3, self.turn, self.river]:
            card_widget.card_changed.connect(self._on_card_changed)
        self.auto_check.toggled.connect(self._on_auto_toggled)

    def _on_card_changed(self):
        """Handle card selection change."""
        if not self._auto_detect:
            self.community_changed.emit(self.get_community())

    def _on_auto_toggled(self, checked: bool):
        """Handle auto-detect toggle."""
        self._auto_detect = checked
        for card_widget in [self.flop1, self.flop2, self.flop3, self.turn, self.river]:
            card_widget.setEnabled(not checked)
        if not checked:
            self.community_changed.emit(self.get_community())

    def get_community(self) -> List[str]:
        """
        Get current community cards as list.

        Returns:
            List of card notations in order
        """
        cards = []
        for widget in [self.flop1, self.flop2, self.flop3, self.turn, self.river]:
            card = widget.get_card()
            if card:
                cards.append(card)
        return cards

    def set_community(self, cards: List[str]):
        """
        Set community cards from list.

        Args:
            cards: List of card notations (up to 5)
        """
        widgets = [self.flop1, self.flop2, self.flop3, self.turn, self.river]
        for i, widget in enumerate(widgets):
            if i < len(cards):
                widget.set_card(cards[i])
            else:
                widget.clear()

    def is_auto_detect(self) -> bool:
        """Check if auto-detect is enabled."""
        return self._auto_detect

    def set_auto_detect(self, enabled: bool):
        """Set auto-detect mode."""
        self.auto_check.setChecked(enabled)
