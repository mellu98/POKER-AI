"""
Player table widget for displaying player information.

Shows position, stack, and status for each player at the table.
"""
from typing import Optional, List, Dict
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ..styles import COLORS, FONTS, get_table_style


class PlayerTableWidget(QGroupBox):
    """
    Table widget showing player positions, stacks, and status.

    Columns: Position | Stack | Status
    """

    # Default positions for 6-max table
    DEFAULT_POSITIONS = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']

    STATUS_COLORS = {
        'Active': COLORS['status_active'],
        'Waiting': COLORS['status_idle'],
        'Folded': COLORS['text_secondary'],
        'All-In': COLORS['accent_warning'],
        'Hero': COLORS['accent_neutral'],
    }

    def __init__(self, title: str = "PLAYERS", parent: Optional[QGroupBox] = None):
        super().__init__(title, parent)
        self._players: List[Dict] = []
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 5)

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Pos', 'Stack', 'Status'])
        self.table.setStyleSheet(get_table_style())

        # Table configuration
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)

        # Column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(2, 70)

        # Set minimum height
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(180)

        layout.addWidget(self.table)

        # Initialize with empty players
        self._initialize_default_players()

    def _initialize_default_players(self):
        """Initialize table with default positions."""
        self._players = [
            {'position': pos, 'stack': 0, 'status': 'Waiting'}
            for pos in self.DEFAULT_POSITIONS
        ]
        self._refresh_table()

    def _refresh_table(self):
        """Refresh table display from player data."""
        self.table.setRowCount(len(self._players))

        for row, player in enumerate(self._players):
            # Position
            pos_item = QTableWidgetItem(player.get('position', ''))
            pos_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, pos_item)

            # Stack
            stack = player.get('stack', 0)
            stack_text = f"{stack:,.0f}" if stack else "-"
            stack_item = QTableWidgetItem(stack_text)
            stack_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 1, stack_item)

            # Status
            status = player.get('status', 'Waiting')
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)

            # Apply status color
            color = self.STATUS_COLORS.get(status, COLORS['text_secondary'])
            status_item.setForeground(QColor(color))

            self.table.setItem(row, 2, status_item)

    def set_players(self, players: List[Dict]):
        """
        Set player data.

        Args:
            players: List of player dicts with 'position', 'stack', 'status'
        """
        self._players = players
        self._refresh_table()

    def update_player(self, position: str, stack: Optional[float] = None, status: Optional[str] = None):
        """
        Update a single player by position.

        Args:
            position: Position string (BTN, SB, etc.)
            stack: New stack size
            status: New status string
        """
        for player in self._players:
            if player.get('position') == position:
                if stack is not None:
                    player['stack'] = stack
                if status is not None:
                    player['status'] = status
                break
        self._refresh_table()

    def set_hero_position(self, position: str):
        """Mark a position as the hero (player)."""
        for player in self._players:
            if player.get('position') == position:
                player['status'] = 'Hero'
            elif player.get('status') == 'Hero':
                player['status'] = 'Waiting'
        self._refresh_table()

    def clear_players(self):
        """Reset all players to default state."""
        self._initialize_default_players()

    def get_players(self) -> List[Dict]:
        """Get current player data."""
        return self._players.copy()
