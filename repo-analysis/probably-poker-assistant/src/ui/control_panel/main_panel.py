"""
Main Control Panel window for Poker Assistant.

Assembles all widgets into a comprehensive control interface that runs
alongside the transparent overlay.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QFrame, QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt, QSettings
from PyQt5.QtGui import QIcon, QCloseEvent

from .styles import (
    COLORS, FONTS, get_full_stylesheet,
    get_push_button_style, get_group_box_style
)
from .widgets import (
    CardSelectorWidget, HandSelectorWidget, CommunitySelectorWidget,
    ActionDisplayWidget, StatisticsPanel, PlayerTableWidget,
    AmountDisplayWidget, CalibrationButtonsWidget, ActionFrequenciesWidget
)


class ControlPanelWindow(QMainWindow):
    """
    Main dockable control panel window for Poker Assistant.

    Provides:
    - Manual card input with dropdown selectors
    - Real-time statistics display
    - Calibration controls
    - Start/Stop controls

    Signals:
    - manual_cards_changed: Emitted when manual cards are changed
    - start_stop_toggled: Emitted when start/stop button is clicked
    - calibration_requested: Emitted when calibration action is requested
    - settings_requested: Emitted when settings button is clicked
    """

    # Signals to GameLoop
    manual_cards_changed = pyqtSignal(list, list)  # hole_cards, community_cards
    start_stop_toggled = pyqtSignal(bool)  # True = start, False = stop
    calibration_requested = pyqtSignal(str)  # action type
    settings_requested = pyqtSignal()

    # Panel state file
    STATE_FILE = 'config/panel_state.json'

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_running = False
        self._setup_ui()
        self._connect_signals()
        self._load_state()

    def _setup_ui(self):
        """Initialize the main UI layout."""
        self.setWindowTitle("POKER ASSISTANT CONTROL PANEL")
        self.setMinimumSize(400, 700)
        self.setStyleSheet(get_full_stylesheet())

        # Central widget with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === SUGGESTED ACTION ===
        self.action_display = ActionDisplayWidget()
        self.action_display.setStyleSheet(get_group_box_style())
        main_layout.addWidget(self.action_display)

        # === CARDS ROW (Hand + Community) ===
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)

        # Your Hand section
        self.hand_selector = HandSelectorWidget("YOUR HAND")
        self.hand_selector.setStyleSheet(get_group_box_style())
        self.hand_selector.setMaximumWidth(200)

        # Community section
        self.community_selector = CommunitySelectorWidget("COMMUNITY")
        self.community_selector.setStyleSheet(get_group_box_style())

        cards_row.addWidget(self.hand_selector)
        cards_row.addWidget(self.community_selector)
        main_layout.addLayout(cards_row)

        # === AMOUNTS ===
        self.amount_display = AmountDisplayWidget("AMOUNTS")
        self.amount_display.setStyleSheet(get_group_box_style())
        main_layout.addWidget(self.amount_display)

        # === STATISTICS ===
        self.statistics_panel = StatisticsPanel("STATISTICS")
        self.statistics_panel.setStyleSheet(get_group_box_style())
        main_layout.addWidget(self.statistics_panel)

        # === GTO ACTION FREQUENCIES ===
        self.action_frequencies = ActionFrequenciesWidget("GTO ACTION FREQUENCIES")
        self.action_frequencies.setStyleSheet(get_group_box_style())
        main_layout.addWidget(self.action_frequencies)

        # === PLAYERS ===
        self.player_table = PlayerTableWidget("PLAYERS")
        self.player_table.setStyleSheet(get_group_box_style())
        main_layout.addWidget(self.player_table)

        # === CALIBRATION ===
        self.calibration_buttons = CalibrationButtonsWidget("CALIBRATION")
        self.calibration_buttons.setStyleSheet(get_group_box_style())
        main_layout.addWidget(self.calibration_buttons)

        # === BOTTOM CONTROLS ===
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        # Start/Stop button
        self.start_stop_btn = QPushButton("\u25B6 START")  # Play symbol
        self.start_stop_btn.setStyleSheet(get_push_button_style('success'))
        self.start_stop_btn.setMinimumHeight(40)

        # Settings button
        self.settings_btn = QPushButton("\u2699 Settings")  # Gear symbol
        self.settings_btn.setStyleSheet(get_push_button_style('primary'))
        self.settings_btn.setMinimumHeight(40)

        bottom_layout.addWidget(self.start_stop_btn, stretch=2)
        bottom_layout.addWidget(self.settings_btn, stretch=1)
        main_layout.addLayout(bottom_layout)

        # Add stretch to push everything up
        main_layout.addStretch()

        scroll.setWidget(central)
        self.setCentralWidget(scroll)

        # === STATUS BAR ===
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['background_darker']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)
        self.setStatusBar(self.status_bar)
        self._update_status("Idle")

    def _connect_signals(self):
        """Connect internal widget signals."""
        # Card selectors
        self.hand_selector.hand_changed.connect(self._on_cards_changed)
        self.community_selector.community_changed.connect(self._on_cards_changed)

        # Start/Stop
        self.start_stop_btn.clicked.connect(self._on_start_stop_clicked)

        # Settings
        self.settings_btn.clicked.connect(self.settings_requested.emit)

        # Calibration
        self.calibration_buttons.auto_resize_clicked.connect(
            lambda: self.calibration_requested.emit('auto_resize')
        )
        self.calibration_buttons.set_anchor_clicked.connect(
            lambda: self.calibration_requested.emit('set_anchor')
        )
        self.calibration_buttons.config_regions_clicked.connect(
            lambda: self.calibration_requested.emit('config_regions')
        )
        self.calibration_buttons.reset_clicked.connect(
            lambda: self.calibration_requested.emit('reset')
        )

    def _on_cards_changed(self):
        """Handle manual card changes."""
        if not self.hand_selector.is_auto_detect():
            hole_cards = self.hand_selector.get_hand()
        else:
            hole_cards = []

        if not self.community_selector.is_auto_detect():
            community_cards = self.community_selector.get_community()
        else:
            community_cards = []

        self.manual_cards_changed.emit(hole_cards, community_cards)

    def _on_start_stop_clicked(self):
        """Handle start/stop button click."""
        self._is_running = not self._is_running
        self.start_stop_toggled.emit(self._is_running)
        self._update_start_stop_button()

    def _update_start_stop_button(self):
        """Update start/stop button appearance."""
        if self._is_running:
            self.start_stop_btn.setText("\u25A0 STOP")  # Stop symbol
            self.start_stop_btn.setStyleSheet(get_push_button_style('danger'))
            self._update_status("Running")
        else:
            self.start_stop_btn.setText("\u25B6 START")  # Play symbol
            self.start_stop_btn.setStyleSheet(get_push_button_style('success'))
            self._update_status("Idle")

    def _update_status(self, status: str, extra: str = ""):
        """Update status bar."""
        text = f"Status: {status}"
        if extra:
            text += f" | {extra}"
        self.status_bar.showMessage(text)

    # === Public API for GameLoop integration ===

    def on_game_state_updated(self, data: Dict[str, Any]):
        """
        Slot: Called when GameLoop emits new game state.

        Args:
            data: Dict with 'decision' and 'game_state' keys
        """
        decision = data.get('decision')
        game_state = data.get('game_state')

        # Update action display
        self.action_display.update_from_decision(decision)

        # Update statistics
        self.statistics_panel.update_from_decision(decision, game_state)

        # Update GTO action frequencies
        self.action_frequencies.update_from_decision(decision)

        # Update amounts if auto-detect
        if game_state:
            self.amount_display.update_from_game_state(game_state)

            # Update cards if auto-detect is enabled
            if self.hand_selector.is_auto_detect() and game_state.hole_cards:
                self.hand_selector.set_hand(game_state.hole_cards)

            if self.community_selector.is_auto_detect() and game_state.community_cards:
                self.community_selector.set_community(game_state.community_cards)

        # Update status
        if decision:
            self._update_status("Running", f"Last: {decision.action.upper()}")

    def on_detection_status_changed(self, running: bool):
        """
        Slot: Called when detection running status changes.

        Args:
            running: True if detection is running
        """
        self._is_running = running
        self._update_start_stop_button()

    def set_running(self, running: bool):
        """Set running state programmatically."""
        self._is_running = running
        self._update_start_stop_button()

    def is_running(self) -> bool:
        """Check if detection is running."""
        return self._is_running

    # === State Persistence ===

    def _get_state_path(self) -> Path:
        """Get path to state file."""
        # Get project root (assumes we're in src/ui/control_panel/)
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / self.STATE_FILE

    def _load_state(self):
        """Load panel state from file."""
        try:
            state_path = self._get_state_path()
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)

                # Restore window geometry
                if 'geometry' in state:
                    geo = state['geometry']
                    self.setGeometry(geo['x'], geo['y'], geo['w'], geo['h'])

                # Restore blinds
                if 'blinds' in state:
                    blinds = state['blinds']
                    self.amount_display.set_amounts(
                        sb=blinds.get('sb'),
                        bb=blinds.get('bb')
                    )
        except Exception:
            # Use defaults if loading fails
            pass

    def _save_state(self):
        """Save panel state to file."""
        try:
            state = {
                'geometry': {
                    'x': self.x(),
                    'y': self.y(),
                    'w': self.width(),
                    'h': self.height(),
                },
                'blinds': {
                    'sb': self.amount_display.blinds_field.get_sb(),
                    'bb': self.amount_display.blinds_field.get_bb(),
                },
            }

            state_path = self._get_state_path()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent):
        """Handle window close - save state and minimize to tray if available."""
        self._save_state()

        # If we have a system tray manager, minimize to tray instead of closing
        if hasattr(self, 'tray_manager') and self.tray_manager:
            event.ignore()
            self.hide()
        else:
            event.accept()

    def set_tray_manager(self, tray_manager):
        """Set reference to system tray manager."""
        self.tray_manager = tray_manager
