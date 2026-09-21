"""
First-Run Wizard for Poker Assistant.

Guides new users through initial setup and provides test mode options.
"""
import sys
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from .styles import COLORS, FONTS


class FirstRunWizard(QDialog):
    """
    First-run wizard dialog for new users.

    Provides options to:
    - Run calibration for live poker window
    - Use test mode with manual card entry
    - Skip setup for now
    """

    # Signals
    calibration_requested = pyqtSignal()
    test_mode_selected = pyqtSignal()
    setup_skipped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_mode = None
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Welcome to Poker Assistant")
        self.setFixedSize(500, 450)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QPushButton {{
                background-color: {COLORS['accent_neutral']};
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_positive']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
            QRadioButton {{
                color: {COLORS['text_primary']};
                font-size: 14px;
                spacing: 10px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title_label = QLabel("Welcome to Poker Assistant")
        title_label.setFont(QFont(FONTS['family'], 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Subtitle
        subtitle = QLabel(
            "This appears to be your first time running the application.\n"
            "Choose how you'd like to get started:"
        )
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']};")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {COLORS['border']};")
        layout.addWidget(line)

        # Options
        self.option_group = QButtonGroup(self)

        # Option 1: Live Calibration
        self.live_radio = QRadioButton("Live Window Calibration (Recommended)")
        self.live_radio.setChecked(True)
        self.option_group.addButton(self.live_radio, 1)
        live_desc = QLabel(
            "Open your poker client first, then run calibration to\n"
            "mark the screen regions for cards, pot, and actions."
        )
        live_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-left: 28px;")
        layout.addWidget(self.live_radio)
        layout.addWidget(live_desc)

        # Option 2: Test Mode
        self.test_radio = QRadioButton("Test Mode (Manual Entry)")
        self.option_group.addButton(self.test_radio, 2)
        test_desc = QLabel(
            "Use manual card entry without screen capture.\n"
            "Great for testing strategy recommendations."
        )
        test_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-left: 28px;")
        layout.addWidget(self.test_radio)
        layout.addWidget(test_desc)

        # Option 3: Skip
        self.skip_radio = QRadioButton("Skip Setup (Advanced Users)")
        self.option_group.addButton(self.skip_radio, 3)
        skip_desc = QLabel(
            "Start the application without initial setup.\n"
            "You can configure settings later."
        )
        skip_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-left: 28px;")
        layout.addWidget(self.skip_radio)
        layout.addWidget(skip_desc)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.continue_btn = QPushButton("Continue")
        self.continue_btn.clicked.connect(self._on_continue)
        self.continue_btn.setMinimumWidth(120)

        self.cancel_btn = QPushButton("Exit")
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
            }}
            QPushButton:hover {{
                background-color: {COLORS['background_light']};
            }}
        """)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setMinimumWidth(100)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.continue_btn)

        layout.addLayout(btn_layout)

    def _on_continue(self):
        """Handle continue button click."""
        selected_id = self.option_group.checkedId()

        if selected_id == 1:
            self.selected_mode = 'calibration'
            self.calibration_requested.emit()
        elif selected_id == 2:
            self.selected_mode = 'test'
            self.test_mode_selected.emit()
        elif selected_id == 3:
            self.selected_mode = 'skip'
            self.setup_skipped.emit()

        self.accept()

    def get_selected_mode(self) -> Optional[str]:
        """Get the selected mode after dialog closes."""
        return self.selected_mode


def needs_first_run_setup(config_dir: Path) -> bool:
    """
    Check if first-run setup is needed.

    Returns True if:
    - No anchor_config.json exists
    - anchor_config.json exists but has no active_anchor
    - No regions.json exists
    """
    anchor_config = config_dir / 'anchor_config.json'
    regions_config = config_dir / 'regions.json'

    # Check anchor config
    if not anchor_config.exists():
        return True

    try:
        import json
        with open(anchor_config, 'r') as f:
            config = json.load(f)

        # Check if there's an active anchor
        if not config.get('active_anchor'):
            return True

        # Check if there are regions defined
        if not config.get('regions'):
            return True

    except (json.JSONDecodeError, IOError):
        return True

    return False


def show_first_run_wizard(parent=None) -> Optional[str]:
    """
    Show the first-run wizard dialog.

    Returns:
        'calibration', 'test', 'skip', or None if cancelled
    """
    wizard = FirstRunWizard(parent)
    result = wizard.exec_()

    if result == QDialog.Accepted:
        return wizard.get_selected_mode()
    return None
