"""
Calibration buttons widget for poker table calibration controls.

Provides buttons for auto-resize, set anchor, configure regions, and reset.
"""
from typing import Optional
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QVBoxLayout, QPushButton, QWidget
)
from PyQt5.QtCore import pyqtSignal

from ..styles import COLORS, get_push_button_style


class CalibrationButtonsWidget(QGroupBox):
    """
    Widget with calibration control buttons.

    Buttons:
    - Auto-Resize: Automatically find and resize to poker window
    - Set Anchor: Set reference anchor point
    - Config Regions: Open region configuration tool
    - Reset: Reset calibration to defaults
    """

    # Signals for button actions
    auto_resize_clicked = pyqtSignal()
    set_anchor_clicked = pyqtSignal()
    config_regions_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    def __init__(self, title: str = "CALIBRATION", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # First row of buttons
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        self.auto_resize_btn = QPushButton("Auto-Resize")
        self.auto_resize_btn.setStyleSheet(get_push_button_style('primary'))
        self.auto_resize_btn.setToolTip("Automatically detect and fit to poker window")

        self.set_anchor_btn = QPushButton("Set Anchor")
        self.set_anchor_btn.setStyleSheet(get_push_button_style('primary'))
        self.set_anchor_btn.setToolTip("Set reference anchor point for detection")

        row1.addWidget(self.auto_resize_btn)
        row1.addWidget(self.set_anchor_btn)

        # Second row of buttons
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        self.config_regions_btn = QPushButton("Config Regions")
        self.config_regions_btn.setStyleSheet(get_push_button_style('primary'))
        self.config_regions_btn.setToolTip("Configure detection regions")

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setStyleSheet(get_push_button_style('danger'))
        self.reset_btn.setToolTip("Reset calibration to defaults")

        row2.addWidget(self.config_regions_btn)
        row2.addWidget(self.reset_btn)

        layout.addLayout(row1)
        layout.addLayout(row2)

    def _connect_signals(self):
        """Connect button signals."""
        self.auto_resize_btn.clicked.connect(self.auto_resize_clicked.emit)
        self.set_anchor_btn.clicked.connect(self.set_anchor_clicked.emit)
        self.config_regions_btn.clicked.connect(self.config_regions_clicked.emit)
        self.reset_btn.clicked.connect(self.reset_clicked.emit)

    def set_enabled(self, enabled: bool):
        """Enable or disable all buttons."""
        self.auto_resize_btn.setEnabled(enabled)
        self.set_anchor_btn.setEnabled(enabled)
        self.config_regions_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)

    def set_calibrating(self, is_calibrating: bool):
        """
        Update button states during calibration.

        Args:
            is_calibrating: True if calibration is in progress
        """
        if is_calibrating:
            self.auto_resize_btn.setEnabled(False)
            self.set_anchor_btn.setText("Cancel")
            self.set_anchor_btn.setStyleSheet(get_push_button_style('danger'))
            self.config_regions_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
        else:
            self.auto_resize_btn.setEnabled(True)
            self.set_anchor_btn.setText("Set Anchor")
            self.set_anchor_btn.setStyleSheet(get_push_button_style('primary'))
            self.config_regions_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
