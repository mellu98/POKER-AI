"""
System tray manager for Poker Assistant Control Panel.

Provides system tray icon, context menu, and minimize-to-tray functionality.
"""
from typing import Optional, Callable
from pathlib import Path
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction, QApplication
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QIcon

from .styles import COLORS


class SystemTrayManager(QObject):
    """
    System tray icon and menu manager.

    Features:
    - Tray icon with app branding
    - Context menu: Show Panel | Start/Stop | Settings | Exit
    - Double-click to restore window
    - Minimize to tray on close (X button)
    """

    # Signals
    show_panel_requested = pyqtSignal()
    start_stop_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    exit_requested = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._is_running = False
        self._panel_window = None
        self._setup_tray()

    def _get_icon_path(self) -> str:
        """Get path to tray icon."""
        # Look for icon in assets folder
        project_root = Path(__file__).parent.parent.parent.parent
        icon_path = project_root / 'assets' / 'icons' / 'poker_icon.ico'

        if icon_path.exists():
            return str(icon_path)

        # Fallback to PNG if ICO not found
        png_path = project_root / 'assets' / 'icons' / 'poker_icon.png'
        if png_path.exists():
            return str(png_path)

        return ""

    def _setup_tray(self):
        """Initialize system tray icon and menu."""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)

        # Set icon
        icon_path = self._get_icon_path()
        if icon_path:
            self.tray_icon.setIcon(QIcon(icon_path))
        else:
            # Use application default icon
            self.tray_icon.setIcon(QApplication.style().standardIcon(
                QApplication.style().SP_ComputerIcon
            ))

        self.tray_icon.setToolTip("Poker Assistant")

        # Create context menu
        self._create_menu()

        # Connect signals
        self.tray_icon.activated.connect(self._on_tray_activated)

        # Show tray icon
        self.tray_icon.show()

    def _create_menu(self):
        """Create tray context menu."""
        self.menu = QMenu()

        # Show Panel action
        self.show_action = QAction("Show Panel", self)
        self.show_action.triggered.connect(self._on_show_panel)
        self.menu.addAction(self.show_action)

        self.menu.addSeparator()

        # Start/Stop action
        self.start_stop_action = QAction("\u25B6 Start", self)  # Play symbol
        self.start_stop_action.triggered.connect(self._on_start_stop)
        self.menu.addAction(self.start_stop_action)

        # Settings action
        self.settings_action = QAction("\u2699 Settings", self)  # Gear symbol
        self.settings_action.triggered.connect(self.settings_requested.emit)
        self.menu.addAction(self.settings_action)

        self.menu.addSeparator()

        # Exit action
        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self._on_exit)
        self.menu.addAction(self.exit_action)

        # Set menu
        self.tray_icon.setContextMenu(self.menu)

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.DoubleClick:
            self._on_show_panel()
        elif reason == QSystemTrayIcon.Trigger:
            # Single click - show menu on some platforms
            pass

    def _on_show_panel(self):
        """Handle show panel request."""
        self.show_panel_requested.emit()
        if self._panel_window:
            self._panel_window.show()
            self._panel_window.raise_()
            self._panel_window.activateWindow()

    def _on_start_stop(self):
        """Handle start/stop request."""
        self.start_stop_requested.emit()

    def _on_exit(self):
        """Handle exit request."""
        self.exit_requested.emit()

    def set_panel_window(self, window):
        """
        Set reference to panel window for show/hide operations.

        Args:
            window: ControlPanelWindow instance
        """
        self._panel_window = window

    def set_running(self, running: bool):
        """
        Update running state and menu appearance.

        Args:
            running: True if detection is running
        """
        self._is_running = running
        if running:
            self.start_stop_action.setText("\u25A0 Stop")  # Stop symbol
            self.tray_icon.setToolTip("Poker Assistant - Running")
        else:
            self.start_stop_action.setText("\u25B6 Start")  # Play symbol
            self.tray_icon.setToolTip("Poker Assistant - Idle")

    def is_running(self) -> bool:
        """Check if detection is running."""
        return self._is_running

    def show_message(self, title: str, message: str,
                     icon: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.Information,
                     duration: int = 3000):
        """
        Show a tray notification message.

        Args:
            title: Notification title
            message: Notification body
            icon: Icon type (Information, Warning, Critical)
            duration: Display duration in milliseconds
        """
        self.tray_icon.showMessage(title, message, icon, duration)

    def hide(self):
        """Hide the tray icon."""
        self.tray_icon.hide()

    def show(self):
        """Show the tray icon."""
        self.tray_icon.show()

    def is_available(self) -> bool:
        """Check if system tray is available on this platform."""
        return QSystemTrayIcon.isSystemTrayAvailable()
