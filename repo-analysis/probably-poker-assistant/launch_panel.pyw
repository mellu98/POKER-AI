"""
Desktop shortcut entry point for Poker Assistant Control Panel.

This file launches the Control Panel alongside the existing transparent overlay.
Use .pyw extension to run without console window on Windows.

Usage:
    Double-click launch_panel.pyw or create a desktop shortcut to it.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

from src.utils.logger import logger
from src.ui.control_panel import ControlPanelWindow, SystemTrayManager
from src.main import GameLoop
from src.ui.display_manager import DisplayManager


class PokerAssistantApp:
    """
    Main application class integrating Control Panel, Overlay, and GameLoop.
    """

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Poker Assistant")
        self.app.setQuitOnLastWindowClosed(False)  # Keep running in tray

        # Initialize components
        self._init_components()
        self._connect_signals()

        logger.info("Poker Assistant Control Panel initialized")

    def _init_components(self):
        """Initialize all application components."""
        # System tray (create first for availability check)
        self.tray_manager = SystemTrayManager()

        # Control Panel
        self.control_panel = ControlPanelWindow()
        self.control_panel.set_tray_manager(self.tray_manager)
        self.tray_manager.set_panel_window(self.control_panel)

        # Overlay display (existing transparent HUD)
        self.display_manager = DisplayManager(None)

        # Game logic thread
        self.game_loop = GameLoop()

    def _connect_signals(self):
        """Connect all component signals."""
        # GameLoop -> Control Panel & Overlay
        self.game_loop.update_signal.connect(self._on_game_update)

        # Control Panel -> GameLoop
        self.control_panel.start_stop_toggled.connect(self._on_start_stop)
        self.control_panel.manual_cards_changed.connect(self._on_manual_cards)
        self.control_panel.calibration_requested.connect(self._on_calibration)
        self.control_panel.settings_requested.connect(self._on_settings)

        # System Tray -> App
        self.tray_manager.show_panel_requested.connect(self._show_panel)
        self.tray_manager.start_stop_requested.connect(self._toggle_running)
        self.tray_manager.settings_requested.connect(self._on_settings)
        self.tray_manager.exit_requested.connect(self._on_exit)

    def _on_game_update(self, data):
        """Handle game state updates from GameLoop."""
        # Forward to control panel
        self.control_panel.on_game_state_updated(data)

        # Forward to overlay
        self.display_manager.signals.update_signal.emit(data)

    def _on_start_stop(self, start: bool):
        """Handle start/stop from control panel."""
        if start:
            if not self.game_loop.isRunning():
                self.game_loop.running = True
                self.game_loop.start()
                logger.info("Game loop started from Control Panel")
        else:
            self.game_loop.running = False
            logger.info("Game loop stopped from Control Panel")

        self.tray_manager.set_running(start)

    def _on_manual_cards(self, hole_cards, community_cards):
        """Handle manual card input from control panel."""
        if hasattr(self.game_loop, 'set_manual_cards'):
            self.game_loop.set_manual_cards(hole_cards, community_cards)
            logger.debug(f"Manual cards set: {hole_cards}, {community_cards}")

    def _on_calibration(self, action: str):
        """Handle calibration requests."""
        logger.info(f"Calibration requested: {action}")

        if action == 'auto_resize':
            # TODO: Implement auto-resize
            self.tray_manager.show_message(
                "Calibration",
                "Auto-resize not yet implemented",
                duration=2000
            )
        elif action == 'set_anchor':
            # TODO: Launch anchor calibration tool
            self.tray_manager.show_message(
                "Calibration",
                "Use tools/calibrate_anchors.py for now",
                duration=2000
            )
        elif action == 'config_regions':
            # TODO: Launch region configuration tool
            self.tray_manager.show_message(
                "Calibration",
                "Use tools/calibrate_anchors.py for now",
                duration=2000
            )
        elif action == 'reset':
            # TODO: Reset calibration
            self.tray_manager.show_message(
                "Calibration",
                "Reset not yet implemented",
                duration=2000
            )

    def _on_settings(self):
        """Handle settings request."""
        logger.info("Settings requested")
        # TODO: Open settings dialog
        self.tray_manager.show_message(
            "Settings",
            "Settings dialog not yet implemented",
            duration=2000
        )

    def _show_panel(self):
        """Show the control panel window."""
        self.control_panel.show()
        self.control_panel.raise_()
        self.control_panel.activateWindow()

    def _toggle_running(self):
        """Toggle running state from tray."""
        is_running = self.game_loop.isRunning() and self.game_loop.running
        self._on_start_stop(not is_running)
        self.control_panel.set_running(not is_running)

    def _on_exit(self):
        """Handle exit request."""
        logger.info("Exit requested")

        # Stop game loop
        if self.game_loop.isRunning():
            self.game_loop.stop()

        # Hide tray
        self.tray_manager.hide()

        # Close windows
        self.control_panel.close()

        # Quit application
        self.app.quit()

    def run(self) -> int:
        """
        Run the application.

        Returns:
            Exit code
        """
        # Show control panel
        self.control_panel.show()

        # Show startup message
        if self.tray_manager.is_available():
            self.tray_manager.show_message(
                "Poker Assistant",
                "Control Panel is ready. Click Start to begin detection.",
                duration=3000
            )

        logger.info("=" * 50)
        logger.info("Poker AI Assistant - Control Panel Running")
        logger.info("=" * 50)

        # Run Qt event loop
        return self.app.exec_()


def main():
    """Main entry point."""
    try:
        app = PokerAssistantApp()
        sys.exit(app.run())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

        # Show error dialog
        try:
            error_app = QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "Poker Assistant Error",
                f"Failed to start: {e}\n\nCheck logs for details."
            )
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    main()
