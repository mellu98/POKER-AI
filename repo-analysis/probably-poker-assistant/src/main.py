"""
Main entry point for Poker AI Assistant.
Integrates capture, detection, strategy, and UI.

Architecture:
- GameLoop (QThread): Background thread for game logic
- DisplayManager: Handles PyQt5 overlay window
- ControlPanelWindow: Main control interface
- SystemTrayManager: System tray integration
- Components: WindowFinder, ScreenGrabber, AnchorManager, CardDetector, TextReader, DecisionEngine
"""
import sys
import time
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.utils.config_loader import config_loader
from src.utils.session_logger import SessionLogger
from src.utils.performance import PerformanceMonitor
from src.capture.window_finder import WindowFinder
from src.capture.screen_grabber import ScreenGrabber
from src.capture.anchor_manager import AnchorManager
from src.detection.card_detector import CardDetector
from src.detection.text_reader import TextReader
from src.detection.game_state import GameStateTracker
from src.strategy.decision_engine import DecisionEngine
from src.ui.display_manager import DisplayManager
from src.ui.control_panel import ControlPanelWindow, SystemTrayManager, needs_first_run_setup, show_first_run_wizard


class GameLoop(QThread):
    """Background thread for game logic pipeline."""
    update_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.running = True

        # Load configuration
        try:
            self.config = config_loader.load('settings.json')
        except Exception:
            self.config = {}

        self.window_title = self.config.get('capture', {}).get('window_title', 'Ignition')
        self.loop_interval = self.config.get('capture', {}).get('interval_seconds', 1.0)

        # Initialize components
        self.window_finder = WindowFinder(self.window_title)
        self.screen_grabber = ScreenGrabber()
        self.anchor_manager = AnchorManager()
        self.card_detector = CardDetector()
        self.text_reader = TextReader()
        self.tracker = GameStateTracker()
        self.decision_engine = DecisionEngine()
        self.session_logger = SessionLogger()
        self.perf = PerformanceMonitor()

        # Frame counter for periodic performance logging
        self.frame_count = 0
        self.perf_log_interval = 100  # Log stats every 100 frames

        # Manual card override (from Control Panel)
        self._manual_hole_cards = []
        self._manual_community_cards = []
        self._use_manual_cards = False

        # Load anchor config
        self.anchor_manager.load_config()
        if not self.anchor_manager.active_anchor_name:
            logger.warning("No active anchor. Run tools/calibrate_anchors.py first.")

    def set_running(self, running: bool):
        """
        Set running state from external control (Control Panel).

        Args:
            running: True to enable processing, False to pause
        """
        self.running = running
        logger.info(f"Game loop running state set to: {running}")

    def set_manual_cards(self, hole_cards: list, community_cards: list):
        """
        Set manual card override from Control Panel.

        When manual cards are set (non-empty), they override detected cards.

        Args:
            hole_cards: List of hole card strings, e.g. ["Ah", "Ks"]
            community_cards: List of community card strings
        """
        self._manual_hole_cards = hole_cards or []
        self._manual_community_cards = community_cards or []
        self._use_manual_cards = bool(hole_cards) or bool(community_cards)
        logger.debug(f"Manual cards set: hole={hole_cards}, community={community_cards}")

    def run(self):
        """Main game loop - runs in background thread."""
        logger.info("Game loop started")

        while self.running:
            try:
                self._process_frame()
                time.sleep(self.loop_interval)
            except Exception as e:
                logger.error(f"Game loop error: {e}")
                time.sleep(2)

    def _process_frame(self):
        """Process single frame: capture -> detect -> decide -> display."""
        with self.perf.track("total_frame"):
            # 1. Find poker window
            with self.perf.track("window_find"):
                if not self.window_finder.find_window():
                    return

            # 2. Capture full screen
            with self.perf.track("screen_capture"):
                screen = self.screen_grabber.capture_screen()
                if screen is None:
                    return

            # 3. Find anchor and get absolute regions
            with self.perf.track("anchor_detection"):
                anchor_pos = self.anchor_manager.find_anchor(screen)
                if anchor_pos is None:
                    logger.debug("Anchor not found on screen")
                    return

                ax, ay, _, _ = anchor_pos
                regions = self.anchor_manager.get_absolute_regions((ax, ay))

            # 4. Extract and detect cards (or use manual override)
            with self.perf.track("card_detection"):
                # Check for manual card override
                if self._use_manual_cards and self._manual_hole_cards:
                    hole_cards = self._manual_hole_cards
                else:
                    hole_img = self.screen_grabber.extract_region(screen, regions.get('hole_cards'))
                    hole_cards = self.card_detector.detect_hand(hole_img, num_cards=2) if hole_img is not None else []

                if self._use_manual_cards and self._manual_community_cards:
                    community_cards = self._manual_community_cards
                else:
                    community_img = self.screen_grabber.extract_region(screen, regions.get('community_cards'))
                    community_cards = self.card_detector.detect_community_cards(community_img) if community_img is not None else []

            # 5. OCR for pot/stack/bet
            with self.perf.track("ocr_reading"):
                pot_img = self.screen_grabber.extract_region(screen, regions.get('pot_amount'))
                stack_img = self.screen_grabber.extract_region(screen, regions.get('player_stack'))
                bet_img = self.screen_grabber.extract_region(screen, regions.get('current_bet'))

                pot_size = self.text_reader.read_pot_amount(pot_img) if pot_img is not None else None
                stack_size = self.text_reader.read_stack_size(stack_img) if stack_img is not None else None
                current_bet = self.text_reader.read_bet_amount(bet_img) if bet_img is not None else None

            # 6. Update game state
            game_state = self.tracker.update_state(
                hole_cards=hole_cards,
                community_cards=community_cards,
                pot_size=pot_size,
                stack_size=stack_size,
                current_bet=current_bet
            )

            # 7. Make decision (only if we have hole cards)
            if len(hole_cards) >= 2:
                with self.perf.track("decision_engine"):
                    decision = self.decision_engine.decide(game_state)

                self.update_signal.emit({'decision': decision, 'game_state': game_state})

                # 8. Log decision for learning
                self.session_logger.log_decision(game_state, decision)

        # Periodic performance logging
        self.frame_count += 1
        if self.frame_count % self.perf_log_interval == 0:
            self.perf.log_summary()
            bottleneck = self.perf.check_bottleneck()
            if bottleneck:
                logger.info(f"Current bottleneck: {bottleneck}")

    def stop(self):
        """Stop the game loop gracefully."""
        self.running = False
        self.session_logger.close()
        self.perf.log_summary()
        self.wait()


def main():
    """Main application entry point."""
    logger.info("=" * 50)
    logger.info("Poker AI Assistant - Starting")
    logger.info("=" * 50)

    try:
        # Initialize Qt Application
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)  # Keep running in tray

        # Check if first-run setup is needed
        config_dir = project_root / 'config'
        if needs_first_run_setup(config_dir):
            logger.info("First-run detected - showing setup wizard")
            setup_mode = show_first_run_wizard()

            if setup_mode is None:
                # User cancelled - exit
                logger.info("Setup wizard cancelled - exiting")
                return 0
            elif setup_mode == 'calibration':
                logger.info("Calibration mode selected - please run calibration tool")
                # Continue to main app, but show message
            elif setup_mode == 'test':
                logger.info("Test mode selected - manual card entry enabled")
            elif setup_mode == 'skip':
                logger.info("Setup skipped - continuing with defaults")

        # Initialize Display Manager (overlay)
        display_manager = DisplayManager(None)

        # Initialize Control Panel
        control_panel = ControlPanelWindow()

        # Initialize System Tray
        tray_manager = SystemTrayManager()
        tray_manager.set_panel_window(control_panel)
        control_panel.set_tray_manager(tray_manager)

        # Initialize Game Loop
        game_thread = GameLoop()

        # === Connect Signals ===

        # GameLoop -> UI updates
        game_thread.update_signal.connect(display_manager.update_overlay)
        game_thread.update_signal.connect(control_panel.on_game_state_updated)

        # Control Panel -> GameLoop
        control_panel.manual_cards_changed.connect(game_thread.set_manual_cards)
        control_panel.start_stop_toggled.connect(game_thread.set_running)
        control_panel.start_stop_toggled.connect(tray_manager.set_running)

        # System Tray -> GameLoop
        def toggle_running():
            is_running = game_thread.running
            game_thread.set_running(not is_running)
            control_panel.set_running(not is_running)
            tray_manager.set_running(not is_running)

        tray_manager.start_stop_requested.connect(toggle_running)

        # System Tray -> Exit
        def shutdown_app():
            logger.info("Exit requested from system tray")
            game_thread.stop()
            tray_manager.hide()
            app.quit()

        tray_manager.exit_requested.connect(shutdown_app)

        # Calibration requests (placeholder - can be expanded)
        def handle_calibration(action: str):
            logger.info(f"Calibration action requested: {action}")
            if action == 'auto_resize':
                tray_manager.show_message("Calibration", "Auto-resize not yet implemented")
            elif action == 'set_anchor':
                tray_manager.show_message("Calibration", "Use tools/calibrate_anchors.py")
            elif action == 'config_regions':
                tray_manager.show_message("Calibration", "Use tools/calibrate_anchors.py")
            elif action == 'reset':
                game_thread.anchor_manager.load_config()
                tray_manager.show_message("Calibration", "Config reloaded")

        control_panel.calibration_requested.connect(handle_calibration)

        # === Start Application ===

        # Start game thread (paused by default)
        game_thread.running = False
        game_thread.start()

        # Show windows
        control_panel.show()

        # Show startup notification
        if tray_manager.is_available():
            tray_manager.show_message(
                "Poker Assistant",
                "Running in system tray. Click Start to begin.",
                duration=3000
            )

        logger.info("All components initialized successfully")
        logger.info("Control Panel visible, overlay ready")
        logger.info("Click 'Start' to begin detection")

        # Run Qt event loop (blocks until exit)
        exit_code = app.exec_()

        # Cleanup
        game_thread.stop()
        logger.info("Poker AI Assistant - Shutdown complete")
        sys.exit(exit_code)

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
