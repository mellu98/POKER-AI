"""
Final System Verification Script for Poker AI Assistant.
Checks environment, imports, configuration, and basic component initialization.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

class TestFullSystem(unittest.TestCase):
    """Integration tests for the full system."""
    
    def setUp(self):
        logger.info(f"Running test: {self._testMethodName}")
    
    def test_imports(self):
        """Verify all modules can be imported."""
        try:
            import src.capture.window_finder
            import src.capture.screen_grabber
            import src.capture.anchor_manager
            import src.detection.card_detector
            import src.strategy.decision_engine
            import src.ui.display_manager
            import PyQt5.QtWidgets
        except ImportError as e:
            self.fail(f"Import failed: {e}")
            
    def test_config_loading(self):
        """Verify configuration files are loadable."""
        from src.utils.config_loader import config_loader
        try:
            settings = config_loader.load('settings.json')
            self.assertIsInstance(settings, dict)
            # anchor_config might not exist if not calibrated, which is fine
        except Exception as e:
            self.fail(f"Config loading failed: {e}")
            
    def test_strategy_engine_init(self):
        """Verify Strategy Engine initializes correctly."""
        from src.strategy.decision_engine import DecisionEngine
        try:
            engine = DecisionEngine()
            self.assertIsNotNone(engine.hand_evaluator)
            self.assertIsNotNone(engine.equity_calc)
        except Exception as e:
            self.fail(f"DecisionEngine init failed: {e}")
            
    def test_anchor_manager_init(self):
        """Verify Anchor Manager initializes."""
        from src.capture.anchor_manager import AnchorManager
        try:
            am = AnchorManager()
            self.assertIsNotNone(am.anchor_dir)
        except Exception as e:
            self.fail(f"AnchorManager init failed: {e}")

    @patch('src.capture.window_finder.win32gui')
    def test_window_finder_mock(self, mock_win32):
        """Test WindowFinder with mocked win32gui."""
        from src.capture.window_finder import WindowFinder
        mock_win32.FindWindow.return_value = 12345
        mock_win32.GetWindowRect.return_value = (0, 0, 800, 600)
        
        # Mock EnumWindows to find our "Test" window immediately
        def side_effect(callback, results):
            results.append((12345, "Test Window"))
        mock_win32.EnumWindows.side_effect = side_effect
        
        wf = WindowFinder("Test")
        nf = wf.find_window()
        rect = wf.get_window_rect()
        
        self.assertTrue(nf)
        self.assertEqual(rect, (0, 0, 800, 600))

if __name__ == "__main__":
    # Configure logging to stdout
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()
