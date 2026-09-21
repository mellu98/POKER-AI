"""
Window detection and management for PokerStars.
Uses Windows API to find and track application window.

Note: This module only works on Windows. On other platforms,
it provides a mock implementation for testing.
"""
import sys
from typing import Optional, Tuple, Dict
from src.utils.logger import logger

# Platform-specific imports
IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    try:
        import win32gui
        import win32con
        import win32ui
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
        logger.warning("pywin32 not installed - window detection disabled")
else:
    WIN32_AVAILABLE = False
    logger.info("Non-Windows platform detected - using mock window finder")


class WindowFinder:
    """Find and track PokerStars window."""

    def __init__(self, window_title: str = "Ignition"):
        """
        Initialize window finder.

        Args:
            window_title: Title or partial title of window to find
        """
        self.window_title = window_title
        self.hwnd: Optional[int] = None
        self.last_position: Optional[Tuple[int, int, int, int]] = None
        self._mock_mode = not WIN32_AVAILABLE

    def find_window(self) -> bool:
        """
        Find PokerStars window.

        Returns:
            True if window found, False otherwise
        """
        if self._mock_mode:
            logger.debug("Mock mode: simulating window not found")
            return False

        def enum_windows_callback(hwnd, results):
            """Callback for enumerating windows."""
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_title.lower() in title.lower():
                    results.append((hwnd, title))

        results = []
        win32gui.EnumWindows(enum_windows_callback, results)

        if results:
            self.hwnd, title = results[0]
            logger.info(f"Found window: {title} (handle: {self.hwnd})")
            return True
        else:
            logger.warning(f"Window '{self.window_title}' not found")
            self.hwnd = None
            return False

    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get window position and size.

        Returns:
            Tuple of (x, y, width, height) or None if window not found
        """
        if self._mock_mode:
            return None

        if not self.hwnd:
            if not self.find_window():
                return None

        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y

            self.last_position = (x, y, width, height)
            return (x, y, width, height)

        except Exception as e:
            logger.error(f"Error getting window rect: {e}")
            self.hwnd = None
            return None

    def get_client_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get client area (excluding title bar and borders).

        Returns:
            Tuple of (x, y, width, height) or None
        """
        if self._mock_mode:
            return None

        if not self.hwnd:
            if not self.find_window():
                return None

        try:
            # Get window rect
            window_rect = win32gui.GetWindowRect(self.hwnd)

            # Get client rect (relative to window)
            client_rect = win32gui.GetClientRect(self.hwnd)

            # Convert client rect to screen coordinates
            client_left, client_top = win32gui.ClientToScreen(self.hwnd, (0, 0))

            width = client_rect[2] - client_rect[0]
            height = client_rect[3] - client_rect[1]

            return (client_left, client_top, width, height)

        except Exception as e:
            logger.error(f"Error getting client rect: {e}")
            return None

    def is_window_valid(self) -> bool:
        """
        Check if window handle is still valid.

        Returns:
            True if window exists and is visible
        """
        if self._mock_mode:
            return False

        if not self.hwnd:
            return False

        try:
            return win32gui.IsWindow(self.hwnd) and win32gui.IsWindowVisible(self.hwnd)
        except:
            return False

    def bring_to_front(self):
        """Bring PokerStars window to foreground."""
        if self._mock_mode:
            return

        if self.hwnd and self.is_window_valid():
            try:
                win32gui.SetForegroundWindow(self.hwnd)
                logger.info("Brought PokerStars window to front")
            except Exception as e:
                logger.error(f"Could not bring window to front: {e}")

    def get_window_info(self) -> Dict:
        """
        Get comprehensive window information.

        Returns:
            Dictionary with window details
        """
        if self._mock_mode:
            return {}

        if not self.is_window_valid():
            return {}

        rect = self.get_window_rect()
        client_rect = self.get_client_rect()

        if not rect or not client_rect:
            return {}

        return {
            "hwnd": self.hwnd,
            "title": win32gui.GetWindowText(self.hwnd),
            "window": {
                "x": rect[0],
                "y": rect[1],
                "width": rect[2],
                "height": rect[3]
            },
            "client": {
                "x": client_rect[0],
                "y": client_rect[1],
                "width": client_rect[2],
                "height": client_rect[3]
            }
        }

    @staticmethod
    def is_available() -> bool:
        """Check if window finding is available on this platform."""
        return WIN32_AVAILABLE
