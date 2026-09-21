"""
Control Panel UI module for Poker Assistant.

Provides a dockable control panel window with:
- Manual card input with dropdown selectors
- Real-time statistics display
- Calibration controls
- System tray integration
- First-run wizard for new users
"""

from .main_panel import ControlPanelWindow
from .system_tray import SystemTrayManager
from .first_run_wizard import FirstRunWizard, needs_first_run_setup, show_first_run_wizard

__all__ = [
    'ControlPanelWindow',
    'SystemTrayManager',
    'FirstRunWizard',
    'needs_first_run_setup',
    'show_first_run_wizard'
]
