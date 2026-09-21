"""
Display Manager for Poker AI Assistant.
Handles the PyQt5 overlay window and painting.
"""
import sys
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush

class OverlaySignal(QObject):
    """Signals for overlay updates."""
    update_signal = pyqtSignal(object)

class PokerOverlay(QMainWindow):
    """Transparent overlay window."""
    
    def __init__(self, region_mapper):
        super().__init__()
        self.region_mapper = region_mapper
        self.decision = None
        self.game_state = None
        
        # Window setup
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Fullscreen
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        self.show()
        
    def update_data(self, data):
        """Update display data."""
        self.decision = data.get('decision')
        self.game_state = data.get('game_state')
        self.repaint()
        
    def paintEvent(self, event):
        """Draw overlay elements."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw HUD box in top-right or near table
        if self.decision:
            self._draw_hud(painter)
            
        # 2. Highlight cards if needed (optional)
        # 3. Draw debug info (optional)
        
    def _draw_hud(self, painter):
        """Draw main heads-up display with pot odds and EV."""
        # Config
        bg_color = QColor(0, 0, 0, 200)
        text_color = QColor(255, 255, 255)
        accent_color = QColor(0, 255, 0)  # Green for positive

        if self.decision.action == 'fold':
            accent_color = QColor(255, 0, 0)
        elif self.decision.action in ['raise', 'bet']:
            accent_color = QColor(0, 150, 255)  # Blue

        # Position (e.g., top left corner)
        x, y = 50, 50
        w, h = 320, 240  # Increased height for pot odds and EV

        # Draw background
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x, y, w, h, 10, 10)

        # Draw Side Bar
        painter.setBrush(QBrush(accent_color))
        painter.drawRoundedRect(x, y, 10, h, 10, 10)

        # Text Header (Action)
        painter.setPen(QPen(text_color))
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        action_text = f"{self.decision.action.upper()}"
        if self.decision.amount_bb:
            action_text += f" {self.decision.amount_bb:.1f} BB"
        painter.drawText(x + 20, y + 30, action_text)

        # Text Details (Hand Strength)
        painter.setFont(QFont("Arial", 12))
        hand_text = f"Hand: {self.decision.hand_evaluation.description}"
        painter.drawText(x + 20, y + 55, hand_text)

        # Equity
        equity_text = f"Equity: {self.decision.equity:.1f}%"
        painter.drawText(x + 20, y + 78, equity_text)

        # Pot Odds (if available)
        if self.decision.pot_odds and self.decision.pot_odds > 0:
            pot_odds_text = f"Pot Odds: {self.decision.pot_odds:.1f}:1"
            painter.drawText(x + 20, y + 101, pot_odds_text)

            # Calculate required equity
            required_equity = 100 / (self.decision.pot_odds + 1)
            req_text = f"Required: {required_equity:.1f}%"
            painter.drawText(x + 160, y + 101, req_text)

            # Calculate and display EV
            ev = self._calculate_ev()
            if ev is not None:
                # Color code EV (green positive, red negative)
                if ev > 0:
                    ev_color = QColor(0, 255, 100)
                elif ev < 0:
                    ev_color = QColor(255, 100, 100)
                else:
                    ev_color = QColor(255, 255, 100)  # Yellow for breakeven

                painter.setPen(QPen(ev_color))
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                ev_text = f"EV: {ev:+.2f} chips"
                painter.drawText(x + 20, y + 124, ev_text)
                painter.setPen(QPen(text_color))
                painter.setFont(QFont("Arial", 12))

                y_reason = y + 152
            else:
                y_reason = y + 128
        else:
            y_reason = y + 105

        # Reasoning
        painter.setFont(QFont("Arial", 10))
        for reason in self.decision.reasoning[:3]:  # Limit to 3 lines
            painter.drawText(x + 20, y_reason, f"• {reason}")
            y_reason += 18

    def _calculate_ev(self):
        """Calculate expected value for the current decision."""
        if not self.game_state or not self.decision.pot_odds:
            return None

        try:
            pot = self.game_state.pot_size or 0
            bet = self.game_state.current_bet or 0
            equity = self.decision.equity / 100  # Convert to decimal

            if bet <= 0:
                return None

            # EV = (win_amount * equity) - (lose_amount * (1 - equity))
            # Win amount = pot + bet (we win pot plus opponent's bet)
            # Lose amount = bet (we lose our call)
            win_amount = pot + bet
            lose_amount = bet

            ev = (win_amount * equity) - (lose_amount * (1 - equity))
            return ev
        except Exception:
            return None

class DisplayManager:
    """Manages the overlay window."""

    def __init__(self, region_mapper):
        # Use existing QApplication if available, otherwise create one
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        self.overlay = PokerOverlay(region_mapper)
        self.signals = OverlaySignal()
        self.signals.update_signal.connect(self.overlay.update_data)

    def update_overlay(self, data):
        """Thread-safe update trigger.

        Args:
            data: Dict with 'decision' and 'game_state' keys
        """
        self.signals.update_signal.emit(data)

    def process_events(self):
        """Process GUI events (if running in same thread loop)."""
        self.app.processEvents()

    def show(self):
        """Show the overlay window."""
        self.overlay.show()

    def hide(self):
        """Hide the overlay window."""
        self.overlay.hide()
