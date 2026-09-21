"""
Interactive calibration tool for defining screen regions.
Uses PyQt5 to create transparent overlay for region selection.
"""
import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QMainWindow, QFileDialog, QSizeGrip)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QImage, QPixmap
from typing import Optional, Tuple, List
import cv2
import numpy as np
import win32gui
import win32con

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.capture.window_finder import WindowFinder
from src.capture.screen_grabber import ScreenGrabber
from src.capture.region_mapper import RegionMapper
from src.utils.logger import logger


class CalibrationOverlay(QWidget):
    """
    Transparent overlay for selecting regions.
    Supports manual moving (drag anywhere) and resizing (bottom-right grip).
    """
    
    def __init__(self, window_rect: Tuple[int, int, int, int], bg_image: Optional[np.ndarray] = None):
        super().__init__()
        self.window_rect = window_rect
        self.bg_image = bg_image  # Static image for calibration mode
        self.selection_start = None
        self.selection_end = None
        self.current_rect = None
        
        # Define regions: Hero + 8 Opponents
        self.regions_to_calibrate = [
            "hero_cards",
            "player_stack",
            "current_bet",
            "action_buttons",
            "community_cards",
            "pot_amount"
        ]
        
        # Add 8 opponents
        for i in range(1, 9):
            self.regions_to_calibrate.append(f"opponent_{i}_area")
            
        self.region_descriptions = {
            "hero_cards": "Draw a box around YOUR two hole cards. Include the full area.",
            "community_cards": "Draw a box around the table center for the 5 community cards.",
            "pot_amount": "Draw a box tightly around the TOTAL POT number text.",
            "player_stack": "Draw a box around YOUR numeric stack size/balance.",
            "current_bet": "Draw a box around YOUR current bet amount.",
            "action_buttons": "Draw a box around the bottom-right actions (Fold/Call/Raise)."
        }
        
        # Add descriptions for opponents
        for i in range(1, 9):
            self.region_descriptions[f"opponent_{i}_area"] = f"Draw a box around OPPONENT {i}'s entire area (Name, Stack, Cards)."

        self.current_region_index = 0
        self.calibrated_regions = {}
        
        # No padding needed for manual mode
        self.padding = 0
        
        # Dragging state
        self.dragging_window = False
        self.drag_position = QPoint()
        
        # Passthrough mode
        self.passthrough_enabled = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup overlay window."""
        x, y, width, height = self.window_rect
        
        self.setWindowTitle("Poker AI - Calibration")
        
        # Set initial geometry
        self.setGeometry(x, y, width, height)
            
        if self.bg_image is None:
             # Live mode - Frameless to ensure transparency works on Windows
             # Keep StaysOnTop so it's visible over the game
             self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
             
             # Enable transparency
             self.setAttribute(Qt.WA_TranslucentBackground)
             
        else:
            # Image mode
            self.setWindowFlags(Qt.Window)
            self.setFixedSize(width, height)
            
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        
        # Instructions label
        self.instruction_label = QLabel(self)
        self.instruction_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 200);
                color: white;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #00ff00;
            }
        """)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        # Anchor to top-center, resizing with window if possible, but fixed geometry is safer for now
        self.instruction_label.setGeometry(10, 10, 600, 150) 
        self.update_instructions()

        # Add Size Grip for resizing (Bottom-Right corner)
        if self.bg_image is None:
            self.size_grip = QSizeGrip(self)
            # Position it in resizeEvent
    
    def resizeEvent(self, event):
        """Handle resize to keep instructions centered."""
        if hasattr(self, 'instruction_label'):
             self.instruction_label.setGeometry(10, 10, self.width() - 20, 150)
        if hasattr(self, 'size_grip') and self.size_grip:
            self.size_grip.setGeometry(self.width() - self.size_grip.width(),
                                       self.height() - self.size_grip.height(),
                                       self.size_grip.width(),
                                       self.size_grip.height())
        super().resizeEvent(event)
        
    def resync_window(self):
        """Find window again and move overlay to match."""
        # Optional helper, but manual move is primary now
        finder = WindowFinder(window_title="Ignition")
        if not finder.find_window():
             finder = WindowFinder(window_title="PokerStars")
             finder.find_window()
             
        rect = finder.get_window_rect()
        if rect:
            x, y, w, h = rect
            self.setGeometry(x, y, w, h)
            self.instruction_label.setText("Snapped to detected window!")

    # ... (skipping update_instructions as it doesn't change much, but I need to make sure I don't cut it off)
    # Actually I should be careful with replace_file_content.
    # I will stick to what I need to change.
    def update_instructions(self):
        """Update instruction text."""
        if self.current_region_index < len(self.regions_to_calibrate):
            region = self.regions_to_calibrate[self.current_region_index]
            desc = self.region_descriptions.get(region, "")
            
            text = f"STEP {self.current_region_index + 1}/{len(self.regions_to_calibrate)}\n"
            if self.current_region_index == 0:
                text += "CONTROLS:\n"
                text += "- Right Click + Drag: MOVE window\n"
                text += "- Bottom-Right Corner: RESIZE window\n"
                text += "- Left Click + Drag: DRAW SELECT box\n\n"
            text += f"TARGET: {region.replace('_', ' ').upper()}\n"
            text += f"{desc}\n"
            text += "(P: Previous / R: Reset / S: Sync)\n"
            text += "** T: Toggle Interact Mode (Click-Through) **"
            
            if self.passthrough_enabled:
                 text += "\n\n[INTERACT MODE ENABLED]\n"
                 text += "Overlay is click-through.\n"
                 text += "Alt+Tab back to this window and press T to disable."
        else:
            text = "Calibration complete!\n\nPress ESC to SAVE and EXIT"
        
        self.instruction_label.setText(text)
    
    def mousePressEvent(self, event):
        """Start selection or drag window."""
        if event.button() == Qt.LeftButton:
            # Start dragging window
            self.dragging_window = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.RightButton:
            # Start selection
            self.selection_start = event.pos()
            self.selection_end = event.pos()
            self.update()
    
    def mouseMoveEvent(self, event):
        """Update selection or drag window."""
        if self.dragging_window:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
        elif self.selection_start:
            self.selection_end = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Finalize selection or stop dragging."""
        if event.button() == Qt.LeftButton:
            self.dragging_window = False
        elif event.button() == Qt.RightButton and self.selection_start:
            self.selection_end = event.pos()
            
            # Normalize coordinates
            x1 = min(self.selection_start.x(), self.selection_end.x())
            y1 = min(self.selection_start.y(), self.selection_end.y())
            x2 = max(self.selection_start.x(), self.selection_end.x())
            y2 = max(self.selection_start.y(), self.selection_end.y())
            width = x2 - x1
            height = y2 - y1
            
            # Save region
            if width > 5 and height > 5:
                # Check if we are done
                if self.current_region_index >= len(self.regions_to_calibrate):
                    return

                # Add current region to calibrated list
                region_name = self.regions_to_calibrate[self.current_region_index]
                
                # Coordinates are relative to the client area (which is what we want)
                self.calibrated_regions[region_name] = (
                    x1, y1, width, height
                )
                
                logger.info(f"Calibrated {region_name}: {self.calibrated_regions[region_name]}")
                
                # Move to next region
                self.current_region_index += 1
                self.update_instructions()
            
            # Reset selection visual (not data)
            self.selection_start = None
            self.selection_end = None
            self.update()
    
    def paintEvent(self, event):
        """Draw overlay elements."""
        painter = QPainter(self)
        
        if self.bg_image is not None:
            # Draw background image
            height, width, channel = self.bg_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.bg_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            painter.drawImage(0, 0, q_img)
        else:
            # Draw almost clear background for hit testing (needed for mouse events)
            # Using alpha 1 makes it practically invisible but interactable
            painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
            
            # Draw a dashed border so user sees the window boundaries
            pen = QPen(QColor(100, 255, 100), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(1, 1, self.width()-2, self.height()-2)
        
        # Draw selection rectangle
        if self.selection_start and self.selection_end:
            pen = QPen(QColor(0, 255, 0), 3)
            painter.setPen(pen)
            
            x1 = min(self.selection_start.x(), self.selection_end.x())
            y1 = min(self.selection_start.y(), self.selection_end.y())
            x2 = max(self.selection_start.x(), self.selection_end.x())
            y2 = max(self.selection_start.y(), self.selection_end.y())
            
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)
            
            # Draw dimensions
            painter.setFont(QFont("Arial", 12))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                rect.topLeft() - QPoint(0, 5),
                f"{rect.width()} x {rect.height()}"
            )
        
        # Draw previously calibrated regions
        pen = QPen(QColor(0, 100, 255), 2)
        painter.setPen(pen)
        
        for region_name, coords in self.calibrated_regions.items():
            x, y, w, h = coords
            rect = QRect(x, y, w, h)
            painter.drawRect(rect)
            
            # Draw label
            painter.drawText(rect.topLeft() - QPoint(0, 5), region_name)

        # Draw resize grip hint (bottom right)
        if self.bg_image is None:
             # Changes color if passthrough
             color = QColor(255, 100, 100) if self.passthrough_enabled else QColor(255, 255, 255)
             painter.setPen(QPen(color, 2))
             w, h = self.width(), self.height()
             for i in range(3):
                 offset = (i + 1) * 5
                 painter.drawLine(w - offset - 5, h, w, h - offset - 5)
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif (event.key() == Qt.Key_Backspace or event.key() == Qt.Key_P) and self.current_region_index > 0:
            print("Previous step triggered")
            # Go back one step
            self.current_region_index -= 1
            region_name = self.regions_to_calibrate[self.current_region_index]
            if region_name in self.calibrated_regions:
                del self.calibrated_regions[region_name]
            self.update_instructions()
            self.update()
        elif event.key() == Qt.Key_S:
            print("Sync triggered")
            self.resync_window()
        elif event.key() == Qt.Key_R:
             print("Reset triggered")
             # Reset current selection
             self.selection_start = None
             self.selection_end = None
             self.update()
        elif event.key() == Qt.Key_T:
            self.toggle_passthrough()

    def toggle_passthrough(self):
        """Toggle click-through mode for interacting with the game."""
        self.passthrough_enabled = not self.passthrough_enabled
        
        # Windows specific: Set/Unset WS_EX_TRANSPARENT
        if self.bg_image is None:
             # Need to get window handle
             hwnd = int(self.winId())
             style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
             
             if self.passthrough_enabled:
                 win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style | win32con.WS_EX_TRANSPARENT)
                 print("Passthrough ENABLED")
             else:
                 win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style & ~win32con.WS_EX_TRANSPARENT)
                 print("Passthrough DISABLED")
        
        self.update_instructions()
        self.update()

def run_calibration_from_image(image_path: str) -> Optional[dict]:
    """Run calibration using a static image."""
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image: {image_path}")
        return None
        
    # Convert BGR to RGB for Qt
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    overlay = CalibrationOverlay((100, 100, width, height), bg_image=img)
    overlay.show()
    app.exec_()
    
    if overlay.calibrated_regions:
        return {
            "window": {
                "title": "Ignition",
                "width": width,
                "height": height
            },
            "regions": overlay.calibrated_regions
        }
    return None

def run_calibration() -> Optional[dict]:
    """Run calibration tool."""
    # Check if we want to use image mode
    print("Select mode:")
    print("1. Live Window (Ignition/PokerStars)")
    print("2. Static Image (screenshots/calibration/calibration_source.png)")
    
    # Simple CLI argument check or prompt
    # Since we can't easily interact with stdin in some envs, let's check for a file flag
    # or just try to find window first, if not found, ask for image
    
    finder = WindowFinder(window_title="Ignition") # Try Ignition first
    found = finder.find_window()
    if not found:
        finder = WindowFinder(window_title="PokerStars")
        found = finder.find_window()
        
    if found:
        choice = "1"
        # However, user explicitly asked to use images. 
        # But we are in a tool call. Let's try to detect if image exists and prefer it if window not found?
        pass 
    else:
        logger.info("Window not found. Checking for calibration image...")
        choice = "2"

    # Force image mode if specified or window missing
    image_path = os.path.join(project_root, "screenshots", "calibration", "calibration_source.png")
    
    if not found and os.path.exists(image_path):
        logger.info(f"Using calibration image: {image_path}")
        return run_calibration_from_image(image_path)
    
    if not found:
        logger.error("No game window found and no calibration image found.")
        return None
        
    # Live window mode
    window_info = finder.get_window_info()
    if not window_info:
        return None
        
    window_rect = (
        window_info['window']['x'],
        window_info['window']['y'],
        window_info['window']['width'],
        window_info['window']['height']
    )
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    overlay = CalibrationOverlay(window_rect)
    overlay.show()
    app.exec_()
    
    if overlay.calibrated_regions:
        return {
            "window": window_info,
            "regions": overlay.calibrated_regions
        }
    return None

    # ChoiceOverride = "2" # Default to image mode for this session since user provided inputs


if __name__ == "__main__":
    try:
        regions = run_calibration()
        if regions:
            import json
            from pathlib import Path
            
            config_dir = Path(project_root) / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Structure for regions.json
            output = {
                "window": {
                    "title": regions["window"].get("title", "Ignition"), 
                    "x": regions["window"]["window"]["x"],
                    "y": regions["window"]["window"]["y"],
                    "width": regions["window"]["window"]["width"],
                    "height": regions["window"]["window"]["height"]
                },
                "regions": regions["regions"]
            }
            
            if "window" in regions and "window" in regions["window"]:
                 output["window"] = regions["window"]["window"]
                 if "title" in regions["window"]:
                     output["window"]["title"] = regions["window"]["title"]

            with open(config_dir / "regions.json", "w") as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Saved calibration to {config_dir / 'regions.json'}")
            
    except Exception as e:
        import traceback
        error_msg = f"CRASH: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("crash_log.txt", "w") as f:
            f.write(error_msg)
        input("Press Enter to exit...") # Keep console open if possible
