"""
Interactive Anchor Calibration Tool.
Allows user to define an anchor image and card regions relative to it.
"""
import cv2
import numpy as np
import time
from pathlib import Path
from src.capture.window_finder import WindowFinder
from src.capture.screen_grabber import ScreenGrabber
from src.capture.anchor_manager import AnchorManager
from src.utils.logger import logger

class AnchorCalibrator:
    def __init__(self, window_title="Ignition"):
        self.wf = WindowFinder(window_title)
        self.sg = ScreenGrabber()
        self.am = AnchorManager()
        self.current_screen = None
        self.selection = None
        self.start_point = None
        self.mode = "ANCHOR" # ANCHOR, HOLE, BOARD
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.selection = (x, y, 0, 0)
        elif event == cv2.EVENT_MOUSEMOVE and self.start_point:
            ix, iy = self.start_point
            self.selection = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        elif event == cv2.EVENT_LBUTTONUP:
            ix, iy = self.start_point
            self.selection = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            self.start_point = None
            self.handle_selection()

    def handle_selection(self):
        x, y, w, h = self.selection
        if w < 5 or h < 5: return
        
        crop = self.current_screen[y:y+h, x:x+w]
        
        if self.mode == "ANCHOR":
            self.am.set_anchor("main_anchor", crop)
            self.anchor_pos = (x, y)
            print("Anchor set! Now draw a box around your HOLE CARDS.")
            self.mode = "HOLE"
        elif self.mode == "HOLE":
            self.am.add_relative_region("hole_cards", self.anchor_pos, (x, y, w, h))
            print("Hole cards region set! Now draw a box around the BOARD (COOMUNITY CARDS).")
            self.mode = "BOARD"
        elif self.mode == "BOARD":
            self.am.add_relative_region("community_cards", self.anchor_pos, (x, y, w, h))
            print("Board region set! Calibration complete. Press 'q' to exit.")
            self.mode = "DONE"

    def run(self):
        if not self.wf.find_window():
            print("Could not find window. Please open Ignition.")
            return

        self.wf.bring_to_front()
        time.sleep(1) # Wait for window to stabilize
        
        rect = self.wf.get_client_rect()
        if not rect: return
        
        self.current_screen = self.sg.capture_region(rect)
        if self.current_screen is None: return
        
        print("\n--- ANCHOR CALIBRATION ---")
        print("1. Find a static part of the UI (e.g., logo, stable menu icon).")
        print("2. Click and drag to select it as the ANCHOR.")
        print("Press 'r' to refresh screen, 'q' to quit.")
        
        cv2.setWindowTitle("Calibration", "Anchor Calibration - Draw ANCHOR")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        
        while True:
            display = self.current_screen.copy()
            
            # Draw instructions
            text = f"Mode: {self.mode}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.selection:
                x, y, w, h = self.selection
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_screen = self.sg.capture_region(rect)
                self.selection = None
                print("Screen refreshed.")
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = AnchorCalibrator()
    calibrator.run()
