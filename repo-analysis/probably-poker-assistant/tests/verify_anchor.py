"""
Verification script for Anchor Manager.
Mocks a screen and anchor to verify coordinate calculations and template matching.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from src.capture.anchor_manager import AnchorManager

def test_anchor_logic():
    # 1. Create a dummy screen (800x600)
    screen = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # 2. Define an "Anchor" (a red square at 100, 100)
    anchor_color = (0, 0, 255)
    cv2.rectangle(screen, (100, 100), (150, 150), anchor_color, -1)
    
    # 3. Define a "Hole Card" (a green square at 200, 200)
    hole_color = (0, 255, 0)
    cv2.rectangle(screen, (200, 200), (280, 250), hole_color, -1)
    
    # 4. Save anchor for testing
    am = AnchorManager(anchor_dir="models/test_anchors")
    anchor_img = screen[100:150, 100:150]
    am.set_anchor("test_anchor", anchor_img)
    
    # 5. Add relative region
    # Anchor is at (100, 100), Hole is at (200, 200, 80, 50)
    am.add_relative_region("hole_cards", (100, 100), (200, 200, 80, 50))
    
    # 6. Verify Detection
    found = am.find_anchor(screen)
    print(f"Anchor found at: {found}")
    
    if found:
        ax, ay, aw, ah = found
        print(f"Anchor find result: x={ax}, y={ay}, w={aw}, h={ah}")
        print(f"Stored relative regions: {am.relative_regions}")
        
        abs_regions = am.get_absolute_regions((ax, ay))
        print(f"Calculated Absolute Regions: {abs_regions}")
        
        # Check if the calculated absolute region matches our green square
        target = tuple(abs_regions["hole_cards"])
        if target == (200, 200, 80, 50):
            print("SUCCESS: Relative coordinate logic is perfect.")
        else:
            print(f"FAILURE: Expected (200, 200, 80, 50), got {target}")

if __name__ == "__main__":
    test_anchor_logic()
