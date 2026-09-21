"""
Anchor Manager for stable UI tracking.
Finds a stable UI element on the screen and calculates relative positions for cards.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
from src.utils.logger import logger
from src.utils.config_loader import config_loader

class AnchorManager:
    """Manages UI anchors to handle window movement and resizing."""
    
    def __init__(self, anchor_dir: str = "models/anchors"):
        """Initialize anchor manager."""
        self.anchor_dir = Path(anchor_dir)
        self.anchor_dir.mkdir(parents=True, exist_ok=True)
        self.active_anchor_name: Optional[str] = None
        self.active_anchor_img: Optional[np.ndarray] = None
        self.relative_regions: Dict[str, Dict] = {}
        
        # Load active configuration if it exists
        self.load_config()

    def load_config(self):
        """Load anchor configuration and regions from config."""
        try:
            config = config_loader.load('anchor_config.json')
            self.active_anchor_name = config.get('active_anchor')
            if self.active_anchor_name:
                anchor_path = self.anchor_dir / f"{self.active_anchor_name}.png"
                if anchor_path.exists():
                    self.active_anchor_img = cv2.imread(str(anchor_path), cv2.IMREAD_GRAYSCALE)
                    logger.info(f"Loaded active anchor: {self.active_anchor_name}")
            
            self.relative_regions = config.get('regions', {})
            # Ensure internal consistency if types were lost during JSON serialization
            for name, data in self.relative_regions.items():
                if isinstance(data, list):
                    pass # Not expected for the top level objects
        except FileNotFoundError:
            logger.info("No anchor_config.json found. Need calibration.")
        except Exception as e:
            logger.error(f"Error loading anchor config: {e}")

    def save_config(self):
        """Save current anchor configuration and relative regions."""
        config = {
            "active_anchor": self.active_anchor_name,
            "regions": self.relative_regions
        }
        config_loader.save('anchor_config.json', config)
        logger.info("Saved anchor configuration.")

    def set_anchor(self, name: str, image: np.ndarray):
        """Set a new anchor image and save it."""
        self.active_anchor_name = name
        self.active_anchor_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        anchor_path = self.anchor_dir / f"{name}.png"
        cv2.imwrite(str(anchor_path), self.active_anchor_img)
        logger.info(f"New anchor saved: {name}")
        self.save_config()

    def find_anchor(self, screen_img: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the active anchor on the screen.
        Returns: (x, y, w, h) of the found anchor or None.
        """
        if self.active_anchor_img is None:
            logger.warning("No active anchor image loaded.")
            return None
            
        gray_screen = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY) if len(screen_img.shape) == 3 else screen_img
        
        # Template matching
        result = cv2.matchTemplate(gray_screen, self.active_anchor_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            h, w = self.active_anchor_img.shape
            logger.debug(f"Anchor '{self.active_anchor_name}' found at {max_loc} with confidence {max_val:.2f}")
            return (max_loc[0], max_loc[1], w, h)
            
        logger.debug(f"Anchor '{self.active_anchor_name}' not found (max confidence: {max_val:.2f})")
        return None

    def add_relative_region(self, name: str, anchor_pos: Tuple[int, int], region_rect: Tuple[int, int, int, int]):
        """
        Add a region relative to the anchor's top-left corner.
        anchor_pos: (x, y) of anchor
        region_rect: (x, y, w, h) of region in screen coordinates
        """
        rx, ry, rw, rh = region_rect
        ax, ay = anchor_pos
        
        self.relative_regions[name] = {
            "off_x": rx - ax,
            "off_y": ry - ay,
            "w": rw,
            "h": rh
        }
        logger.info(f"Added relative region '{name}'")
        self.save_config()

    def get_absolute_regions(self, anchor_pos: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Convert relative regions to absolute screen coordinates given an anchor position.
        """
        ax, ay = anchor_pos
        abs_regions = {}
        
        for name, data in self.relative_regions.items():
            abs_regions[name] = (
                int(ax + data['off_x']),
                int(ay + data['off_y']),
                int(data['w']),
                int(data['h'])
            )
        return abs_regions
