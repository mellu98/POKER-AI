"""
Fast screen capture using MSS library.
Captures specific regions of the screen efficiently.
"""
import mss
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Union
from src.utils.logger import logger

class ScreenGrabber:
    """Efficient screen region capture."""

    def __init__(self):
        """Initialize screen grabber."""
        # Don't create MSS here - it's not thread-safe
        # MSS will be created fresh for each capture operation
        self.last_capture = None
        self.last_full_screen = None
        self.capture_count = 0

    def capture_screen(self, monitor_index: int = 0) -> Optional[np.ndarray]:
        """
        Capture the entire screen or a specific monitor.

        Args:
            monitor_index: Index of monitor to capture (0 = all monitors, 1+ = specific monitor)

        Returns:
            Numpy array (BGR format) of full screen or None on error
        """
        try:
            # Create MSS instance fresh for thread safety
            with mss.mss() as sct:
                # Get monitor info (0 = all monitors combined, 1+ = individual monitors)
                monitors = sct.monitors
                if monitor_index >= len(monitors):
                    logger.warning(f"Monitor index {monitor_index} out of range, using primary")
                    monitor_index = 1 if len(monitors) > 1 else 0

                monitor = monitors[monitor_index]

                # Capture screenshot
                screenshot = sct.grab(monitor)

                # Convert to numpy array
                img = np.array(screenshot)

                # Convert BGRA to BGR (remove alpha, keep BGR order from MSS)
                img = img[:, :, :3]

                self.last_full_screen = img
                self.capture_count += 1

                logger.debug(f"Captured full screen: {img.shape}")
                return img

        except Exception as e:
            logger.error(f"Error capturing full screen: {e}")
            return None

    def extract_region(self,
                       screen: np.ndarray,
                       region: Union[Tuple[int, int, int, int], Dict, None]) -> Optional[np.ndarray]:
        """
        Extract a region from an already-captured screen image.

        Args:
            screen: Full screen numpy array (BGR format)
            region: Either tuple (x, y, width, height) or dict with those keys, or None

        Returns:
            Cropped numpy array or None if invalid region
        """
        if screen is None:
            logger.warning("Cannot extract region from None screen")
            return None

        if region is None:
            logger.warning("Region is None, cannot extract")
            return None

        # Handle both tuple and dict formats
        if isinstance(region, dict):
            x = region.get('x', region.get('off_x', 0))
            y = region.get('y', region.get('off_y', 0))
            width = region.get('width', region.get('w', 0))
            height = region.get('height', region.get('h', 0))
        elif isinstance(region, (tuple, list)) and len(region) == 4:
            x, y, width, height = region
        else:
            logger.error(f"Invalid region format: {region}")
            return None

        # Validate bounds
        screen_height, screen_width = screen.shape[:2]

        if x < 0 or y < 0:
            logger.warning(f"Region has negative coordinates: ({x}, {y})")
            x = max(0, x)
            y = max(0, y)

        if x + width > screen_width:
            logger.warning(f"Region extends beyond screen width: {x + width} > {screen_width}")
            width = screen_width - x

        if y + height > screen_height:
            logger.warning(f"Region extends beyond screen height: {y + height} > {screen_height}")
            height = screen_height - y

        if width <= 0 or height <= 0:
            logger.warning(f"Region has invalid dimensions: {width}x{height}")
            return None

        # Extract the region
        extracted = screen[y:y+height, x:x+width].copy()

        logger.debug(f"Extracted region ({x}, {y}, {width}, {height}): shape {extracted.shape}")
        return extracted
    
    def capture_region(self,
                       x: int,
                       y: int,
                       width: int,
                       height: int) -> Optional[np.ndarray]:
        """
        Capture specific screen region.

        Args:
            x: Left coordinate
            y: Top coordinate
            width: Region width
            height: Region height

        Returns:
            Numpy array (BGR format) or None on error
        """
        try:
            # Define monitor region
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }

            # Create MSS instance fresh for thread safety
            with mss.mss() as sct:
                # Capture screenshot
                screenshot = sct.grab(monitor)

                # Convert to numpy array (RGB)
                img = np.array(screenshot)

                # Convert RGB to BGR (OpenCV format)
                img = img[:, :, :3]  # Remove alpha channel
                img = img[:, :, ::-1]  # RGB to BGR

                self.last_capture = img
                self.capture_count += 1

                return img

        except Exception as e:
            logger.error(f"Error capturing region ({x},{y},{width},{height}): {e}")
            return None
    
    def capture_multiple_regions(self, regions: dict) -> dict:
        """
        Capture multiple regions efficiently.
        
        Args:
            regions: Dict of region_name -> (x, y, width, height)
        
        Returns:
            Dict of region_name -> numpy array
        """
        captures = {}
        
        for name, coords in regions.items():
            if len(coords) == 4:
                x, y, w, h = coords
                capture = self.capture_region(x, y, w, h)
                if capture is not None:
                    captures[name] = capture
                else:
                    logger.warning(f"Failed to capture region: {name}")
        
        return captures
    
    def save_capture(self, filepath: str, image: np.ndarray = None):
        """
        Save captured image to file.
        
        Args:
            filepath: Path to save image
            image: Image to save (uses last_capture if None)
        """
        if image is None:
            image = self.last_capture
        
        if image is None:
            logger.error("No image to save")
            return
        
        try:
            # Convert BGR to RGB for saving
            image_rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(image_rgb)
            pil_image.save(filepath)
            logger.info(f"Saved capture to {filepath}")
        except Exception as e:
            logger.error(f"Error saving capture: {e}")
    
    def get_stats(self) -> dict:
        """
        Get capture statistics.
        
        Returns:
            Dict with capture stats
        """
        return {
            "total_captures": self.capture_count,
            "last_capture_shape": self.last_capture.shape if self.last_capture is not None else None
        }
    
    def __del__(self):
        """Cleanup - no longer needed as MSS is created fresh each time."""
        pass
