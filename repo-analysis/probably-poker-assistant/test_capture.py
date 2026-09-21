"""
Test screen capture system.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.capture.window_finder import WindowFinder
from src.capture.screen_grabber import ScreenGrabber
from src.capture.region_mapper import RegionMapper
from src.ui.calibration_tool import run_calibration
from src.utils.logger import logger
import time

def test_window_finder():
    """Test window detection."""
    logger.info("="*50)
    logger.info("Testing Window Finder")
    logger.info("="*50)
    
    finder = WindowFinder()
    
    if finder.find_window():
        info = finder.get_window_info()
        logger.info(f"Window: {info['title']}")
        logger.info(f"Position: {info['window']}")
        logger.info(f"Client area: {info['client']}")
        return True
    else:
        logger.error("Could not find PokerStars window")
        logger.info("Please open PokerStars and try again")
        return False

def test_screen_capture(finder: WindowFinder):
    """Test screen capture."""
    logger.info("\n" + "="*50)
    logger.info("Testing Screen Capture")
    logger.info("="*50)
    
    grabber = ScreenGrabber()
    rect = finder.get_client_rect()
    
    if not rect:
        logger.error("Could not get window rect")
        return False
    
    x, y, width, height = rect
    logger.info(f"Capturing region: ({x}, {y}, {width}, {height})")
    
    # Capture screenshot
    img = grabber.capture_region(x, y, width, height)
    
    if img is not None:
        logger.info(f"Captured image shape: {img.shape}")
        
        # Save test screenshot
        from pathlib import Path
        screenshots_dir = Path("screenshots/test_hands")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = screenshots_dir / "test_capture.png"
        grabber.save_capture(str(filepath), img)
        
        logger.info(f"✓ Screenshot saved to {filepath}")
        return True
    else:
        logger.error("Failed to capture screenshot")
        return False

def test_region_mapper():
    """Test region mapping."""
    logger.info("\n" + "="*50)
    logger.info("Testing Region Mapper")
    logger.info("="*50)
    
    mapper = RegionMapper()
    
    if mapper.is_calibrated():
        logger.info("✓ Regions already calibrated")
        regions = mapper.get_all_regions()
        for name, coords in regions.items():
            logger.info(f"  {name}: {coords}")
        return True
    else:
        logger.warning("Regions not calibrated yet")
        logger.info("Run calibration to set up regions")
        return False

def test_calibration():
    """Test calibration tool."""
    logger.info("\n" + "="*50)
    logger.info("Testing Calibration Tool")
    logger.info("="*50)
    
    logger.info("Starting calibration GUI...")
    logger.info("Follow on-screen instructions")
    
    result = run_calibration()
    
    if result:
        logger.info("✓ Calibration successful")
        
        # Save calibrated regions
        mapper = RegionMapper()
        mapper.window_info = result['window']
        
        for name, coords in result['regions'].items():
            x, y, w, h = coords
            mapper.set_region(name, x, y, w, h)
        
        mapper.calibrated = True
        mapper.save_regions()
        
        logger.info("✓ Regions saved to config")
        return True
    else:
        logger.warning("Calibration cancelled or failed")
        return False

def main():
    """Run all tests."""
    logger.info("PHASE 2: Screen Capture System Tests")
    logger.info("="*50)
    
    # Test 1: Window Finder
    if not test_window_finder():
        logger.error("Window finder test failed")
        return 1
    
    finder = WindowFinder()
    finder.find_window()
    
    # Test 2: Screen Capture
    if not test_screen_capture(finder):
        logger.error("Screen capture test failed")
        return 1
    
    # Test 3: Region Mapper
    test_region_mapper()
    
    # Test 4: Calibration (optional - requires user interaction)
    logger.info("\n" + "="*50)
    response = input("Run calibration tool? (y/n): ").strip().lower()
    if response == 'y':
        test_calibration()
    
    logger.info("\n" + "="*50)
    logger.info("✓ PHASE 2 COMPLETE")
    logger.info("="*50)
    logger.info("Next: Proceed to Phase 3 (Card Detection)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
