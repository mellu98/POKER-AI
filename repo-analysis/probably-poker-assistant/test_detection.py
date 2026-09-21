"""
Test card detection and OCR system.
"""
import sys
from pathlib import Path
import cv2
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.capture.window_finder import WindowFinder
from src.capture.screen_grabber import ScreenGrabber
from src.capture.region_mapper import RegionMapper
from src.detection.card_detector import CardDetector
from src.detection.text_reader import TextReader
from src.detection.game_state import GameStateTracker
from src.utils.logger import logger

def test_card_detection():
    """Test card detection."""
    logger.info("="*50)
    logger.info("Testing Card Detection")
    logger.info("="*50)
    
    # Initialize detector
    detector = CardDetector()
    
    if not detector.templates:
        logger.error("No card templates found")
        logger.info("Run: python tools/create_card_templates.py")
        return False
    
    # Find window and capture
    finder = WindowFinder()
    if not finder.find_window():
        logger.error("PokerStars window not found")
        return False
    
    grabber = ScreenGrabber()
    mapper = RegionMapper()
    
    if not mapper.is_calibrated():
        logger.error("Regions not calibrated")
        return False
    
    # Capture hole cards
    hole_region = mapper.get_region("hole_cards")
    if hole_region:
        hole_img = grabber.capture_region(*hole_region.to_tuple())
        if hole_img is not None:
            # Detect cards
            cards = detector.detect_hole_cards(hole_img)
            logger.info(f"✓ Detected hole cards: {cards}")
            
            # Save annotated image
            screenshots_dir = Path("screenshots/test_hands")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(screenshots_dir / "detected_hole_cards.png"), hole_img)
    
    # Capture community cards
    community_region = mapper.get_region("community_cards")
    if community_region:
        community_img = grabber.capture_region(*community_region.to_tuple())
        if community_img is not None:
            cards = detector.detect_community_cards(community_img)
            logger.info(f"✓ Detected community cards: {cards}")
            
            screenshots_dir = Path("screenshots/test_hands")
            cv2.imwrite(str(screenshots_dir / "detected_community_cards.png"), community_img)
    
    return True

def test_ocr():
    """Test OCR for chip/pot reading."""
    logger.info("\n" + "="*50)
    logger.info("Testing OCR")
    logger.info("="*50)
    
    reader = TextReader()
    
    finder = WindowFinder()
    if not finder.find_window():
        return False
    
    grabber = ScreenGrabber()
    mapper = RegionMapper()
    
    # Read pot
    pot_region = mapper.get_region("pot_amount")
    if pot_region:
        pot_img = grabber.capture_region(*pot_region.to_tuple())
        if pot_img is not None:
            pot = reader.read_pot_amount(pot_img)
            if pot:
                logger.info(f"✓ Read pot: {pot}")
            else:
                logger.warning("Could not read pot")
    
    # Read stack
    stack_region = mapper.get_region("player_stack")
    if stack_region:
        stack_img = grabber.capture_region(*stack_region.to_tuple())
        if stack_img is not None:
            stack = reader.read_stack_size(stack_img)
            if stack:
                logger.info(f"✓ Read stack: {stack}")
            else:
                 logger.warning("Could not read stack")
    
    return True

def test_full_pipeline():
    """Test complete detection pipeline."""
    logger.info("\n" + "="*50)
    logger.info("Testing Full Pipeline")
    logger.info("="*50)
    
    # Initialize components
    finder = WindowFinder()
    grabber = ScreenGrabber()
    mapper = RegionMapper()
    detector = CardDetector()
    reader = TextReader()
    tracker = GameStateTracker()
    
    # Find window
    if not finder.find_window():
        logger.error("PokerStars not found")
        return False
    
    # Capture all regions
    regions = mapper.get_all_regions()
    captures = grabber.capture_multiple_regions(regions)
    
    # Detect cards
    hole_cards = []
    community_cards = []
    
    if "hole_cards" in captures and captures["hole_cards"] is not None:
        hole_cards = detector.detect_hole_cards(captures["hole_cards"])
    
    if "community_cards" in captures and captures["community_cards"] is not None:
        community_cards = detector.detect_community_cards(captures["community_cards"])
    
    # Read amounts
    pot = None
    stack = None
    bet = None
    
    if "pot_amount" in captures and captures["pot_amount"] is not None:
        pot = reader.read_pot_amount(captures["pot_amount"])
    
    if "player_stack" in captures and captures["player_stack"] is not None:
        stack = reader.read_stack_size(captures["player_stack"])
    
    if "current_bet" in captures and captures["current_bet"] is not None:
        bet = reader.read_bet_amount(captures["current_bet"])
    
    # Update game state
    state = tracker.update_state(hole_cards, community_cards, pot, stack, bet)
    
    logger.info("\n" + "="*50)
    logger.info("GAME STATE:")
    logger.info("="*50)
    logger.info(f"Hole Cards: {state.hole_cards}")
    logger.info(f"Community: {state.community_cards}")
    logger.info(f"Round: {state.betting_round.value}")
    logger.info(f"Pot: ${state.pot_size:.2f}")
    logger.info(f"Stack: ${state.stack_size:.2f}")
    logger.info(f"Bet: ${state.current_bet:.2f}")
    
    return True

def main():
    """Run all detection tests."""
    logger.info("PHASE 3: Card Detection System Tests")
    logger.info("="*50)
    
    # Test card detection
    if not test_card_detection():
        logger.warning("Card detection test had issues")
    
    # Test OCR
    if not test_ocr():
        logger.warning("OCR test had issues")
    
    # Test full pipeline
    logger.info("\nFor full pipeline test: Ensure PokerStars is showing a hand with cards visible")
    print("Press Enter when ready or 's' to skip...")
    user_input = input()
    if user_input.lower() == 's':
        return 0

    if test_full_pipeline():
        logger.info("\n" + "="*50)
        logger.info("✓ PHASE 3 COMPLETE")
        logger.info("="*50)
        logger.info("Next: Proceed to Phase 4 (Hand Evaluation)")
        return 0
    else:
        logger.error("Full pipeline test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
