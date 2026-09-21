"""
Card detection using template matching.
Identifies playing cards by matching rank and suit templates.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from src.utils.logger import logger


class CardDetector:
    """
    Identifies playing cards using template matching.
    Uses separate templates for Ranks (A, K, Q...) and Suits (h, d, c, s).
    """

    # Default templates directory relative to project root
    DEFAULT_TEMPLATES_DIR = "models/templates"

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize card detector.

        Args:
            templates_dir: Path to templates directory. If None, uses default.
        """
        if templates_dir is None:
            # Find project root by looking for config directory
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "config").exists():
                    templates_dir = str(parent / self.DEFAULT_TEMPLATES_DIR)
                    break
            else:
                # Fallback to relative path
                templates_dir = self.DEFAULT_TEMPLATES_DIR

        self.templates_dir = Path(templates_dir)
        self.rank_templates: Dict[str, np.ndarray] = {}
        self.suit_templates: Dict[str, np.ndarray] = {}
        self.confidence_threshold = 0.75  # Configurable threshold
        self.load_templates()
        
    def load_templates(self):
        """Load rank and suit templates from disk."""
        rank_dir = self.templates_dir / "ranks"
        suit_dir = self.templates_dir / "suits"
        
        # Load Ranks
        if rank_dir.exists():
            for f in rank_dir.glob("*.png"):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.rank_templates[f.stem] = img
        
        # Load Suits
        if suit_dir.exists():
            for f in suit_dir.glob("*.png"):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.suit_templates[f.stem] = img
                    
        logger.info(f"Loaded {len(self.rank_templates)} ranks and {len(self.suit_templates)} suits")

    def detect_card(self, card_image: np.ndarray) -> Optional[str]:
        """
        Identify a single card image.
        
        Args:
            card_image: BGR image of a single card
            
        Returns:
            String representing card (e.g. 'Ah', 'Td') or None if unknown
        """
        if card_image is None or card_image.size == 0:
            return None
            
        # Preprocess
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        
        # We process the top corner for rank and suit as per build_template_db logic
        # Crop roughly where we expect them
        h, w = gray.shape
        roi_w = int(w * 0.45)
        roi_h = int(h * 0.65) # Extended to cover suit
        roi = gray[0:roi_h, 0:roi_w]
        
        # Identify Rank
        best_rank = self._match_template(roi, self.rank_templates)
        
        # Identify Suit
        # Suit is usually below the rank. We can search the whole ROI or split.
        best_suit = self._match_template(roi, self.suit_templates)
        
        if best_rank and best_suit:
            return f"{best_rank}{best_suit}"
            
        return None
        
    def _match_template(self, image: np.ndarray, templates: Dict[str, np.ndarray]) -> Optional[str]:
        """Find best matching template."""
        best_score = -1.0
        best_label = None
        
        for label, template in templates.items():
            # Template matching requires template <= image
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                continue
                
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_label = label
                
        # Log match scores for debugging (only at debug level)
        if best_score > 0.5:
            logger.debug(f"Template match: {best_label} with score {best_score:.4f}")

        if best_score >= self.confidence_threshold:
            return best_label

        return None

    def detect_hand(self, hand_image: np.ndarray, num_cards: int = 2) -> List[str]:
        """
        Detect multiple cards from a hand region image.

        Args:
            hand_image: BGR image containing the cards (e.g., hole cards region)
            num_cards: Expected number of cards in the image

        Returns:
            List of detected card strings (e.g., ['Ah', 'Kd'])
        """
        if hand_image is None or hand_image.size == 0:
            logger.warning("Empty hand image provided")
            return []

        detected_cards = []
        h, w = hand_image.shape[:2]

        if num_cards <= 0:
            return []

        # Calculate approximate card width based on number of cards
        # Assume cards are arranged horizontally with some overlap
        card_width = w // num_cards

        # Add some padding to account for spacing between cards
        padding = int(card_width * 0.1)

        for i in range(num_cards):
            # Calculate region for this card
            x_start = max(0, i * card_width - padding)
            x_end = min(w, (i + 1) * card_width + padding)

            if x_end <= x_start:
                continue

            # Extract individual card region
            card_region = hand_image[:, x_start:x_end]

            if card_region.size == 0:
                continue

            # Detect this card
            card = self.detect_card(card_region)

            if card:
                detected_cards.append(card)
                logger.debug(f"Detected card {i+1}: {card}")
            else:
                logger.debug(f"Could not detect card {i+1} in region")

        logger.info(f"Detected hand: {detected_cards}")
        return detected_cards

    def detect_community_cards(self, board_image: np.ndarray) -> List[str]:
        """
        Detect community cards (flop, turn, river) from board region.

        Args:
            board_image: BGR image of the community cards area

        Returns:
            List of detected card strings (0-5 cards)
        """
        if board_image is None or board_image.size == 0:
            return []

        # Community cards can be 0 (preflop), 3 (flop), 4 (turn), or 5 (river)
        # We'll try to detect up to 5 cards
        h, w = board_image.shape[:2]

        # Estimate card width - community cards are typically evenly spaced
        estimated_card_width = w // 5
        detected_cards = []

        for i in range(5):
            x_start = i * estimated_card_width
            x_end = (i + 1) * estimated_card_width

            card_region = board_image[:, x_start:x_end]

            if card_region.size == 0:
                continue

            card = self.detect_card(card_region)

            if card:
                detected_cards.append(card)

        # Filter based on what makes sense
        # If we detect 1 or 2 cards, something is wrong (can't have 1-2 community cards)
        if len(detected_cards) in [1, 2]:
            logger.warning(f"Unusual community card count: {len(detected_cards)}, returning empty")
            return []

        logger.info(f"Detected community cards: {detected_cards}")
        return detected_cards

    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for template matching."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Card detection threshold set to {self.confidence_threshold}")


if __name__ == "__main__":
    # Test card detector
    detector = CardDetector()
    print(f"Loaded {len(detector.rank_templates)} rank templates")
    print(f"Loaded {len(detector.suit_templates)} suit templates")
    print(f"Templates dir: {detector.templates_dir}")
