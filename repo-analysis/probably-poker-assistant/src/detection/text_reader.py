"""
OCR for reading chip counts, pot sizes, and bet amounts.
"""
import cv2
import numpy as np
import pytesseract
import re
from typing import Optional
import sys
from pathlib import Path

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import logger

class TextReader:
    """Read text from poker table using OCR."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize text reader.
        
        Args:
            tesseract_path: Path to Tesseract executable
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Default Windows path
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR.
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            return image
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Increase contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Resize for better OCR (Tesseract works best with larger text)
        scale_factor = 2
        height, width = denoised.shape
        resized = cv2.resize(denoised, (width * scale_factor, height * scale_factor))
        
        return resized
    
    def read_number(self, image: np.ndarray) -> Optional[float]:
        """
        Read number from image (chip count, bet, pot).
        
        Args:
            image: Image containing number
        
        Returns:
            Parsed number or None
        """
        if image is None or image.size == 0:
            return None

        # Preprocess
        processed = self.preprocess_image(image)
        if processed is None:
            return None
        
        # OCR with number-only configuration
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,$KM'
        try:
            text = pytesseract.image_to_string(processed, config=custom_config)
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}. Is Tesseract installed?")
            return None
        
        # Clean and parse
        text = text.strip().replace(' ', '').replace(',', '')
        
        if not text:
            return None

        logger.debug(f"OCR raw text: '{text}'")
        
        # Parse number (handle K, M suffixes)
        try:
            val = 0.0
            if 'K' in text.upper():
                val = float(re.sub(r'[^\d.]', '', text.upper().replace('K', ''))) * 1000
            elif 'M' in text.upper():
                 val = float(re.sub(r'[^\d.]', '', text.upper().replace('M', ''))) * 1_000_000
            else:
                val = float(re.sub(r'[^\d.]', '', text))
            
            logger.debug(f"Parsed number: {val}")
            return val
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse number from '{text}': {e}")
            return None
    
    def read_pot_amount(self, image: np.ndarray) -> Optional[float]:
        """
        Read pot amount.
        
        Args:
            image: Image of pot display
        
        Returns:
            Pot amount or None
        """
        amount = self.read_number(image)
        if amount is not None:
            logger.info(f"Pot: {amount}")
        return amount
    
    def read_stack_size(self, image: np.ndarray) -> Optional[float]:
        """
        Read player stack size.
        
        Args:
            image: Image of stack display
        
        Returns:
            Stack size or None
        """
        amount = self.read_number(image)
        if amount is not None:
            logger.info(f"Stack: {amount}")
        return amount
    
    def read_bet_amount(self, image: np.ndarray) -> Optional[float]:
        """
        Read current bet amount.
        
        Args:
            image: Image of bet display
        
        Returns:
            Bet amount or None
        """
        amount = self.read_number(image)
        if amount is not None:
            logger.info(f"Bet: {amount}")
        return amount
