
import cv2
import numpy as np
import os
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "models" / "card_templates"
OUT_DIR = PROJECT_ROOT / "models" / "templates"
RANK_DIR = OUT_DIR / "ranks"
SUIT_DIR = OUT_DIR / "suits"

def ensure_dirs():
    RANK_DIR.mkdir(parents=True, exist_ok=True)
    SUIT_DIR.mkdir(parents=True, exist_ok=True)

def process_templates():
    ensure_dirs()
    
    # Iterate over all pngs
    files = list(RAW_DIR.glob("*.png"))
    # Clear existing templates for a clean rebuild
    for d in [RANK_DIR, SUIT_DIR]:
        if d.exists():
            for f in d.glob("*.png"):
                f.unlink()
    
    print(f"Found {len(files)} raw templates.")
    
    for f in files:
        name = f.stem # e.g. "Ah", "10c"
        
        if len(name) == 2:
            rank_char = name[0] # 'A'
            suit_char = name[1] # 'h'
        elif len(name) == 3 and name.startswith("10"):
            rank_char = "T" # Map 10 to T
            suit_char = name[2] # 'c'
        else:
            print(f"Skipping {name}: invalid format (needs 'Ah' or '10h')")
            continue
            
        # Validate chars
        if rank_char not in "23456789TJQKA":
            print(f"Skipping {name}: invalid rank {rank_char} from {name}")
            continue
        if suit_char not in "hdcs":
            print(f"Skipping {name}: invalid suit {suit_char} from {name}")
            continue

        img = cv2.imread(str(f))
        if img is None: continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Isolate the bright card body using Otsu's Binarization
        # This separates the card from the table background automatically
        _, card_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        card_contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Cards are the largest white blobs
        card_candidates = []
        for cnt in card_contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 40 and ch > 60:
                card_candidates.append((x, y, cw, ch))
        
        if not card_candidates:
            roi = gray
        else:
            # Take the LARGEST candidate (The Card) to avoid small high-up UI elements
            card_candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
            cx, cy, cw, ch = card_candidates[0]
            # Crop to top-left of card (Rank and Suit)
            # 50% width and 65% height to ensure full 10 and JQK are caught
            roi = gray[cy:cy + int(ch*0.65), cx:cx + int(cw*0.50)]

        # Debug save ROI
        cv2.imwrite(str(OUT_DIR / f"debug_roi_{name}.png"), roi)

        # 2. Extract symbols from the card ROI
        # Rank and Suit are the darkest things on the card
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for significant chunks that are LIKELY Rank or Suit
        valid_contours = []
        if hierarchy is not None:
            hi = hierarchy[0]
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                # If it's a child (has a parent) it's more likely to be a symbol inside the card body
                # Or if it's external but within size bounds.
                parent_idx = hi[i][3]
                if w > 4 and h > 8:
                    # Prefer children of the main ROI or significant chunks
                    valid_contours.append((x, y, w, h))
        
        # Determine the split point: Half-split based on the card body's top-left ROI
        h_roi, w_roi = roi.shape
        mid_y = int(h_roi * 0.5)
        
        # Upper half is for the Rank, bottom half is for the Suit
        rank_half = roi[0:mid_y, :]
        suit_half = roi[mid_y:, :]

        def extract_symbol(sub_roi, min_w=4, min_h=8):
            """Finds the bounding box of symbols in a sub-ROI and crops it."""
            _, s_thresh = cv2.threshold(sub_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Find all contours
            s_contours, _ = cv2.findContours(s_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            sh, sw = sub_roi.shape
            valid = []
            for c in s_contours:
                x, y, w, h = cv2.boundingRect(c)
                # PIXEL FILTER: Symbols start at least 5-10 pixels from the left ROI edge.
                # ROI Margin is usually at x=0.
                if w >= min_w and h >= min_h and x > 5 and w < sw * 0.8:
                    valid.append((x, y, w, h))
            
            if not valid:
                return sub_roi, False
            
            # Merge all significant symbols
            x1 = min(c[0] for c in valid)
            y1 = min(c[1] for c in valid)
            x2 = max(c[0] + c[2] for c in valid)
            y2 = max(c[1] + c[3] for c in valid)
            
            pad = 1
            cropped = sub_roi[max(0, y1-pad):min(sub_roi.shape[0], y2+pad), 
                              max(0, x1-pad):min(sub_roi.shape[1], x2+pad)]
            return cropped, True 

        # Split point: Half-split based on the card body's top-left ROI
        h_roi, w_roi = roi.shape
        mid_y = int(h_roi * 0.5)
        
        # Upper half is for the Rank, bottom half is for the Suit
        rank_half = roi[0:mid_y, :]
        suit_half = roi[mid_y:, :]
        
        rank_img, rank_ok = extract_symbol(rank_half)
        suit_img, suit_ok = extract_symbol(suit_half)

        # Save Rank (Only if success or not already existing)
        rank_path = RANK_DIR / f"{rank_char}.png"
        if rank_ok:
            cv2.imwrite(str(rank_path), rank_img)
            
        # Save Suit (Only if success or not already existing)
        suit_path = SUIT_DIR / f"{suit_char}.png"
        if suit_ok:
            cv2.imwrite(str(suit_path), suit_img)
        
        print(f"Processed {name} -> Rank: {rank_char}, Suit: {suit_char} (R:{rank_ok}, S:{suit_ok})")

if __name__ == "__main__":
    process_templates()
