"""
Prepare images for calibration by resizing to standard size.
"""
import cv2
import sys
from pathlib import Path

def resize_images(target_width: int = 1200, target_height: int = 800):
    """
    Resize images in screenshots/calibration to target size.
    """
    project_root = Path(__file__).parent.parent
    calibration_dir = project_root / "screenshots" / "calibration"
    
    if not calibration_dir.exists():
        print(f"Directory not found: {calibration_dir}")
        return
        
    # Process reference_ignition.jpg
    input_path = calibration_dir / "reference_ignition.jpg"
    output_path = calibration_dir / "calibration_source.png"
    
    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        # Try to find any jpg/png in the folder
        images = list(calibration_dir.glob("*.jpg")) + list(calibration_dir.glob("*.png"))
        if images:
            input_path = images[0]
            print(f"Using alternative image: {input_path.name}")
        else:
            print("No images found to process.")
            return

    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Failed to load image: {input_path}")
        return
        
    # Resize
    resized = cv2.resize(img, (target_width, target_height))
    cv2.imwrite(str(output_path), resized)
    
    print(f"Successfully created {output_path}")
    print(f"Size: {target_width}x{target_height}")
    print("Use this image for calibration.")

if __name__ == "__main__":
    resize_images()
