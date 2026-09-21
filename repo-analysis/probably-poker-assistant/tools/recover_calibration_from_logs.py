
import re
import json
import os
from pathlib import Path

def recover_from_logs():
    log_path = Path("logs/detection_accuracy.log")
    if not log_path.exists():
        print("Log file not found")
        return

    regions = {}
    
    # Regex to match: [Timestamp] INFO - poker_ai - Calibrated region_name: (x, y, w, h)
    pattern = re.compile(r"Calibrated\s+(\w+):\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")

    print(f"Reading log: {log_path}")
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                region_name = match.group(1)
                x, y, w, h = map(int, match.groups()[1:])
                regions[region_name] = {"x": x, "y": y, "width": w, "height": h}
                print(f"Found {region_name}: {x}, {y}, {w}, {h}")

    if not regions:
        print("No regions found in logs.")
        return

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Reconstruct window info (approximated from logs or default)
    # The logs show "Found window ... handle: ..." but not size usually, unless we printed it.
    # But calibration stores relative to window client area.
    # We can assume a default or use the max extents of regions?
    # Actually, regions.json usually needs window width/height for relative calculations if needed.
    # Since we can't be sure, let's look for "Window rect" or similar if we logged it.
    # CalibrationOverlay logs: `Calibrated {region}: {rect}`
    
    # We'll just define a standard window size or leave it 0 if unknown, 
    # but the detection logic might rely on it.
    # Let's set a minimal valid window size wrapping the regions.
    
    max_x = 0
    max_y = 0
    for r in regions.values():
        max_x = max(max_x, r["x"] + r["width"])
        max_y = max(max_y, r["y"] + r["height"])
        
    output = {
        "window": {
            "title": "Ignition",
            "x": 0, "y": 0,
            "width": max_x + 50, # Padding
            "height": max_y + 50
        },
        "regions": regions
    }
    
    out_path = config_dir / "regions_recovered.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    # Also overwrite regions.json
    with open(config_dir / "regions.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Recovered {len(regions)} regions to {out_path}")

if __name__ == "__main__":
    recover_from_logs()
