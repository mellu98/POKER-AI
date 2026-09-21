"""
Define and manage screen regions for card detection.
Maps poker table layout to screen coordinates.
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.config_loader import config_loader

@dataclass
class Region:
    """Represents a screen region."""
    name: str
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to coordinate tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }

class RegionMapper:
    """Manage poker table screen regions."""
    
    def __init__(self):
        """Initialize region mapper."""
        self.regions: Dict[str, Region] = {}
        self.window_info = None
        self.calibrated = False
        
        # Try to load saved regions
        self.load_regions()
    
    def load_regions(self) -> bool:
        """
        Load regions from config file.
        
        Returns:
            True if regions loaded successfully
        """
        try:
            config = config_loader.load('regions.json')
            
            # Check if regions are calibrated (have non-zero values)
            regions_data = config.get('regions', {})
            has_calibration = any(
                r.get('x', 0) != 0 or r.get('y', 0) != 0
                for r in regions_data.values()
            )
            
            if has_calibration:
                # Load window info
                self.window_info = config.get('window', {})
                
                # Load regions
                for name, data in regions_data.items():
                    if all(k in data for k in ['x', 'y', 'width', 'height']):
                        self.regions[name] = Region(
                            name=name,
                            x=data['x'],
                            y=data['y'],
                            width=data['width'],
                            height=data['height']
                        )
                
                self.calibrated = True
                logger.info(f"Loaded {len(self.regions)} regions from config")
                return True
            else:
                logger.warning("No calibration data found in regions.json")
                return False
                
        except FileNotFoundError:
            logger.warning("regions.json not found - calibration required")
            return False
        except Exception as e:
            logger.error(f"Error loading regions: {e}")
            return False
    
    def save_regions(self):
        """Save regions to config file."""
        try:
            regions_dict = {
                name: region.to_dict()
                for name, region in self.regions.items()
            }
            
            config = {
                "window": self.window_info or {},
                "regions": regions_dict
            }
            
            config_loader.save('regions.json', config)
            logger.info("Saved regions to config")
            
        except Exception as e:
            logger.error(f"Error saving regions: {e}")
    
    def set_region(self, name: str, x: int, y: int, width: int, height: int):
        """
        Set or update a region.
        
        Args:
            name: Region identifier
            x, y, width, height: Region coordinates
        """
        self.regions[name] = Region(name, x, y, width, height)
        logger.info(f"Set region '{name}': ({x}, {y}, {width}, {height})")
    
    def get_region(self, name: str) -> Optional[Region]:
        """
        Get region by name.
        
        Args:
            name: Region identifier
        
        Returns:
            Region object or None
        """
        return self.regions.get(name)
    
    def get_all_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Get all regions as coordinate tuples.
        
        Returns:
            Dict of region_name -> (x, y, width, height)
        """
        return {
            name: region.to_tuple()
            for name, region in self.regions.items()
        }
    
    def is_calibrated(self) -> bool:
        """
        Check if regions are calibrated.
        
        Returns:
            True if regions exist and have valid coordinates
        """
        return self.calibrated and len(self.regions) > 0
    
    def create_default_regions(self, window_width: int, window_height: int):
        """
        Create default region layout based on typical PokerStars dimensions.
        
        Args:
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        # These are estimates - user should calibrate for their specific setup
        center_x = window_width // 2
        center_y = window_height // 2
        
        # Player hole cards (bottom center)
        self.set_region(
            "hole_cards",
            x=center_x - 100,
            y=int(window_height * 0.75),
            width=200,
            height=100
        )
        
        # Community cards (center)
        self.set_region(
            "community_cards",
            x=center_x - 200,
            y=int(window_height * 0.45),
            width=400,
            height=100
        )
        
        # Pot amount (top center)
        self.set_region(
            "pot_amount",
            x=center_x - 50,
            y=int(window_height * 0.35),
            width=100,
            height=30
        )
        
        # Player stack (bottom right)
        self.set_region(
            "player_stack",
            x=int(window_width * 0.70),
            y=int(window_height * 0.85),
            width=150,
            height=30
        )
        
        # Current bet (near action buttons)
        self.set_region(
            "current_bet",
            x=int(window_width * 0.70),
            y=int(window_height * 0.65),
            width=100,
            height=30
        )
        
        # Action buttons area
        self.set_region(
            "action_buttons",
            x=int(window_width * 0.60),
            y=int(window_height * 0.80),
            width=400,
            height=80
        )
        
        logger.info("Created default regions (calibration recommended)")
