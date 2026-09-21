"""
Track current game state (betting round, position, etc.).
"""
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import logger

class BettingRound(Enum):
    """Poker betting rounds."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class Position(Enum):
    """Player positions."""
    UTG = "UTG"
    UTG1 = "UTG+1"
    MP = "MP"
    MP1 = "MP+1"
    CO = "CO"
    BTN = "BTN"
    SB = "SB"
    BB = "BB"

@dataclass
class GameState:
    """Current state of poker game."""
    hole_cards: List[str]
    community_cards: List[str]
    pot_size: float
    stack_size: float
    current_bet: float
    betting_round: BettingRound
    position: Optional[Position] = None
    num_opponents: int = 1
    
    def __str__(self):
        return (f"GameState(round={self.betting_round.value}, "
                f"hole={self.hole_cards}, board={self.community_cards}, "
                f"pot={self.pot_size}, stack={self.stack_size})")

class GameStateTracker:
    """Track and update game state."""
    
    def __init__(self):
        """Initialize game state tracker."""
        self.current_state: Optional[GameState] = None
        self.previous_state: Optional[GameState] = None
    
    def determine_betting_round(self, community_cards: List[str]) -> BettingRound:
        """
        Determine betting round from community cards.
        
        Args:
            community_cards: List of community cards
        
        Returns:
            Current betting round
        """
        if not community_cards:
             return BettingRound.PREFLOP

        num_cards = len(community_cards)
        
        if num_cards == 0:
            return BettingRound.PREFLOP
        elif num_cards == 3:
            return BettingRound.FLOP
        elif num_cards == 4:
            return BettingRound.TURN
        elif num_cards == 5:
            return BettingRound.RIVER
        else:
            # Fallback for transient states or detection errors
            if num_cards < 3:
                return BettingRound.PREFLOP
            elif num_cards > 5:
                # Should not happen typically
                return BettingRound.RIVER
            
            # Default fallback
            return BettingRound.PREFLOP
    
    def update_state(self, 
                    hole_cards: List[str],
                    community_cards: List[str],
                    pot_size: Optional[float],
                    stack_size: Optional[float],
                    current_bet: Optional[float]) -> GameState:
        """
        Update and return current game state.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot_size: Current pot size
            stack_size: Player's stack
            current_bet: Current bet amount
        
        Returns:
            Updated GameState object
        """
        # Save previous state
        self.previous_state = self.current_state
        
        # Determine betting round
        betting_round = self.determine_betting_round(community_cards)
        
        # Create new state
        self.current_state = GameState(
            hole_cards=hole_cards or [],
            community_cards=community_cards or [],
            pot_size=pot_size or 0,
            stack_size=stack_size or 0,
            current_bet=current_bet or 0,
            betting_round=betting_round
        )
        
        logger.info(str(self.current_state))
        
        return self.current_state
    
    def has_state_changed(self) -> bool:
        """
        Check if state has changed since last update.
        
        Returns:
            True if state changed
        """
        if not self.previous_state or not self.current_state:
            return True
        
        # Compare key elements
        changed = (
            self.current_state.betting_round != self.previous_state.betting_round or
            self.current_state.community_cards != self.previous_state.community_cards or
            abs(self.current_state.pot_size - self.previous_state.pot_size) > 0.01  # Use tolerance for floats
        )
        
        return changed
