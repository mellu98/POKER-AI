"""
Calculate pot odds and determine profitable plays.
"""
from typing import Tuple
from src.utils.logger import logger

class PotOddsCalculator:
    """Calculate and compare pot odds vs equity."""
    
    @staticmethod
    def calculate_pot_odds(pot_size: float, amount_to_call: float) -> float:
        """
        Calculate pot odds.
        
        Args:
            pot_size: Current pot size
            amount_to_call: Amount needed to call
        
        Returns:
            Pot odds ratio (e.g., 3.0 means 3:1)
        """
        if amount_to_call <= 0:
            return float('inf')
        
        total_pot = pot_size + amount_to_call
        odds = total_pot / amount_to_call
        
        logger.debug(f"Pot odds: {odds:.1f}:1")
        return odds
    
    @staticmethod
    def pot_odds_to_percentage(pot_odds: float) -> float:
        """
        Convert pot odds to required equity percentage.
        
        Args:
            pot_odds: Pot odds ratio
        
        Returns:
            Required equity percentage
        """
        if pot_odds == float('inf'):
            return 0.0
        
        required = (1 / (pot_odds + 1)) * 100
        return required
    
    @staticmethod
    def is_profitable_call(equity: float, pot_odds: float, margin: float = 5.0) -> bool:
        """
        Determine if call is profitable.
        
        Args:
            equity: Win equity percentage
            pot_odds: Pot odds ratio
            margin: Safety margin (%)
        
        Returns:
            True if call is profitable
        """
        required_equity = PotOddsCalculator.pot_odds_to_percentage(pot_odds)
        profitable = equity >= (required_equity + margin)
        
        logger.debug(f"Equity: {equity:.1f}%, Required: {required_equity:.1f}%, Profitable: {profitable}")
        
        return profitable
    
    @staticmethod
    def calculate_ev(pot_size: float, 
                    amount_to_call: float, 
                    equity: float) -> float:
        """
        Calculate expected value of call.
        
        Args:
            pot_size: Current pot
            amount_to_call: Amount to call
            equity: Win probability (%)
        
        Returns:
            Expected value in chips
        """
        total_pot = pot_size + amount_to_call
        win_amount = total_pot
        lose_amount = amount_to_call
        
        ev = (equity/100 * win_amount) - ((100-equity)/100 * lose_amount)
        
        logger.debug(f"EV: ${ev:.2f}")
        return ev
