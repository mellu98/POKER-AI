"""
Calculate win probability using Monte Carlo simulation.
"""
import random
from typing import List
from src.strategy.hand_evaluator import HandEvaluator
from src.utils.logger import logger

class EquityCalculator:
    """Calculate hand equity via Monte Carlo."""
    
    def __init__(self):
        """Initialize equity calculator."""
        self.evaluator = HandEvaluator()
        self.deck = self._create_deck()
    
    def _create_deck(self) -> List[str]:
        """Create full 52-card deck."""
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        suits = ['h', 'd', 'c', 's']
        return [f"{rank}{suit}" for rank in ranks for suit in suits]
    
    def calculate_equity(self,
                        hole_cards: List[str],
                        community_cards: List[str],
                        num_opponents: int = 1,
                        iterations: int = 1000) -> float:
        """
        Calculate win equity using Monte Carlo simulation.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Known community cards
            num_opponents: Number of opponents
            iterations: Simulation iterations
        
        Returns:
            Win probability (0-100)
        """
        if not hole_cards or len(hole_cards) != 2:
            return 0.0
        
        wins = 0
        ties = 0
        
        # Remove known cards from deck
        remaining_deck = [c for c in self.deck 
                         if c not in hole_cards and c not in community_cards]
        
        for _ in range(iterations):
            # Shuffle deck
            simulation_deck = remaining_deck.copy()
            random.shuffle(simulation_deck)
            
            # Deal remaining community cards
            cards_needed = 5 - len(community_cards)
            simulated_community = community_cards + simulation_deck[:cards_needed]
            deck_index = cards_needed
            
            # Evaluate player hand
            player_eval = self.evaluator.evaluate(hole_cards, simulated_community)
            
            # Simulate opponent hands
            opponent_won = False
            for _ in range(num_opponents):
                # Ensure we have enough cards for opponents
                if deck_index + 2 > len(simulation_deck):
                    break
                    
                opponent_hole = simulation_deck[deck_index:deck_index+2]
                deck_index += 2
                
                opponent_eval = self.evaluator.evaluate(opponent_hole, simulated_community)
                
                # Compare hands
                if opponent_eval.hand_strength > player_eval.hand_strength:
                    opponent_won = True
                    break
                elif opponent_eval.hand_strength == player_eval.hand_strength:
                    ties += 1
            
            if not opponent_won:
                wins += 1
        
        # Calculate equity
        equity = ((wins + ties * 0.5 / (num_opponents or 1)) / iterations) * 100
        
        logger.info(f"Equity: {equity:.1f}% ({wins} wins, {ties} ties in {iterations} sims)")
        
        return equity
