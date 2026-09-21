"""
Session Logger for Poker AI Assistant.

Logs all decisions and game states for analysis and learning.
Stores data in JSON Lines format for easy parsing.

Usage:
    logger = SessionLogger()
    logger.log_decision(game_state, decision, timestamp)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import asdict

from src.utils.logger import logger


class SessionLogger:
    """
    Logs poker session data for analysis and learning.

    Output format: JSON Lines (.jsonl)
    Each line is a self-contained JSON object with:
    - timestamp
    - game_state (cards, pot, stack, position)
    - decision (action, sizing, confidence, reasoning)
    """

    def __init__(self, output_dir: str = "database/learning"):
        """
        Initialize session logger.

        Args:
            output_dir: Directory to store session files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.output_dir / f"session_{self.session_id}.jsonl"

        self.decision_count = 0
        self.enabled = True

        logger.info(f"SessionLogger initialized: {self.session_file}")

    def log_decision(self,
                     game_state: Any,
                     decision: Any,
                     timestamp: Optional[datetime] = None) -> bool:
        """
        Log a decision to the session file.

        Args:
            game_state: Current GameState object
            decision: Decision object from decision engine
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if logged successfully, False otherwise
        """
        if not self.enabled:
            return False

        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Build log entry
            entry = {
                "timestamp": timestamp.isoformat(),
                "session_id": self.session_id,
                "decision_id": self.decision_count,
                "game_state": self._serialize_game_state(game_state),
                "decision": self._serialize_decision(decision)
            }

            # Append to session file
            with open(self.session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

            self.decision_count += 1
            logger.debug(f"Logged decision #{self.decision_count}")
            return True

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            return False

    def log_hand_result(self,
                        hand_id: str,
                        result: str,
                        chips_delta: float,
                        final_pot: Optional[float] = None) -> bool:
        """
        Log the outcome of a completed hand.

        Args:
            hand_id: Identifier for the hand
            result: "won", "lost", or "folded"
            chips_delta: Chips won/lost
            final_pot: Final pot size

        Returns:
            True if logged successfully
        """
        if not self.enabled:
            return False

        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "type": "hand_result",
                "hand_id": hand_id,
                "result": result,
                "chips_delta": chips_delta,
                "final_pot": final_pot
            }

            with open(self.session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

            logger.debug(f"Logged hand result: {result} ({chips_delta:+.2f})")
            return True

        except Exception as e:
            logger.error(f"Failed to log hand result: {e}")
            return False

    def _serialize_game_state(self, game_state: Any) -> Dict:
        """Convert GameState to serializable dict."""
        try:
            return {
                "hole_cards": game_state.hole_cards,
                "community_cards": game_state.community_cards,
                "pot_size": game_state.pot_size,
                "stack_size": game_state.stack_size,
                "current_bet": game_state.current_bet,
                "betting_round": game_state.betting_round.value if hasattr(game_state.betting_round, 'value') else str(game_state.betting_round),
                "position": game_state.position.value if game_state.position and hasattr(game_state.position, 'value') else str(game_state.position),
                "num_opponents": getattr(game_state, 'num_opponents', 1)
            }
        except Exception as e:
            logger.warning(f"Error serializing game_state: {e}")
            return {"error": str(e)}

    def _serialize_decision(self, decision: Any) -> Dict:
        """Convert Decision to serializable dict."""
        try:
            return {
                "action": decision.action,
                "amount_bb": decision.amount_bb,
                "amount_chips": decision.amount_chips,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "hand_description": decision.hand_evaluation.description if decision.hand_evaluation else None,
                "hand_strength": decision.hand_evaluation.hand_strength if decision.hand_evaluation else None,
                "equity": decision.equity,
                "pot_odds": decision.pot_odds,
                "source": getattr(decision, 'source', 'unknown')
            }
        except Exception as e:
            logger.warning(f"Error serializing decision: {e}")
            return {"error": str(e)}

    def get_session_stats(self) -> Dict:
        """
        Get statistics for current session.

        Returns:
            Dict with session statistics
        """
        return {
            "session_id": self.session_id,
            "session_file": str(self.session_file),
            "decision_count": self.decision_count,
            "enabled": self.enabled
        }

    def enable(self):
        """Enable logging."""
        self.enabled = True
        logger.info("Session logging enabled")

    def disable(self):
        """Disable logging."""
        self.enabled = False
        logger.info("Session logging disabled")

    def close(self):
        """
        Close the session and write summary.
        """
        if self.decision_count > 0:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "type": "session_summary",
                "session_id": self.session_id,
                "total_decisions": self.decision_count
            }

            try:
                with open(self.session_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(summary) + '\n')
                logger.info(f"Session closed: {self.decision_count} decisions logged")
            except Exception as e:
                logger.error(f"Failed to write session summary: {e}")
