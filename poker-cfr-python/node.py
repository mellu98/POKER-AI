"""
CFR Node.
Each node corresponds to an information set and stores:
- regret_sum: cumulative regrets for each action
- strategy_sum: cumulative strategy contributions (used to compute average strategy)
"""
import numpy as np

NUM_ACTIONS = 2


class Node:
    """A node in the CFR game tree."""

    def __init__(self):
        self.regret_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.strategy_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        """
        Return the current strategy via Regret Matching.
        Strategy is proportional to positive regrets.
        """
        strategy = np.where(self.regret_sum > 0, self.regret_sum, 0)
        normalizing_sum = np.sum(strategy)
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        Return the average strategy over all iterations.
        This converges to the Nash equilibrium strategy.
        """
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
