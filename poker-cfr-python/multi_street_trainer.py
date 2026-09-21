"""
CFR Trainer for Multi-Street Texas Hold'em.
Uses equity-based card abstraction per street.
"""
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

from multi_street_game import (
    deal_progressive_hand, is_terminal, payoff, current_player,
    is_street_terminal, available_actions,
)
from equity import equity_bucket, preflop_equity_bucket
from node import Node

NUM_BUCKETS = 10


def make_key(street: int, bucket: int, history: str) -> Tuple[int, int, str]:
    return (street, bucket, history)


class MultiStreetTrainer:
    def __init__(self, num_buckets: int = NUM_BUCKETS):
        self.num_buckets = num_buckets
        self.node_map: Dict[Tuple[int, int, str], Node] = {}

    def _get_node(self, key: Tuple[int, int, str]) -> Node:
        if key not in self.node_map:
            self.node_map[key] = Node()
        return self.node_map[key]

    def cfr(self, cards, history_by_street: Tuple[str, ...], street: int,
            p0: float, p1: float, buckets) -> float:
        if is_terminal(history_by_street):
            return payoff(cards, history_by_street, 0)

        h = history_by_street[street]

        if is_street_terminal(h):
            new_history = list(history_by_street)
            if len(new_history) <= street + 1:
                new_history.append('')
            return self.cfr(cards, tuple(new_history), street + 1, p0, p1, buckets)

        player = current_player(h)
        bucket = buckets[street][player]
        key = make_key(street, bucket, h)
        node = self._get_node(key)
        strategy = node.get_strategy()
        util = np.zeros(2, dtype=np.float64)
        node_util = 0.0

        for a in range(2):
            action = available_actions(h)[a]
            next_h = h + action
            new_history = list(history_by_street)
            new_history[street] = next_h
            if player == 0:
                util[a] = self.cfr(cards, tuple(new_history), street,
                                   p0 * strategy[a], p1, buckets)
            else:
                util[a] = self.cfr(cards, tuple(new_history), street,
                                   p0, p1 * strategy[a], buckets)
            node_util += strategy[a] * util[a]

        for a in range(2):
            if player == 0:
                regret = util[a] - node_util
                node.regret_sum[a] += p1 * regret
                node.strategy_sum[a] += p0 * strategy[a]
            else:
                regret = node_util - util[a]
                node.regret_sum[a] += p0 * regret
                node.strategy_sum[a] += p1 * strategy[a]

        return node_util

    def train(self, num_iterations: int):
        for _ in tqdm(range(num_iterations), desc="Training Multi-Street CFR"):
            p0, p1, board_by_street = deal_progressive_hand()
            buckets = {}
            for s in range(4):
                board = board_by_street[s]
                if s == 0:
                    b0 = preflop_equity_bucket(p0, self.num_buckets)
                    b1 = preflop_equity_bucket(p1, self.num_buckets)
                else:
                    b0 = equity_bucket(p0, board, self.num_buckets)
                    b1 = equity_bucket(p1, board, self.num_buckets)
                buckets[s] = (b0, b1)

            cards = (p0, p1, board_by_street[3])
            self.cfr(cards, ('', '', '', ''), 0, 1.0, 1.0, buckets)

        return self.node_map

    def print_strategies(self):
        for key, node in sorted(self.node_map.items()):
            avg = node.get_average_strategy()
            street, bucket, history = key
            print(f"St{street} Bk{bucket:2d} Hist '{history or 'start':4s}' "
                  f"-> 0:{avg[0]:.4f} 1:{avg[1]:.4f}")


if __name__ == '__main__':
    trainer = MultiStreetTrainer(num_buckets=10)
    trainer.train(10_000)
    trainer.print_strategies()
