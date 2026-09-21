"""
CLI to play Multi-Street Texas Hold'em against the CFR bot.
"""
import random
from typing import Dict, Tuple

from texas_engine import Card, deal_hand
from multi_street_game import (
    deal_progressive_hand, is_terminal, payoff, current_player,
    is_street_terminal, available_actions,
)
from equity import equity_bucket, preflop_equity_bucket
from node import Node
from multi_street_trainer import MultiStreetTrainer, make_key

STREET_NAMES = ['Preflop', 'Flop', 'Turn', 'River']


def load_or_train_strategy(iterations: int = 20_000, num_buckets: int = 10):
    print("Training the Multi-Street Texas Hold'em CFR bot...")
    print("This may take a few minutes (preflop cache will be built on first run).\n")
    trainer = MultiStreetTrainer(num_buckets=num_buckets)
    trainer.train(iterations)
    print("Training complete!\n")
    return trainer.node_map


def sample_action(strategy):
    r = random.random()
    return 0 if r < strategy[0] else 1


def print_strategy(key, strategy):
    street, bucket, history = key
    action_names = available_actions(history)
    parts = [f"{action_names[i]}: {strategy[i]:.1%}" for i in range(len(strategy))]
    print(f"  Bot strategy (St{street}, Bk{bucket}, {history or 'start'}) -> {', '.join(parts)}")


def play_game(node_map: Dict[Tuple[int, int, str], Node], num_buckets: int,
              human_chips: int, bot_chips: int):
    p0_hole, p1_hole, board_by_street = deal_progressive_hand()
    history_by_street = ['', '', '', '']

    print("=" * 60)
    print(f"Your hole cards:  {p0_hole[0]} {p0_hole[1]}")

    for street in range(4):
        if is_terminal(tuple(history_by_street)):
            break

        board = board_by_street[street]
        if board:
            print(f"{STREET_NAMES[street]} board: {' '.join(str(c) for c in board)}")
        else:
            print(f"{STREET_NAMES[street]}")

        human_bucket = (preflop_equity_bucket(p0_hole, num_buckets) if street == 0
                        else equity_bucket(p0_hole, board, num_buckets))
        print(f"Your equity bucket: {human_bucket}")
        print("-" * 60)

        while not is_street_terminal(history_by_street[street]):
            h = history_by_street[street]
            player = current_player(h)
            actions = available_actions(h)

            if player == 0:
                if actions == ['k', 'b']:
                    action = input("Your action [k=check, b=bet]: ").strip().lower()
                    while action not in ('k', 'b'):
                        action = input("Invalid. Enter 'k' to check or 'b' to bet: ").strip().lower()
                else:
                    action = input("Your action [f=fold, c=call]: ").strip().lower()
                    while action not in ('f', 'c'):
                        action = input("Invalid. Enter 'f' to fold or 'c' to call: ").strip().lower()
                history_by_street[street] += action
                verb = "check" if action == 'k' else "bet" if action == 'b' else "fold" if action == 'f' else "call"
                print(f"You {verb}.")
            else:
                bucket = (preflop_equity_bucket(p1_hole, num_buckets) if street == 0
                          else equity_bucket(p1_hole, board, num_buckets))
                key = make_key(street, bucket, h)
                node = node_map.get(key)
                if node is None:
                    print("  Bot has no strategy for this state! Defaulting to first action.")
                    action_idx = 0
                else:
                    strategy = node.get_average_strategy()
                    print_strategy(key, strategy)
                    action_idx = sample_action(strategy)

                action = actions[action_idx]
                history_by_street[street] += action
                verb = "checks" if action == 'k' else "bets" if action == 'b' else "folds" if action == 'f' else "calls"
                print(f"Bot {verb}.")

            if is_terminal(tuple(history_by_street)):
                break

    print("-" * 60)
    cards = (p0_hole, p1_hole, board_by_street[3])
    human_payoff = payoff(cards, tuple(history_by_street), 0)

    print(f"Final history: {' / '.join(history_by_street)}")
    print(f"Bot's hole cards: {p1_hole[0]} {p1_hole[1]}")

    folder = None
    for i, h in enumerate(history_by_street):
        if 'f' in h:
            folder = h.index('f') % 2
            street_folded = i
            break

    if folder is not None:
        if folder == 0:
            print(f"You folded on the {STREET_NAMES[street_folded]}. Bot wins the pot.")
        else:
            print(f"Bot folded on the {STREET_NAMES[street_folded]}. You win the pot!")
    else:
        from evaluator import evaluate_class
        print("Showdown!")
        print(f"Your best hand:  {evaluate_class(p0_hole, board_by_street[3])}")
        print(f"Bot best hand:   {evaluate_class(p1_hole, board_by_street[3])}")
        if human_payoff > 0:
            print("You win the hand!")
        elif human_payoff < 0:
            print("Bot wins the hand.")
        else:
            print("It's a draw.")

    human_chips += int(human_payoff)
    bot_chips -= int(human_payoff)
    print(f"Result: {'+' if human_payoff > 0 else ''}{int(human_payoff)} chips")
    print(f"Your chips: {human_chips} | Bot chips: {bot_chips}")
    return human_chips, bot_chips


def main():
    print("Welcome to Multi-Street Texas Hold'em CFR Bot!")
    print("Rules: 2 players, 52 cards, 4 betting rounds.")
    print("Actions per street: CHECK / BET 1 chip (or FOLD / CALL if facing a bet).")
    print("The bot uses Equity Buckets as card abstraction per street.\n")

    node_map = load_or_train_strategy(iterations=20_000, num_buckets=10)

    human_chips = 0
    bot_chips = 0

    while True:
        human_chips, bot_chips = play_game(node_map, 10, human_chips, bot_chips)
        again = input("\nPlay another hand? [y/n]: ").strip().lower()
        if again != 'y':
            print(f"\nFinal score — You: {human_chips} | Bot: {bot_chips}")
            print("Goodbye!")
            break


if __name__ == '__main__':
    main()
