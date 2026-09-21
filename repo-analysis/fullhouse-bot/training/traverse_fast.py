"""Python Deep CFR traversal."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from .game import (
    N_ACTIONS,
    S_TERMINAL,
    S_PAYOFFS,
    S_TOACT,
    new_hand,
    legal_actions,
    step,
    STARTING_STACK,
    BIG_BLIND,
)
from bot.features import encode_state, load_preflop_equity
from .network import DuelingAdvantageNet
from .buffer import ReservoirBuffer

_EQUITY_TABLES: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.dot(x, weight.T) + bias


def _require_finite(name: str, x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return x


def _ensure_equity_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _EQUITY_TABLES
    if _EQUITY_TABLES is None:
        _EQUITY_TABLES = load_preflop_equity()
    return _EQUITY_TABLES


class NumpyPolicy:
    """Numpy-only dueling forward pass."""

    __slots__ = (
        "trunk_w0",
        "trunk_b0",
        "trunk_ln0_g",
        "trunk_ln0_b",
        "trunk_w1",
        "trunk_b1",
        "trunk_ln1_g",
        "trunk_ln1_b",
        "val_w0",
        "val_b0",
        "val_w1",
        "val_b1",
        "adv_w0",
        "adv_b0",
        "adv_w1",
        "adv_b1",
    )

    def __init__(self, net: DuelingAdvantageNet):
        trunk_params = []
        for module in net.trunk:
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                trunk_params.append(
                    (
                        module.weight.detach().cpu().numpy().copy(),
                        module.bias.detach().cpu().numpy().copy(),
                    )
                )
        self.trunk_w0, self.trunk_b0 = trunk_params[0]
        self.trunk_ln0_g, self.trunk_ln0_b = trunk_params[1]
        self.trunk_w1, self.trunk_b1 = trunk_params[2]
        self.trunk_ln1_g, self.trunk_ln1_b = trunk_params[3]

        val_params = []
        for module in net.value_head:
            if isinstance(module, nn.Linear):
                val_params.append(
                    (
                        module.weight.detach().cpu().numpy().copy(),
                        module.bias.detach().cpu().numpy().copy(),
                    )
                )
        self.val_w0, self.val_b0 = val_params[0]
        self.val_w1, self.val_b1 = val_params[1]

        adv_params = []
        for module in net.advantage_head:
            if isinstance(module, nn.Linear):
                adv_params.append(
                    (
                        module.weight.detach().cpu().numpy().copy(),
                        module.bias.detach().cpu().numpy().copy(),
                    )
                )
        self.adv_w0, self.adv_b0 = adv_params[0]
        self.adv_w1, self.adv_b1 = adv_params[1]

    @staticmethod
    def _layer_norm(
        x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        mean = x.mean()
        var = ((x - mean) ** 2).mean()
        return _require_finite(
            "layer_norm", gamma * (x - mean) / np.sqrt(var + eps) + beta
        )

    def strategy(self, features: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        features = _require_finite("features", np.asarray(features, dtype=np.float32))
        legal_mask = _require_finite(
            "legal_mask", np.asarray(legal_mask, dtype=np.float32)
        )
        n_legal = legal_mask.sum()
        if n_legal <= 0:
            raise ValueError("legal_mask has no legal actions")

        x = _require_finite(
            "trunk linear 0", _linear(features, self.trunk_w0, self.trunk_b0)
        )
        x = self._layer_norm(x, self.trunk_ln0_g, self.trunk_ln0_b)
        x = _require_finite("trunk activation 0", np.where(x > 0, x, 0.01 * x))

        x = _require_finite("trunk linear 1", _linear(x, self.trunk_w1, self.trunk_b1))
        x = self._layer_norm(x, self.trunk_ln1_g, self.trunk_ln1_b)
        x = _require_finite("trunk activation 1", np.where(x > 0, x, 0.01 * x))

        v = _require_finite("value linear 0", _linear(x, self.val_w0, self.val_b0))
        v = _require_finite("value activation", np.where(v > 0, v, 0.01 * v))
        v = _require_finite("value linear 1", _linear(v, self.val_w1, self.val_b1))

        a = _require_finite("adv linear 0", _linear(x, self.adv_w0, self.adv_b0))
        a = _require_finite("adv activation", np.where(a > 0, a, 0.01 * a))
        a = _require_finite("adv linear 1", _linear(a, self.adv_w1, self.adv_b1))

        a_masked = _require_finite("masked advantages", a * legal_mask)
        a_mean = a_masked.sum() / n_legal
        out = _require_finite(
            "dueling output", (v + a_masked - a_mean * legal_mask) * legal_mask
        )

        np.maximum(out, 0.0, out=out)
        out *= legal_mask
        total = out.sum()
        if total > 0:
            return _require_finite("strategy", out / total)
        return legal_mask.copy() / n_legal


def _mix_with_uniform(
    strategy: np.ndarray, legal_mask: np.ndarray, epsilon: float
) -> np.ndarray:
    """Blend a strategy with uniform legal-action exploration."""
    if epsilon <= 0.0:
        return strategy
    n_legal = legal_mask.sum()
    if n_legal <= 0:
        return strategy
    eps = float(np.clip(epsilon, 0.0, 1.0))
    return (1.0 - eps) * strategy + eps * (legal_mask / n_legal)


def sample_match_stacks(rng: np.random.Generator, n_players: int) -> np.ndarray:
    """Sample a chip-conserving stack vector resembling later match hands."""
    total = STARTING_STACK * n_players
    min_stack = BIG_BLIND
    if total <= min_stack * n_players:
        return np.full(n_players, STARTING_STACK, dtype=np.int32)
    weights = rng.exponential(1.0, size=n_players)
    weights = np.maximum(weights, 1e-6)
    remaining = total - min_stack * n_players
    stacks = min_stack + np.floor(remaining * weights / weights.sum()).astype(np.int32)
    stacks[-1] += total - int(stacks.sum())
    rng.shuffle(stacks)
    return stacks.astype(np.int32)


def _encode_fast(
    state: np.ndarray,
    hole_cards: np.ndarray,
    deck: np.ndarray,
    seat: int,
    legal: np.ndarray,
) -> np.ndarray:
    """Encode game state to 51-dim dense features via bot.features."""
    return encode_state(state, hole_cards, deck, seat, legal, _ensure_equity_tables())


def traverse_external(
    state: np.ndarray,
    deck: np.ndarray,
    hole_cards: np.ndarray,
    traverser: int,
    policy: NumpyPolicy,
    feat_list: list,
    adv_list: list,
    strat_feat_list: list,
    strat_list: list,
    strat_weight_list: list,
    rng: np.random.Generator,
    opponent_exploration: float = 0.0,
    traverser_reach: float = 1.0,
) -> float:
    """External sampling: explore ALL traverser actions, sample opponents."""
    if state[S_TERMINAL] == 1:
        return float(state[S_PAYOFFS + traverser])

    seat = int(state[S_TOACT])
    legal = legal_actions(state, seat)
    n_legal = int(legal.sum())
    if n_legal == 0:
        raise RuntimeError(f"game engine returned no legal actions for seat {seat}")

    features = _encode_fast(state, hole_cards, deck, seat, legal)
    legal_f = legal.astype(np.float32)
    sigma = policy.strategy(features, legal_f)

    if seat != traverser:
        sigma = _mix_with_uniform(sigma, legal_f, opponent_exploration)
        r = rng.random()
        cumsum = 0.0
        action = 0
        for a in range(N_ACTIONS):
            cumsum += sigma[a]
            if r < cumsum:
                action = a
                break
        else:
            if cumsum < 1.0 - 1e-6:
                raise RuntimeError(
                    f"opponent strategy mass sums to {cumsum}, expected 1.0"
                )
            legal_actions_idx = np.flatnonzero(legal)
            if legal_actions_idx.size == 0:
                raise RuntimeError(
                    "no legal actions available while sampling opponent action"
                )
            action = int(legal_actions_idx[-1])
        child = state.copy()
        step(child, deck, hole_cards, action)
        return traverse_external(
            child,
            deck,
            hole_cards,
            traverser,
            policy,
            feat_list,
            adv_list,
            strat_feat_list,
            strat_list,
            strat_weight_list,
            rng,
            opponent_exploration,
            traverser_reach,
        )

    # Traverser: explore ALL legal actions (exact counterfactual values)
    strat_feat_list.append(features)
    strat_list.append(sigma.astype(np.float32))
    strat_weight_list.append(np.float32(traverser_reach))

    child_values = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            child = state.copy()
            step(child, deck, hole_cards, a)
            child_values[a] = traverse_external(
                child,
                deck,
                hole_cards,
                traverser,
                policy,
                feat_list,
                adv_list,
                strat_feat_list,
                strat_list,
                strat_weight_list,
                rng,
                opponent_exploration,
                traverser_reach * float(sigma[a]),
            )

    value = float(np.dot(sigma, child_values))
    advantages = np.zeros(N_ACTIONS, dtype=np.float32)
    for a in range(N_ACTIONS):
        if legal[a]:
            advantages[a] = (child_values[a] - value) / 100.0

    feat_list.append(features)
    adv_list.append(advantages)
    return value


def run_traversals_fast(
    n_traversals: int,
    n_players: int,
    traverser: int,
    net: DuelingAdvantageNet,
    buffer: ReservoirBuffer,
    iteration: int,
    rng: np.random.Generator,
    equity_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    opponent_exploration: float = 0.0,
    strategy_buffer: ReservoirBuffer | None = None,
    randomize_stacks: bool = True,
) -> float:
    """Run external-sampling traversals: numpy inference, no PyTorch in hot loop."""
    global _EQUITY_TABLES
    if equity_tables is not None:
        _EQUITY_TABLES = equity_tables
    else:
        _ensure_equity_tables()

    policy = NumpyPolicy(net)
    feat_list: list[np.ndarray] = []
    adv_list: list[np.ndarray] = []
    strat_feat_list: list[np.ndarray] = []
    strat_list: list[np.ndarray] = []
    strat_weight_list: list[np.ndarray] = []
    total_value = 0.0

    for _ in range(n_traversals):
        dealer = int(rng.integers(0, n_players))
        stacks = sample_match_stacks(rng, n_players) if randomize_stacks else None
        state, deck, hole_cards = new_hand(n_players, dealer, rng, stacks)
        v = traverse_external(
            state,
            deck,
            hole_cards,
            traverser,
            policy,
            feat_list,
            adv_list,
            strat_feat_list,
            strat_list,
            strat_weight_list,
            rng,
            opponent_exploration,
        )
        total_value += v

    if feat_list:
        buffer.add_batch(np.array(feat_list), np.array(adv_list), iteration)
    if strategy_buffer is not None and strat_feat_list:
        strat_weights = np.array(strat_weight_list, dtype=np.float32)
        keep = strat_weights > np.float32(1e-8)
        strategy_buffer.add_batch(
            np.array(strat_feat_list)[keep],
            np.array(strat_list)[keep],
            iteration,
            sample_weights=strat_weights[keep],
        )

    return total_value / max(n_traversals, 1)
