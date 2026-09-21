"""Regression tests for strategy-sample reach weighting."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.training

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bot.features import FEATURE_DIM, LEGAL_MASK_OFFSET, N_ACTIONS
from training import train as train_mod
from training import traverse_fast
from training import worker as worker_mod
from training.buffer import ReservoirBuffer
from training.network import AverageStrategyNet


class _ToyPolicy:
    def strategy(self, features: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        node = int(features[0])
        if node == 0:
            return np.array([0.25, 0.75, 0.0, 0.0, 0.0], dtype=np.float32)
        if node == 1:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if node == 2:
            return np.array([0.1, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)
        if node == 3:
            return np.array([0.6, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
        raise AssertionError(f"unexpected toy node {node}")


def _manual_strategy_step(
    net: AverageStrategyNet,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> float:
    net.train()
    feat_t = torch.from_numpy(features)
    target_t = torch.from_numpy(targets)
    w_t = torch.from_numpy(weights)
    legal_mask = feat_t[:, LEGAL_MASK_OFFSET : LEGAL_MASK_OFFSET + N_ACTIONS]

    target_t = target_t * legal_mask
    target_t = target_t / target_t.sum(dim=1, keepdim=True).clamp(min=1e-12)

    logits = net(feat_t, legal_mask)
    log_probs = torch.log_softmax(logits, dim=1)
    loss_per_sample = -(target_t * log_probs).sum(dim=1)
    loss = (loss_per_sample * w_t).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    net.eval()
    return float(loss.item())


def _patch_toy_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    node_slot = max(traverse_fast.S_PAYOFFS + 8, traverse_fast.S_TOACT + 8, 64)
    state_size = node_slot + 1
    legal_by_node = {
        0: np.array([1, 1, 0, 0, 0], dtype=np.int8),
        1: np.array([1, 0, 0, 0, 0], dtype=np.int8),
        2: np.array([1, 1, 0, 0, 0], dtype=np.int8),
        3: np.array([1, 1, 0, 0, 0], dtype=np.int8),
    }

    def make_state(
        node: int, to_act: int, *, terminal: bool = False, payoff: int = 0
    ) -> np.ndarray:
        state = np.zeros(state_size, dtype=np.int32)
        state[traverse_fast.S_TERMINAL] = int(terminal)
        state[traverse_fast.S_TOACT] = to_act
        state[traverse_fast.S_PAYOFFS + 0] = payoff
        state[node_slot] = node
        return state

    def new_hand(
        n_players: int,
        dealer: int,
        rng: np.random.Generator,
        stacks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del n_players, dealer, rng, stacks
        return (
            make_state(0, 0),
            np.zeros(0, dtype=np.int8),
            np.zeros((2, 2), dtype=np.int8),
        )

    def legal_actions(state: np.ndarray, seat: int) -> np.ndarray:
        del seat
        return legal_by_node[int(state[node_slot])].copy()

    def step(
        state: np.ndarray, deck: np.ndarray, hole_cards: np.ndarray, action: int
    ) -> None:
        del deck, hole_cards
        node = int(state[node_slot])
        state[traverse_fast.S_PAYOFFS + 0] = 0
        if node == 0 and action == 0:
            state[traverse_fast.S_TOACT] = 1
            state[node_slot] = 1
        elif node == 0 and action == 1:
            state[traverse_fast.S_TOACT] = 0
            state[node_slot] = 2
        elif node == 1 and action == 0:
            state[traverse_fast.S_TOACT] = 0
            state[node_slot] = 3
        elif node == 2 and action == 0:
            state[traverse_fast.S_TERMINAL] = 1
            state[traverse_fast.S_PAYOFFS + 0] = 5
        elif node == 2 and action == 1:
            state[traverse_fast.S_TERMINAL] = 1
            state[traverse_fast.S_PAYOFFS + 0] = -5
        elif node == 3 and action == 0:
            state[traverse_fast.S_TERMINAL] = 1
            state[traverse_fast.S_PAYOFFS + 0] = 9
        elif node == 3 and action == 1:
            state[traverse_fast.S_TERMINAL] = 1
            state[traverse_fast.S_PAYOFFS + 0] = -9
        else:
            raise AssertionError(f"unexpected transition node={node} action={action}")

    def encode_fast(
        state: np.ndarray,
        hole_cards: np.ndarray,
        deck: np.ndarray,
        seat: int,
        legal: np.ndarray,
    ) -> np.ndarray:
        del hole_cards, deck, seat
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        features[0] = float(state[node_slot])
        for action in range(N_ACTIONS):
            features[LEGAL_MASK_OFFSET + action] = float(legal[action])
        return features

    monkeypatch.setattr(traverse_fast, "new_hand", new_hand)
    monkeypatch.setattr(worker_mod, "new_hand", new_hand)
    monkeypatch.setattr(traverse_fast, "legal_actions", legal_actions)
    monkeypatch.setattr(traverse_fast, "step", step)
    monkeypatch.setattr(traverse_fast, "_encode_fast", encode_fast)
    monkeypatch.setattr(
        worker_mod, "_policy_from_arrays", lambda weights, biases: _ToyPolicy()
    )


def test_strategy_reach_weights_survive_worker_to_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_toy_traversal(monkeypatch)

    result = worker_mod.worker_run(
        {
            "worker_id": 7,
            "n_traversals": 1,
            "traverser": 0,
            "n_players": 2,
            "weights": [],
            "biases": [],
            "equity_tables": None,
            "opponent_exploration": 0.0,
            "randomize_stacks": False,
        }
    )
    strategy_buffer = ReservoirBuffer(
        8,
        FEATURE_DIM,
        N_ACTIONS,
        store_sample_weights=True,
    )
    advantage_buffer = ReservoirBuffer(8, FEATURE_DIM, N_ACTIONS)
    n_adv, n_strategy = train_mod._collect_worker_batches(
        [result],
        advantage_buffer,
        strategy_buffer,
        iteration=4,
    )

    assert n_adv == 3
    assert n_strategy == 3
    assert np.allclose(
        strategy_buffer.sample_weights[:3],
        np.array([1.0, 0.25, 0.75], dtype=np.float32),
    )

    sample_indices = np.array([0, 1, 2], dtype=np.int64)

    def take_all_indices(
        low: int, high: int | None = None, size: int | tuple[int, ...] | None = None
    ):
        assert low == 0
        assert high == 3
        assert size == 3
        return sample_indices.copy()

    monkeypatch.setattr(np.random, "randint", take_all_indices)
    features, targets, weights = strategy_buffer.sample(batch_size=3)
    assert np.allclose(weights, np.array([1.0, 0.25, 0.75], dtype=np.float32))

    torch.manual_seed(0)
    net = AverageStrategyNet(FEATURE_DIM, 8, N_ACTIONS)
    expected_net = copy.deepcopy(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    expected_optimizer = torch.optim.SGD(expected_net.parameters(), lr=0.05)

    expected_loss = _manual_strategy_step(
        expected_net, expected_optimizer, features, targets, weights
    )
    actual_loss = train_mod.train_strategy_network(
        net,
        strategy_buffer,
        optimizer,
        n_steps=1,
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert actual_loss == pytest.approx(expected_loss, rel=1e-6, abs=1e-7)
    for actual_param, expected_param in zip(
        net.parameters(), expected_net.parameters()
    ):
        assert torch.allclose(actual_param, expected_param)


def test_strategy_buffer_checkpoint_round_trip_preserves_sample_weights(
    tmp_path: Path,
) -> None:
    strategy_buffer = ReservoirBuffer(
        8,
        FEATURE_DIM,
        N_ACTIONS,
        store_sample_weights=True,
    )
    features = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    features[:, LEGAL_MASK_OFFSET : LEGAL_MASK_OFFSET + 2] = 1.0
    strategies = np.array(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    sample_weights = np.array([1.0, 0.25, 0.75], dtype=np.float32)
    strategy_buffer.add_batch(
        features, strategies, iteration=9, sample_weights=sample_weights
    )

    checkpoint_path = tmp_path / "latest.pt"
    torch.save({"strategy_buffer": strategy_buffer.state_dict()}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    restored = ReservoirBuffer(
        8,
        FEATURE_DIM,
        N_ACTIONS,
        store_sample_weights=True,
    )
    restored.load_state_dict(checkpoint["strategy_buffer"])

    assert np.array_equal(restored.features[:3], features)
    assert np.array_equal(restored.advantages[:3], strategies)
    assert np.array_equal(restored.iterations[:3], np.full(3, 9.0, dtype=np.float32))
    assert np.array_equal(restored.sample_weights[:3], sample_weights)
