"""Reservoir buffer and GPU cache for Deep CFR samples."""

from __future__ import annotations

import numpy as np
import torch


class ReservoirBuffer:
    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        n_actions: int = 5,
        *,
        store_sample_weights: bool = False,
    ):
        self.capacity = capacity
        self.store_sample_weights = store_sample_weights
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.advantages = np.zeros((capacity, n_actions), dtype=np.float32)
        self.iterations = np.zeros(capacity, dtype=np.float32)
        self.sample_weights = np.ones(capacity, dtype=np.float32)
        self.size = 0
        self.total_seen = 0

    def add(
        self,
        feature: np.ndarray,
        advantage: np.ndarray,
        iteration: int,
        sample_weight: float = 1.0,
    ) -> None:
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            r = np.random.randint(0, self.total_seen + 1)
            if r >= self.capacity:
                self.total_seen += 1
                return
            idx = r
        self.features[idx] = feature
        self.advantages[idx] = advantage
        self.iterations[idx] = float(iteration)
        self.sample_weights[idx] = float(sample_weight)
        self.total_seen += 1

    def add_batch(
        self,
        features: np.ndarray,
        advantages: np.ndarray,
        iteration: int,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        n = features.shape[0]
        if n == 0:
            return
        if sample_weights is None:
            sample_weights = np.ones(n, dtype=np.float32)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float32)
            if sample_weights.shape[0] != n:
                raise ValueError(
                    f"sample_weights length {sample_weights.shape[0]} does not match batch size {n}"
                )

        free = self.capacity - self.size
        if free >= n:
            end = self.size + n
            self.features[self.size : end] = features
            self.advantages[self.size : end] = advantages
            self.iterations[self.size : end] = float(iteration)
            self.sample_weights[self.size : end] = sample_weights
            self.size = end
            self.total_seen += n
            return

        if free > 0:
            self.features[self.size : self.capacity] = features[:free]
            self.advantages[self.size : self.capacity] = advantages[:free]
            self.iterations[self.size : self.capacity] = float(iteration)
            self.sample_weights[self.size : self.capacity] = sample_weights[:free]
            self.size = self.capacity
            self.total_seen += free

        remaining = n - max(free, 0)
        start_seen = self.total_seen
        totals = np.arange(start_seen + 1, start_seen + remaining + 1, dtype=np.int64)
        rand_vals = (np.random.random(remaining) * totals).astype(np.int64)
        accept_mask = rand_vals < self.capacity
        if accept_mask.any():
            positions = rand_vals[accept_mask]
            src_start = max(free, 0)
            src_indices = np.where(accept_mask)[0] + src_start
            self.features[positions] = features[src_indices]
            self.advantages[positions] = advantages[src_indices]
            self.iterations[positions] = float(iteration)
            self.sample_weights[positions] = sample_weights[src_indices]
        self.total_seen += remaining

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("Buffer is empty")
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))
        features_batch = self.features[indices]
        weights = self.iterations[indices]
        max_w = weights.max().clip(min=1.0)
        weights = (weights / max_w) ** 1.5
        if self.store_sample_weights:
            weights = weights * self.sample_weights[indices]
        return features_batch, self.advantages[indices], weights

    def state_dict(self) -> dict:
        state = {
            "features": self.features[: self.size].copy(),
            "advantages": self.advantages[: self.size].copy(),
            "iterations": self.iterations[: self.size].copy(),
            "count": self.size,
            "total_seen": self.total_seen,
        }
        if self.store_sample_weights:
            state["sample_weights"] = self.sample_weights[: self.size].copy()
        return state

    def load_state_dict(self, d: dict) -> None:
        count = d["count"]
        self.features[:count] = d["features"]
        self.advantages[:count] = d["advantages"]
        self.iterations[:count] = d["iterations"]
        if self.store_sample_weights:
            if "sample_weights" not in d:
                raise RuntimeError(
                    "Strategy replay buffer checkpoint is missing sample_weights; "
                    "resuming would change the training distribution."
                )
            self.sample_weights[:count] = d["sample_weights"]
        else:
            self.sample_weights[:count] = 1.0
        self.size = count
        self.total_seen = d["total_seen"]

    def __len__(self) -> int:
        return self.size


class GPUBuffer:
    def __init__(self, cpu_buffers: list[ReservoirBuffer], device: torch.device):
        self.cpu_buffers = cpu_buffers
        self.device = device
        self._synced_size = 0

        total_cap = sum(b.capacity for b in cpu_buffers)
        feat_dim = cpu_buffers[0].features.shape[1]
        n_act = cpu_buffers[0].advantages.shape[1]

        self._gpu_features = torch.zeros(
            total_cap, feat_dim, dtype=torch.float32, device=device
        )
        self._gpu_advantages = torch.zeros(
            total_cap, n_act, dtype=torch.float32, device=device
        )
        self._gpu_iterations = torch.zeros(
            total_cap, dtype=torch.float32, device=device
        )
        vram_gb = (
            self._gpu_features.nbytes
            + self._gpu_advantages.nbytes
            + self._gpu_iterations.nbytes
        ) / 1e9
        print(
            f"[GPUBuffer] Pre-allocated {total_cap:,} slots across "
            f"{len(cpu_buffers)} players ({vram_gb:.1f} GB VRAM)"
        )

    def sync(self):
        offset = 0
        self._player_offsets = []
        self._player_sizes = []
        for buf in self.cpu_buffers:
            n = buf.size
            self._player_offsets.append(offset)
            self._player_sizes.append(n)
            if n == 0:
                continue
            self._gpu_features[offset : offset + n].copy_(
                torch.from_numpy(buf.features[:n])
            )
            self._gpu_advantages[offset : offset + n].copy_(
                torch.from_numpy(buf.advantages[:n])
            )
            self._gpu_iterations[offset : offset + n].copy_(
                torch.from_numpy(buf.iterations[:n])
            )
            offset += n
        torch.cuda.synchronize(self.device)
        self._synced_size = offset
        sizes = np.array(self._player_sizes, dtype=np.float64)
        total = sizes.sum()
        self._player_probs = sizes / total if total > 0 else None

    def sample(self, batch_size: int):
        size = self._synced_size
        if size == 0:
            raise ValueError("Buffer is empty")

        player = int(np.random.choice(len(self._player_sizes), p=self._player_probs))
        start = self._player_offsets[player]
        psize = self._player_sizes[player]
        if psize == 0:
            start = 0
            psize = size

        indices = torch.randint(
            start, start + psize, (min(batch_size, psize),), device=self.device
        )
        features = self._gpu_features[indices]

        weights = self._gpu_iterations[indices]
        max_w = weights.max().clamp(min=1.0)
        weights = (weights / max_w) ** 1.5

        return features, self._gpu_advantages[indices], weights

    def __len__(self):
        return self._synced_size
