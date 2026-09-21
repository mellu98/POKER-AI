"""Neural networks used by the Deep CFR trainer and runtime export."""

from __future__ import annotations

import torch
import torch.nn as nn

N_ACTIONS = 5


class DuelingAdvantageNet(nn.Module):
    def __init__(self, input_dim: int = 51, trunk_dim: int = 256, head_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(trunk_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.LeakyReLU(0.01),
        )
        self.value_head = nn.Sequential(
            nn.Linear(trunk_dim, head_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(head_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(trunk_dim, head_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(head_dim, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        value = self.value_head(shared)
        adv = self.advantage_head(shared)

        adv_masked = adv * legal_mask
        n_legal = legal_mask.sum(dim=1, keepdim=True).clamp(min=1)
        adv_mean = adv_masked.sum(dim=1, keepdim=True) / n_legal
        return (value + adv_masked - adv_mean * legal_mask) * legal_mask

    def get_strategy(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(x, legal_mask)
            adv = torch.clamp(out, min=0.0) * legal_mask
            total = adv.sum(dim=-1, keepdim=True)
            return torch.where(
                total > 0,
                adv / total,
                legal_mask / legal_mask.sum(dim=-1, keepdim=True).clamp(min=1),
            )


class AverageStrategyNet(nn.Module):
    """Average strategy network for Deep CFR deployment.

    This approximates the iteration-weighted average policy directly. Unlike
    the advantage network, deployment uses a masked softmax over this model's
    logits instead of regret matching a single checkpoint's advantages.
    """

    def __init__(
        self, input_dim: int = 51, trunk_dim: int = 256, n_actions: int = N_ACTIONS
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(trunk_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.LeakyReLU(0.01),
        )
        self.policy_head = nn.Linear(trunk_dim, n_actions)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.policy_head(self.trunk(x))
        return logits.masked_fill(legal_mask <= 0, -1e9)

    def get_strategy(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, legal_mask)
            probs = torch.softmax(logits, dim=-1) * legal_mask
            total = probs.sum(dim=-1, keepdim=True)
            return torch.where(
                total > 0,
                probs / total.clamp(min=1e-12),
                legal_mask / legal_mask.sum(dim=-1, keepdim=True).clamp(min=1),
            )
