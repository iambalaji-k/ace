# engine/model_v2.py
"""Upgraded neural network architecture (AceNetV2) for RL Agent 2.0.

Features a Residual MLP shared trunk with LayerNorm, GELU activations,
and separate output heads for policy and value. Optimized for CPU execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual Block with LayerNorm, GELU activation, and linear mappings."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AceNetV2(nn.Module):
    """Residual MLP model for Ace RL Agent 2.0.

    Input representation: 459 features (from state encoder v2).
    Shared trunk: 4 residual blocks of size 512.
    Policy Head: Linear(512, 256) -> GELU -> Linear(256, 54).
    Value Head: Linear(512, 256) -> GELU -> Linear(256, 1).
    """

    def __init__(self, input_dim: int = 459, action_dim: int = 54):
        super().__init__()

        # Input projection layer
        self.input_layer = nn.Linear(input_dim, 512)
        self.input_gelu = nn.GELU()

        # Shared residual trunk (4 blocks, 512 width)
        self.shared_trunk = nn.Sequential(
            *[ResidualBlock(512) for _ in range(4)]
        )

        # Policy head branch
        self.policy_fc = nn.Linear(512, 256)
        self.policy_gelu = nn.GELU()
        self.policy_out = nn.Linear(256, action_dim)

        # Value head branch
        self.value_fc = nn.Linear(512, 256)
        self.value_gelu = nn.GELU()
        self.value_out = nn.Linear(256, 1)

    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the forward pass, returning logits and value estimation."""
        # Project input to shared dimension
        x = self.input_gelu(self.input_layer(state_tensor))

        # Pass through residual shared trunk
        x = self.shared_trunk(x)

        # Policy head
        p = self.policy_gelu(self.policy_fc(x))
        logits = self.policy_out(p)

        # Value head (expected value, unbounded range)
        v = self.value_gelu(self.value_fc(x))
        value = self.value_out(v)

        return logits, value

    def get_masked_policy(self, state_tensor: torch.Tensor, legal_indices: torch.Tensor) -> torch.Tensor:
        """Computes probabilities over actions, masking out illegal actions."""
        logits, _ = self.forward(state_tensor)

        # Create a negative infinity mask for illegal actions
        mask = torch.full_like(logits, -1e9)
        mask.scatter_(1, legal_indices, 0.0)

        # Apply softmax to masked logits
        masked_logits = logits + mask
        return F.softmax(masked_logits, dim=-1)
