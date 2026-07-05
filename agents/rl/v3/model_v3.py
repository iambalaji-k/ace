# agents/rl/v3/model_v3.py
"""Upgraded neural network architecture (AceNetV3) for RL Agent 3.0/3.1.

Features:
1. Residual MLP shared trunk with LayerNorm and GELU activations.
2. Separate Policy Heads:
   - Steal Head: Linear(256, 2) for binary steal decisions (indices 0 and 1).
   - Play Head: Linear(256, 52) for card plays (indices 2 to 53).
3. Dual Value Heads (Critics):
   - Round Value Head: Predicts round-level placement rewards.
   - Match Value Head: Predicts match-level placement rewards.
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


class AceNetV3(nn.Module):
    """Residual MLP model with separate steal policy and dual value heads."""

    def __init__(self, input_dim: int = 1879, action_dim: int = 54):
        super().__init__()

        # Input projection layer
        self.input_layer = nn.Linear(input_dim, 512)
        self.input_gelu = nn.GELU()

        # Shared residual trunk (4 blocks, 512 width)
        self.shared_trunk = nn.Sequential(
            *[ResidualBlock(512) for _ in range(4)]
        )

        # Policy branches
        self.policy_fc = nn.Linear(512, 256)
        self.policy_gelu = nn.GELU()
        self.play_out = nn.Linear(256, 52)  # Card plays: 2..53

        self.steal_fc = nn.Linear(512, 256)
        self.steal_gelu = nn.GELU()
        self.steal_out = nn.Linear(256, 2)  # Steal/decline decisions: 0..1

        # Dual Value branches
        self.round_value_fc = nn.Linear(512, 256)
        self.round_value_gelu = nn.GELU()
        self.round_value_out = nn.Linear(256, 1)

        self.match_value_fc = nn.Linear(512, 256)
        self.match_value_gelu = nn.GELU()
        self.match_value_out = nn.Linear(256, 1)

    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: Unified (B, 54) policy logits where:
                    - [:, 0:2] are from the Steal Head.
                    - [:, 2:] are from the Play Head.
            round_value: (B, 1) expected round-level reward.
            match_value: (B, 1) expected match-level reward.
        """
        # Project input to shared dimension
        x = self.input_gelu(self.input_layer(state_tensor))

        # Pass through residual shared trunk
        x = self.shared_trunk(x)

        # Policy outputs
        p = self.policy_gelu(self.policy_fc(x))
        play_logits = self.play_out(p)

        s = self.steal_gelu(self.steal_fc(x))
        steal_logits = self.steal_out(s)

        # Combine logits to shape (B, 54)
        B = state_tensor.shape[0]
        logits = torch.zeros((B, 54), device=state_tensor.device, dtype=state_tensor.dtype)
        logits[:, 0:2] = steal_logits
        logits[:, 2:] = play_logits

        # Value outputs
        rv = self.round_value_gelu(self.round_value_fc(x))
        round_value = self.round_value_out(rv)

        mv = self.match_value_gelu(self.match_value_fc(x))
        match_value = self.match_value_out(mv)

        return logits, round_value, match_value

    def get_masked_policy(self, state_tensor: torch.Tensor, legal_indices: torch.Tensor) -> torch.Tensor:
        """Computes probabilities over actions, masking out illegal actions."""
        logits, _, _ = self.forward(state_tensor)

        # Create a negative infinity mask for illegal actions
        mask = torch.full_like(logits, -1e9)
        mask.scatter_(1, legal_indices, 0.0)

        # Apply softmax to masked logits
        masked_logits = logits + mask
        return F.softmax(masked_logits, dim=-1)
