# engine/model.py
"""CPU-optimized Dual Actor-Critic Neural Network for the Ace Engine."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AceNet(nn.Module):
    """Lightweight 2-layer MLP for policy and value estimation on CPU."""
    
    def __init__(self, input_dim: int = 343, action_dim: int = 54):
        super().__init__()
        
        # Shared representations representation
        self.shared_dense1 = nn.Linear(input_dim, 128)
        self.shared_dense2 = nn.Linear(128, 128)
        
        # Policy Head (Outputs logits over all 54 possible actions)
        # Action space: 0-51 (PlayCard), 52 (Steal), 53 (DeclineSteal)
        self.policy_head = nn.Linear(128, action_dim)
        
        # Value Head (Outputs estimated expected game outcome in [-1.0, 1.0])
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the forward pass on the state tensor."""
        x = F.relu(self.shared_dense1(state_tensor))
        x = F.relu(self.shared_dense2(x))
        
        logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        
        return logits, value

    def get_masked_policy(self, state_tensor: torch.Tensor, legal_indices: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities after zeroing out illegal actions (masking)."""
        logits, _ = self.forward(state_tensor)
        
        # Create a negative infinity mask for illegal actions
        mask = torch.full_like(logits, -1e9)
        mask.scatter_(1, legal_indices, 0.0)
        
        # Add mask to logits and compute softmax probabilities
        masked_logits = logits + mask
        return F.softmax(masked_logits, dim=-1)
