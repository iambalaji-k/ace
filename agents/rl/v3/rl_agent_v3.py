# agents/rl/v3/rl_agent_v3.py
"""Reinforcement Learning Agent (V3) using the AceNetV3 PyTorch model.

Standardized on action_encoding discrete indexing (Decline=0, Steal=1, Plays=2..53).
"""

import os
import torch
import numpy as np
import random
from typing import Sequence, Optional
from agents.random.agent import BaseAgent
from engine.types import Action, EngineState
from agents.rl.v3.encoder_v3 import encode_state_v3
from agents.rl.v3.model_v3 import AceNetV3
from engine.action_encoding import action_to_index, index_to_action


class RLAgentV3(BaseAgent):
    """Upgraded Reinforcement Learning agent guided by AceNetV3 policy predictions."""

    def __init__(
        self,
        player_id: int,
        checkpoint_path: Optional[str] = None,
        explore: bool = False,
        temperature: float = 1.0,
        model: Optional[AceNetV3] = None,
        seed: Optional[int] = None
    ) -> None:
        super().__init__(player_id, seed=seed)
        self.explore = explore
        self.temperature = temperature

        # Instantiate or share the PyTorch model
        if model is not None:
            self.model = model
        else:
            self.model = AceNetV3()
            if checkpoint_path:
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Model checkpoint path not found: {checkpoint_path}")
                try:
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False))
                except Exception as e:
                    raise RuntimeError(f"Failed to load model state_dict from {checkpoint_path}: {e}")

        self.model.eval()  # Set model to evaluation mode by default

    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        """Selects an action using AceNetV3 policy predictions."""
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 1. Encode state into flat numerical representation
        state_vec = encode_state_v3(player_view, self.player_id)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # Add batch dim

        # 2. Map legal actions to their action indices using unified action_encoding
        action_indices = [action_to_index(a) for a in legal_actions]
        legal_indices_tensor = torch.tensor([action_indices], dtype=torch.long)

        # 3. Compute masked policy probabilities
        with torch.no_grad():
            probs = self.model.get_masked_policy(state_tensor, legal_indices_tensor)
            probs = probs.squeeze(0).numpy()

        # Filter probabilities to only the legal action slots
        legal_probs = np.array([probs[idx] for idx in action_indices], dtype=np.float32)

        # Handle potential zero sum/numerical edge cases
        prob_sum = legal_probs.sum()
        if prob_sum > 0:
            legal_probs /= prob_sum
        else:
            legal_probs = np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)

        # 4. Action Selection
        if self.explore:
            # Temperature scaling for exploration
            if self.temperature != 1.0:
                # Add small epsilon to prevent log of zero
                legal_probs = np.clip(legal_probs, 1e-8, 1.0)
                legal_probs = np.power(legal_probs, 1.0 / self.temperature)
                legal_probs /= legal_probs.sum()
            chosen_action = self.rng.choices(legal_actions, weights=legal_probs, k=1)[0]
        else:
            # Deterministic argmax selection
            chosen_action = legal_actions[np.argmax(legal_probs)]

        return chosen_action
