# engine/rl_agent.py
"""Reinforcement Learning Agent using the AceNet PyTorch model."""

import os
import torch
import numpy as np
import random
from typing import Sequence, Optional
from engine.agent import BaseAgent
from engine.types import Action, EngineState, PlayCardAction, StealAction, DeclineStealAction
from engine.encoder import encode_state
from engine.model import AceNet

def action_to_idx(action: Action) -> int:
    """Maps action objects to index positions [0-53]."""
    if isinstance(action, PlayCardAction):
        return action.card
    elif isinstance(action, StealAction):
        return 52
    elif isinstance(action, DeclineStealAction):
        return 53
    raise ValueError(f"Unknown action type: {action}")

class RLAgent(BaseAgent):
    """Reinforcement Learning agent guided by AceNet output predictions."""
    
    def __init__(
        self,
        player_id: int,
        checkpoint_path: Optional[str] = None,
        explore: bool = False,
        temperature: float = 1.0,
        model: Optional[AceNet] = None
    ) -> None:
        super().__init__(player_id)
        self.explore = explore
        self.temperature = temperature
        
        # Instantiate or share the PyTorch model
        if model is not None:
            self.model = model
        else:
            self.model = AceNet()
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
                except Exception:
                    pass
                    
        self.model.eval()  # Set model to evaluation mode by default
        
    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        """Selects an action using AceNet policy predictions."""
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        # 1. Encode state into flat numerical representation
        state_vec = encode_state(player_view, self.player_id)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        
        # 2. Map legal actions to their action indices
        action_indices = [action_to_idx(a) for a in legal_actions]
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
                legal_probs = np.power(legal_probs, 1.0 / self.temperature)
                legal_probs /= legal_probs.sum()
            chosen_action = random.choices(legal_actions, weights=legal_probs, k=1)[0]
        else:
            # Deterministic argmax selection
            chosen_action = legal_actions[np.argmax(legal_probs)]
            
        return chosen_action
