# tests/test_rl.py
"""Unit tests verifying the Reinforcement Learning model, state encoder, and agent."""

import pytest
import numpy as np
import torch
from engine.rules import AceEngine
from engine.types import PlayCardAction, RoundStarting
from engine.encoder import encode_state
from engine.model import AceNet
from engine.rl_agent import RLAgent, action_to_idx

def test_state_encoder_dimensions() -> None:
    """Verifies that the state encoder output has correct shape and types."""
    num_players = 4
    state = AceEngine.create_match(
        match_id=1,
        num_players=num_players,
        num_rounds=1,
        match_seed=100
    )
    state, _ = AceEngine.advance(state)
    
    encoded = encode_state(state, player_id=0)
    
    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.float32
    assert encoded.shape == (343,)

def test_model_outputs_and_masking() -> None:
    """Verifies model forward pass and action masking output probability constraints."""
    model = AceNet()
    state_tensor = torch.randn(1, 343)
    
    # 1. Forward Pass
    logits, value = model(state_tensor)
    assert logits.shape == (1, 54)
    assert value.shape == (1, 1)
    assert -1.0 <= value.item() <= 1.0
    
    # 2. Masking (simulate only action indices 2, 10, 52 are legal)
    legal_indices = torch.tensor([[2, 10, 52]], dtype=torch.long)
    probs = model.get_masked_policy(state_tensor, legal_indices)
    probs = probs.squeeze(0).detach().numpy()
    
    # Assert legal actions have non-zero probability
    assert probs[2] > 0.0
    assert probs[10] > 0.0
    assert probs[52] > 0.0
    
    # Assert illegal actions have exactly 0.0 probability
    assert np.allclose(probs[0], 0.0, atol=1e-6)
    assert np.allclose(probs[53], 0.0, atol=1e-6)
    # Sum of probabilities must equal 1.0
    assert np.allclose(probs.sum(), 1.0, atol=1e-5)

def test_rl_agent_action_selection() -> None:
    """Verifies that RLAgent selects valid actions and maps outputs correctly."""
    num_players = 4
    state = AceEngine.create_match(
        match_id=2,
        num_players=num_players,
        num_rounds=1,
        match_seed=200
    )
    state, _ = AceEngine.advance(state)
    
    agent = RLAgent(player_id=0, explore=False)
    
    legal_actions = AceEngine.get_legal_actions(state)
    action = agent.select_action(state, legal_actions)
    
    assert action in legal_actions
    assert isinstance(action_to_idx(action), int)
