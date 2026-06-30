# tests/test_rl_v2.py
"""Unit tests verifying RL Agent 2.0 Core: encoder_v2, model_v2, RLAgentV2 strict checkpointing, and dynamic MCTS pools."""

import sys
sys.path.append('.')
import numpy as np
import torch
import os
from engine.rules import AceEngine
from engine.types import PlayCardAction, Action
from engine.encoder_v2 import encode_state_v2
from engine.model_v2 import AceNetV2
from engine.rl_agent_v2 import RLAgentV2
from engine.mcts_agent import MCTSAgent
from engine.action_encoding import action_to_index


def test_encoder_v2_dimensions() -> None:
    """Verifies that the upgraded state encoder has correct shape of 459 across player counts 3 to 6."""
    for num_players in [3, 4, 5, 6]:
        state = AceEngine.create_match(
            match_id=1,
            num_players=num_players,
            num_rounds=1,
            match_seed=100
        )
        state, _ = AceEngine.advance(state)
        
        encoded = encode_state_v2(state, player_id=0)
        
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.float32
        assert encoded.shape == (459,)


def test_model_v2_architecture() -> None:
    """Verifies that AceNetV2 handles forward pass dimensions and masked action mapping correctly."""
    model = AceNetV2()
    state_tensor = torch.randn(1, 459)
    
    # 1. Forward Pass
    logits, value = model(state_tensor)
    assert logits.shape == (1, 54)
    assert value.shape == (1, 1)
    
    # 2. Masking probabilities
    legal_indices = torch.tensor([[0, 1, 5, 10]], dtype=torch.long)
    probs = model.get_masked_policy(state_tensor, legal_indices)
    assert probs.shape == (1, 54)
    
    # Assert that illegal actions are zeroed out (probabilities < 1e-7)
    probs_np = probs.squeeze(0).detach().numpy()
    for idx in range(54):
        if idx not in [0, 1, 5, 10]:
            assert probs_np[idx] < 1e-7
        else:
            assert probs_np[idx] > 0.0
    
    # Assert sum of legal probabilities is exactly 1.0
    assert np.allclose(probs_np.sum(), 1.0)


def test_strict_checkpoint_failure() -> None:
    """Verifies that RLAgentV2 raises FileNotFoundError or RuntimeError when requested checkpoints fail."""
    # 1. Non-existent file
    try:
        RLAgentV2(player_id=0, checkpoint_path="engine/non_existent_ckpt.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
        
    # 2. Corrupt / invalid weight file format
    corrupt_file = "engine/corrupt_test_ckpt.pt"
    with open(corrupt_file, "w") as f:
        f.write("corrupt file content")
        
    try:
        try:
            RLAgentV2(player_id=0, checkpoint_path=corrupt_file)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
    finally:
        if os.path.exists(corrupt_file):
            os.remove(corrupt_file)


def test_mcts_dynamic_player_pool() -> None:
    """Verifies that MCTSAgent rollout agents pool allocates dynamically up to 6 players, avoiding KeyErrors."""
    agent = MCTSAgent(player_id=0, max_iterations=5, time_limit=0.01)
    
    for num_players in [3, 4, 5, 6]:
        state = AceEngine.create_match(
            match_id=1,
            num_players=num_players,
            num_rounds=1,
            match_seed=120
        )
        state, _ = AceEngine.advance(state)
        legal_actions = AceEngine.get_legal_actions(state)
        
        action = agent.select_action(state, legal_actions)
        assert isinstance(action, Action)
        
        for p in range(num_players):
            assert p in agent.rollout_agents


if __name__ == "__main__":
    print("====================================================")
    print("===      RL AGENT 2.0 CORE UNIT TEST SUITE       ===")
    print("====================================================\n")
    
    print("Running test_encoder_v2_dimensions...")
    test_encoder_v2_dimensions()
    print("-> PASS")
    
    print("Running test_model_v2_architecture...")
    test_model_v2_architecture()
    print("-> PASS")
    
    print("Running test_strict_checkpoint_failure...")
    test_strict_checkpoint_failure()
    print("-> PASS")
    
    print("Running test_mcts_dynamic_player_pool...")
    test_mcts_dynamic_player_pool()
    print("-> PASS")
    
    print("\nSUCCESS: All unit tests completed and passed successfully!")

