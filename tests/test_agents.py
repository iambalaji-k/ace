# tests/test_agents.py
"""Unit tests for agents, observation layer, and action encoding APIs.

Verifies random agent determinism, action-to-index lossless mappings,
observation creation, and flat vector state encoding.
"""

from typing import Sequence
import pytest  # pyright: ignore[reportMissingImports]
from agents.random.agent import RandomAgent
from engine.types import Action, DeclineStealAction, StealAction, PlayCardAction
from engine.rules import AceEngine, Success
from engine.observation import build_player_observation, encode_observation
from engine.action_encoding import action_to_index, index_to_action, legal_action_mask


def test_random_agent_compliance() -> None:
    # 1. Setup random agent with fixed seed
    agent = RandomAgent(player_id=0, seed=42)

    # Mock custom set of legal actions
    legal_actions: Sequence[Action] = (
        DeclineStealAction(player_id=0),
        StealAction(player_id=0),
        PlayCardAction(player_id=0, card=10)
    )

    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=42)
    state, _ = AceEngine.advance(state)

    # Assert selected action is member of the input list
    action = agent.select_action(state, legal_actions)
    assert action in legal_actions

    # 2. Assert determinism across identical instances
    agent_1 = RandomAgent(player_id=0, seed=123)
    agent_2 = RandomAgent(player_id=0, seed=123)

    action_1 = agent_1.select_action(state, legal_actions)
    action_2 = agent_2.select_action(state, legal_actions)
    assert action_1 == action_2


def test_random_agent_empty_legal_actions() -> None:
    agent = RandomAgent(player_id=0, seed=42)
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=42)
    state, _ = AceEngine.advance(state)

    # Empty action sequence must raise an IndexError from python's random generator
    with pytest.raises(IndexError):
        agent.select_action(state, ())


def test_action_encoding_lossless() -> None:
    player_id = 2
    
    # 1. Test DeclineSteal mapping
    a_decline = DeclineStealAction(player_id=player_id)
    idx_decline = action_to_index(a_decline)
    assert idx_decline == 0
    assert index_to_action(player_id, idx_decline) == a_decline
    
    # 2. Test Steal mapping
    a_steal = StealAction(player_id=player_id)
    idx_steal = action_to_index(a_steal)
    assert idx_steal == 1
    assert index_to_action(player_id, idx_steal) == a_steal
    
    # 3. Test PlayCard mapping
    for card_id in range(52):
        a_play = PlayCardAction(player_id=player_id, card=card_id)
        idx_play = action_to_index(a_play)
        assert idx_play == 2 + card_id
        assert index_to_action(player_id, idx_play) == a_play

    # 4. Out of bounds index
    with pytest.raises(ValueError):
        index_to_action(player_id, 54)
    with pytest.raises(ValueError):
        index_to_action(player_id, -1)


def test_observation_building_and_encoding() -> None:
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=42)
    state, _ = AceEngine.advance(state)
    
    # Build observation for player 0 (currently player 0's turn)
    obs = build_player_observation(state, player_id=0)
    assert obs.player_id == 0
    assert obs.is_your_turn is True
    assert len(obs.hand) == 13
    assert len(obs.opponent_hand_sizes) == 4
    assert obs.opponent_hand_sizes == (13, 13, 13, 13)
    assert len(obs.legal_actions) == 2  # Steal or Decline
    assert obs.game_phase_type == "STEAL_PHASE"
    
    # Check encoding
    enc = encode_observation(obs)
    assert len(enc.vector) == 126
    assert len(enc.action_mask) == 54
    # Action mask should be True at index 0 (DeclineSteal) and 1 (Steal), False elsewhere
    assert enc.action_mask[0] is True
    assert enc.action_mask[1] is True
    assert sum(enc.action_mask) == 2
    
    # Vector: hand indexes should be 1.0
    for card in obs.hand:
        assert enc.vector[card] == 1.0
    
    # Vector: turn indicator at index 125 should be 1.0
    assert enc.vector[125] == 1.0

    # Build observation for opponent player 1
    obs_opponent = build_player_observation(state, player_id=1)
    assert obs_opponent.is_your_turn is False
    assert len(obs_opponent.legal_actions) == 0
    
    enc_opponent = encode_observation(obs_opponent)
    assert enc_opponent.vector[125] == 0.0
    assert sum(enc_opponent.action_mask) == 0


def test_legal_action_mask_utility() -> None:
    legal = (
        DeclineStealAction(player_id=1),
        PlayCardAction(player_id=1, card=4),
        PlayCardAction(player_id=1, card=39)
    )
    mask = legal_action_mask(legal)
    assert len(mask) == 54
    assert mask[0] is True  # Decline
    assert mask[1] is False # Steal
    assert mask[2 + 4] is True # Card 4
    assert mask[2 + 39] is True # Card 39
    assert sum(mask) == 3
