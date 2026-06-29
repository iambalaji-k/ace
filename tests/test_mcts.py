# tests/test_mcts.py
"""Unit tests verifying the Monte Carlo Tree Search (MCTS) Agent and hand determinization engine."""

import pytest
import random
from engine.rules import AceEngine
from engine.types import PlayCardAction, RoundStarting, StealAction, DeclineStealAction
from engine.heuristic_agent import CardTracker
from engine.mcts_agent import MCTSAgent, project_determinization
from engine.card import get_suit

def test_project_determinization_constraints() -> None:
    """Verifies that project_determinization satisfies void and known card constraints."""
    rng = random.Random(42)
    num_players = 4
    num_rounds = 1
    
    # 1. Create a clean match state
    state = AceEngine.create_match(
        match_id=1,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=101
    )
    
    # Advance past RoundStarting
    state, _ = AceEngine.advance(state)
    
    # Get CardTracker
    tracker = CardTracker(num_players=num_players)
    tracker.reconstruct(
        viewer_id=0,
        round_state=state.round_state,
        match_state=state.match_state
    )
    
    # Inject a known fact: Player 1 has card ID 15 (which is suit code 1: Hearts)
    known_card = 15
    tracker.card_locations[known_card] = "known_1"
    tracker.player_known_cards[1].add(known_card)
    
    # Inject a void constraint: Player 2 is void in Spades (suit 0)
    tracker.is_void[2][0] = True
    
    # 2. Run determinization projection
    det_state = project_determinization(
        viewer_id=0,
        tracker=tracker,
        state=state,
        rng=rng
    )
    
    # 3. Assertions
    det_round = det_state.round_state
    
    # A. Hand sizes must match the original state exactly
    for p in range(num_players):
        assert len(det_round.players[p].hand) == len(state.round_state.players[p].hand)
        
    # B. The known card MUST be in Player 1's hand
    assert known_card in det_round.players[1].hand
    
    # C. Player 2 MUST NOT hold any Spades (suit 0)
    for card in det_round.players[2].hand:
        assert get_suit(card) != 0

def test_mcts_agent_action_selection() -> None:
    """Verifies that MCTSAgent selects valid actions and runs without runtime errors."""
    num_players = 4
    state = AceEngine.create_match(
        match_id=2,
        num_players=num_players,
        num_rounds=1,
        match_seed=202
    )
    state, _ = AceEngine.advance(state)
    
    agent = MCTSAgent(
        player_id=0,
        seed=202,
        max_iterations=10,  # Small iteration count for fast tests
        time_limit=0.05
    )
    
    legal_actions = AceEngine.get_legal_actions(state)
    action = agent.select_action(state, legal_actions)
    
    assert action in legal_actions
    assert isinstance(action, (PlayCardAction, StealAction, DeclineStealAction))
