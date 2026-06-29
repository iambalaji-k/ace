# tests/test_heuristics.py
"""Unit tests for CardTracker and HeuristicAgent."""

import pytest  # pyright: ignore[reportMissingImports]
from engine.types import Action, DeclineStealAction, StealAction, PlayCardAction
from engine.rules import AceEngine
from engine.heuristic_agent import CardTracker, HeuristicAgent


def test_card_tracker_initialization() -> None:
    tracker = CardTracker(num_players=4)
    assert tracker.num_players == 4
    assert len(tracker.card_locations) == 52
    assert all(loc == "unknown" for loc in tracker.card_locations)


def test_heuristic_agent_creation() -> None:
    agent = HeuristicAgent(player_id=1, seed=123)
    assert agent.player_id == 1
    assert "H102" in agent.weights
    assert agent.weights["H114"] == 9999.0


def test_heuristic_agent_action_selection() -> None:
    agent = HeuristicAgent(player_id=0, seed=42)
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=42)
    state, _ = AceEngine.advance(state)

    legal_actions = (
        DeclineStealAction(player_id=0),
        StealAction(player_id=0)
    )

    action = agent.select_action(state, legal_actions)
    assert action in legal_actions


def test_heuristic_agent_tie_breaking() -> None:
    agent = HeuristicAgent(player_id=0, seed=42)
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=42)
    state, _ = AceEngine.advance(state)

    # Mock a set of identical play actions
    legal_actions = (
        PlayCardAction(player_id=0, card=0),
        PlayCardAction(player_id=0, card=1)
    )

    action = agent.select_action(state, legal_actions)
    assert action in legal_actions


def test_heuristic_agent_v2_creation() -> None:
    from engine.heuristic_agent_v2 import HeuristicAgentV2
    agent = HeuristicAgentV2(player_id=1, seed=123)
    assert agent.player_id == 1
    assert "H102" in agent.weights
    assert agent.weights["H114"] == 9999.0
