# tests/test_rules.py
"""Integration tests and full match flow rules checking.

Verifies state transitions, player turn validations, legal actions,
steal triggers, suit follow rules, re-entries, and scoring.
"""

from engine.rules import AceEngine, Success, Error
from engine.types import (
    StealAction, DeclineStealAction, PlayCardAction, RoundStarting,
    AwaitingStealDecision, AwaitingCardPlay, MatchComplete
)


def test_match_creation() -> None:
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    assert state.match_state.match_id == 1
    assert len(state.match_state.players) == 4
    assert state.match_state.num_rounds == 2
    assert state.match_state.status == "INIT"
    assert state.round_state is None
    assert isinstance(state.runtime_state.current_phase, RoundStarting)


def test_round_starting_advance() -> None:
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

    assert state.match_state.status == "IN_PROGRESS"
    assert state.round_state is not None
    assert state.round_state.round_number == 1
    assert state.round_state.status == "IN_PROGRESS"
    assert len(state.round_state.active_player_ids) == 4

    # The current trick must be trick 1, STEAL_PHASE
    trick = state.round_state.current_trick
    assert trick is not None
    assert trick.trick_number == 1
    assert trick.status == "STEAL_PHASE"
    assert trick.lead_player_id == state.runtime_state.current_player_id

    # The phase must be awaiting steal
    assert isinstance(state.runtime_state.current_phase, AwaitingStealDecision)
    assert len(state.runtime_state.pending_legal_actions) == 2  # Steal or Decline


def test_decline_steal_and_play() -> None:
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    lead_id = state.round_state.current_trick.lead_player_id
    action = DeclineStealAction(player_id=lead_id)
    res = AceEngine.apply_action(state, action)

    assert isinstance(res, Success)
    state = res.new_state
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.status == "PLAY_PHASE"
    assert isinstance(state.runtime_state.current_phase, AwaitingCardPlay)

    # Lead player must play a card from their hand
    lead_hand = next(p.hand for p in state.round_state.players if p.player_id == lead_id)
    card_to_play = lead_hand[0]
    action_play = PlayCardAction(player_id=lead_id, card=card_to_play)

    res_play = AceEngine.apply_action(state, action_play)
    assert isinstance(res_play, Success)
    state = res_play.new_state
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert len(state.round_state.current_trick.plays) == 1
    assert state.round_state.current_trick.plays[0].card == card_to_play


def test_steal_mechanics() -> None:
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    lead_id = state.round_state.current_trick.lead_player_id
    phase = state.runtime_state.current_phase
    assert isinstance(phase, AwaitingStealDecision)
    target_id = phase.steal_target

    stealer_state = next(p for p in state.round_state.players if p.player_id == lead_id)
    victim_state = next(p for p in state.round_state.players if p.player_id == target_id)

    stealer_initial_hand = list(stealer_state.hand)
    victim_initial_hand = list(victim_state.hand)

    action = StealAction(player_id=lead_id)
    res = AceEngine.apply_action(state, action)
    assert isinstance(res, Success)
    state = res.new_state

    assert state.round_state is not None
    new_stealer_state = next(p for p in state.round_state.players if p.player_id == lead_id)
    new_victim_state = next(p for p in state.round_state.players if p.player_id == target_id)

    # Victim must have empty hand and be inactive
    assert len(new_victim_state.hand) == 0
    assert not new_victim_state.is_active
    assert new_victim_state.is_round_winner
    assert target_id not in state.round_state.active_player_ids

    # Stealer must have victim's cards sorted canonically
    expected_hand = tuple(sorted(stealer_initial_hand + victim_initial_hand))
    assert new_stealer_state.hand == expected_hand


def test_illegal_moves() -> None:
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

    # Wrong player action
    curr_player_id = state.runtime_state.current_player_id
    assert curr_player_id is not None
    wrong_player = (curr_player_id + 1) % 4
    action = DeclineStealAction(player_id=wrong_player)
    res = AceEngine.apply_action(state, action)
    assert isinstance(res, Error)
    assert res.error_code == "NOT_YOUR_TURN"

    # Decline steal by correct player
    lead_id = state.runtime_state.current_player_id
    assert lead_id is not None
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=lead_id))
    assert isinstance(res, Success)
    state = res.new_state

    # Play a card not in hand
    assert state.round_state is not None
    lead_hand = next(p.hand for p in state.round_state.players if p.player_id == lead_id)
    card_not_in_hand = next(c for c in range(52) if c not in lead_hand)
    action_invalid = PlayCardAction(player_id=lead_id, card=card_not_in_hand)
    res_invalid = AceEngine.apply_action(state, action_invalid)
    assert isinstance(res_invalid, Error)
    assert res_invalid.error_code == "ILLEGAL_CARD"


def test_full_match_flow() -> None:
    # 4 players, 2 rounds, seed 42 (matching the walkthrough match seed)
    state = AceEngine.create_match(match_id=42, num_players=4, num_rounds=2, match_seed=42)
    # Advance to Round 1 Start
    state, _ = AceEngine.advance(state)
    assert state.round_state is not None
    assert state.round_state.round_number == 1
    
    actions_taken = 0
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)
        if isinstance(phase, RoundStarting):
            state, _ = AceEngine.advance(state)
            continue

        legal = AceEngine.get_legal_actions(state)
        assert len(legal) > 0
        action = legal[0]

        res = AceEngine.apply_action(state, action)
        assert isinstance(res, Success)
        state = res.new_state
        actions_taken += 1
        assert actions_taken < 1000  # Avoid infinite loop safeguard

    result = AceEngine.get_result(state)
    assert result is not None
    assert result.total_rounds == 2
    assert len(result.rankings) == 4
    # The ranks should be 1, 2, 3, 4
    ranks = [r.rank for r in result.rankings]
    assert ranks == [1, 2, 3, 4]


def test_steal_auto_loss() -> None:
    # 3 players, 1 round, seed 123
    state = AceEngine.create_match(match_id=1, num_players=3, num_rounds=1, match_seed=123)
    state, _ = AceEngine.advance(state)

    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    lead_id = state.round_state.current_trick.lead_player_id

    # 1st Steal (steals from first player to left)
    action1 = StealAction(player_id=lead_id)
    res = AceEngine.apply_action(state, action1)
    assert isinstance(res, Success)
    state = res.new_state
    
    # Still active: lead player and remaining player
    assert state.round_state is not None
    assert len(state.round_state.active_player_ids) == 2
    assert isinstance(state.runtime_state.current_phase, AwaitingStealDecision)

    # 2nd Steal (steals from last player to left)
    action2 = StealAction(player_id=lead_id)
    res = AceEngine.apply_action(state, action2)
    assert isinstance(res, Success)
    state = res.new_state

    # Match should be complete because this was a 1 round match, and stealing from all players ended the round
    assert state.match_state.status == "COMPLETE"
    assert isinstance(state.runtime_state.current_phase, MatchComplete)
    result = AceEngine.get_result(state)
    assert result is not None
    
    # The lead player who stole everyone's cards is the loser of the round, so they have 0 rounds won, 1 round lost, points 0, and rank 3
    lead_ranking = next(r for r in result.rankings if r.player_id == lead_id)
    assert lead_ranking.rank == 3
    assert lead_ranking.half_points == 0
    assert lead_ranking.rounds_lost == 1
    assert lead_ranking.rounds_won == 0
