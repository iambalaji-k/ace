# tests/test_rules.py
import pytest
from engine.rules import AceEngine, Success, Error, ValidationError
from engine.types import StealAction, DeclineStealAction, PlayCardAction, RoundStarting, AwaitingStealDecision, AwaitingCardPlay, MatchComplete
from engine.card import str_to_card, get_suit


def test_match_creation():
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    assert state.match_state.match_id == 1
    assert len(state.match_state.players) == 4
    assert state.match_state.num_rounds == 2
    assert state.match_state.status == "INIT"
    assert state.round_state is None
    assert isinstance(state.runtime_state.current_phase, RoundStarting)


def test_round_starting_advance():
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, events = AceEngine.advance(state)

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


def test_decline_steal_and_play():
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

    lead_id = state.round_state.current_trick.lead_player_id
    action = DeclineStealAction(player_id=lead_id)
    res = AceEngine.apply_action(state, action)

    assert isinstance(res, Success)
    state = res.new_state
    assert state.round_state.current_trick.status == "PLAY_PHASE"
    assert isinstance(state.runtime_state.current_phase, AwaitingCardPlay)

    # Lead player must play a card from their hand
    lead_hand = next(p.hand for p in state.round_state.players if p.player_id == lead_id)
    card_to_play = lead_hand[0]
    action_play = PlayCardAction(player_id=lead_id, card=card_to_play)

    res_play = AceEngine.apply_action(state, action_play)
    assert isinstance(res_play, Success)
    state = res_play.new_state
    assert len(state.round_state.current_trick.plays) == 1
    assert state.round_state.current_trick.plays[0].card == card_to_play


def test_steal_mechanics():
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

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

    new_stealer_state = next(p for p in state.round_state.players if p.player_id == lead_id)
    new_victim_state = next(p for p in state.round_state.players if p.player_id == target_id)

    # Victim must have empty hand and be inactive
    assert len(new_victim_state.hand) == 0
    assert not new_victim_state.is_active
    assert new_victim_state.is_round_winner
    assert target_id not in state.round_state.active_player_ids

    # Stealer must have victim's cards sorted canonically
    expected_hand = sorted(stealer_initial_hand + victim_initial_hand)
    assert new_stealer_state.hand == expected_hand

    # Current phase should still be steal decision, but with next active left
    new_phase = state.runtime_state.current_phase
    assert isinstance(new_phase, AwaitingStealDecision)
    assert new_phase.player_id == lead_id
    assert new_phase.steal_target != target_id


def test_illegal_moves():
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=2, match_seed=123)
    state, _ = AceEngine.advance(state)

    # Wrong player action
    wrong_player = (state.runtime_state.current_player_id + 1) % 4
    action = DeclineStealAction(player_id=wrong_player)
    res = AceEngine.apply_action(state, action)
    assert isinstance(res, Error)
    assert res.error_code == "NOT_YOUR_TURN"

    # Decline steal by correct player
    lead_id = state.runtime_state.current_player_id
    state = AceEngine.apply_action(state, DeclineStealAction(player_id=lead_id)).new_state

    # Play a card not in hand
    lead_hand = next(p.hand for p in state.round_state.players if p.player_id == lead_id)
    card_not_in_hand = next(c for c in range(52) if c not in lead_hand)
    action_invalid = PlayCardAction(player_id=lead_id, card=card_not_in_hand)
    res_invalid = AceEngine.apply_action(state, action_invalid)
    assert isinstance(res_invalid, Error)
    assert res_invalid.error_code == "ILLEGAL_CARD"


def test_full_match_flow():
    # 4 players, 2 rounds, seed 42 (matching the walkthrough match seed)
    state = AceEngine.create_match(match_id=42, num_players=4, num_rounds=2, match_seed=42)
    # Advance to Round 1 Start
    state, events = AceEngine.advance(state)
    assert state.round_state.round_number == 1
    
    actions_taken = 0
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)
        if isinstance(phase, RoundStarting):
            state, evs = AceEngine.advance(state)
            events.extend(evs)
            continue
            
        if isinstance(phase, AwaitingStealDecision):
            # Decline steal to keep game simple
            action = DeclineStealAction(player_id=phase.player_id)
        elif isinstance(phase, AwaitingCardPlay):
            # Play first legal action
            legal = AceEngine.get_legal_actions(state)
            action = legal[0]
        else:
            break
            
        res = AceEngine.apply_action(state, action)
        assert isinstance(res, Success), f"Action failed at step {actions_taken}: {action}"
        state = res.new_state
        events.extend(res.events)
        actions_taken += 1
        
    assert state.match_state.status == "COMPLETE"
    result = AceEngine.get_result(state)
    assert result is not None
    assert result.total_rounds == 2
    assert len(result.rankings) == 4
    # The ranks should be 1, 2, 3, 4
    ranks = [r.rank for r in result.rankings]
    assert ranks == [1, 2, 3, 4]


def test_steal_auto_loss():
    # 3 players, 1 round, seed 123
    state = AceEngine.create_match(match_id=1, num_players=3, num_rounds=1, match_seed=123)
    state, _ = AceEngine.advance(state)

    lead_id = state.round_state.current_trick.lead_player_id

    # 1st Steal (steals from first player to left)
    action1 = StealAction(player_id=lead_id)
    state = AceEngine.apply_action(state, action1).new_state
    
    # Still active: lead player and remaining player
    assert len(state.round_state.active_player_ids) == 2
    assert isinstance(state.runtime_state.current_phase, AwaitingStealDecision)

    # 2nd Steal (steals from last player to left)
    action2 = StealAction(player_id=lead_id)
    state = AceEngine.apply_action(state, action2).new_state

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


