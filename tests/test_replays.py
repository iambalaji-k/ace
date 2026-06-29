# tests/test_replays.py
"""Unit tests for the Replay System.

Verifies serialization, playback fidelity, undo/redo, timeline jumps,
and branching (forking) capabilities of the ReplayPlayer.
"""

import os
from engine.rules import AceEngine, Success
from engine.types import PlayCardAction, DeclineStealAction, StealAction, RoundStarting
from engine.replay import Replay, ReplayAction, serialize_replay, deserialize_replay, export_replay, import_replay
from engine.replay_player import ReplayPlayer


def test_replay_serialization_and_fidelity(tmp_path):
    # 1. Run a live match with seed 42 to completion (or 50 moves)
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=42)
    state, events = AceEngine.advance(state)
    
    actions_log = []
    seq_counter = 0
    
    # We will play 20 moves randomly/legally to build a replay
    while not AceEngine.is_terminal(state) and seq_counter < 30:
        legal = AceEngine.get_legal_actions(state)
        # Select first legal action for deterministic testing
        action = legal[0]
        
        # Log the action
        act_type = ""
        card_val = None
        if isinstance(action, DeclineStealAction):
            act_type = "DeclineSteal"
        elif isinstance(action, StealAction):
            act_type = "Steal"
        elif isinstance(action, PlayCardAction):
            act_type = "PlayCard"
            card_val = action.card
            
        actions_log.append(ReplayAction(
            sequence=seq_counter,
            action_type=act_type,
            player_id=action.player_id,
            card=card_val
        ))
        
        res = AceEngine.apply_action(state, action)
        assert isinstance(res, Success)
        state = res.new_state
        seq_counter += 1
        
        # Advance if needed
        while isinstance(state.runtime_state.current_phase, RoundStarting) and not AceEngine.is_terminal(state):
            state, _ = AceEngine.advance(state)

    live_final_hash = state.get_state_hash()

    # 2. Build Replay object
    replay = Replay(
        version="0.1.0",
        match_id=1,
        num_players=4,
        num_rounds=1,
        match_seed=42,
        actions=actions_log
    )

    # 3. Test Serialization
    serialized = serialize_replay(replay)
    deserialized = deserialize_replay(serialized)
    assert deserialized.match_seed == 42
    assert len(deserialized.actions) == len(actions_log)

    # 4. Test Export / Import file
    temp_filepath = os.path.join(tmp_path, "temp_test_replay.json")
    export_replay(replay, temp_filepath)
    assert os.path.exists(temp_filepath)
    
    imported = import_replay(temp_filepath)
    assert imported.match_seed == 42

    # 5. Playback using ReplayPlayer
    player = ReplayPlayer(imported)
    while player.step():
        pass
        
    replay_final_hash = player.state.get_state_hash()

    # Assert perfect determinism
    assert live_final_hash == replay_final_hash


def test_replay_playback_traversal():
    # Setup a fixed 2-action replay
    actions = [
        ReplayAction(sequence=0, action_type="DeclineSteal", player_id=0),
        # P0 holds A♠ (card 0) and plays it
        ReplayAction(sequence=1, action_type="PlayCard", player_id=0, card=0)
    ]
    replay = Replay(
        version="0.1.0",
        match_id=2,
        num_players=4,
        num_rounds=1,
        match_seed=12345, # Player 0 holds A♠ for seed 12345
        actions=actions
    )

    player = ReplayPlayer(replay)
    assert player.current_index == 0

    # Step 1: Decline steal
    success = player.step()
    assert success
    assert player.current_index == 1

    # Step 2: Play card
    success = player.step()
    assert success
    assert player.current_index == 2

    # Step 3: Undo play card
    success = player.undo()
    assert success
    assert player.current_index == 1

    # Step 4: Jump to 0 (initial state)
    player.jump_to(0)
    assert player.current_index == 0

    # Step 5: Jump to 2
    player.jump_to(2)
    assert player.current_index == 2


def test_replay_branching_analysis():
    # Setup match and decline steal
    actions = [
        ReplayAction(sequence=0, action_type="DeclineSteal", player_id=0)
    ]
    replay = Replay(
        version="0.1.0",
        match_id=3,
        num_players=4,
        num_rounds=1,
        match_seed=12345,
        actions=actions
    )

    player = ReplayPlayer(replay)
    player.step() # current_index = 1
    
    # Fork at index 1 by playing a different card than what would normally happen
    # P0 holds card 0 (A♠) but we will force him to play card 0
    # Let's verify that a new branch is created
    forked_replay = player.fork_at(1, PlayCardAction(player_id=0, card=0))
    
    assert len(forked_replay.actions) == 2
    assert forked_replay.actions[1].action_type == "PlayCard"
    assert forked_replay.actions[1].card == 0
