# tests/test_compliance_harness.py
"""Automated compliance test harness.

Parses JSON compliance test cases and validates them against the AceEngine simulator.
"""

import json
import glob
from engine.rules import AceEngine, Success, Error
from engine.types import (
    StealAction, DeclineStealAction, PlayCardAction, Action, RoundStarting
)


def test_compliance_suite():
    # Find all compliance test case JSON files
    test_files = glob.glob("tests/compliance/TEST-*.json")
    assert len(test_files) >= 2, "Expected at least 2 compliance JSON test files"

    for file_path in test_files:
        with open(file_path, "r", encoding="utf-8") as f:
            test_case = json.load(f)

        test_id = test_case["test_id"]
        config = test_case["config"]
        actions = test_case["actions"]
        assertions = test_case["assertions"]

        # 1. Setup Match
        seed = int(config["match_seed"])
        state = AceEngine.create_match(
            match_id=int(test_id.split("-")[1]),
            num_players=config["num_players"],
            num_rounds=config["num_rounds"],
            match_seed=seed
        )

        # 2. Check PRNG conformance (if expected in assertions)
        if "expected_prng_outputs" in assertions:
            from engine.prng import pcg_seed, pcg_next
            s = pcg_seed(seed)
            outputs = []
            for _ in range(len(assertions["expected_prng_outputs"])):
                s, out = pcg_next(s)
                outputs.append(out)
            assert outputs == assertions["expected_prng_outputs"], f"{test_id} PRNG output mismatch"
            continue

        # 3. Advance to start Round 1
        events = []
        state, evs = AceEngine.advance(state)
        events.extend(evs)

        # 4. Apply actions sequence
        for idx, act in enumerate(actions):
            # Check for auto-advances (e.g. ROUND_INIT)
            while isinstance(state.runtime_state.current_phase, RoundStarting) and not AceEngine.is_terminal(state):
                state, evs = AceEngine.advance(state)
                events.extend(evs)

            player_id = act["player_id"]
            action_obj: Action
            if act["type"] == "DeclineSteal":
                action_obj = DeclineStealAction(player_id=player_id)
            elif act["type"] == "Steal":
                action_obj = StealAction(player_id=player_id)
            elif act["type"] == "PlayCard":
                action_obj = PlayCardAction(player_id=player_id, card=act["card"])
            else:
                raise ValueError(f"Unknown action type: {act['type']}")

            res = AceEngine.apply_action(state, action_obj)
            assert isinstance(res, Success), f"{test_id} action {idx} failed: {act}. Got Error: {getattr(res, 'message', '') if isinstance(res, Error) else ''}"
            state = res.new_state
            events.extend(res.events)

        # 5. Assertions validation
        assert AceEngine.is_terminal(state) == assertions["terminal"], f"{test_id} terminal status mismatch"

        if "round_status" in assertions:
            assert state.round_state is not None
            assert state.round_state.status == assertions["round_status"], f"{test_id} round status mismatch"

        if "current_player_id" in assertions:
            assert state.runtime_state.current_player_id == assertions["current_player_id"], f"{test_id} current player ID mismatch"

        if "expected_events" in assertions:
            event_types = [e.event_type for e in events]
            assert event_types == assertions["expected_events"], f"{test_id} event sequence mismatch. Got: {event_types}"

        if "final_player_states" in assertions:
            assert state.round_state is not None
            for p_id_str, expected_p_state in assertions["final_player_states"].items():
                p_id = int(p_id_str)
                r_player = next(p for p in state.round_state.players if p.player_id == p_id)
                assert len(r_player.hand) == expected_p_state["hand_size"], f"{test_id} player {p_id} hand size mismatch"
                assert r_player.is_active == expected_p_state["is_active"], f"{test_id} player {p_id} active status mismatch"
                m_player = next(p for p in state.match_state.players if p.player_id == p_id)
                assert m_player.consecutive_loss_count == expected_p_state["consecutive_loss_count"], f"{test_id} player {p_id} consecutive loss counter mismatch"
