# Phase 8 — Simulator Hardening & AI Interface Improvements Plan

This document details the stage-by-stage execution plan to implement the architectural, safety, and interface improvements suggested in [suggested_improvements.md](file:///D:/Vibe%20Coding/ace/suggested_improvements.md).

---

## Stage 1: Strict State Immutability (Safety & Threading)
* **Goal**: Close the Python list mutation loophole by replacing all `List[...]` fields inside `engine/types.py` dataclasses with `Tuple[...]` fields.
* **Steps**:
  1. In `engine/types.py`, refactor data types:
     * `RoundPlayerState.hand: List[int]` $\rightarrow$ `Tuple[int, ...]`
     * `RoundState.players: List[RoundPlayerState]` $\rightarrow$ `Tuple[RoundPlayerState, ...]`
     * `RoundState.active_player_ids: List[int]` $\rightarrow$ `Tuple[int, ...]`
     * `RoundState.trick_history: List[CompletedTrick]` $\rightarrow$ `Tuple[CompletedTrick, ...]`
     * `RoundState.discard_pile: List[int]` $\rightarrow$ `Tuple[int, ...]`
     * `RuntimeState.pending_legal_actions: List[Action]` $\rightarrow$ `Tuple[Action, ...]`
  2. Refactor rules construction sites in `engine/rules.py` to wrap outputs in `tuple()` before state instantiations.
  3. Refactor mock hand setups inside `tests/test_rules.py` and `tests/test_walkthrough_compliance.py` to construct hands/players as tuples.
  4. Run `basedpyright` and `pytest` to verify compilation and execution.

---

## Stage 2: AI Observation & Action Encoding Layers (Interface stability)
* **Goal**: Create clean, decoupled APIs for agents to read states and output actions.
* **Steps**:
  1. Create `engine/observation.py` implementing:
     * `PlayerObservation` dataclass (completely decoupled representation).
     * `build_player_observation(state: EngineState, player_id: int) -> PlayerObservation`.
     * `EncodedObservation` (float vector + action mask tuple).
     * `encode_observation(obs: PlayerObservation) -> EncodedObservation`.
  2. Create `engine/action_encoding.py` (or add to `agents/random/agent.py`) implementing:
     * Action mapping logic:
       * `0` $\rightarrow$ `DeclineSteal`
       * `1` $\rightarrow$ `Steal`
       * `2..53` $\rightarrow$ `PlayCard(card_id = index - 2)`
     * `action_to_index(action: Action) -> int`
     * `index_to_action(player_id: int, action_index: int) -> Action`
     * `legal_action_mask(legal_actions: Sequence[Action]) -> Tuple[bool, ...]`
  3. Add test coverage in `tests/test_agents.py` to verify that observations correctly mask hidden details and that action indices map back and forth with zero loss.

---

## Stage 3: Test Suite Portability (tmp_path fixture)
* **Goal**: Refactor temporary file exports in test files to use pytest's safe, sandboxed `tmp_path` directory context.
* **Steps**:
  1. In `tests/test_replays.py`, modify `test_replay_serialization_and_fidelity` to accept the `tmp_path` argument.
  2. Replace `temp_filepath = "replays/temp_test_replay.json"` with `temp_filepath = os.path.join(tmp_path, "temp_test_replay.json")`.
  3. Remove manual `os.remove` calls since pytest cleans up `tmp_path` automatically.
  4. Run pytest to ensure tests execute cleanly without leftover junk.

---

## Stage 4: Integrate Stable APIs with Demos
* **Goal**: Update simulator tools to leverage the new observation/action abstractions.
* **Steps**:
  1. Update `scripts/run_bot_match.py` to use `build_player_observation` when querying agents.
  2. Verify that random bots still execute successfully and output correct scoreboards.
  3. Update linter configurations and resolve any new compiler/check warnings.
