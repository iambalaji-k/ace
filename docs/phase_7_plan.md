# Phase 7 — Random AI Agent Implementation Plan

This document details the plan to verify and complete the **Random AI Agent** (Phase 7) baseline and create tools to execute automated matches.

---

## 1. Conformance & Interface Check

We will review `engine/agent.py` to ensure it complies with Phase 7 requirements:
- **Encapsulation**: The agent MUST NOT receive the raw `EngineState`. It receives the projected `player_view` where opponent cards are masked as `-1` and the discard history is cleared.
- **Move Safety**: The agent selects an action uniformly from `legal_actions`.

---

## 2. Automated Match Execution Script (`scripts/run_bot_match.py`)

To allow the user to watch automated games between bots, we will implement a command-line script:
- **Inputs**: User configures the number of players, number of rounds, seed, and optional delay between moves (in seconds, e.g. `0.5`).
- **Match Loop**:
  - Run the match.
  - Print the actions chosen by the agents.
  - Print the hands of all players (revealing cards for visualization) alongside events (e.g. `P0 plays A♠`).
  - Conclude with the scoreboard.

---

## 3. Conformance Unit Testing (`tests/test_agents.py`)

We will implement dedicated tests:
- **Determinism**: Assert that `RandomAgent` with seed `S` makes identical decisions for the same state and legal actions.
- **Compliance**: Verify that the selected action is always a member of the provided `legal_actions` list.
- **Handling Empty Actions**: Ensure the agent raises an error if `legal_actions` is empty.
- **State Masking Proof**: Verify that the agent functions correctly even if all other players' cards are masked.
