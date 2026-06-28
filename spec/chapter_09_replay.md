# AETS Chapter 9: Replay System

## 9.1 Replay Serialization
A match replay record MUST contain all the information necessary to reproduce the state of the match from start to finish.
A Replay file MUST be serialized as a single JSON object containing:
- `version`: The specification version (e.g. `"0.1.0"`).
- `config`: The `MatchConfig` (seed, number of players, number of rounds).
- `actions`: The ordered list of all player actions applied during the match.

## 9.2 Deterministic Playback
A replay player MUST be capable of reconstructing the state at any point by starting from the initial configuration and applying the list of actions sequentially. Given the same seed, the PRNG and rules transitions will yield identical state sequences.

## 9.3 Undo, Redo, and Jump Operations
- **Undo**: Reverts the state by 1 action. The engine SHALL reconstruct the state by applying actions up to sequence `seq - 1`.
- **Redo**: Re-applies the next action from the replay action log.
- **Jump-to-state**: Instantly loads the state at a specific action sequence number `K` by executing the action log from sequence `0` up to `K`.

## 9.4 Branching Analysis
The replay system MUST support branching:
1. The user can jump to a specific action sequence `K`.
2. The user can submit a different action than the one recorded.
3. The engine MUST create a new match branch, truncating the action log after `K` and appending the new action.
4. The engine SHALL resume simulation from this new branch.
