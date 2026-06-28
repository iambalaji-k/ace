# AETS Chapter 2: Canonical Data Model

## 2.1 Object Immutability
All state representation objects in the simulator MUST be immutable. State transitions SHALL return a newly constructed state object instead of modifying the existing one.

## 2.2 Seating and Player Identity
- Each player in the match SHALL be assigned a unique player index (from `0` to `N-1`, where `N` is the number of players).
- The player index SHALL correspond exactly to their seating seat.
- Seating order MUST be clockwise ascending player index: `0 -> 1 -> 2 -> ... -> N-1 -> 0`.
- Seating order SHALL be immutable throughout the entire match.

## 2.3 State Structures

### 2.3.1 PlayerState (Match-Level)
Holds persistent player counters:
- `player_id: int` (equal to seat index)
- `consecutive_loss_count: int`
- `rounds_won: int`
- `rounds_lost: int`
- `rounds_drawn: int`
- `half_points: int` (representing points × 2)

### 2.3.2 MatchState
Holds match-level variables:
- `match_id: int`
- `num_rounds: int`
- `current_round: int` (1-indexed)
- `match_seed: uint64`
- `players: list[PlayerState]`
- `seating_order: list[int]`
- `round_results: list[RoundResult]`
- `status: str` (MUST be `"INIT"`, `"IN_PROGRESS"`, or `"COMPLETE"`)

### 2.3.3 RoundState
Round-specific variables:
- `round_number: int`
- `round_seed: uint64`
- `players: list[RoundPlayerState]`
- `active_player_ids: list[int]` (seats of players currently active, in seat order)
- `current_trick: Optional[TrickState]`
- `trick_history: list[CompletedTrick]`
- `lead_player_id: int`
- `discard_pile: list[int]` (card IDs, face-down)
- `status: str` (MUST be `"INIT"`, `"IN_PROGRESS"`, or `"COMPLETE"`)

### 2.3.4 RuntimeState
Transient execution context (Refinement 4):
- `action_sequence_number: int`
- `current_phase: GamePhase`
- `current_player_id: Optional[int]`
- `pending_legal_actions: list[Action]`
- `prng_state: uint64`

### 2.3.5 EngineState
The outer envelope holding the complete state of a match:
- `match_state: MatchState`
- `round_state: Optional[RoundState]`
- `runtime_state: RuntimeState`

## 2.4 Serialization
- The canonical wire format for serializing state and events SHALL be **JSON**.
- Seed values (uint64) MUST be serialized as JSON strings to avoid float64 precision loss in web environments.
- Card values SHALL be serialized as integers in the range `0` through `51`.
- Player IDs SHALL be serialized as integers.
- State objects MUST expose a deterministic SHA-256 state hash method.
