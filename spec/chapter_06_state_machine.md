# AETS Chapter 6: State Machine

## 6.1 State Definitions and Transitions
The simulator MUST strictly transition through the following 10 states:

```text
MATCH_INIT -> ROUND_INIT -> TRICK_STEAL_PHASE -> TRICK_PLAY_PHASE -> TRICK_DISCARD/PICKUP -> TRICK_EVAL -> ROUND_END -> ROUND_RESULT -> MATCH_END
```

### 6.1.1 MATCH_INIT
- **Description**: Sets up the match configurations and seats.
- **Exit Condition**: Match configuration valid -> `ROUND_INIT`.

### 6.1.2 ROUND_INIT
- **Description**: Handles reserved aces allocation, shuffles deck, deals cards, and determines the initial lead player.
- **Exit Condition**: Dealing complete -> `TRICK_STEAL_PHASE` of Trick 1.

### 6.1.3 TRICK_STEAL_PHASE
- **Description**: Awaiting lead player's steal or decline decision.
- **Exit Condition**:
  - Lead player declines -> `TRICK_PLAY_PHASE`.
  - Steals from all players (auto-loss) -> `ROUND_END`.

### 6.1.4 TRICK_PLAY_PHASE
- **Description**: Players play cards sequentially clockwise.
- **Exit Condition**:
  - All active players follow suit -> `TRICK_DISCARD`.
  - A player breaks suit -> `TRICK_PICKUP`.

### 6.1.5 TRICK_DISCARD
- **Description**: Clears played cards to the face-down discard pile.
- **Exit Condition**: Always -> `TRICK_EVAL`.

### 6.1.6 TRICK_PICKUP
- **Description**: Gives all trick plays to the collector's hand.
- **Exit Condition**: Always -> `TRICK_EVAL`.

### 6.1.7 TRICK_EVAL
- **Description**: Updates player active/inactive flags. Evaluates round ending conditions. Determines the next trick leader.
- **Exit Condition**:
  - $\ge 2$ active players -> `TRICK_STEAL_PHASE` of the next trick.
  - 1 active player (or 0 active players in draw) -> `ROUND_END`.

### 6.1.8 ROUND_END
- **Description**: Declares the round loser (if any) and round winners.
- **Exit Condition**: Always -> `ROUND_RESULT`.

### 6.1.9 ROUND_RESULT
- **Description**: Updates match-level player loss counters and accumulates points.
- **Exit Condition**:
  - More rounds remain -> `ROUND_INIT`.
  - Final round completes -> `MATCH_END`.

### 6.1.10 MATCH_END
- **Description**: Calculates rankings and concludes the match.
- **Exit Condition**: Terminal state.
