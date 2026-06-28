# AETS Chapter 3: Match Lifecycle

## 3.1 Match Definition
A Match SHALL consist of one or more sequential Rounds. The number of rounds MUST be specified before the match starts and MUST NOT change during execution.

## 3.2 Match Initialization
Upon match creation, the engine MUST:
1. Verify `num_players` is between `3` and `6` inclusive.
2. Verify `num_rounds` is $\ge 1$.
3. Set all players' `consecutive_loss_count`, `rounds_won`, `rounds_lost`, `rounds_drawn`, and `half_points` to `0`.
4. Initialize `current_round` to `1` and status to `"INIT"`.
5. Seed the match PRNG.

## 3.3 Match Progression
The match progresses by executing rounds sequentially. The `current_round` counter MUST increment by 1 each time a round completes.

## 3.4 Match Termination
The match MUST end immediately after the configured number of rounds has completed. Upon termination:
1. Match status MUST transition to `"COMPLETE"`.
2. The runtime phase MUST transition to `MatchComplete(result)`.
3. The engine SHALL calculate player rankings and determine the final match winner.

## 3.5 Match Scoring and Tiebreakers
- **Points Awarded**:
  - Round Win: **2 half-points** (equivalent to 1.0 point).
  - Round Draw: **1 half-point** (equivalent to 0.5 point).
  - Round Loss: **0 half-points** (equivalent to 0.0 points).
- **Match Winner Determination**:
  - The player with the highest total `half_points` at the end of the match SHALL be ranked 1st (winner).
- **Tiebreaker Rules**: If two or more players have equal half-points, the engine MUST resolve ties in the following order:
  1. **Fewer Round Losses**: The player with fewer `rounds_lost` is ranked higher.
  2. **Seat Index**: The player with the lower seat index is ranked higher.
