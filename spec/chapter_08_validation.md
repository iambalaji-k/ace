# AETS Chapter 8: Validation and Error Handling

## 8.1 Verification Registry (Invariants)
The simulator MUST check all 12 core invariants after every state transition:

- **INV-001**: Every card ID (`0–51`) exists exactly once across:
  - all players' hands
  - plays in the current trick
  - the discard pile
- **INV-002**: `len(active_players) >= 0` at all times during a round.
- **INV-003**: Every Active Player has `len(hand) >= 1` at the start of each trick (in `TRICK_STEAL_PHASE`).
- **INV-004**: Seating order is immutable throughout the match.
- **INV-005**: `sum(all hand sizes) + len(discard_pile) + len(current_trick_cards) = 52` at all times.
- **INV-006**: Exactly one Round Loser per completed round, unless draw (zero losers).
- **INV-007**: A player's consecutive loss counter $\ge 0$ at all times.
- **INV-008**: Reserved ace count = `min(player.consecutive_loss_count, 4)` for each player at round start.
- **INV-009**: The lead player for the first trick of a round holds A♠.
- **INV-010**: The number of completed rounds never exceeds `match.num_rounds`.
- **INV-011**: A player in `TRICK_PLAY_PHASE` who holds a card of the Lead Suit MUST play a card of that suit.
- **INV-012**: At most one player has `consecutive_loss_count > 0` at any given time.

## 8.2 Error Handling Strategy (Option C)
The simulator MUST enforce a strict separation between player errors and simulator engine bugs.

### 8.2.1 Player Moves (Illegal Actions)
If a player submits an action that is illegal (e.g. playing a card they don't hold, or breaking suit when they have cards of the lead suit):
1. The engine MUST reject the action.
2. The engine MUST NOT update its state.
3. The engine MUST return a structured `ActionResult.Error` containing an error code (e.g. `ILLEGAL_CARD`, `MUST_FOLLOW_SUIT`), an explanation message, and the list of current legal actions.
4. There is no penalty or timeout; the player is allowed to retry.

### 8.2.2 Engine Panics (Invariant Violations)
If any of the 12 invariants (`INV-001` through `INV-012`) fail to validate after a transition:
1. The engine MUST immediately halt execution (panic).
2. The engine MUST dump a diagnostic file containing:
   - The ID of the failed invariant.
   - The current full state snapshot.
   - The complete event log.
   - The current PRNG state.
