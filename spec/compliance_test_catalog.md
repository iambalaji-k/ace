# AETS Spec Chapter 12: Compliance Test Catalog

This catalog specifies the test cases required to verify conformance of any implementation with the Ace Engine Technical Specification.

---

## TEST-0001: PCG-XSH-RR-64/32 Conformance
- **Rule IDs**: AETS-1.3
- **Description**: Verifies the implementation of the PCG PRNG state transitions and output values given a known seed.
- **Initial Configuration**:
  - Seed: `42`
- **Actions**:
  1. Call `pcg_seed(42)`.
  2. Generate first 3 random outputs.
- **Expected Assertions**:
  - The LCG state sequences and outputs MUST match reference values.

---

## TEST-0002: Fisher-Yates Shuffle Determinism
- **Rule IDs**: AETS-1.4
- **Description**: Verifies that the Fisher-Yates Durstenfeld backward shuffle algorithm operates deterministically.
- **Initial Configuration**:
  - Seed: `42`
  - Input list: `[0, 1, 2, ..., 51]`
- **Actions**:
  1. Shuffle the input list using `deterministic_shuffle` with the PRNG state seeded with `42`.
- **Expected Assertions**:
  - Shuffled list MUST match the deterministic output array exactly.

---

## TEST-0003: Standard Deal Allocation & Lead Player
- **Rule IDs**: AETS-2.2, AETS-4.2
- **Description**: Verifies that dealing with 4 players and 0 loss counters distributes cards evenly and determines the lead player.
- **Initial Configuration**:
  - Num Players: `4`
  - Num Rounds: `1`
  - Seed: `12345`
- **Preconditions**:
  - Match is in `RoundStarting` phase.
- **Actions**:
  1. Advance state to start Round 1.
- **Expected Events**:
  - `MATCH_STARTED`, `ROUND_STARTED`, `CARDS_DEALT` (x4), `TRICK_STARTED`.
- **Expected Assertions**:
  - Each player MUST hold exactly 13 cards.
  - Hands MUST be sorted in canonical order.
  - The player holding `A♠` (card `0`) MUST be designated the lead player.

---

## TEST-0004: Skip Deal Rotations (1 Reserved Ace)
- **Rule IDs**: AETS-4.4
- **Description**: Verifies dealing with 4 players where player 3 (recipient) receives A♠ and is skipped for 1 deal rotation.
- **Initial Configuration**:
  - Num Players: `4`
  - Num Rounds: `2`
  - Seed: `42`
- **Preconditions**:
  - Player 3 has `consecutive_loss_count = 1` at round init.
- **Actions**:
  1. Start Round 2.
- **Expected Assertions**:
  - Player 3 receives `A♠` as a reserved card.
  - Player 3 is skipped during the first deal rotation.
  - All hands MUST hold exactly 13 cards after dealing completes.

---

## TEST-0005: Lead Decline Steal
- **Rule IDs**: AETS-5.2
- **Description**: Verifies that declining a steal transitions the trick from STEAL_PHASE to PLAY_PHASE.
- **Initial Configuration**:
  - Num Players: `4`
  - Seed: `12345`
- **Preconditions**:
  - Game is in `AwaitingStealDecision` for player `lead_id`.
- **Actions**:
  1. Player `lead_id` declines to steal.
- **Expected Events**:
  - `STEAL_DECLINED`.
- **Expected Assertions**:
  - Current phase MUST transition to `AwaitingCardPlay` for player `lead_id`.
  - Pending legal actions MUST be playing cards from player `lead_id`'s hand.

---

## TEST-0006: Single Steal Execution
- **Rule IDs**: AETS-5.2
- **Description**: Verifies that executing a steal transfers all cards to the stealer, and the victim is immediately set to Inactive.
- **Initial Configuration**:
  - Num Players: `4`
  - Seed: `12345`
- **Preconditions**:
  - Current phase is `AwaitingStealDecision` for player `lead_id`.
- **Actions**:
  1. Player `lead_id` executes a steal.
- **Expected Events**:
  - `STEAL_EXECUTED`, `PLAYER_INACTIVE`.
- **Expected Assertions**:
  - Victim's hand size MUST be `0`.
  - Victim's status MUST be inactive (`is_active = False`) and is a Round Winner.
  - Stealer's hand MUST contain all the victim's initial cards, sorted canonically.

---

## TEST-0007: Steal Induced Auto-Loss
- **Rule IDs**: AETS-5.2
- **Description**: Verifies that if the lead player steals cards from all other players, the lead player is immediately declared the Round Loser and the match finishes.
- **Initial Configuration**:
  - Num Players: `3`
  - Num Rounds: `1`
  - Seed: `123`
- **Preconditions**:
  - Round 1, Trick 1, `AwaitingStealDecision` for `lead_id`.
- **Actions**:
  1. `lead_id` steals from target 1.
  2. `lead_id` steals from target 2.
- **Expected Events**:
  - `STEAL_EXECUTED`, `PLAYER_INACTIVE` (x2), `ROUND_ENDED`, `MATCH_ENDED`.
- **Expected Assertions**:
  - Match status MUST transition to `COMPLETE`.
  - Rankings MUST declare player `lead_id` as the loser (rank 3).

---

## TEST-0008: Illegal Off-Suit Play Rejection
- **Rule IDs**: AETS-5.3, AETS-8.2
- **Description**: Verifies that the engine rejects an off-suit play if the player holds cards of the lead suit.
- **Initial Configuration**:
  - Num Players: `4`
  - Seed: `123`
- **Preconditions**:
  - Lead suit is Club (♣). Player holds `K♣` and `7♦`.
- **Actions**:
  1. Player attempts to play `7♦`.
- **Expected Assertions**:
  - Action MUST be rejected with error `MUST_FOLLOW_SUIT`.
  - Engine state MUST remain unchanged.

---

## TEST-0009: Interrupted Trick Pickup
- **Rule IDs**: AETS-5.4.2
- **Description**: Verifies that the first suit-break ends the trick, played cards are collected by the player of the highest lead-suit card, and they lead the next trick.
- **Initial Configuration**:
  - Num Players: `4`
  - Seed: `42`
- **Preconditions**:
  - Play sequence: P0 plays `8♠` (lead), P1 plays `6♠`, P2 plays `4♠`.
  - P3 holds no Spades and plays `Q♣` (off-suit).
- **Actions**:
  1. P3 plays `Q♣`.
- **Expected Events**:
  - `CARD_PLAYED`, `TRICK_COMPLETED` (outcome: INTERRUPTED).
- **Expected Assertions**:
  - P0 (highest card in lead suit: `8♠`) collects all 4 cards (`8♠`, `6♠`, `4♠`, `Q♣`).
  - Next phase MUST transition to Trick 2 `AwaitingStealDecision` for P0.

---

## TEST-0010: Successful Trick Discard
- **Rule IDs**: AETS-5.4.1
- **Description**: Verifies that a successful trick where all follow suit discards cards and the winner leads next.
- **Initial Configuration**:
  - Num Players: `4`
  - Seed: `42`
- **Preconditions**:
  - Play sequence: P0 plays `A♠`, P1 plays `K♠`, P2 plays `Q♠`, P3 plays `J♠`.
- **Actions**:
  1. P3 plays `J♠`.
- **Expected Events**:
  - `CARD_PLAYED`, `TRICK_COMPLETED` (outcome: DISCARDED).
- **Expected Assertions**:
  - Played cards are cleared to the discard pile.
  - P0 (played highest card `A♠`) is the trick winner and leads Trick 2.
