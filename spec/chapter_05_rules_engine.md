# AETS Chapter 5: Rules Engine

## 5.1 Definition of a Turn
A **Turn** is the interval during a Trick in which exactly one Active Player is expected to make one decision.
- In `STEAL_PHASE`, the lead player makes a steal decision (Steal or Decline).
- In `PLAY_PHASE`, the active player whose turn it is plays a card.

## 5.2 Steal Phase (Special Rule 1)
Before any cards are played in a trick, the Lead Player MAY execute a Steal:
1. **Target**: The steal target MUST be the **immediate active left** player (the next Active Player clockwise).
2. **Steal Action**: A steal MUST transfer ALL cards from the victim's hand to the stealer's hand. Stolen cards MUST merge into the stealer's hand in canonical sort order.
3. **Victim Invalidation**: Because the victim's hand is now empty, they MUST immediately become Inactive (Round Winner) before the trick play starts.
4. **Repeated Steals**: The lead player MAY steal repeatedly. After a steal completes, the target is recalculated clockwise from the stealer (skipping newly Inactive players).
5. **Auto-Loss**: If the lead player steals from all other active players, the lead player is immediately declared the **Round Loser** and the round ends without entering the `PLAY_PHASE`.
6. **Decline**: When the lead player declines to steal, the trick transitions to `PLAY_PHASE`.

## 5.3 Play Phase & Suit Following
- The Lead Player plays the first card, establishing the **Lead Suit**.
- Clockwise, each remaining active player MUST play a card:
  - If a player holds one or more cards of the Lead Suit, they **MUST follow suit** by playing a card of that suit.
  - Playing off-suit when a player is capable of following suit is an **illegal move** and MUST be rejected by the engine.
  - If a player holds no cards of the Lead Suit, they **MAY play any card** from their hand. This is a **suit break**.

## 5.4 Trick Resolution

### 5.4.1 Successful Trick (All Follow Suit)
If all active players successfully followed suit:
1. All played cards MUST be moved to a face-down discard pile.
2. The player who played the **highest rank card** in the Lead Suit is the trick winner.
3. The trick winner SHALL lead the next trick.

### 5.4.2 Interrupted Trick (Suit Break)
The trick is **interrupted immediately** upon the first suit break:
1. The player who broke suit plays their off-suit card.
2. All remaining players in the trick rotation do NOT play.
3. **Collector Identification**: The player who played the **highest rank card** in the Lead Suit is the collector.
4. **Collection**: The collector collects ALL cards played in the trick (including the off-suit card and lead-suit cards played before the break) and merges them into their hand in canonical sort order.
5. The collector SHALL lead the next trick.
