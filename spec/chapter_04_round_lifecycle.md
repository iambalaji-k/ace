# AETS Chapter 4: Round Lifecycle

## 4.1 Definition
A Round is a single complete game played using one 52-card deck.

## 4.2 Round Initialization (ROUND_INIT)
When a round starts, the engine MUST execute the following sequence:
1. **Derive Seed**: Calculate round seed from match seed: `round_seed = pcg_advance(match_seed, round_number)`.
2. **Reserved Ace Allocation**: Determine which aces to reserve (see §4.3).
3. **Deck Setup**: Construct full deck of 52 cards, then remove the reserved aces.
4. **Shuffle**: Shuffle the remaining cards using the round's seed.
5. **Give Reserved Aces**: Place reserved aces directly into the recipient's hand.
6. **Deal with Skip**: Deal remaining cards clockwise starting from player 0, skipping the reserved-ace recipient for $N$ rotations, where $N$ is the number of reserved aces.
7. **Lead Player Determination**: Identify the player holding A♠ (card `0`) as the lead player.

## 4.3 Reserved Aces Allocation Rule
The player with a non-zero `consecutive_loss_count` from the end of the previous round SHALL receive reserved aces according to the following mapping:
- `consecutive_loss_count = 1`: A♠
- `consecutive_loss_count = 2`: A♠, A♣
- `consecutive_loss_count = 3`: A♠, A♣, A♥
- `consecutive_loss_count = 4+`: A♠, A♣, A♥, A♦

At most one player can have a non-zero loss count at any time. If all players have 0 losses, no reserved aces are allocated.

## 4.4 Deal Skip Rotations
- Let $R$ be the reserved-ace recipient.
- Let $N$ be the number of reserved aces allocated to $R$.
- During the deal, when the card deal rotation clockwise reaches player $R$, if the count of skips executed for $R$ is less than $N$, the engine MUST skip player $R$, increment the skip counter, and deal the card to the next player clockwise instead.
- Once $N$ skips are executed, player $R$ SHALL be dealt cards normally.

## 4.5 Active / Inactive State Transitions
- All players SHALL start the round as Active.
- A player whose hand becomes empty **during** a trick plays MUST remain Active until the trick completes.
- At the **trick boundary** (evaluation phase), any player who has `0` cards in hand MUST transition to **Inactive**.
- Inactive players SHALL be declared **Round Winners**.
- A player who empties their hand but is forced to collect cards due to an interrupted trick (re-entry) MUST remain Active. Re-entry SHALL NOT disqualify a player from becoming a Round Winner in a subsequent trick.

## 4.6 Round End Conditions
- A round MUST end immediately when either:
  - **Exactly one** Active player remains: That player is declared the **Round Loser**. All other (Inactive) players are **Round Winners**.
  - **Zero** Active players remain: The round is declared a **Draw** (no loser, all players are Winners).

## 4.7 Loss Counter Updates
At the end of a round:
- The Round Loser's `consecutive_loss_count` MUST increment by 1.
- All Round Winners' `consecutive_loss_count` MUST reset to 0.
- In a Draw round, all players are Winners, so all loss counters MUST reset to 0.
