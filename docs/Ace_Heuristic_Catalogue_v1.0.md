# Heuristic Catalogue v1.0

This catalogue defines the complete list of modular heuristics for the `HeuristicAgent`. Each entry contains its unique ID, purpose, applicable game phase, preconditions, evaluation formula, weight, conflicts, and rationale.

**Total entries: 87** (1 Core Utility + 86 Heuristics)

---

## User Strategy Cross-Reference

The following table maps each of the 19 user-defined strategies to the heuristic(s) that implement them.

| User Strategy # | Summary | Implementing Heuristic(s) |
|:---|:---|:---|
| 1 | No stealing unless left player lacks our suits | H104 |
| 2 | Only steal when not-stealing is disadvantageous | H110, H111 |
| 3 | Steal only when cards benefit future tricks | H102, H112 |
| 4 | Opening: high probability all have all suits → discard high ranks first | H202, H401 |
| 5 | Play lower ranks if next player may interrupt | H303, H305, H307 |
| 6 | Plan based on cards taken by others from interrupted tricks | Core Utility ($C_{\text{known},P}$), H310, H214 |
| 7 | Guess remaining cards and play strategically to avoid highest | H304, H305, H307, H313 |
| 8 | 13 cards per suit: compute remaining distribution | Core Utility |
| 9 | Don't lead a suit if another player is void, unless lure possible | H203, H206 |
| 10 | When interrupting: discard highest or drain suit strategically | H401, H402, H406, H411 |
| 11 | Don't play highest in middle/endgame (guessing not accurate) | H305 |
| 12 | Opening: discard highest cards in all suits | H202, H401 |
| 13 | Middle: cautious of highest card in trick | H305 |
| 14 | Endgame: more known cards → exploit information | H314, H315 |
| 15 | When interrupting: dump highest or drain suit for void creation | H401, H411, H402 |
| 16 | Dispose high ranks early to end up with low ranks late | H202, H401, H408 |
| 17 | When interrupting: give collector a suit they don't have | H403 |
| 18 | Endgame steal: valuable when victim has ≤2 cards, low ranks, not our missing suits | H105 |
| 19 | Hoard high cards of suit we dominate, but expose the Ace early | H211 |

---

## Card Counting & Probability Estimator (Core Utility)

The agent maintains a running tally of card distribution for each of the four suits $S \in \{\text{Spade}, \text{Club}, \text{Heart}, \text{Diamond}\}$:
* $C_{\text{own}}(S)$: Count of suit $S$ cards in our own hand.
* $C_{\text{discard}}(S)$: Count of suit $S$ cards publicly discarded (from completed tricks with outcome `DISCARDED`).
* $C_{\text{known}, P}(S)$: Count of suit $S$ cards known to be in player $P$'s hand (reconstructed from: reserved aces allocation, collected cards from interrupted tricks, observed steals, and cards played but collected back via interruption).
* $C_{\text{played\_then\_discarded}, P}(S)$: Tracks whether known-collected cards have since been played/discarded in later tricks (reduces $C_{\text{known},P}$).
* $U(S)$: Remaining unknown cards of suit $S$ in circulation:
  $$U(S) = 13 - C_{\text{own}}(S) - C_{\text{discard}}(S) - \sum_{P \neq \text{self}} C_{\text{known}, P}(S)$$

### Void Estimation:
* If $C_{\text{own}}(S) + C_{\text{discard}}(S) + \sum C_{\text{known}}(S) = 13$, then all cards of suit $S$ are accounted for.
* If a player $P$ has broken suit $S$ in a previous trick, player $P$ is **confirmed void** in $S$ at the time of that break.
* If $U(S)$ is very low, the probability of any given player holding suit $S$ is low → high interruption risk.

### Rank Estimation:
* Known cards have exact ranks. Unknown cards can be probabilistically estimated based on which ranks of suit $S$ have been seen (in discards, plays, and collections).
* Example: If cards 2♣, 3♣, 5♣, 7♣, 9♣, J♣, Q♣, K♣ have all been seen, then only A♣, 4♣, 6♣, 8♣, 10♣ remain in circulation.

### Game Phase Definition:
* **Opening**: Tricks 1 through $\lfloor T/3 \rfloor$ where $T$ is approximate total tricks in the round.
* **Middle**: Tricks $\lfloor T/3 \rfloor + 1$ through $\lfloor 2T/3 \rfloor$.
* **Endgame**: Tricks beyond $\lfloor 2T/3 \rfloor$, or when $\le 2$ active players remain, or average hand size $\le 4$.

---

## Part 1: Steal Phase Heuristics (H100 - H199)

### H102: Expected Suit Concentration
* **Purpose**: Prefer stealing if the victim's known/suspected hand cards align with our own hand suits to build a concentrated hand, OR to avoid getting interrupted by the immediate left player (since stealing their hand if they don't have our suits prevents them from interrupting our leads). Also, if we suspect they hold a lower rank card of our suit, stealing it lets us hoard it to lure other players to play higher and get interrupted by void players.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: Victim's known cards (from past collections/steals) are recorded.
* **Evaluation Formula**:
  * `StealAction`:
    $$\text{Score} = (\text{concentration}_{\text{merged}} - \text{concentration}_{\text{current}}) \times 50.0 + \text{suspected\_low\_ranks\_bonus} \times 40.0$$
  * where $\text{concentration} = \max_S \frac{C(S)}{|\text{hand}|}$
* **Weight**: $50.0$
* **Conflicts**: None.
* **Rationale**: Concentrated hands allow high control over trick leads, and acquiring low cards of our own suits lets us control the play order.

### H103: Steal Card Count Penalty
* **Purpose**: Discourage stealing large hands because holding more cards increases the distance to winning (emptying our hand).
* **Applicable Game Phase**: Any
* **Preconditions**: `victim_hand_size > 1`.
* **Evaluation Formula**:
  * `StealAction`: $\text{Score} = -(\text{victim\_hand\_size}) \times 40.0$
  * `DeclineStealAction`: $0.0$
* **Weight**: $40.0$ (per card)
* **Conflicts**: None.
* **Rationale**: Stealing 8 cards makes it almost impossible to win the round quickly.

### H104: Suit Intersection Steal Filter
* **Purpose**: Do not steal unless the victim does not have any of the suits we currently hold in our hand, EXCEPT when we guess/infer the victim has a lower rank card of a suit we hold, which can be stolen and used strategically in later tricks.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: Phase is `AwaitingStealDecision`.
* **Evaluation Formula**:
  * `StealAction`:
    * Victim suspected to hold cards in our suits AND no low-rank cards of our suits: $-100.0$
    * Victim void in all our suits, OR has suspected low-rank cards of our suits: $+100.0$
* **Weight**: $100.0$
* **Conflicts**: None.
* **Rationale**: Avoids diluting our suit concentration unless the victim holds low cards we can exploit.

### H105: Endgame Targeted Steal
* **Purpose**: Maximize stealing value in the endgame against low-card opponents. Valuable when the victim has ≤2 cards, low ranks, compatible suits, and not higher-ranked cards.
* **Applicable Game Phase**: Endgame
* **Preconditions**: `victim_hand_size <= 2`.
* **Evaluation Formula**:
  * `StealAction`:
    * Base: $+150.0$
    * Bonus if victim's known cards are low ranks ($R \le 6$): $+50.0$
    * Penalty if victim's known cards include suits we are void in: $-60.0$
    * Penalty if victim's known cards include high ranks ($R \ge 12$): $-40.0$
* **Weight**: $150.0$
* **Conflicts**: Overrides `H103` when victim has $\le 2$ cards.
* **Rationale**: In the endgame, taking a tiny hand of low-rank compatible cards is very strong and prevents the victim from winning.

### H107: Steal to Mitigate Bad Leads
* **Purpose**: Steal from the next player if we are leading but hold only high-risk singletons or cards that will cause us to collect if led.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are the lead player. Our hand has no safe lead options (every suit we could lead has known void opponents).
* **Evaluation Formula**:
  * `StealAction`: $+90.0$
* **Weight**: $90.0$
* **Rationale**: If leading is highly likely to result in us collecting cards, stealing resets our hand.

### H109: Decline Steal when Hand is Balanced
* **Purpose**: Decline stealing if our hand is already flat and balanced (no singletons, no voids) to avoid disrupting our structure.
* **Applicable Game Phase**: Any
* **Preconditions**: Our hand has $\ge 2$ cards in every suit we hold (no singletons or voids).
* **Evaluation Formula**:
  * `DeclineStealAction`: $+70.0$
* **Weight**: $70.0$
* **Rationale**: A balanced hand is easier to shed safely without taking risks.

### H110: Steal Disadvantage Assessment
* **Purpose**: Only steal when not-stealing would leave us at a strategic disadvantage. Evaluates the cost of declining by estimating how likely the victim's future plays will hurt us.
* **Applicable Game Phase**: Any
* **Preconditions**: Phase is `AwaitingStealDecision`.
* **Evaluation Formula**:
  * `DeclineStealAction`:
    * If victim has suits we are void in (they can play those, and we might have to collect later): $-80.0$
    * If victim has singleton (likely to go void soon and interrupt our leads): $-60.0$
  * `StealAction`:
    * Proportional to the number of threats neutralized: $+40.0 \times \text{threats\_neutralized}$
* **Weight**: $80.0$
* **Rationale**: Stealing should be a defensive move, not just an offensive one. If the victim is about to become dangerous, stealing pre-empts that.

### H111: Steal Timing Optimizer
* **Purpose**: Evaluate whether stealing now vs. later in the round produces better outcomes. In the opening, stealing is risky because hands are large. In the middle/endgame, stealing is more predictable because we know more about the victim's hand.
* **Applicable Game Phase**: Any
* **Preconditions**: Phase is `AwaitingStealDecision`.
* **Evaluation Formula**:
  * `StealAction`:
    * Opening phase: $-50.0$ (penalize early steals when info is low)
    * Middle phase: $+0.0$ (neutral)
    * Endgame phase: $+30.0$ (bonus for late steals when info is high)
* **Weight**: $50.0$
* **Rationale**: In the opening, we lack information about the victim's hand, making steals unpredictable.

### H112: Steal for Future Trick Benefit
* **Purpose**: Steal only when the cards from the victim will benefit us in the next tricks — specifically when the victim's cards give us safe leads or low follow cards.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Victim's known cards are partially reconstructed.
* **Evaluation Formula**:
  * `StealAction`:
    * For each known victim card that would be a safe lead for us: $+30.0$
    * For each known victim card that is a low rank ($R \le 5$): $+20.0$
    * For each known victim card that is an Ace or King of a suit we are void in: $-40.0$
* **Weight**: $30.0$
* **Rationale**: Stealing is worthwhile only when it improves our hand for upcoming play.

### H113: Chain Steal Evaluation
* **Purpose**: Evaluate whether stealing multiple times in sequence is beneficial. If the first steal gives us a strong hand, decline further steals. If the first steal was bad, consider a second steal if the next victim has compatible cards.
* **Applicable Game Phase**: Any
* **Preconditions**: We have already executed at least one steal this trick.
* **Evaluation Formula**:
  * If current hand (post-steal) is well-structured: `DeclineStealAction`: $+100.0$
  * If current hand is still poor: `StealAction`: $+40.0$ (reduced from base because chain steals add many cards)
* **Weight**: $100.0$
* **Conflicts**: Must check auto-loss condition — stealing all remaining players causes immediate loss.
* **Rationale**: Chain steals can be powerful but also dangerous. Must avoid auto-loss.

### H114: Auto-Loss Prevention
* **Purpose**: Never steal if doing so would leave us as the only active player (auto-loss). The engine declares the lead player the Round Loser if they steal from all active players.
* **Applicable Game Phase**: Any
* **Preconditions**: Only 2 active players remain (us and the steal target).
* **Evaluation Formula**:
  * `StealAction`: $-9999.0$ (hard block)
  * `DeclineStealAction`: $+500.0$
* **Weight**: $9999.0$ (absolute override)
* **Conflicts**: Overrides ALL other steal heuristics.
* **Rationale**: Auto-loss is the worst possible outcome. Never allow it.

### H115: Steal to Deny Void
* **Purpose**: Steal from a player who is about to become void in a critical suit. If they go void, they will interrupt our leads. Stealing prevents this by giving them new cards.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Victim has exactly 1 card of a suit that matters to us.
* **Evaluation Formula**:
  * `StealAction`: $+60.0$
* **Weight**: $60.0$
* **Rationale**: Preemptively neutralizes a future interruption threat. After being stolen from, the victim goes inactive (empty hand), so the interruption source is removed entirely.

### H117: Reserved Ace Steal Consideration - unless they have discarded the ace already.
* **Purpose**: If the steal target received reserved aces (visible from match state: consecutive_loss_count > 0 at round start), they are known to hold specific aces - unless they have discarded the ace already.
* **Applicable Game Phase**: Opening
* **Preconditions**: Victim had `consecutive_loss_count > 0` at round start.
* **Evaluation Formula**:
  * Known aces in victim's hand: If those aces are in suits we hold and they haven't discarded them: $+40.0$
  * If those aces are in suits we are void in: $-30.0$
* **Weight**: $40.0$
* **Rationale**: Reserved aces provide guaranteed knowledge about the victim's hand at round start, unless they have discarded the ace already.

---

## Part 2: Lead Phase Heuristics (H200 - H299)

### H202: High Rank Lead (Opening)
* **Purpose**: Prefer leading high rank cards in our suits at the beginning of the game, since in the opening all players are highly likely to have all suits (low interruption risk).
* **Applicable Game Phase**: Opening
* **Preconditions**: We are leading the trick.
* **Evaluation Formula**:
  * For each candidate card of rank $R$ (where $2 = 2, \ldots, \text{Ace} = 14$):
    $$\text{Score} = (R - 2) \times 12.0$$
* **Weight**: $12.0$
* **Rationale**: In the opening, players are highly likely to have all suits, so high cards can be safely discarded.

### H203: Opponent Void Avoidance
* **Purpose**: Avoid leading a suit if any subsequent active player is confirmed or strongly suspected to be void in it. Leading into a void guarantees an interruption.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are leading the trick. Void data is available.
* **Evaluation Formula**:
  * If candidate lead suit is void for any subsequent active player: $-150.0$
  * If candidate lead suit is void for the **immediate next** player: $-200.0$ (even worse — they interrupt first, and we are likely the collector)
* **Weight**: $200.0$
* **Rationale**: Leading a suit an opponent is void in directly enables a suit break/interruption.

### H204: Low Rank Lead (Middle/Endgame)
* **Purpose**: Prefer leading low rank cards in middle and endgame phases to minimize collection risk.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are leading the trick.
* **Evaluation Formula**:
  * For each card of rank $R$:
    $$\text{Score} = (14 - R) \times 10.0$$
* **Weight**: $10.0$
* **Conflicts**: Overrides `H202` during Middle and Endgame.
* **Rationale**: In later phases, the risk of suit interruption is high, so playing low minimizes the damage if we end up collecting.

### H205: Suit Depletion Risk Lead
* **Purpose**: Avoid leading a suit if card counting indicates that very few cards of that suit remain in circulation among other players.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are leading the trick.
* **Evaluation Formula**:
  * If $U(S) \le 2$ (excluding our hand): $-120.0$
  * If $U(S) = 0$ (all remaining cards of suit $S$ are in our hand or accounted for): $-250.0$
* **Weight**: $250.0$
* **Rationale**: If no other player can follow suit, leading it guarantees interruption.

### H206: Lure Lead
* **Purpose**: Lead a low card of a suit to force subsequent players to play higher ranks. If the trick gets interrupted, the player with the highest card of the lead suit collects — not us.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are leading. At least one subsequent player is suspected void in the lead suit (creating interruption risk), but at least one other player is expected to follow suit.
* **Evaluation Formula**:
  * If we lead a card with $R \le 5$: $+70.0$
  * If at least one subsequent player has a higher known card of this suit: $+40.0$ additional
* **Weight**: $70.0$
* **Rationale**: Playing low lures others to play higher ranks. If interrupted, they collect instead of us.

### H208: Void-Promoting Lead
* **Purpose**: Lead a suit in which we hold very few cards to exhaust it and create a void for future interruption capability.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: We are leading and hold 1 or 2 cards in the candidate suit.
* **Evaluation Formula**:
  * 1 card in suit: $+80.0$ (immediate void after this trick)
  * 2 cards in suit: $+40.0$
* **Weight**: $80.0$
* **Conflicts**: Bypassed if H203 penalizes the lead (void opponent).
* **Rationale**: Creating voids unlocks the ability to break suit and dump high cards.

### H209: Target Specific Void Probe
* **Purpose**: Deliberately lead a suit that a dangerous opponent is void in, knowing they will break suit and cause the highest lead-suit follower to collect. Use this to punish a specific player.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Target opponent is void in suit $S$. At least one other player is expected to follow with a high card of $S$.
* **Evaluation Formula**:
  * If we lead with a low card of $S$: $+90.0$
  * If we lead with a high card of $S$: $-50.0$ (we might collect)
* **Weight**: $90.0$
* **Rationale**: Forces the target to break suit. A follow player with a high card collects.

### H210: Safe Leader with Known Collector
* **Purpose**: Lead a suit where another player is known to hold the highest card, guaranteeing they collect if the trick is interrupted.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: The highest card of suit $S$ is known to be in player $P$'s hand ($P \ne \text{self}$).
* **Evaluation Formula**:
  * Lead suit $S$ with a low card: $+100.0$
* **Weight**: $100.0$
* **Rationale**: $P$ holds the ceiling. Even if interrupted, $P$ collects.

### H211: Suit Hoarding & Ace Exposer
* **Purpose**: If we hold $\ge 5$ cards of a suit $S$, we dominate that suit. Hoard the high cards (King, Queen) for future interruptions, but play the Ace of $S$ early.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: We hold $\ge 5$ cards of suit $S$ and hold the Ace of $S$.
* **Evaluation Formula**:
  * Lead/play Ace of $S$ early: $+115.0$
  * Hoard King/Queen of $S$: $+95.0$ (resist playing them)
* **Weight**: $115.0$
* **Rationale**: An unplayed Ace is easily tracked by observant players. Disposing it early prevents opponents from deducing we hold it. Meanwhile, the King/Queen are less trackable and serve as interruption tools later.

### H213: Short-Suit Lead (Opening)
* **Purpose**: In the opening, lead a suit where we have a singleton or doubleton to quickly create a void.
* **Applicable Game Phase**: Opening
* **Preconditions**: We are leading.
* **Evaluation Formula**:
  * Lead singleton: $+85.0$
  * Lead from doubleton: $+50.0$
* **Weight**: $85.0$
* **Rationale**: Creates early voids for high flexibility in the middle game.

### H214: Avoid Leading Opponent's Collected Suit
* **Purpose**: Avoid leading a suit if an opponent has recently collected many cards of that suit from interrupted tricks.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Opponent $P$ collected $\ge 3$ cards of suit $S$.
* **Evaluation Formula**:
  * Lead suit $S$: $-100.0$
* **Weight**: $100.0$
* **Rationale**: Avoids playing into the collector's strength — they can easily follow and outrank us.

### H215: Safe Exit Lead
* **Purpose**: Lead a card guaranteed to be lower than another player's known card of that suit, ensuring we do not collect if the trick is interrupted.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Another player has a known higher card of the lead suit.
* **Evaluation Formula**:
  * Lead exit card: $+90.0$
* **Weight**: $90.0$
* **Rationale**: Shifts the collection burden to other players.

### H216: Lead Suit with Most Cards (Opening)
* **Purpose**: In the opening, leading from our longest suit is safe because we are most likely to have multiple rounds of play in that suit.
* **Applicable Game Phase**: Opening
* **Preconditions**: We are leading.
* **Evaluation Formula**:
  * Lead from suit with $\ge 5$ cards: $+40.0$
  * Lead from suit with $\ge 7$ cards: $+60.0$
* **Weight**: $60.0$
* **Rationale**: Long suits provide sustained safe leading opportunities.

### H217: Avoid Leading Suit We Just Collected
* **Purpose**: If we just collected cards from an interrupted trick, avoid leading the suit we collected. Other players saw those cards come to us and may have already gone void in that suit.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We collected cards of suit $S$ in the last 1-2 tricks.
* **Evaluation Formula**:
  * Lead suit $S$: $-80.0$
* **Weight**: $80.0$
* **Rationale**: Opponents know our hand composition and may be void in that suit.

### H218: Lead Suit Where All Active Players Have Cards
* **Purpose**: Lead a suit where we are confident all active players can follow, guaranteeing the trick will be discarded.
* **Applicable Game Phase**: Any
* **Preconditions**: Card counting shows all active players have $\ge 1$ card of suit $S$ (confirmed or high probability).
* **Evaluation Formula**:
  * Lead suit $S$: $+130.0$
* **Weight**: $130.0$
* **Rationale**: A guaranteed discard is the safest possible trick outcome for us.

### H219: Ace Lead When All Can Follow
* **Purpose**: If we hold an Ace of a suit and we are confident all active players can follow that suit, leading the Ace is a safe way to dispose of it.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: All active players are confirmed/suspected to have suit $S$. We hold Ace of $S$.
* **Evaluation Formula**:
  * Lead Ace of $S$: $+140.0$
* **Weight**: $140.0$
* **Rationale**: The Ace is the highest rank. If the trick completes without interruption, it gets discarded. Leading it when safe removes our riskiest card.

### H220: Lead to Force Opponent Collection
* **Purpose**: Lead a suit where we hold only low cards and a specific opponent is known to hold high cards of that suit. If interrupted, they collect.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Opponent $P$ has known high cards ($R \ge 10$) of suit $S$. We hold only low cards ($R \le 5$) of $S$.
* **Evaluation Formula**:
  * Lead low card of $S$: $+85.0$
* **Weight**: $85.0$
* **Rationale**: We are safe because the opponent's higher card absorbs the collection.

### H221: Avoid Leading When Only 2 Active Players Remain
* **Purpose**: When only 2 players remain, every trick either discards (both follow) or one interrupts. If we lead a suit the opponent is void in, they will break suit and we collect since we played the only card of the lead suit (the highest by default). Be extremely cautious.
* **Applicable Game Phase**: Endgame
* **Preconditions**: Only 2 active players remain.
* **Evaluation Formula**:
  * Lead suit $S$ where opponent might be void: $-200.0$
  * Lead suit $S$ where opponent definitely has cards: $+50.0$
* **Weight**: $200.0$
* **Rationale**: In 2-player situations, leading a void suit is catastrophic — we are guaranteed to collect.

### H222: Lead Most Common Remaining Suit
* **Purpose**: Lead a suit that has the most cards still in circulation (among all players), maximizing the probability that everyone can follow.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are leading.
* **Evaluation Formula**:
  * Lead suit $S$ with highest $U(S)$: $+50.0$
* **Weight**: $50.0$
* **Rationale**: High circulation suits are safest to lead.

### H223: Post-Collection Safe Lead
* **Purpose**: After collecting cards from an interrupted trick (our hand just grew), select the safest lead by choosing the suit for which non interruption is highly likely.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We just collected cards and must now lead.
* **Evaluation Formula**:
  * Lead lowest card of suit $S$ that maximizes the expected follow probability of all other active players: $+150.0$
  * If any active player is confirmed void in suit $S$: $-200.0$
* **Weight**: $150.0$
* **Rationale**: After collecting cards, our hand is bloated. We must lead a suit that all active players can follow, ensuring the trick is discarded and we do not collect any more cards.

---

## Part 3: Follow Suit Heuristics (H300 - H399)

### H301: Safe Discard Dump (Trick Will Be Discarded)
* **Purpose**: Dump highest rank cards when we are the last player in the trick and no one has broken suit, guaranteeing the trick will be discarded.
* **Applicable Game Phase**: Any
* **Preconditions**: We are the last active player to play in this trick. No prior player has broken suit.
* **Evaluation Formula**:
  * For card rank $R$: $\text{Score} = R \times 15.0$
* **Weight**: $15.0$
* **Rationale**: Guaranteed safe discard opportunity. Dump our most dangerous cards.

### H303: Suspected Interruption Underplay
* **Purpose**: Play lower ranks than the current highest played card if a subsequent player is suspected to be void in the lead suit (likely to interrupt).
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are following suit. A subsequent player is suspected void.
* **Evaluation Formula**:
  * Play card higher than current highest played lead-suit card: $-250.0$
* **Weight**: $250.0$
* **Rationale**: If a subsequent player interrupts, the highest lead-suit card holder collects.

### H304: Interruption Risk Underplay
* **Purpose**: Play lower ranks when card counting reveals the lead suit is nearly depleted.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are following suit. $U(\text{lead suit})$ among remaining players $\le 2$.
* **Evaluation Formula**:
  * Play card $R \ge 10$: $-200.0$
* **Weight**: $200.0$
* **Rationale**: Depleted suits have high interruption probability.

### H305: Middle/Endgame Caution (Avoid Playing Highest)
* **Purpose**: General defensive rule — avoid playing the highest card in the trick during middle/endgame when guessing opponents' cards is uncertain.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are following suit. Trick is not guaranteed to be discarded.
* **Evaluation Formula**:
  * Play card that would be the highest rank of lead suit in the trick: $-100.0$
* **Weight**: $100.0$
* **Rationale**: Conservative play when information is imperfect.

### H306: Spade Follow Conservation
* **Purpose**: Play the lowest card possible. Preserve high Spades for future interruptions.
* **Applicable Game Phase**: Middle game
* **Preconditions**: Must follow suit.
* **Evaluation Formula**:
  * For each Spade card of rank $R$: $\text{Score} = (14 - R) \times 15.0$ (lower rank = higher score)
* **Weight**: $15.0$
* **Rationale**: High Spades are too valuable to waste on following.

### H307: Sequential Underplay
* **Purpose**: Follow suit by playing a card immediately lower than the highest played card, discarding as high as possible while remaining safe.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are following suit.
* **Evaluation Formula**:
  * Play card rank $R_{\text{play}} < R_{\text{highest\_played}}$ with minimal gap: $+80.0$
* **Weight**: $80.0$
* **Rationale**: Maximizes how much we shed while staying under the ceiling.

### H308: Hand Balancing Follow
* **Purpose**: When following suit and choosing among multiple playable cards, prefer playing from the suit with the most cards to keep our hand balanced, UNLESS playing from a near-empty suit creates a void sooner.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: Multiple playable cards of lead suit.
* **Evaluation Formula**:
  * If playing this specific card leaves us with 0 cards of a near-empty secondary suit (via indirect rebalancing): $+90.0$
  * If playing the highest available card keeps more suits balanced: $+50.0$
* **Weight**: $50.0$
* **Conflicts**: Overridden if void creation is possible.
* **Rationale**: Keeps hand flat but prioritizes void creation if exhaustion is imminent.

### H310: Follow with Known Cards First
* **Purpose**: Play cards that opponents already know we hold (from past collections or steals) to hide our unknown/private cards.
* **Applicable Game Phase**: Any
* **Preconditions**: We hold cards of the lead suit that are "known" (opponents saw us collect them).
* **Evaluation Formula**:
  * Play known card: $+70.0$
* **Weight**: $70.0$
* **Rationale**: Information concealment — keeps our private hand hidden.

### H311: Underplay Subsequent Player's High Card
* **Purpose**: If a subsequent player is known to hold the Ace/highest available card of the lead suit, play our King/Queen safely because they must follow with an even higher card.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Subsequent player has a known card higher than ours in the lead suit.
* **Evaluation Formula**:
  * Play high card (King/Queen) below the known highest: $+80.0$
* **Weight**: $80.0$
* **Rationale**: We can shed high cards safely when we know someone above us in rank must follow.

### H313: Rank Estimation Follow
* **Purpose**: Use card counting to estimate the ranks remaining in the lead suit among subsequent players. Play just below the estimated highest remaining rank to minimize collection risk.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are following suit. We have estimated remaining ranks in the lead suit.
* **Evaluation Formula**:
  * Let $R_{\text{est\_max}}$ = estimated highest remaining rank of lead suit among subsequent players.
  * Play card with $R < R_{\text{est\_max}}$: $+60.0$
  * Play card with $R > R_{\text{est\_max}}$: $-120.0$
* **Weight**: $120.0$
* **Rationale**: Uses information from the Core Utility to make informed follow decisions.

### H314: Endgame Known-Card Exploitation
* **Purpose**: In the endgame, more cards are known (from discards, collections, plays). Use this information to play aggressively when we know for certain that no subsequent player can beat our card.
* **Applicable Game Phase**: Endgame
* **Preconditions**: We are following suit. All remaining cards of lead suit are known.
* **Evaluation Formula**:
  * If we hold the highest remaining card of lead suit AND no subsequent player is void (trick will be discarded): play it for $+100.0$ (dump the high card safely).
  * If subsequent player IS void: play our lowest card: $+50.0$.
* **Weight**: $100.0$
* **Rationale**: Perfect information allows aggressive play when safe.

### H315: Endgame Perfect Information Follow
* **Purpose**: In the endgame, when we know every remaining card's location, compute the optimal play deterministically.
* **Applicable Game Phase**: Endgame
* **Preconditions**: All cards in play are accounted for (no unknowns: $U(S) = 0$ for all $S$).
* **Evaluation Formula**:
  * Compute the exact outcome for each playable card. Choose the one that minimizes our collection probability.
  * Score: $+200.0$ for the optimal card, $-200.0$ for all others.
* **Weight**: $200.0$
* **Rationale**: Perfect information should produce perfect play.

### H316: Follow High When Trick is Safe
* **Purpose**: If we are following suit and can determine that the trick will definitely be discarded (all remaining players have the lead suit and no one will break), dump our highest card of that suit.
* **Applicable Game Phase**: Any
* **Preconditions**: We are following suit. All subsequent active players have confirmed cards of lead suit $S$.
* **Evaluation Formula**:
  * Play highest card of $S$: $+120.0$
* **Weight**: $120.0$
* **Rationale**: Safe dump opportunity — maximize what we shed.

### H317: Position-Aware Follow (Early vs. Late in Trick)
* **Purpose**: If we play early in the trick order (2nd player out of 4+), we are at higher risk because many players follow after us. Play conservatively. If we play late (last or second-to-last), we have more information.
* **Applicable Game Phase**: Any
* **Preconditions**: We are following suit.
* **Evaluation Formula**:
  * If we are 2nd player (right after lead): additional caution $-30.0$ on high cards.
  * If we are last player: additional freedom $+30.0$ on highest card (we know no one else can outrank us in the lead suit after us).
* **Weight**: $30.0$
* **Rationale**: Positional awareness.

---

## Part 4: Break Suit Heuristics (H400 - H499)

### H401: High Rank Off-Suit Dump
* **Purpose**: When breaking suit, dump our highest card to rid ourselves of dangerous high ranks.
* **Applicable Game Phase**: Any
* **Preconditions**: We cannot follow suit.
* **Evaluation Formula**:
  * For off-suit card of rank $R$ (excluding Spades): $\text{Score} = R \times 20.0$
* **Weight**: $20.0$
* **Rationale**: Gets rid of dangerous high cards.

### H402: Preserve Future Voids
* **Purpose**: Bonus for playing the last card of a suit, creating a new void.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Playing this card leaves us with 0 cards of its suit.
* **Evaluation Formula**:
  * Play last card of suit in hand: $+120.0$
* **Weight**: $120.0$
* **Rationale**: Voids allow future suit breaks.

### H403: Collector Void Disruption
* **Purpose**: When interrupting/breaking suit, discard a card of a suit that the current trick collector is void in, forcing them to hold a new suit and destroying their void.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are breaking suit. The collector of this trick is determined (they played the highest lead-suit card).
* **Evaluation Formula**:
  * Discard card of suit $S$ where collector is known void: $+100.0$
* **Weight**: $100.0$
* **Rationale**: Disrupts the collector's hand structure.

### H405: First-Break Suit Selection
* **Purpose**: When we are the first player to break suit (we are void in the lead suit), choose which off-suit card to play carefully. Per §5.4.2, the trick interrupts **immediately** when we break suit — no subsequent players play. The collector is the player who played the highest card of the **lead suit** (not us, since we played off-suit). Our card still goes to the collector.
* **Applicable Game Phase**: Any
* **Preconditions**: We cannot follow suit and are about to cause the trick's interruption.
* **Evaluation Formula**:
  * Play highest card (dump high rank on the collector): $+80.0$ (synergy with H401)
  * Play card of a suit the collector is void in (disruption, synergy with H403): $+100.0$
  * Play card from a near-empty suit (void creation, synergy with H402): $+90.0$
  * Avoid playing a Spade unless only Spades remain: $-60.0$
* **Weight**: $100.0$
* **Rationale**: Since we are causing the interruption, we get to choose what card the collector absorbs. This is a powerful offensive opportunity. The collector is always the highest lead-suit card player, never us (we played off-suit).

### H406: Specific Suit Draining Discard
* **Purpose**: Discard a card from a suit we hold very few cards of, accelerating void creation.
* **Applicable Game Phase**: Opening, Middle
* **Preconditions**: We are breaking suit.
* **Evaluation Formula**:
  * Discard card from suit where we hold exactly 2 cards: $+70.0$
  * Discard card from suit where we hold exactly 1 card: $+100.0$ (creates void immediately)
* **Weight**: $100.0$
* **Rationale**: Accelerates future void creation.

### H408: King/Ace Sacrifice Discard
* **Purpose**: When breaking suit, dump high value cards of any suit to eliminate our highest-risk cards.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are breaking suit. The collector is another player.
* **Evaluation Formula**:
  * Play highest value card of any suit: $+110.0$
* **Weight**: $110.0$
* **Rationale**: Overloads the collector with dangerous high ranks, while cleaning our hand.

### H410: Discard to Create Multi-Suit Voids
* **Purpose**: Discard a card that brings us closer to having voids in multiple suits simultaneously.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are breaking suit. We hold cards in multiple near-empty suits.
* **Evaluation Formula**:
  * Discard card that creates a void: $+90.0$
  * Discard card that leaves a suit at exactly 1 (near-void): $+50.0$
* **Weight**: $90.0$
* **Rationale**: Multiple voids maximize interruption flexibility.

### H411: Strategic Low-Rank Drain for Void Acceleration
* **Purpose**: When interrupting a trick, prefer discarding a low-rank card from a suit we hold very few of, even though it's not the highest, because discarding it accelerates void creation in that suit faster than dumping a high card from a long suit.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are breaking suit. We hold $\le 2$ cards of some suit $S$.
* **Evaluation Formula**:
  * Discard low card from suit with $\le 2$ remaining: $+75.0$
  * Override H401's preference for high ranks if void acceleration is more valuable.
* **Weight**: $75.0$
* **Rationale**: Sometimes a 4♣ (when we only have 4♣ and 7♣ in clubs) is more valuable to discard than a K♦ (when we have 5 diamonds), because voiding clubs gives us future interruption power.

### H412: Break with Collector's Missing Suit
* **Purpose**: When we break suit and the collector has already been determined (the player who played the highest lead-suit card), discard a card of a suit the collector has previously broken (i.e., a suit they were void in). This gives them a card in a suit they lack, destroying their void advantage.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: We are breaking suit. Collector is determined. Collector previously broke suit $S$ (so they were void in $S$).
* **Evaluation Formula**:
  * Discard card of suit $S$ (collector's known void): $+100.0$
  * If the card is also a high rank: $+130.0$ (double benefit)
* **Weight**: $130.0$
* **Rationale**: The collector must absorb ALL trick cards including our break card. Giving them a card of a suit they don't have destroys their void and weakens their future play.

### H413: Avoid Breaking with Suit We Want to Lead
* **Purpose**: Don't discard a card from a suit we plan to lead in future tricks. Keep our long-suit intact for controlled play.
* **Applicable Game Phase**: Any
* **Preconditions**: We are breaking suit.
* **Evaluation Formula**:
  * Discard card from our longest suit: $-40.0$
* **Weight**: $40.0$
* **Rationale**: Preserves our strongest suit for future leads.

---

## Part 5: Positional & Situational Heuristics (H500 - H599)

### H501: Play Order Position Awareness
* **Purpose**: Adjust aggression based on our position in the trick's play order.
* **Applicable Game Phase**: Any
* **Preconditions**: We are playing (not leading).
* **Evaluation Formula**:
  * If we are 2nd in play order (right after leader): Additional caution $-20.0$ on high cards.
  * If we are 2nd-to-last: Neutral.
  * If we are last: Can play aggressively (dump high cards if no break has occurred): $+40.0$.
* **Weight**: $40.0$
* **Rationale**: Later positions have more information about the trick state.

### H502: Active Player Count Adjustment
* **Purpose**: Adjust strategy based on how many active players remain.
* **Applicable Game Phase**: Any
* **Preconditions**: Active player count known.
* **Evaluation Formula**:
  * 4+ active players: Standard play, no adjustment.
  * 3 active players: Increase caution on all follow/lead scores by $-15.0$ for high cards (fewer players → more likely someone is void).
  * 2 active players: Maximum caution. Activate H221.
* **Weight**: $15.0$
* **Rationale**: Fewer players means higher void probability per player.

### H503: Hand Size Relative Assessment
* **Purpose**: Adjust strategy based on our hand size relative to opponents. If we have many more cards than average, play conservatively (we need to shed more cards to win). If we have fewer cards, play aggressively.
* **Applicable Game Phase**: Middle, Endgame
* **Preconditions**: Hand sizes of all active players are visible (though cards are masked, the count is visible).
* **Evaluation Formula**:
  * If our hand size $> 1.5 \times$ average opponent hand size: Conservative modifier $-20.0$ on aggressive plays.
  * If our hand size $< 0.7 \times$ average opponent hand size: Aggressive modifier $+30.0$ on safe dump plays.
* **Weight**: $30.0$
* **Rationale**: Hand-size relative awareness informs tempo.

### H504: Win Proximity Bonus
* **Purpose**: When we have very few cards left ($\le 3$), every play must maximize the chance of emptying our hand. Avoid actions that risk collecting cards.
* **Applicable Game Phase**: Endgame
* **Preconditions**: Our hand size $\le 3$.
* **Evaluation Formula**:
  * Any play that might cause us to collect: $-150.0$ additional penalty.
  * Any play that safely discards: $+50.0$ additional bonus.
* **Weight**: $150.0$
* **Rationale**: Protecting a near-win position.

### H505: Post-Steal Hand Re-evaluation
* **Purpose**: After stealing, immediately re-evaluate our hand structure and adjust the internal phase assessment (opening/middle/endgame) since our hand has changed dramatically.
* **Applicable Game Phase**: Any
* **Preconditions**: We just executed a steal.
* **Evaluation Formula**: No direct score. Meta-heuristic that triggers re-computation of all suit counts, void states, and phase classification.
* **Rationale**: Stealing radically changes our hand; stale evaluations are dangerous.

### H506: Reserved Ace Holder Awareness
* **Purpose**: The player who holds reserved aces (from consecutive losses) always leads the first trick (they have A♠). This is public knowledge. Track which player received reserved aces and adjust our strategy.
* **Applicable Game Phase**: Opening
* **Preconditions**: A player has `consecutive_loss_count > 0` at round start.
* **Evaluation Formula**:
  * If reserved-ace holder leads: expect the trick to be safe. Dump our highest card: $+60.0$.
  * If reserved-ace holder has 2+ aces: they have A♠ and A♣. Expect them to lead these suits. Prepare to follow.
* **Weight**: $60.0$
* **Rationale**: Reserved aces provide public information about the first leader's hand.

### H507: Consecutive Loss Prevention
* **Purpose**: If WE have a non-zero `consecutive_loss_count`, we received reserved aces and MUST NOT lose this round (consecutive losses escalate). Play more conservatively.
* **Applicable Game Phase**: Any
* **Preconditions**: Our `consecutive_loss_count > 0`.
* **Evaluation Formula**:
  * Global modifier: multiply all "risk of collecting" penalties by $1.5$.
  * Additional steal avoidance: $-50.0$ on steal actions (stealing adds cards, making our already-weak position worse).
* **Weight**: $50.0$
* **Rationale**: Escalating losses mean receiving more reserved aces in the next round, making it even harder to win.

### H508: Round Draw Awareness
* **Purpose**: If multiple players are close to emptying their hands simultaneously, a draw is possible (all go inactive at the same trick boundary → no loser). In some cases, facilitating a draw is better than risking being the last player.
* **Applicable Game Phase**: Endgame
* **Preconditions**: Multiple opponents have $\le 2$ cards.
* **Evaluation Formula**:
  * If a draw is achievable and we would otherwise be at risk of losing: play to facilitate simultaneous emptying: $+40.0$ for safe, symmetrical plays.
* **Weight**: $40.0$
* **Rationale**: A draw gives everyone 1 half-point (including us), which is better than a loss (0 points).

### H510: Trick Count Phase Estimator
* **Purpose**: Dynamically estimate the current game phase (Opening/Middle/Endgame) based on trick count, average hand size, and number of active players.
* **Applicable Game Phase**: Any (meta-heuristic)
* **Evaluation Formula**:
  * $\text{Opening}$: trick number $\le \lfloor \text{total\_tricks\_estimate} / 3 \rfloor$ AND average hand size $\ge 8$.
  * $\text{Endgame}$: $\le 3$ active players OR average hand size $\le 4$ OR $\ge 70\%$ of total tricks completed.
  * $\text{Middle}$: everything else.
* **Rationale**: Accurate phase classification drives all phase-conditional heuristics.

---

## Part 6: Opponent Modeling Heuristics (H600 - H699)

### H601: Void Tracking (Confirmed)
* **Purpose**: Record confirmed voids: if a player $P$ broke suit $S$ in a previous trick, $P$ is confirmed void in $S$ (at that moment). Track whether $P$ has since acquired cards of $S$ (via collection or steal).
* **Applicable Game Phase**: Any
* **Evaluation Formula**: No direct score. Feeds data into H203, H303, H304, and other heuristics.
* **Rationale**: Confirmed voids are the most reliable data for interruption prediction.

### H602: Void Tracking (Probabilistic)
* **Purpose**: Estimate void probabilities when voids are not confirmed. If $U(S) = 3$ and there are 3 opponents, each has roughly a $\frac{U(S)}{U(S) + \text{known}}$ chance of holding suit $S$.
* **Applicable Game Phase**: Middle, Endgame
* **Evaluation Formula**: No direct score. Provides probability estimates to H303, H205, etc.
* **Rationale**: When confirmed voids are unavailable, probabilistic estimation is the next best thing.

### H603: Opponent Play Pattern Tracking
* **Purpose**: Track what each opponent tends to play. If an opponent consistently plays low when following suit, they are likely hoarding high cards. If they always dump high, they are trying to shed.
* **Applicable Game Phase**: Middle, Endgame
* **Evaluation Formula**: No direct score. Adjusts rank estimation probabilities.
* **Rationale**: Behavioral patterns reveal hidden information.

### H604: Opponent Hand Composition Inference
* **Purpose**: Based on what an opponent has played, estimate what they still hold. If Player 2 has played 3 Hearts but never any Clubs over 5 tricks, they likely have many Clubs.
* **Applicable Game Phase**: Middle, Endgame
* **Evaluation Formula**: No direct score. Feeds into suit-based heuristics.
* **Rationale**: Negative inference (what they haven't played) is powerful.

### H605: Stolen Hand Knowledge
* **Purpose**: When a player steals from another, both the stealer and victim's cards may be partially known. Track what was stolen (if we were the stealer/victim) or note that the stealer now holds the victim's cards (if we were an observer, the specific cards are hidden per §7).
* **Applicable Game Phase**: Any
* **Evaluation Formula**: No direct score. Updates $C_{\text{known}, P}$.
* **Rationale**: Steals provide a burst of information about hand composition.

### H606: Collected Card Tracking
* **Purpose**: When a trick is interrupted and a player collects the cards, record the exact cards they collected (all trick plays are visible in `trick_history`). Track whether those cards have been played in subsequent tricks.
* **Applicable Game Phase**: Any
* **Evaluation Formula**: No direct score. Updates $C_{\text{known}, P}$.
* **Rationale**: Collected cards are the primary source of known-hand information.

### H607: Discard Pile Reconstruction
* **Purpose**: Although the discard pile is hidden from players (per `get_player_view`), we can reconstruct it by aggregating all plays from completed tricks with outcome `DISCARDED`.
* **Applicable Game Phase**: Any
* **Evaluation Formula**: No direct score. Updates $C_{\text{discard}}$.
* **Rationale**: Knowing which cards are permanently out of play is essential for card counting.

### H608: Card Elimination Matrix
* **Purpose**: Maintain a 4×13 matrix tracking the status of every card: `IN_OUR_HAND`, `DISCARDED`, `KNOWN_IN_PLAYER_P`, `UNKNOWN`.
* **Applicable Game Phase**: Any
* **Evaluation Formula**: No direct score. Core data structure for all card counting.
* **Rationale**: The foundation of all inference.

---

## Part 7: Match-Level & Multi-Round Heuristics (H700 - H799)

### H701: Match Standing Awareness
* **Purpose**: Adjust risk tolerance based on our match standing. If we are leading the match, play conservatively to protect our lead. If we are trailing, take more risks.
* **Applicable Game Phase**: Any
* **Preconditions**: Match state (`half_points`, `rounds_won`, `rounds_lost`) is available.
* **Evaluation Formula**:
  * If we are match leader: multiply collection-risk penalties by $1.3$ (more cautious).
  * If we are trailing by $\ge 2$ points: multiply collection-risk penalties by $0.7$ (more aggressive).
* **Weight**: Modifier, not standalone.
* **Rationale**: Match context should influence round strategy.

### H702: Final Round Pressure
* **Purpose**: In the last round of the match, adjust strategy to maximize points. If we need a win to overtake the leader, play aggressively. If a draw is sufficient, play for a draw.
* **Applicable Game Phase**: Any (last round)
* **Preconditions**: `current_round == num_rounds`.
* **Evaluation Formula**:
  * If win is needed: aggressive modifier $+20.0$ on risky-but-high-reward plays.
  * If draw is sufficient: conservative modifier $+30.0$ on safe plays.
* **Weight**: $30.0$
* **Rationale**: Last-round stakes are highest.

### H703: Target the Trailing Player
* **Purpose**: If another player has consecutive losses, they received reserved aces (public knowledge). They are in a weak position. Do not help them by giving them easy tricks.
* **Applicable Game Phase**: Any
* **Preconditions**: A player has `consecutive_loss_count > 0`.
* **Evaluation Formula**:
  * Plays that would help the trailing player shed cards safely: $-30.0$
  * Plays that force the trailing player to collect: $+30.0$
* **Weight**: $30.0$
* **Rationale**: Keeping the trailing player down maintains our competitive position.

### H704: Protect Against Consecutive Loss
* **Purpose**: If we have `consecutive_loss_count > 0`, we must avoid losing again at all costs. Losing again means receiving even more reserved aces next round, creating a death spiral.
* **Applicable Game Phase**: Any
* **Preconditions**: Our `consecutive_loss_count > 0`.
* **Evaluation Formula**:
  * Global modifier: all collection-risk penalties increased by $50\%$.
  * Steal avoidance: additional $-50.0$ on steals.
* **Weight**: Modifier.
* **Rationale**: Prevents the escalating loss spiral.

### H705: Opponent Near-Victory Threat
* **Purpose**: Track opponents who have won the most rounds. If an opponent is near the match victory threshold, prioritize plays that force them to collect cards (slowing their progress) over plays that are merely safe for us.
* **Applicable Game Phase**: Any
* **Preconditions**: An opponent has $\text{rounds\_won} \ge \lfloor \text{num\_rounds} \times 0.6 \rfloor$.
* **Evaluation Formula**:
  * Plays that increase the threat opponent's hand size: $+25.0$
  * Plays that let the threat opponent shed cards easily: $-25.0$
* **Weight**: $25.0$
* **Rationale**: Prevents an opponent from running away with the match.

### H706: Early Round Steal Exploitation
* **Purpose**: In the very first trick of a round, the lead player decides whether to steal. If we are the lead and our hand is weak (many high cards, no voids), consider stealing to restructure. But if our hand is strong (many voids, low cards), decline.
* **Applicable Game Phase**: Opening (Trick 1 specifically)
* **Preconditions**: Trick number = 1. We are the lead player.
* **Evaluation Formula**:
  * If hand has $\ge 3$ singletons or $\ge 1$ void: `DeclineStealAction`: $+80.0$
  * If hand is flat with only high cards: `StealAction`: $+50.0$
* **Weight**: $80.0$
* **Rationale**: The first trick is the only time we know the least about opponents. Make the steal decision based purely on our own hand quality.

### H707: Loss Spiral Recovery Play
* **Purpose**: If we have `consecutive_loss_count >= 2`, we received multiple reserved aces. These aces are known to opponents. Play them immediately (lead them) in the opening when all players can follow, to shed the tracked cards.
* **Applicable Game Phase**: Opening
* **Preconditions**: We hold reserved aces. Our `consecutive_loss_count >= 2`.
* **Evaluation Formula**:
  * Lead a reserved ace when all players likely have that suit: $+130.0$
* **Weight**: $130.0$
* **Rationale**: Reserved aces are publicly known. Holding them is dangerous because opponents will track them and plan around them.

### H708: Two-Player Endgame Perfect Play
* **Purpose**: When only 2 active players remain, compute optimal play by exhaustive analysis. With 2 players, every trick is either a clean discard (both follow) or an immediate interruption (one breaks). This is a small game tree.
* **Applicable Game Phase**: Endgame (2 players)
* **Preconditions**: Exactly 2 active players remain.
* **Evaluation Formula**:
  * Compute all possible trick outcomes for each card. Select the card that minimizes our expected hand growth.
  * Optimal card: $+250.0$
  * All others: $-250.0$
* **Weight**: $250.0$
* **Rationale**: With only 2 players, the game is simple enough for exhaustive analysis.

---

## Part 8: Meta-Heuristics & Conflict Resolution (H900 - H999)

### H901: Heuristic Priority Chain
* **Purpose**: When multiple heuristics produce conflicting scores, apply a priority ordering:
  1. **Hard blocks** (H114 Auto-Loss Prevention): Always respected.
  2. **Critical threats** (H504 Win Proximity): Override medium-weight heuristics.
  3. **Strategic**: Score-additive.
  4. **Positional/meta**: Modifier weights.
* **Rationale**: Ensures that safety-critical heuristics cannot be overridden by tactical preferences.

### H902: Score Normalization
* **Purpose**: Normalize the final aggregated score across all heuristics to prevent runaway scores from dominating. Use a softmax or clamped-sum approach.
* **Evaluation Formula**:
  * $\text{FinalScore}(a) = \sum_{h \in \text{active\_heuristics}} w_h \times s_h(a)$
  * Clamp individual heuristic scores to $[-500, +500]$ before summing.
* **Rationale**: Prevents a single heuristic from dominating the decision.

### H903: Tie-Breaking Rule
* **Purpose**: When two or more actions have identical final scores, break ties by:
  1. Prefer the action that minimizes hand size (for plays).
  2. Prefer the action that maintains the most suit diversity.
  3. Random (using agent's PRNG).
* **Rationale**: Deterministic tie-breaking prevents random behavior in critical situations.

### H904: Confidence-Weighted Scoring
* **Purpose**: Weight each heuristic's score by the confidence of its underlying information.
* **Evaluation Formula**:
  * Heuristics based on confirmed data (confirmed voids, known cards): confidence = $1.0$.
  * Heuristics based on probabilistic estimation: confidence = $0.5$ to $0.8$.
  * Heuristics based on pure pattern guessing: confidence = $0.3$.
  * $\text{AdjustedScore} = \text{Score} \times \text{confidence}$
* **Rationale**: Prevents overconfident decisions based on uncertain information.
