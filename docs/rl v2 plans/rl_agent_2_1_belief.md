# Plan: RL Agent 2.1 Advanced Cognitive (v2.1)

This plan outlines the design and integration of advanced cognitive layers, specifically encoder upgrades, a probabilistic belief tracker, opponent behavioral features, and auxiliary prediction heads.

---

## Phase 1: Encoder Upgrades

### Goal
Enrich the state representation with explicit context-heavy features, reducing raw card inference requirements and standardizing on ego-centric (relative) features.

### Details
* **Ego-Centric Seating**:
  * Encode all seat-dependent features (known cards, voids, scores) relative to the learner's seat (Player + 1, Player + 2, etc.) rather than absolute player IDs (0, 1, 2, etc.). This makes the features translation-invariant across seating arrangements.
* **Score Normalization**:
  * Normalize point scores by `2 * num_rounds` (the maximum possible points) rather than an arbitrary constant of 500, to scale features accurately across matches of varying round lengths.
* **Additional Encoder Features**:
  * `player_count`: Dynamic number of players at the table.
  * `seat_presence_mask`: Binary mask indicating which of the 6 slots are populated.
  * `active_player_mask`: Binary mask showing who is still active in the current round.
  * `hand_sizes`: Distribution of remaining cards in hand per player.
  * `current_trick_cards`: Card IDs currently played in the active trick.
  * `trick_position`: One-hot encoded position (lead, middle, last).
  * `phase_selector`: One-hot indicator representing whether the step is in the Steal Phase or the Card Play Phase.
  * `suit_counts`: Count of cards remaining per suit in the own hand.

---

## Phase 2: Belief Tracker Integration

### Goal
Provide estimated probabilistic beliefs about hidden card states.

### Details
* Probabilistic features to generate:
  * `player_card_probs[opponent][card]`: Estimated probability that an opponent holds a specific card.
  * `player_void_probs[opponent][suit]`: Probability that an opponent is void in a given suit.
  * `suit_lead_danger[suit]`: Collection risk profile for leading a specific suit.
  * `interruption_chance[suit]`: Probability that a lead of a suit will trigger an interruption.
* Calculations: Use heuristic probabilities (from CardTracker and ProbabilityEstimator) first, then transition to learned estimators.

---

## Phase 3: Opponent Profiling

### Goal
Adapt decision-making based on observed opponent play habits.

### Details
* Track behavioral metrics:
  * `opponent_steal_rate`: Steal execution frequency.
  * `opponent_risk_tolerance`: Underplaying frequency.
  * `opponent_recent_aggression`: Frequency of high rank leads.
* Input injection: Concatenate these profile features directly to the neural network state vector.

---

## Phase 4: Auxiliary Prediction Heads

### Goal
Force representation learning of hidden states via multi-task auxiliary learning.

### Details
* Model extensions: Add extra output heads sharing the main representation trunk of AceNetV2.
* Targets:
  * **Card Ownership Head**: Predicts probability of card ownership for all opponents (binary cross-entropy loss against actual hidden cards during training).
  * **Void Prediction Head**: Predicts opponent void suits per player per suit.
  * **Trick Risk Head**: Predicts expected trick outcomes.
* Optimization: Compute auxiliary losses during training and add them with small weights (e.g. 0.1) to the main PPO loss function.
