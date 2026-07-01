# Plan: RL Agent 2.3 Advanced Deception & Exploitation (v2.3)

This plan outlines the design and integration of exploit-margin training, deception rewards, and persistent human profile memory.

---

## Phase 1: Exploit-Margin Training

### Goal
Fine-tune the model to maximize its win rate specifically against HeuristicAgentV2.

### Details
* Table composition during fine-tuning:
  * 50% vs mixed opponent league.
  * 30% vs HeuristicAgentV2 heavy tables (e.g. 1 RL, 3 Heuristic V2).
  * 20% pure self-play.
* Reward formulation: Add a reward component proportional to the survival margin over HeuristicAgentV2 players (positive if Heuristic V2 loses early, negative if Heuristic V2 outlasts the agent).

---

## Phase 2: Deception & Bluffing Rewards

### Goal
Enable deceptive maneuvers (underplaying high cards, leading to imply voidness/strength) by shaping information leakage.

### Details
* Information leakage penalty:
  * Compute the entropy of the opponent's belief model regarding the agent's cards and void suits.
  * Apply a small penalty (e.g. $-0.02$) if an action dramatically reduces opponent uncertainty about the agent's hand.
* Misdirection bonus:
  * Reward the agent (e.g. $+0.05$) when an opponent plays a suboptimal card or collection-prone lead as a result of misestimating the agent's void suits/card strengths.

---

## Phase 3: Human Profile Memory

### Goal
Maintain and query opponent habits across multiple match instances.

### Details
* Profile Database: Maintain a persistent key-value store (e.g., JSON or SQLite) mapping player identifiers to behavioral metrics (steal frequency, risk tolerance, suit preferences, underplay tendencies).
* Integration:
  * Load the opponent profiles at the start of a match.
  * Update the profiles after each trick/round.
  * Feed the loaded features directly into the agent's opponent modeling encoder slots to activate targeted "Exploit Mode" strategies.
