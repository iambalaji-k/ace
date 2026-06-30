# Plan: RL Agent 2.0 Core Foundation (v2.0) - Refined

This plan outlines the core implementation of **RL Agent 2.0 Core** to establish a stable, high-performance training foundation for the Ace card game.

---

## Phase 1: Action Encoding & Model Loading Cleanup

### Goal
Standardize the action-to-index mapping and ensure strict model loading behavior.

### Details
* **Inconsistency Resolution**:
  * Remove local mapping functions in agent/training files.
  * Import and use `engine/action_encoding.py` functions (`action_to_index`, `index_to_action`, and `legal_action_mask`) in all V2 modules.
* **Strict Model Loading**:
  * Modify `RLAgentV2` to raise an explicit exception if `checkpoint_path` is provided but cannot be loaded. Remove silent try-except fallbacks.

---

## Phase 2: Rigorous Evaluation Harness

### Goal
Replace the noisy 10-match evaluation with a statistically sound evaluation runner that uses paired seeds and robust confidence intervals.

### Details
* **Matches and Seating**:
  * Run **504 matches** (84 cycles of all 6 seating permutations).
* **Paired Seed Evaluation**:
  * Run all seating permutations of a cycle using the **same match seed** to establish a paired comparison, reducing variance.
* **Harness Metrics**:
  * Calculate the difference in survival/win rate per match between RLAgentV2 and HeuristicAgentV2.
  * Compute the mean difference and a **paired-seed confidence interval** across matches rather than treating round outcomes as independent Bernoulli trials.

---

## Phase 3: Model Architecture (AceNetV2)

### Goal
Implement a medium-capacity Residual MLP model optimized for CPU training.

### Details
* Input dimension: 459 (to support 6-player encoder V2).
* Hidden dimensions: 512.
* Activation: GELU.
* Normalization: LayerNorm.
* Shared Trunk: 4 residual blocks.
* Separate policy (actor) and value (critic) output towers.

---

## Phase 4: Imitation Pretraining

### Goal
Bootstrap model policy prior using expert demonstrations, avoiding data leakage.

### Details
* **Trajectory collection**:
  * Play HeuristicAgentV2 against itself and record state-action trajectories.
* **Data Leakage Fix**:
  * Split the dataset into train and validation sets **by match seed** (not by individual states) to prevent adjacent frames of the same match from leaking into the validation set.
* **Pretraining**:
  * Train the policy head of AceNetV2 via supervised Cross-Entropy Loss to predict HeuristicAgentV2 actions.
  * Save pretrained weights to `engine/rl_champion_v2.pt`.

---

## Phase 5: GAE & PPO Training Pipeline

### Goal
Implement Proximal Policy Optimization with Generalized Advantage Estimation, ensuring correct round boundary handling and metrics tracking.

### Details
* **GAE Round Boundary Fix**:
  * Finalize the pending learner transition immediately with `done=True` whenever a round ends (even if the round ended on an opponent's turn).
  * This prevents GAE from linking advantages across separate rounds.
* **PPO Metrics**:
  * Log and monitor PPO training metrics: approximate KL-divergence, clip fraction, explained variance, policy loss, and value loss.
  * Implement target-KL early stopping (e.g. KL > 0.015) to prevent policy collapse.
* **Training Resumption**:
  * Save the optimizer state, current epoch number, and RNG state alongside the model checkpoint to allow seamless training resumption.

---

## Phase 6: Step-Level Shaped Rewards & Transition Accumulation

### Goal
Provide clear credit assignment signals by accumulating rewards across opponent turns.

### Details
* **Transition Accumulation**:
  * A learner transition spans from one learner decision to the next decision, **accumulating all intermediate rewards** (such as collections, round wins, or steals occurring on opponent turns) in a buffer.
  * When the learner acts, assign the accumulated buffer to the *previous* transition and reset the buffer.
* **Shaped Rewards**:
  * **Round Win (Hand Empty or Stolen From)**: $+1.0$ (non-overlapping, clean round outcome utility).
  * **Round Loss (Last Active Player)**: $-1.0$.
  * **Trick Discard (Survival)**: $+0.02$.
  * **Trick Interruption (Collection)**: $-0.05 \times \text{cards collected}$.
  * **Forcing Interruption**: $+0.1$ if the learner plays off-suit, causing another player to collect.
  * **Steal Safety Penalty**: $-0.05 \times \text{cards taken}$ to penalize hand overloading.
  * **Endgame Risk mitigation**: $+0.005$ per low-power card (power rank $\le 10$) in the hand per step when avg hand size is $\le 4$.

---

## Phase 7: Opponent League & Dynamic Tables

### Goal
Train against a diverse, non-biased opponent distribution under varying table sizes and match lengths, with candidate gating.

### Details
* **Table Distribution**:
  * 55% Self-Play (Active model or past checkpoints).
  * 15% Heuristic V2.
  * 15% MCTS Agent (with iterations constrained to 20 for CPU speed).
  * 15% RL v1.0.
* **Dynamic Sizes**:
  * Randomly select players between **3 and 6** at match init.
  * Randomly select rounds between **3 and 7** at match init.
  * **Player Count Fallback**: Since RL v1.0 is hardcoded for 4 players, it must only be seated in 4-player matches. Fallback to HeuristicAgentV2 / Random if player count is not 4.
  * **MCTS 5-6 Player Fix**: Dynamically allocate MCTS rollout agents in `mcts_agent.py` up to `num_players` to prevent KeyErrors.
* **Champion Evaluation Gate**:
  * Keep the training model (`engine/rl_active_v2.pt`) separate from the validated champion (`engine/rl_champion_v2.pt`).
  * Periodically (e.g. every 20 epochs), run a 100-match seat-balanced tournament between the active candidate and the champion. Promote the candidate only if it outperforms the incumbent.
