# Plan: RL Agent 2.2 Advanced Planning (v2.2)

This plan outlines the design and integration of search-guided planning, belief sampling, endgame brute-force solvers, and policy distillation.

---

## Phase 1: Belief Sampling (Hidden State Search)

### Goal
Lookahead search over plausible hidden opponent card layouts.

### Details
* Deal sampling:
  * Sample $M$ (e.g. 50–100) possible opponent hands consistent with observed voids, known cards, and discards.
* Evaluation:
  * For each legal action:
    * Simulate play over all $M$ worlds.
    * Use the pretrained policy/value network as a rollout or leaf-node prior to evaluate intermediate states.
    * Choose the action with the highest average expected value (EV) across all worlds.

---

## Phase 2: Endgame Solver

### Goal
Brute-force resolve the game tree in late-round positions.

### Details
* Trigger: Activates when the total number of cards remaining in all active player hands is $\le 12$.
* Execution: Run a minimax search of the remaining tricks to find the exact optimal sequence of plays.
* Benefit: Prevents late-round miscalculations and guarantees optimal play when cards are scarce.

---

## Phase 3: Teacher-Student Distillation

### Goal
Incorporate search improvements back into the fast neural policy for searchless execution.

### Details
* Teacher: Search-guided agent (RL policy + belief sampling + endgame solver).
* Student: Standard AceNetV2 model.
* Distillation:
  * Run matches using the Teacher to select actions.
  * Record states and the final Teacher action distributions.
  * Train the Student policy head using Kullback-Leibler (KL) divergence to match the Teacher's action probabilities.
