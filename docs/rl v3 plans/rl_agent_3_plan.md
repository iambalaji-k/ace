# Master Plan Index: RL Agent 2.0

To keep implementation organized and maintainable, the development roadmap has been divided into four separate detailed plans. Each plan represents a distinct phase of development:

1. **[Core Foundation (v2.0)](file:///C:/Users/HP/.gemini/antigravity-cli/brain/a1d9fafa-e801-41c5-938a-b2f111bd1e2f/rl_agent_2_0_core.md)**
   * Standardize action encoding.
   * Implementation of the 500+ match rotated seats evaluation harness.
   * Baseline model setup (AceNetV2: 512 width, 4 residual blocks).
   * Supervised pretraining (Imitation Learning).
   * PPO + GAE implementation.
   * Step-level shaped rewards.
   * Opponent league training with mixed distribution, dynamic player count (3-6), and variable rounds.

2. **[Advanced Cognitive (v2.1)](file:///C:/Users/HP/.gemini/antigravity-cli/brain/a1d9fafa-e801-41c5-938a-b2f111bd1e2f/rl_agent_2_1_belief.md)**
   * Context-heavy state encoder upgrades.
   * Probabilistic belief tracker.
   * Opponent behavioral profiling features.
   * Auxiliary prediction heads.

3. **[Advanced Planning (v2.2)](file:///C:/Users/HP/.gemini/antigravity-cli/brain/a1d9fafa-e801-41c5-938a-b2f111bd1e2f/rl_agent_2_2_search.md)**
   * Belief sampling lookahead search.
   * Endgame brute-force solver.
   * Search-guided distillation.

4. **[Advanced Deception & Exploitation (v2.3)](file:///C:/Users/HP/.gemini/antigravity-cli/brain/a1d9fafa-e801-41c5-938a-b2f111bd1e2f/rl_agent_2_3_deception.md)**
   * targeted exploit-margin training against HeuristicAgentV2.
   * Deception and information leakage rewards.
   * Persistent human profile database.
