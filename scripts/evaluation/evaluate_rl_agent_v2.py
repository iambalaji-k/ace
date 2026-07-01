# scripts/evaluate_rl_agent_v2.py
"""Rigorous evaluation tournament script for RL Agent 2.0 Core.

Simulates 504 matches (84 cycles of all 6 seating permutations).
Features paired seed comparisons (running all seating layouts with the same seed)
and paired t-test confidence intervals to account for correlated round outcomes.
"""

import sys
import time
import math
import numpy as np
from typing import List, Type, Dict, Any
sys.path.append('.')

from engine.tournament import TournamentConfig, _run_single_match
from agents.random.agent import BaseAgent
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.rl.v2.rl_agent_v2 import RLAgentV2


def main():
    print("====================================================")
    print("===   EVALUATION TOURNAMENT: RL V2.0 VS HEURISTIC V2 ===")
    print("====================================================\n")

    num_cycles = 84
    num_players = 4
    num_rounds = 5
    base_seed = 9999
    checkpoint_path = "checkpoints/rl_champion_v2.pt"

    print(f"Tournament Configuration:")
    print(f"  Total Cycles:     {num_cycles}")
    print(f"  Matches per Cycle: 6 (all permutations)")
    print(f"  Total Matches:    {num_cycles * 6}")
    print(f"  Rounds per Match: {num_rounds}")
    print(f"  Total Rounds:     {num_cycles * 6 * num_rounds}")
    print(f"  Base Seed:        {base_seed}")
    print(f"  RL Checkpoint:    {checkpoint_path}\n")

    # Define the 6 seat-balanced team permutations (0 = RLAgentV2, 1 = HeuristicAgentV2)
    permutations = [
        [0, 1, 0, 1],  # RL, H2, RL, H2
        [1, 0, 1, 0],  # H2, RL, H2, RL
        [0, 0, 1, 1],  # RL, RL, H2, H2
        [1, 1, 0, 0],  # H2, H2, RL, RL
        [0, 1, 1, 0],  # RL, H2, H2, RL
        [1, 0, 0, 1]   # H2, RL, RL, H2
    ]

    class EvaluatedRLAgent(RLAgentV2):
        """Evaluation wrapper for RLAgentV2 executing deterministically."""
        def __init__(self, player_id: int, seed=None):
            super().__init__(player_id, checkpoint_path=checkpoint_path, explore=False, seed=seed)

    class EvaluatedHeuristicAgent(HeuristicAgentV2):
        """Evaluation wrapper for HeuristicAgentV2."""
        def __init__(self, player_id: int, seed=None):
            super().__init__(player_id, seed=seed)

    agent_types = [EvaluatedRLAgent, EvaluatedHeuristicAgent]

    # Overall counters
    rl_rounds_lost = 0
    rl_rounds_played = 0
    rl_points = []

    h2_rounds_lost = 0
    h2_rounds_played = 0
    h2_points = []

    # Paired metrics per cycle (to calculate robust confidence intervals)
    cycle_rl_survival_rates = []
    cycle_h2_survival_rates = []
    cycle_diffs = []

    start_time = time.time()

    for c_idx in range(num_cycles):
        # All 6 permutations in this cycle share the exact same deal seed
        cycle_seed = base_seed + c_idx

        c_rl_lost = 0
        c_rl_played = 0
        c_h2_lost = 0
        c_h2_played = 0

        for perm_idx, layout in enumerate(permutations):
            # Seeding: base seed + cycle offset + deterministic layout seed offset
            config = TournamentConfig(
                num_matches=1,
                num_players=num_players,
                num_rounds=num_rounds,
                base_seed=cycle_seed,
                agent_classes=[agent_types[val] for val in layout]
            )

            state = _run_single_match(config, 0)

            # Record points
            for p_idx, agent_type_code in enumerate(layout):
                pts = state.match_state.players[p_idx].half_points / 2.0
                if agent_type_code == 0:
                    rl_points.append(pts)
                else:
                    h2_points.append(pts)

            # Record round outcomes
            for res in state.match_state.round_results:
                if not res.is_draw and res.loser_id is not None:
                    loser_type = layout[res.loser_id]
                    if loser_type == 0:
                        rl_rounds_lost += 1
                        c_rl_lost += 1
                    else:
                        h2_rounds_lost += 1
                        c_h2_lost += 1

                rl_rounds_played += 2
                c_rl_played += 2
                h2_rounds_played += 2
                c_h2_played += 2

        # Record cycle statistics
        c_rl_surv = 1.0 - (c_rl_lost / c_rl_played)
        c_h2_surv = 1.0 - (c_h2_lost / c_h2_played)
        cycle_rl_survival_rates.append(c_rl_surv)
        cycle_h2_survival_rates.append(c_h2_surv)
        cycle_diffs.append(c_rl_surv - c_h2_surv)

    duration = time.time() - start_time

    # Calculate overall survival rates
    overall_rl_survival = 1.0 - (rl_rounds_lost / rl_rounds_played)
    overall_h2_survival = 1.0 - (h2_rounds_lost / h2_rounds_played)

    # Compute Paired Seed Standard Error and Confidence Intervals
    mean_diff = np.mean(cycle_diffs)
    std_diff = np.std(cycle_diffs, ddof=1)
    se_diff = std_diff / math.sqrt(num_cycles)
    
    # 95% Confidence Interval for the survival difference (t-distribution critical value for df=83 is ~1.989)
    t_critical = 1.989
    ci_margin = t_critical * se_diff

    # Points metrics
    rl_mean_points = np.mean(rl_points)
    rl_std_points = np.std(rl_points)
    h2_mean_points = np.mean(h2_points)
    h2_std_points = np.std(h2_points)

    print("="*60)
    print("                   EVALUATION SUMMARY")
    print("="*60)
    print(f"Matches Run:      {num_cycles * 6}")
    print(f"Total Match Time: {duration:.2f} seconds ({duration/(num_cycles*6):.3f}s per match)")
    print("-"*60)
    print(f"RLAgentV2 (Core):")
    print(f"  Survival Rate:  {overall_rl_survival * 100.0:.2f}%")
    print(f"  Rounds Lost:    {rl_rounds_lost} / {rl_rounds_played}")
    print(f"  Mean Points:    {rl_mean_points:.3f} ± {rl_std_points:.3f}")
    print("-"*60)
    print(f"HeuristicAgentV2 (Evolved):")
    print(f"  Survival Rate:  {overall_h2_survival * 100.0:.2f}%")
    print(f"  Rounds Lost:    {h2_rounds_lost} / {h2_rounds_played}")
    print(f"  Mean Points:    {h2_mean_points:.3f} ± {h2_std_points:.3f}")
    print("-"*60)
    print(f"Paired Difference Statistics:")
    print(f"  Survival Rate Margin: {mean_diff * 100.0:+.2f}% ± {ci_margin * 100.0:.2f}% (95% CI)")
    print(f"  95% CI Range:         [{(mean_diff - ci_margin)*100.0:+.2f}%, {(mean_diff + ci_margin)*100.0:+.2f}%]")
    print("="*60)

    if mean_diff - ci_margin > 0:
        print(f"\nSUCCESS: RLAgentV2 is STATISTICALLY SUPERIOR to HeuristicAgentV2!")
    elif mean_diff + ci_margin < 0:
        print(f"\nREGRESSION: HeuristicAgentV2 is STATISTICALLY SUPERIOR to RLAgentV2.")
    else:
        print(f"\nSTATISTICALLY INCONCLUSIVE: Performance margin is within confidence bounds.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
