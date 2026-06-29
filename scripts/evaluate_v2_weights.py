# scripts/evaluate_v2_weights.py
"""Head-to-head evaluation script comparing HeuristicAgentV2 (GA weights) against HeuristicAgentV1."""

import sys
import time
from typing import List, Type
sys.path.append('.')

from engine.tournament import TournamentConfig, _run_single_match
from engine.agent import BaseAgent
from engine.heuristic_agent import HeuristicAgent
from engine.heuristic_agent_v2 import HeuristicAgentV2

def main():
    print("====================================================")
    print("===   EVALUATION TOURNAMENT: HEURISTIC V2 VS V1   ===")
    print("====================================================\n")

    num_matches = 100
    num_players = 4
    num_rounds = 5
    base_seed = 42

    # Players 0 and 2 are HeuristicAgentV2 (Evolved)
    # Players 1 and 3 are HeuristicAgent (Baseline V1)
    agent_classes: List[Type[BaseAgent]] = [
        HeuristicAgentV2,
        HeuristicAgent,
        HeuristicAgentV2,
        HeuristicAgent
    ]

    config = TournamentConfig(
        num_matches=num_matches,
        num_players=num_players,
        num_rounds=num_rounds,
        base_seed=base_seed,
        agent_classes=agent_classes
    )

    print(f"Simulating {num_matches} matches (total {num_matches * num_rounds} rounds)...")
    start_time = time.time()

    # Track rounds lost (last active player) for each player seat
    rounds_lost = [0] * num_players
    total_rounds = 0

    for m_idx in range(num_matches):
        state = _run_single_match(config, m_idx)
        for res in state.match_state.round_results:
            total_rounds += 1
            if not res.is_draw and res.loser_id is not None:
                rounds_lost[res.loser_id] += 1

    duration = time.time() - start_time
    
    print("\n" + "="*50)
    print("Tournament Results Summary")
    print("="*50)
    print(f"Total Matches Simulated: {num_matches}")
    print(f"Total Rounds Simulated: {total_rounds}")
    print(f"Execution Time: {duration:.2f} seconds")
    print("-"*50)
    
    # Non-loss rate is the proportion of rounds where the player did NOT finish last
    for i in range(num_players):
        loss_rate = (rounds_lost[i] / total_rounds) * 100.0
        survival_rate = 100.0 - loss_rate
        agent_type = "HeuristicV2 (Evolved)" if i in (0, 2) else "HeuristicV1 (Baseline)"
        print(f"Player {i} [{agent_type}]:")
        print(f"  Survival Rate: {survival_rate:.2f}% | Rounds Lost: {rounds_lost[i]}/{total_rounds}")

    print("="*50)
    
    v2_total_loss = rounds_lost[0] + rounds_lost[2]
    v1_total_loss = rounds_lost[1] + rounds_lost[3]
    
    v2_avg_survival = 100.0 - (v2_total_loss / (2.0 * total_rounds) * 100.0)
    v1_avg_survival = 100.0 - (v1_total_loss / (2.0 * total_rounds) * 100.0)
    
    print(f"Team HeuristicV2 (Evolved) Avg Survival: {v2_avg_survival:.2f}%")
    print(f"Team HeuristicV1 (Baseline) Avg Survival: {v1_avg_survival:.2f}%")
    print("-"*50)
    
    if v2_total_loss < v1_total_loss:
        improvement = ((v1_total_loss - v2_total_loss) / v1_total_loss) * 100.0 if v1_total_loss > 0 else 0.0
        print(f"\nSUCCESS! Evolved HeuristicV2 Team survived better than Baseline HeuristicV1 by {improvement:.2f}% fewer losses!")
    else:
        print("\nHeuristicV2 did not outperform Baseline HeuristicV1. Tuning required.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
