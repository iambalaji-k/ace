# scripts/evaluate_rl_agent.py
"""Evaluation script comparing the trained RLAgent against HeuristicAgentV2."""

import sys
import time
from typing import List, Type
sys.path.append('.')

from engine.tournament import TournamentConfig, _run_single_match
from agents.random.agent import BaseAgent
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.rl.v1.rl_agent import RLAgent

def main():
    print("====================================================")
    print("===   EVALUATION TOURNAMENT: RL VS HEURISTIC V2   ===")
    print("====================================================\n")

    num_matches = 10
    num_players = 4
    num_rounds = 5
    base_seed = 888
    checkpoint_path = "checkpoints/rl_champion.pt"

    class RLAgentWrap(RLAgent):
        def __init__(self, player_id: int, seed=None):
            # Evaluate deterministically (explore=False) using the trained model
            super().__init__(player_id, checkpoint_path=checkpoint_path, explore=False)

    agent_classes: List[Type[BaseAgent]] = [
        RLAgentWrap,
        HeuristicAgentV2,
        RLAgentWrap,
        HeuristicAgentV2
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
    print("RL Agent vs HeuristicV2 Tournament Results")
    print("="*50)
    print(f"Total Matches Simulated: {num_matches}")
    print(f"Total Rounds Simulated: {total_rounds}")
    print(f"Execution Time: {duration:.2f} seconds")
    print("-"*50)
    
    for i in range(num_players):
        loss_rate = (rounds_lost[i] / total_rounds) * 100.0
        survival_rate = 100.0 - loss_rate
        agent_type = "RLAgent (Phase 11)" if i in (0, 2) else "HeuristicV2 (Evolved)"
        print(f"Player {i} [{agent_type}]:")
        print(f"  Survival Rate: {survival_rate:.2f}% | Rounds Lost: {rounds_lost[i]}/{total_rounds}")

    print("="*50)
    rl_loss = rounds_lost[0] + rounds_lost[2]
    h2_loss = rounds_lost[1] + rounds_lost[3]
    
    rl_survival = 100.0 - (rl_loss / (2.0 * total_rounds) * 100.0)
    h2_survival = 100.0 - (h2_loss / (2.0 * total_rounds) * 100.0)
    
    print(f"Team RLAgent Avg Survival: {rl_survival:.2f}%")
    print(f"Team HeuristicV2 Avg Survival: {h2_survival:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
