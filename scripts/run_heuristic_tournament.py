# scripts/run_heuristic_tournament.py
"""Tournament script to evaluate HeuristicAgent performance against RandomAgents."""

import time
from typing import List, Type
from engine.tournament import TournamentConfig, _run_single_match, PlayerStats
from engine.agent import BaseAgent, RandomAgent
from engine.heuristic_agent import HeuristicAgent
from engine.rules import AceEngine


def run_tournament():
    print("Initializing Heuristic vs Random Tournament (100 Matches)...")
    
    # We will use 4 players: Player 0 is HeuristicAgent, Players 1, 2, 3 are RandomAgent
    agent_classes: List[Type[BaseAgent]] = [
        HeuristicAgent,
        RandomAgent,
        RandomAgent,
        RandomAgent
    ]
    
    config = TournamentConfig(
        num_matches=100,  # Run 100 matches to keep execution quick
        num_players=4,
        num_rounds=5,     # 5 rounds per match
        base_seed=42,
        agent_classes=agent_classes
    )
    
    start_time = time.time()
    
    heuristic_wins = 0
    total_rounds = 0
    heuristic_points = 0.0
    
    for m_idx in range(config.num_matches):
        state = _run_single_match(config, m_idx)
        
        # Collect round stats
        for res in state.match_state.round_results:
            total_rounds += 1
            if not res.is_draw:
                if res.loser_id != 0:
                    heuristic_wins += 1
            else:
                heuristic_wins += 1  # Draw counts as win for everyone except loser (everyone gets 0.5 points)
                
        # Total points for player 0
        heuristic_points += state.match_state.players[0].half_points / 2.0
        
    execution_time = time.time() - start_time
    win_rate = (heuristic_wins / total_rounds) * 100.0
    avg_points = heuristic_points / config.num_matches
    
    print("\n" + "="*40)
    print("Tournament Completed!")
    print(f"Total Matches: {config.num_matches}")
    print(f"Total Rounds Simulated: {total_rounds}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print("-"*40)
    print(f"Heuristic Agent Win Rate (Non-Loss Rate): {win_rate:.2f}%")
    print(f"Heuristic Agent Avg Points per Match: {avg_points:.2f}")
    print("="*40)
    
    if win_rate >= 60.0:
        print("Success! Heuristic Agent outperforms Random Agent by >= 60%.")
    else:
        print("Heuristic Agent did not reach target of 60% win rate.")


if __name__ == "__main__":
    run_tournament()
