# scripts/evaluation/benchmark_all_agents.py
import sys
import time
import random
import numpy as np
from typing import List, Dict, Any, Type
sys.path.append('.')

# Disable PyTorch multithreading to avoid CPU thrashing during evaluation
import torch
torch.set_num_threads(1)

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, EngineState
from agents.random.agent import BaseAgent, RandomAgent
from engine.events import get_player_view

# Import the 5 candidate agents
from agents.heuristic.v1.heuristic_agent import HeuristicAgent
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.mcts.v1.mcts_agent import MCTSAgent
from agents.rl.v1.rl_agent import RLAgent
from agents.rl.v2.rl_agent_v2 import RLAgentV2

class BenchmarkRLAgentV2(RLAgentV2):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, checkpoint_path="checkpoints/rl_champion_v2.pt", explore=False, seed=seed)

class BenchmarkRLAgentV1(RLAgent):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, checkpoint_path="checkpoints/rl_champion.pt", explore=False)

class BenchmarkMCTSAgent(MCTSAgent):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, seed=seed, max_iterations=300, time_limit=0.4)

class BenchmarkHeuristicAgentV2(HeuristicAgentV2):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, seed=seed)

class BenchmarkHeuristicAgentV1(HeuristicAgent):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, seed=seed)

AGENT_CLASSES = {
    "Heuristic V1": BenchmarkHeuristicAgentV1,
    "Heuristic V2": BenchmarkHeuristicAgentV2,
    "MCTS Agent": BenchmarkMCTSAgent,
    "RL Agent V1": BenchmarkRLAgentV1,
    "RL Agent V2": BenchmarkRLAgentV2
}

def main():
    print("====================================================")
    print("===      BENCHMARK TOURNAMENT: ALL 5 AGENTS      ===")
    print("====================================================\n")

    num_matches = 100
    base_seed = 7777

    # Initialize scoreboard for each agent type
    stats = {
        name: {
            "matches_played": 0,
            "rounds_played": 0,
            "rounds_lost": 0,
            "total_points": 0.0
        } for name in AGENT_CLASSES
    }

    start_time = time.time()

    for m_idx in range(num_matches):
        match_seed = base_seed + m_idx
        # Select 4 distinct agents out of the 5
        agent_names = list(AGENT_CLASSES.keys())
        selected_names = random.sample(agent_names, 4)
        
        # Randomize round count (3 to 7 rounds)
        num_rounds = random.randint(3, 7)
        
        # Instantiate state
        state = AceEngine.create_match(
            match_id=m_idx,
            num_players=4,
            num_rounds=num_rounds,
            match_seed=match_seed
        )
        
        # Create agents at their seats
        agents = []
        for seat_id, name in enumerate(selected_names):
            agents.append(AGENT_CLASSES[name](player_id=seat_id, seed=match_seed + 100 + seat_id))
            stats[name]["matches_played"] += 1

        state, _ = AceEngine.advance(state)

        # Match loop
        while not AceEngine.is_terminal(state):
            phase = AceEngine.get_game_phase(state)
            if isinstance(phase, RoundStarting):
                state, _ = AceEngine.advance(state)
                continue

            player_id = state.runtime_state.current_player_id
            if player_id is None:
                state, _ = AceEngine.advance(state)
                continue

            player_view = get_player_view(state, player_id)
            legal_actions = AceEngine.get_legal_actions(state)

            agent = agents[player_id]
            action = agent.select_action(player_view, legal_actions)

            res = AceEngine.apply_action(state, action)
            if not isinstance(res, Success):
                res = AceEngine.apply_action(state, legal_actions[0])
            state = res.new_state

        # Match completed: record stats
        # 1. Round outcomes
        for res_round in state.match_state.round_results:
            # Each player played this round
            for seat_id, name in enumerate(selected_names):
                stats[name]["rounds_played"] += 1
            if not res_round.is_draw and res_round.loser_id is not None:
                loser_name = selected_names[res_round.loser_id]
                stats[loser_name]["rounds_lost"] += 1

        # 2. Points
        match_points = {}
        for seat_id, name in enumerate(selected_names):
            pts = state.match_state.players[seat_id].half_points / 2.0
            stats[name]["total_points"] += pts
            match_points[name] = pts

        # Verbose match output
        print(f"[Match {m_idx + 1:03d}/{num_matches:03d}] Seed: {match_seed} | Rounds: {num_rounds} | Lineup: {selected_names} | Points: {match_points}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nTournament complete in {elapsed:.2f} seconds.")

    # Print final scoreboard
    print("\n" + "="*70)
    print("                    FINAL BENCHMARK SCOREBOARD")
    print("="*70)
    print(f"{'Agent Name':<15} | {'Matches':<8} | {'Rounds':<8} | {'Survival %':<12} | {'Avg Points':<10}")
    print("-"*70)
    
    sorted_agents = sorted(
        stats.items(),
        key=lambda x: (1.0 - x[1]["rounds_lost"] / max(1, x[1]["rounds_played"]), x[1]["total_points"] / max(1, x[1]["matches_played"])),
        reverse=True
    )

    for name, data in sorted_agents:
        m_played = data["matches_played"]
        r_played = data["rounds_played"]
        r_lost = data["rounds_lost"]
        pts = data["total_points"]
        
        survival_rate = (1.0 - r_lost / max(1, r_played)) * 100.0
        avg_pts = pts / max(1, m_played)
        
        print(f"{name:<15} | {m_played:<8} | {r_played:<8} | {survival_rate:>10.2f}% | {avg_pts:>10.3f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
