import sys
import os
import time
from typing import List, Type, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set PYTHONPATH so we can import engine and agents
sys.path.append(os.getcwd())

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, EngineState
from engine.events import get_player_view
from agents.random.agent import BaseAgent, RandomAgent
from agents.rl.v2.rl_agent_v2 import RLAgentV2
from agents.mcts.v1.mcts_agent import MCTSAgent
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2

def prompt_input(prompt: str, default: Any, val_type: Type) -> Any:
    """Helper to prompt for input with a default value and type validation."""
    try:
        user_val = input(f"{prompt} [{default}]: ").strip()
        if not user_val:
            return default
        return val_type(user_val)
    except ValueError:
        print(f"Invalid input. Defaulting to {default}.")
        return default

class TestRLAgentV2(RLAgentV2):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, checkpoint_path="checkpoints/rl_champion_v2.pt", explore=False, seed=seed)

class ConfigurableMCTSAgent(MCTSAgent):
    # Set dynamically from user inputs
    max_iters = 50
    t_limit = 0.10

    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, seed=seed, max_iterations=ConfigurableMCTSAgent.max_iters, time_limit=ConfigurableMCTSAgent.t_limit)

class TestHeuristicAgentV2(HeuristicAgentV2):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id=player_id, seed=seed)

class TestRandomAgent(RandomAgent):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id=player_id, seed=seed)

AGENT_TYPES = {
    "1": ("RL V2", TestRLAgentV2),
    "2": ("MCTS", ConfigurableMCTSAgent),
    "3": ("Heuristic V2", TestHeuristicAgentV2),
    "4": ("Random", TestRandomAgent),
}

def run_single_match(match_index: int, num_players: int, num_rounds: int, base_seed: int, seat_config: dict) -> Tuple[EngineState, float]:
    """Run a single match and return the final state and execution duration."""
    start_time = time.time()
    match_seed = base_seed + match_index

    # 1. Create match state
    state = AceEngine.create_match(
        match_id=match_index,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=match_seed
    )

    # 2. Instantiate agents based on seat configuration
    agents = []
    for i in range(num_players):
        agent_name, agent_class = AGENT_TYPES[seat_config[i]]
        agents.append(agent_class(player_id=i, seed=match_seed + 1000 + i))

    # Start Round 1
    state, _ = AceEngine.advance(state)

    # 3. Match execution loop
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)

        if isinstance(phase, RoundStarting):
            state, _ = AceEngine.advance(state)
            continue

        player_id = state.runtime_state.current_player_id
        assert player_id is not None

        player_view = get_player_view(state, player_id)
        legal_actions = AceEngine.get_legal_actions(state)

        agent = agents[player_id]
        action = agent.select_action(player_view, legal_actions)

        res = AceEngine.apply_action(state, action)
        if not isinstance(res, Success):
            res = AceEngine.apply_action(state, legal_actions[0])

        assert isinstance(res, Success)
        state = res.new_state

    duration = time.time() - start_time
    return state, duration

def main():
    print("===========================================================")
    print("===     CONFIGURABLE AGENT TOURNAMENT SYSTEM             ===")
    print("===========================================================")
    print("Please configure the tournament parameters below:")
    
    num_matches = prompt_input("  Number of Matches   ", 50, int)
    num_rounds = prompt_input("  Rounds per Match    ", 3, int)
    max_iters = prompt_input("  MCTS Max Iterations ", 50, int)
    t_limit = prompt_input("  MCTS Time Limit (s) ", 0.10, float)
    
    # Apply configurations
    ConfigurableMCTSAgent.max_iters = max_iters
    ConfigurableMCTSAgent.t_limit = t_limit

    # Per-seat agent configuration
    print("\nAgent types: [1] RL V2  [2] MCTS  [3] Heuristic V2  [4] Random")
    seat_config = {}
    for seat in range(4):
        default_agent = "1" if seat in [0, 2] else "2"
        choice = prompt_input(f"  Seat {seat} agent", default_agent, str)
        if choice not in AGENT_TYPES:
            print(f"  Invalid choice '{choice}', defaulting to '1' (RL V2)")
            choice = "1"
        seat_config[seat] = choice

    # Validate: at least 2 different agent types for meaningful comparison
    agent_types_used = set(seat_config.values())
    if len(agent_types_used) < 2:
        print("\nWarning: All seats have the same agent type. Results will show self-play.")
        confirm = prompt_input("  Continue anyway? (y/n)", "y", str)
        if confirm.lower() != "y":
            print("Aborted.")
            return

    # Display configuration summary
    print("\nStarting Tournament...")
    print(f"  Matches:    {num_matches}")
    print(f"  Rounds:     {num_rounds}")
    print(f"  MCTS Iters: {max_iters}")
    print(f"  MCTS Time:  {t_limit}s")
    print(f"  Seats:      {', '.join(f'Seat {i}={AGENT_TYPES[seat_config[i]][0]}' for i in range(4))}\n")
    
    start_time = time.time()
    
    # Stats trackers
    player_points = {i: [] for i in range(4)}
    player_wins = {i: 0 for i in range(4)}
    player_losses = {i: 0 for i in range(4)}
    
    completed_matches = 0
    
    for idx in range(num_matches):
        try:
            state, duration = run_single_match(idx, 4, num_rounds, 9000, seat_config)
            completed_matches += 1

            # Determine match loser/winner details
            final_players = state.match_state.players
            sorted_players = sorted(final_players, key=lambda x: x.half_points, reverse=True)
            match_loser = sorted_players[-1].player_id
            match_winner = sorted_players[0].player_id

            # Format names dynamically
            loser_name = f"Player {match_loser} ({AGENT_TYPES[seat_config[match_loser]][0]})"
            winner_name = f"Player {match_winner} ({AGENT_TYPES[seat_config[match_winner]][0]})"
            
            # Verbose logging
            print(f"  [MATCH {completed_matches:03d}/{num_matches:03d}] "
                  f"Winner: {winner_name:<18} | "
                  f"Loser: {loser_name:<18} | "
                  f"Duration: {duration:.2f}s")
            
            # Record stats
            result = AceEngine.get_result(state)
            for player in result.rankings:
                p_id = player.player_id
                player_points[p_id].append(player.half_points / 2.0)
                player_wins[p_id] += player.rounds_won
                player_losses[p_id] += player.rounds_lost
                
        except Exception as e:
            print(f"  [ERROR] Match {idx} failed: {e}")
                
    total_duration = time.time() - start_time
    total_rounds = completed_matches * num_rounds

    # Build individual player stats
    player_stats = []
    for pid in range(4):
        agent_name = AGENT_TYPES[seat_config[pid]][0]
        win_ratio = (player_wins[pid] / total_rounds) * 100 if total_rounds > 0 else 0.0
        loss_ratio = player_losses[pid] / total_rounds * 100 if total_rounds > 0 else 0.0
        mean_points = sum(player_points[pid]) / completed_matches if completed_matches > 0 else 0.0
        player_stats.append({
            "id": pid,
            "agent": agent_name,
            "win_pct": win_ratio,
            "loss_pct": loss_ratio,
            "points": mean_points,
        })

    # Sort by win percentage descending
    player_stats.sort(key=lambda x: x["win_pct"], reverse=True)

    print("\n" + "="*60)
    print("===              INDIVIDUAL PLAYER STATS              ===")
    print("="*60)
    print(f"Total Execution Time: {total_duration:.2f}s")
    print(f"Completed Matches:    {completed_matches}")
    print(f"Total Rounds:         {total_rounds}\n")

    for pid in range(4):
        s = player_stats[pid]
        print(f"Player {s['id']} ({s['agent']:<14}):")
        print(f"  - Win Ratio:  {s['win_pct']:.1f}%")
        print(f"  - Loss Ratio: {s['loss_pct']:.1f}%")
        print(f"  - Avg Points: {s['points']:.2f}")
        print()

    print("="*60)
    print("===                   LEADERBOARD                     ===")
    print("="*60)

    for rank, s in enumerate(player_stats, 1):
        print(f"  #{rank}  Player {s['id']} ({s['agent']:<14}) | Win: {s['win_pct']:.1f}% | Points: {s['points']:.2f}")

    print()
    if player_stats[0]["win_pct"] > player_stats[1]["win_pct"]:
        diff = player_stats[0]["win_pct"] - player_stats[1]["win_pct"]
        print(f"CONCLUSION: Player {player_stats[0]['id']} ({player_stats[0]['agent']}) is STRONGER by {diff:.1f}% Win Ratio.")
    else:
        print("CONCLUSION: No clear winner — results are within margin of error.")
    print("="*60)

if __name__ == "__main__":
    main()
