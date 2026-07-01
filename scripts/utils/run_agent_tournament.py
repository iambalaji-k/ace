import sys
import os
import time
import random
from typing import List, Type, Dict, Any, Tuple

# Set PYTHONPATH so we can import engine and agents
sys.path.append(os.getcwd())

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, EngineState
from engine.events import get_player_view
from engine.card import card_to_str
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

class ConfigurableMCTSAgent(MCTSAgent):
    max_iters = 300
    t_limit = 0.60

    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id, seed=seed, max_iterations=ConfigurableMCTSAgent.max_iters, time_limit=ConfigurableMCTSAgent.t_limit)

class TestHeuristicAgentV2(HeuristicAgentV2):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id=player_id, seed=seed)

class TestRandomAgent(RandomAgent):
    def __init__(self, player_id: int, seed=None):
        super().__init__(player_id=player_id, seed=seed)

class HumanAgent(BaseAgent):
    """An interactive agent that allows a human player to play via console inputs."""
    def select_action(self, player_view: Dict[str, Any], legal_actions: List[Any]) -> Any:
        print("\n" + "="*50)
        print(f"===   YOUR TURN (Player {self.player_id})   ===")
        print("="*50)
        
        # Display hand
        hand = player_view.get("hand", [])
        print("Your Hand:")
        for i, card in enumerate(hand):
            print(f"  {card_to_str(card)}")

        # Display legal moves
        print("\nLegal Moves:")
        for idx, act in enumerate(legal_actions):
            print(f"  [{idx}] {act}")
            
        choice = -1
        while choice < 0 or choice >= len(legal_actions):
            try:
                val = input(f"\nSelect action index (0-{len(legal_actions)-1}): ").strip()
                choice = int(val)
            except ValueError:
                choice = -1
        return legal_actions[choice]

def get_available_checkpoints() -> List[str]:
    """Scans the checkpoints directory for all available RL checkpoints, ignoring legacy V1."""
    ckpt_dir = "checkpoints"
    if not os.path.exists(ckpt_dir):
        return []
    files = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt") and f != "rl_champion.pt":
            files.append(os.path.join(ckpt_dir, f))
    # Natural sort order
    files.sort()
    return files

def run_single_match(match_index: int, num_players: int, num_rounds: int, base_seed: int, seat_config: dict, agent_menu: dict) -> Tuple[EngineState, float]:
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
        agent_name, agent_class = agent_menu[seat_config[i]]
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
    print("===     CONFIGURABLE LEAGUE TOURNAMENT SYSTEM            ===")
    print("===========================================================")
    print("Please configure the tournament parameters below:")
    
    num_matches = prompt_input("  Number of Matches   ", 5, int)
    if num_matches < 1:
        print(f"  Invalid number of matches {num_matches}. Defaulting to 5.")
        num_matches = 5

    num_players = prompt_input("  Number of Players (3-6)   ", 4, int)
    if num_players < 3 or num_players > 6:
        print(f"  Invalid player count {num_players}. Defaulting to 4.")
        num_players = 4

    num_rounds = prompt_input("  Rounds per Match    ", 3, int)
    if num_rounds < 1:
        print(f"  Invalid number of rounds {num_rounds}. Defaulting to 3.")
        num_rounds = 3

    max_iters = prompt_input("  MCTS Max Iterations ", 300, int)
    if max_iters < 1:
        print(f"  Invalid MCTS max iterations {max_iters}. Defaulting to 300.")
        max_iters = 300

    t_limit = prompt_input("  MCTS Time Limit (s) ", 0.60, float)
    if t_limit <= 0:
        print(f"  Invalid MCTS time limit {t_limit}. Defaulting to 0.60s.")
        t_limit = 0.60

    enable_rotation = prompt_input("  Enable Seating Rotation? (y/n) ", "n", str).strip().lower() == "y"

    # Configure MCTS settings
    ConfigurableMCTSAgent.max_iters = max_iters
    ConfigurableMCTSAgent.t_limit = t_limit

    # Dynamically build the menu of available agents based on player count and checkpoints directory
    agent_menu = {}
    menu_idx = 1

    # Standard general agents (3-6 players)
    agent_menu[str(menu_idx)] = ("MCTS (configurable)", ConfigurableMCTSAgent)
    menu_idx += 1
    agent_menu[str(menu_idx)] = ("Heuristic V2 (co-evolved rules)", TestHeuristicAgentV2)
    menu_idx += 1
    agent_menu[str(menu_idx)] = ("Random Agent", TestRandomAgent)
    menu_idx += 1
    agent_menu[str(menu_idx)] = ("Human Interactive Player", HumanAgent)
    menu_idx += 1

    # 4-player specific legacy agents
    if num_players == 4:
        try:
            from agents.heuristic.v1.heuristic_agent import HeuristicAgent
            class TestHeuristicAgentV1(HeuristicAgent):
                def __init__(self, player_id: int, seed=None):
                    super().__init__(player_id=player_id, seed=seed)
            agent_menu[str(menu_idx)] = ("Heuristic V1 (4-player only)", TestHeuristicAgentV1)
            menu_idx += 1
        except ImportError:
            pass

        try:
            from agents.rl.v1.rl_agent import RLAgent
            class TestRLAgentV1(RLAgent):
                def __init__(self, player_id: int, seed=None):
                    super().__init__(player_id=player_id, checkpoint_path="checkpoints/rl_champion.pt", explore=False)
            agent_menu[str(menu_idx)] = ("RL V1 (Legacy, 4-player only)", TestRLAgentV1)
            menu_idx += 1
        except ImportError:
            pass

    # Dynamic RL Checkpoint loading
    available_ckpts = get_available_checkpoints()
    for ckpt_path in available_ckpts:
        name = f"RL Checkpoint: {os.path.basename(ckpt_path)}"
        
        # Factory generator to create a dynamic class binding to the checkpoint path
        def make_dynamic_agent(path):
            class DynamicRLAgent(RLAgentV2):
                def __init__(self, player_id: int, seed=None):
                    super().__init__(player_id, checkpoint_path=path, explore=False, seed=seed)
            return DynamicRLAgent

        agent_menu[str(menu_idx)] = (name, make_dynamic_agent(ckpt_path))
        menu_idx += 1

    # Print the selection menu
    print("\nAvailable Agent Types:")
    for k, v in agent_menu.items():
        print(f"  [{k}] {v[0]}")

    # Prompt seat configurations
    seat_config = {}
    for seat in range(num_players):
        default_choice = "4" if seat == 0 else ("1" if seat % 2 == 1 else "2")
        # Ensure default choice is valid in menu
        if default_choice not in agent_menu:
            default_choice = "1"
        choice = prompt_input(f"  Configure Seat {seat} agent", default_choice, str)
        if choice not in agent_menu:
            print(f"  Invalid choice '{choice}', defaulting to '1' (MCTS)")
            choice = "1"
        seat_config[seat] = choice

    # Generate unique player names (e.g., "Heuristic V2 1", "Heuristic V2 2")
    agent_instance_count = {}
    player_names = {}
    for seat in range(num_players):
        agent_name = agent_menu[seat_config[seat]][0]
        agent_instance_count[agent_name] = agent_instance_count.get(agent_name, 0) + 1
        player_names[seat] = f"{agent_name} {agent_instance_count[agent_name]}"

    # Print tournament parameters summary
    print("\nStarting Tournament...")
    print(f"  Matches:    {num_matches}")
    print(f"  Players:    {num_players}")
    print(f"  Rounds:     {num_rounds}")
    print(f"  Rotation:   {'Enabled' if enable_rotation else 'Disabled'}")
    for seat in range(num_players):
        print(f"  Seat {seat} = {player_names[seat]}")
    print()

    start_time = time.time()
    
    # Stats tracked per named player
    player_stats = {}
    
    completed_matches = 0
    
    for idx in range(num_matches):
        try:
            # Build seat-to-player mapping for this match (handles rotation)
            seat_to_player = {}
            for seat in range(num_players):
                if enable_rotation:
                    rotated_seat = (seat + idx) % num_players
                else:
                    rotated_seat = seat
                seat_to_player[rotated_seat] = player_names[seat]

            # Build current_seats for match execution (seat -> agent menu key)
            current_seats = {}
            for seat in range(num_players):
                if enable_rotation:
                    rotated_seat = (seat + idx) % num_players
                else:
                    rotated_seat = seat
                current_seats[rotated_seat] = seat_config[seat]

            state, duration = run_single_match(idx, num_players, num_rounds, 9000, current_seats, agent_menu)
            completed_matches += 1

            # Determine match results
            final_players = state.match_state.players
            sorted_players = sorted(final_players, key=lambda x: x.half_points, reverse=True)
            match_loser_id = sorted_players[-1].player_id
            match_winner_id = sorted_players[0].player_id

            winner_name = seat_to_player[match_winner_id]
            loser_name = seat_to_player[match_loser_id]
            
            print(f"  [MATCH {completed_matches:03d}/{num_matches:03d}] "
                  f"Winner: {winner_name:<30} | "
                  f"Loser: {loser_name:<30} | "
                  f"Duration: {duration:.2f}s")
            
            # Record stats per named player
            result = AceEngine.get_result(state)
            for player in result.rankings:
                p_id = player.player_id
                pname = seat_to_player[p_id]
                
                if pname not in player_stats:
                    player_stats[pname] = {"wins": 0, "losses": 0, "points": [], "matches_played": 0, "round_wins": 0, "round_losses": 0}
                
                player_stats[pname]["points"].append(player.half_points / 2.0)
                player_stats[pname]["round_wins"] += player.rounds_won
                player_stats[pname]["round_losses"] += player.rounds_lost
                player_stats[pname]["matches_played"] += 1
                
                if p_id != match_loser_id:
                    player_stats[pname]["wins"] += 1
                if p_id == match_loser_id:
                    player_stats[pname]["losses"] += 1
                
        except Exception as e:
            print(f"  [ERROR] Match {idx} failed: {e}")
            import traceback
            traceback.print_exc()
                
    total_duration = time.time() - start_time

    def build_table(stats_dict):
        rows = []
        for name, stats in stats_dict.items():
            if stats["matches_played"] == 0:
                continue
            total_rounds = stats["round_wins"] + stats["round_losses"]
            rows.append({
                "name": name,
                "played": stats["matches_played"],
                "m_win": stats["wins"],
                "m_loss": stats["losses"],
                "m_win_pct": (stats["wins"] / stats["matches_played"]) * 100 if stats["matches_played"] > 0 else 0.0,
                "avg_pts": sum(stats["points"]) / len(stats["points"]) if stats["points"] else 0.0,
                "rounds": total_rounds,
                "r_win": stats["round_wins"],
                "r_loss": stats["round_losses"],
                "r_win_pct": (stats["round_wins"] / total_rounds) * 100 if total_rounds > 0 else 0.0,
                "r_loss_pct": (stats["round_losses"] / total_rounds) * 100 if total_rounds > 0 else 0.0,
            })
        return rows

    # --- HEADER ---
    print("\n" + "="*80)
    print("                      LEAGUE TOURNAMENT RESULTS")
    print("="*80)
    print(f"  Execution Time: {total_duration:.2f}s  |  Matches: {completed_matches}  |  Players/Match: {num_players}  |  Rounds/Match: {num_rounds}")
    print("="*80)

    # --- PLAYER-WISE TABLES ---
    player_rows = build_table(player_stats)
    player_rows.sort(key=lambda x: x["r_win_pct"], reverse=True)

    print("\n  TABLE 1: MATCH PERFORMANCE (per player instance)")
    print("  " + "-"*78)
    print(f"  {'Player':<40} {'Played':>6} {'M_Win':>6} {'M_Loss':>6} {'M_Win%':>8} {'Avg Pts':>9}")
    print("  " + "-"*78)
    for s in player_rows:
        print(f"  {s['name']:<40} {s['played']:>6} {s['m_win']:>6} {s['m_loss']:>6} {s['m_win_pct']:>7.1f}% {s['avg_pts']:>9.2f}")
    print("  " + "-"*78)
    print()

    print("  TABLE 2: ROUND PERFORMANCE (per player instance)")
    print("  " + "-"*78)
    print(f"  {'Player':<40} {'Rounds':>6} {'R_Win':>6} {'R_Loss':>6} {'R_Win%':>8} {'R_Loss%':>9}")
    print("  " + "-"*78)
    for s in player_rows:
        print(f"  {s['name']:<40} {s['rounds']:>6} {s['r_win']:>6} {s['r_loss']:>6} {s['r_win_pct']:>7.1f}% {s['r_loss_pct']:>8.1f}%")
    print("  " + "-"*78)

    # --- CONCLUSION ---
    if len(player_rows) >= 2:
        top1 = player_rows[0]
        top2 = player_rows[1]
        diff = top1["r_win_pct"] - top2["r_win_pct"]
        print(f"\n  VERDICT: {top1['name']} is strongest by {diff:.1f}% round win rate over {top2['name']}.")
    elif len(player_rows) == 1:
        print(f"\n  VERDICT: Only one player participated.")
    print()

if __name__ == "__main__":
    main()
