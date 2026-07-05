# scratch/profile_mcts.py
import sys
sys.path.append('.')
import time
import numpy as np

from engine.rules import AceEngine
from agents.mcts.v1.mcts_agent import MCTSAgent

def profile_mcts():
    print("====================================================")
    print("===      MCTS AGENT PERFORMANCE PROFILER         ===")
    print("====================================================\n")

    # Set up MCTS agent with 300 iterations and a huge time limit to ensure it completes all iterations
    agent = MCTSAgent(player_id=1, max_iterations=300, time_limit=60.0, seed=123)

    # Initialize a match state
    state = AceEngine.create_match(match_id=1, num_players=4, num_rounds=1, match_seed=123)

    # Advance until player 1 has to act
    while state.round_state is None or state.runtime_state.current_player_id != 1:
        state, _ = AceEngine.advance(state)

    player_view = state
    legal_actions = AceEngine.get_legal_actions(state)

    print(f"Profiling 5 trials of select_action() at 300 iterations...")
    times = []
    
    for trial in range(5):
        start = time.time()
        action = agent.select_action(player_view, legal_actions)
        duration = time.time() - start
        times.append(duration)
        print(f"  Trial {trial+1}: {duration:.4f} seconds (completed 300 iterations)")

    print(f"\nResults:")
    print(f"  Average Time for 300 iterations: {np.mean(times):.4f} seconds")
    print(f"  Min Time: {np.min(times):.4f} seconds")
    print(f"  Max Time: {np.max(times):.4f} seconds")
    print(f"  Average time per iteration: {np.mean(times)/300.0*1000.0:.2f} ms")

if __name__ == "__main__":
    profile_mcts()
