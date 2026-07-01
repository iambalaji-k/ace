# scripts/profile_engine.py
"""Profiling script to measure execution times of core engine components."""

import sys
import time
import random
from typing import List, Dict
sys.path.append('.')

from engine.rules import AceEngine, Success
from engine.types import EngineState
from agents.heuristic.v1.heuristic_agent import CardTracker
from agents.rl.v2.encoder_v2 import encode_state_v2
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.mcts.v1.mcts_agent import MCTSAgent

# Global timers dictionary
TIMERS: Dict[str, float] = {
    "total_match": 0.0,
    "encode_state_v2": 0.0,
    "tracker_reconstruct": 0.0,
    "heuristic_select": 0.0,
    "mcts_select": 0.0
}
COUNTS: Dict[str, int] = {
    "total_match": 0,
    "encode_state_v2": 0,
    "tracker_reconstruct": 0,
    "heuristic_select": 0,
    "mcts_select": 0
}

# Monkey patch tracker reconstruct to time it
orig_reconstruct = CardTracker.reconstruct
def patched_reconstruct(self, viewer_id, round_state, match_state):
    start = time.perf_counter()
    res = orig_reconstruct(self, viewer_id, round_state, match_state)
    TIMERS["tracker_reconstruct"] += time.perf_counter() - start
    COUNTS["tracker_reconstruct"] += 1
    return res
CardTracker.reconstruct = patched_reconstruct

# Monkey patch encoder to time it
orig_encode = encode_state_v2
def patched_encode(state, player_id):
    start = time.perf_counter()
    res = orig_encode(state, player_id)
    TIMERS["encode_state_v2"] += time.perf_counter() - start
    COUNTS["encode_state_v2"] += 1
    return res
encode_state_v2 = patched_encode

def main():
    print("====================================================")
    print("===            ENGINE PROFILING RUN              ===")
    print("====================================================\n")

    num_matches = 15
    num_players = 4
    num_rounds = 3
    base_seed = 777

    # Instantiate agents
    # Let's seat 2 HeuristicAgentV2, 1 MCTSAgent (low iterations), 1 random/heuristic agent
    heur_agent = HeuristicAgentV2(player_id=0)
    mcts_agent = MCTSAgent(player_id=1, max_iterations=20, time_limit=0.05)
    bots = {
        0: heur_agent,
        1: mcts_agent,
        2: HeuristicAgentV2(player_id=2),
        3: HeuristicAgentV2(player_id=3),
    }

    start_total = time.perf_counter()
    for m in range(num_matches):
        match_start = time.perf_counter()
        state = AceEngine.create_match(
            match_id=m,
            num_players=num_players,
            num_rounds=num_rounds,
            match_seed=base_seed + m
        )
        state, _ = AceEngine.advance(state)

        while not AceEngine.is_terminal(state):
            player_id = state.runtime_state.current_player_id
            if player_id is None:
                state, _ = AceEngine.advance(state)
                continue

            legal_actions = AceEngine.get_legal_actions(state)
            agent = bots[player_id]

            agent_start = time.perf_counter()
            try:
                action = agent.select_action(state, legal_actions)
            except Exception as e:
                print(f"\n[CRASH] Exception occurred during select_action for player {player_id}:")
                print(e)
                # Print debug info
                tracker = CardTracker(num_players)
                tracker.reconstruct(player_id, state.round_state, state.match_state)
                print("\n--- CardTracker Debug ---")
                print(f"card_locations: {tracker.card_locations}")
                print(f"player_known_cards: {tracker.player_known_cards}")
                print(f"discards: {tracker.discards}")
                # Print hands in state
                print("\n--- State Hands ---")
                for p in range(num_players):
                    print(f"Player {p} hand: {state.round_state.players[p].hand}")
                print(f"Discard pile: {state.round_state.discard_pile}")
                if state.round_state.current_trick:
                    print(f"Current trick plays: {[play.card for play in state.round_state.current_trick.plays]}")
                raise e
            agent_dur = time.perf_counter() - agent_start

            if player_id == 1:
                TIMERS["mcts_select"] += agent_dur
                COUNTS["mcts_select"] += 1
            else:
                TIMERS["heuristic_select"] += agent_dur
                COUNTS["heuristic_select"] += 1

            res = AceEngine.apply_action(state, action)
            if isinstance(res, Success):
                state = res.new_state
                state, _ = AceEngine.advance(state)
            else:
                break

        TIMERS["total_match"] += time.perf_counter() - match_start
        COUNTS["total_match"] += 1

    total_time = time.perf_counter() - start_total

    print("=" * 60)
    print(f"PROFILING RESULTS: {num_matches} Matches ({num_matches * num_rounds} Rounds)")
    print(f"Total Wall Time: {total_time:.2f} seconds")
    print("=" * 60)
    print(f"{'Component':<25} | {'Count':<8} | {'Total Time (s)':<15} | {'Avg Time (ms)':<15}")
    print("-" * 65)
    for name in ["total_match", "encode_state_v2", "tracker_reconstruct", "heuristic_select", "mcts_select"]:
        c = COUNTS[name]
        t = TIMERS[name]
        avg = (t / c * 1000) if c > 0 else 0.0
        print(f"{name:<25} | {c:<8} | {t:<15.4f} | {avg:<15.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
