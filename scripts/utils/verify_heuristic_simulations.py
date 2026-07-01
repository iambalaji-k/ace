# scripts/verify_heuristic_simulations.py
"""Rollout Verification Framework for calibration of HeuristicAgent.

Runs random game playout simulations from active states to compare empirical win rates
against heuristic predictions, generating a ranking calibration report.
"""

import sys
import random
from typing import List, Dict, Tuple
sys.path.append('.')

from engine.rules import AceEngine, Success, Error
from engine.types import Action, PlayCardAction, StealAction, DeclineStealAction, EngineState, AwaitingStealDecision, AwaitingCardPlay
from engine.card import card_to_str
from agents.heuristic.v1.heuristic_agent import HeuristicAgent
from agents.random.agent import RandomAgent

def run_rollout(state: EngineState, start_action: Action, num_playouts: int, player_id: int) -> float:
    """Run N random playout simulations from the given state after applying start_action."""
    wins = 0.0
    for _ in range(num_playouts):
        # Setup clean copy of the state
        sim_state = state
        
        # Apply the initial candidate action
        res = AceEngine.apply_action(sim_state, start_action)
        if isinstance(res, Error):
            continue
        sim_state = res.new_state
        
        # Instantiate random players for playout
        num_players = len(sim_state.match_state.players)
        random_agents = [RandomAgent(p) for p in range(num_players)]
        
        # Playout the rest of the game randomly
        while not AceEngine.is_terminal(sim_state):
            phase = AceEngine.get_game_phase(sim_state)
            if hasattr(phase, 'player_id'):
                p_id = phase.player_id
                legal = AceEngine.get_legal_actions(sim_state)
                # Random choice
                act = random_agents[p_id].select_action(sim_state, legal)
                res_act = AceEngine.apply_action(sim_state, act)
                if isinstance(res_act, Success):
                    sim_state = res_act.new_state
                else:
                    break
            else:
                # Auto-advance
                sim_state, _ = AceEngine.advance(sim_state)
                
        # Evaluate outcome for target player
        result = AceEngine.get_result(sim_state)
        if result:
            ranking = next((r for r in result.rankings if r.player_id == player_id), None)
            if ranking:
                # utility: 1.0 for survival (win), 0.5 for draw, 0.0 for loss
                utility = ranking.half_points  # half_points is 1.0 for round win, 0.5 for draw, 0.0 for loss
                wins += utility
                    
    return (wins / (num_playouts * 2.0)) * 100.0

def main():
    print("====================================================")
    print("===   HEURISTIC CALIBRATION & ROLLOUT FRAMEWORK   ===")
    print("====================================================\n")
    
    # 1. Create a game state in the middle of a match (try seeds until a non-terminal state is found)
    num_players = 4
    match_seed = 41
    state = None
    active_player = None
    legal_actions = []
    
    while True:
        match_seed += 1
        state = AceEngine.create_match(match_id=1, num_players=num_players, num_rounds=1, match_seed=match_seed)
        
        # Advance past round setup to the first real turn
        while True:
            phase = AceEngine.get_game_phase(state)
            if isinstance(phase, (AwaitingStealDecision, AwaitingCardPlay)):
                break
            state, _ = AceEngine.advance(state)
        
        # Let the game progress a few tricks to get a complex middle-game state
        is_valid = True
        for _ in range(4):
            legal = AceEngine.get_legal_actions(state)
            if not legal:
                is_valid = False
                break
            act = random.choice(legal)
            res = AceEngine.apply_action(state, act)
            if isinstance(res, Success):
                state = res.new_state
                
            # Handle intermediate auto-advances
            while True:
                ph = AceEngine.get_game_phase(state)
                if isinstance(ph, (AwaitingStealDecision, AwaitingCardPlay)) or AceEngine.is_terminal(state):
                    break
                state, _ = AceEngine.advance(state)
                
            if AceEngine.is_terminal(state):
                is_valid = False
                break
                
        if not is_valid:
            continue
            
        phase = AceEngine.get_game_phase(state)
        if hasattr(phase, 'player_id') and phase.player_id is not None:
            active_player = phase.player_id
            legal_actions = AceEngine.get_legal_actions(state)
            if len(legal_actions) >= 2:
                print(f"Setup complete! Found non-terminal state at seed={match_seed}.")
                break
                
    print(f"Analyzing {len(legal_actions)} options for Player {active_player} at mid-game state...\n")
    
    # 2. Evaluate with Heuristic Agent
    agent = HeuristicAgent(player_id=active_player, seed=42)
    evals = agent.evaluate_legal_actions(state, legal_actions)
    
    # Map actions to heuristic score
    heuristic_map = {ev.action: ev for ev in evals}
    
    # 3. Run Rollouts (50 simulations per legal action)
    rollout_runs = 100
    print(f"Simulating {rollout_runs} completions for each of the {len(legal_actions)} actions...")
    
    results = []
    for act in legal_actions:
        action_name = ""
        if isinstance(act, PlayCardAction):
            raw_card = card_to_str(act.card)
            action_name = f"Play {raw_card}".replace("♠", "S").replace("♣", "C").replace("♥", "H").replace("♦", "D")
        elif isinstance(act, StealAction):
            action_name = "Steal"
        elif isinstance(act, DeclineStealAction):
            action_name = "Decline"
            
        win_rate = run_rollout(state, act, rollout_runs, active_player)
        h_eval = heuristic_map.get(act)
        h_score = h_eval.total_score if h_eval else 0.0
        
        results.append((action_name, h_score, win_rate, h_eval))
        
    # Sort results by rollout win rate (actual strength) descending
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Print calibration report
    print("\n" + "="*80)
    print(f"=== ROLLOUT CALIBRATION REPORT FOR PLAYER {active_player} ===")
    print("="*80)
    print(f"{'Action':<15} | {'Heuristic Score':<15} | {'Actual Win Rate':<15} | {'Active Rules'}")
    print("-"*80)
    
    for action_name, h_score, win_rate, h_eval in results:
        rules_str = ""
        if h_eval and h_eval.breakdown:
            rules_str = ", ".join(f"{k}:{v:+.0f}" for k, v in h_eval.breakdown.items() if abs(v) > 0.1)
        else:
            rules_str = "None"
            
        print(f"{action_name:<15} | {h_score:<15.1f} | {win_rate:<13.1f}% | {rules_str}")
    print("="*80)
    print("\nInterpretation:")
    print("- If the actual win rates rank in the same order as heuristic scores, the weights are calibrated!")
    print("- Heuristic scores that have a mismatch with win rates indicate rules that need adjustment.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
