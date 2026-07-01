# scripts/play/play_interactive.py
import sys
import os
import random
import time
from typing import List, Dict, Any, Type, Optional
sys.path.append('.')

import torch
torch.set_num_threads(1)  # Speed up PyTorch single-inference

from engine.rules import AceEngine, Success
from engine.types import (
    Action, StealAction, DeclineStealAction, PlayCardAction, RoundStarting,
    AwaitingStealDecision, AwaitingCardPlay, EngineState
)
from engine.card import card_to_str, get_suit, get_rank
from agents.heuristic.v1.heuristic_agent import HeuristicAgent
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.mcts.v1.mcts_agent import MCTSAgent
from agents.rl.v1.rl_agent import RLAgent
from agents.rl.v2.rl_agent_v2 import RLAgentV2
from engine.events import get_player_view

# ANSI color codes for rich console aesthetics
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

SUIT_COLORS = {
    0: CYAN,    # Spades
    1: GREEN,   # Clubs
    2: RED,     # Hearts
    3: YELLOW   # Diamonds
}

def color_card(card: int) -> str:
    """Format a card with its corresponding suit color and symbol."""
    card_s = card_to_str(card)
    s = get_suit(card)
    return f"{SUIT_COLORS[s]}{BOLD}{card_s}{RESET}"

# Wrappers for the agents to load paths and manage parameters cleanly
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

BOT_TYPES = {
    "1": ("Heuristic V1 (Baseline)", BenchmarkHeuristicAgentV1),
    "2": ("Heuristic V2 (Evolved)", BenchmarkHeuristicAgentV2),
    "3": ("MCTS Agent", BenchmarkMCTSAgent),
    "4": ("RL Agent V1 (Champion)", BenchmarkRLAgentV1),
    "5": ("RL Agent V2 (Champion)", BenchmarkRLAgentV2)
}

def print_scoreboard(state: EngineState, bots_meta: Dict[int, str]):
    """Print a clean match scoreboard."""
    print(f"\n{BOLD}{MAGENTA}=== MATCH SCOREBOARD ==={RESET}")
    for p in state.match_state.players:
        role = "Human (You)" if p.player_id == 0 else bots_meta[p.player_id]
        pts = p.half_points / 2.0
        print(f"  Player {p.player_id} ({role:<24}) | Points: {pts:<5} | Wins: {p.rounds_won} | Losses: {p.rounds_lost}")
    print("=" * 35 + "\n")

def get_input_choice(prompt: str, valid_choices: List[str], default: str) -> str:
    """Helper to get validated input choice from user."""
    while True:
        val = input(f"{prompt} (default '{default}'): ").strip()
        if not val:
            return default
        if val in valid_choices:
            return val
        print(f"Invalid input. Choose from {valid_choices}")

def main():
    print(f"{BOLD}{CYAN}================================================={RESET}")
    print(f"{BOLD}{CYAN}===      ACE CARD GAME: INTERACTIVE ARENA     ==={RESET}")
    print(f"{BOLD}{CYAN}================================================={RESET}\n")
    print("Welcome! You are Player 0. Customize your opponents and play matches.\n")

    # 1. Matches and Rounds input
    try:
        matches_str = input("Enter number of matches to play (default 1): ").strip()
        num_matches = int(matches_str) if matches_str else 1
    except ValueError:
        num_matches = 1
        print("Invalid number. Set to 1.")

    try:
        rounds_str = input("Enter number of rounds per match (default 5): ").strip()
        num_rounds = int(rounds_str) if rounds_str else 5
    except ValueError:
        num_rounds = 5
        print("Invalid number. Set to 5.")

    # 2. Opponents configuration
    print("\nHow do you want to configure your opponents?")
    print(" [1] All RL V2 Champion (Recommended)")
    print(" [2] All Heuristic V2 (Evolved)")
    print(" [3] Individual Custom (Select for each seat)")
    print(" [4] Balanced Mix (Seat 1: RL V2, Seat 2: Heuristic V2, Seat 3: MCTS)")
    
    config_choice = get_input_choice("Select option [1-4]", ["1", "2", "3", "4"], "1")
    
    bots_classes = {}
    bots_meta = {}

    if config_choice == "1":
        for i in (1, 2, 3):
            bots_classes[i] = BenchmarkRLAgentV2
            bots_meta[i] = "RL Agent V2"
    elif config_choice == "2":
        for i in (1, 2, 3):
            bots_classes[i] = BenchmarkHeuristicAgentV2
            bots_meta[i] = "Heuristic V2"
    elif config_choice == "4":
        bots_classes[1] = BenchmarkRLAgentV2
        bots_meta[1] = "RL Agent V2"
        bots_classes[2] = BenchmarkHeuristicAgentV2
        bots_meta[2] = "Heuristic V2"
        bots_classes[3] = BenchmarkMCTSAgent
        bots_meta[3] = "MCTS Agent"
    else:  # Custom Configuration
        for seat in (1, 2, 3):
            print(f"\nSelect Agent type for Player {seat}:")
            for k, (name, _) in BOT_TYPES.items():
                print(f"  [{k}] {name}")
            ch = get_input_choice(f"Choice for Player {seat}", list(BOT_TYPES.keys()), "5")
            bots_classes[seat] = BOT_TYPES[ch][1]
            bots_meta[seat] = BOT_TYPES[ch][0]

    print(f"\n{BOLD}{GREEN}Opponents configured!{RESET}")
    for seat, name in bots_meta.items():
        print(f"  Player {seat}: {name}")

    print(f"\nStarting {num_matches} matches...")
    time.sleep(1)

    for m_idx in range(num_matches):
        match_seed = random.randint(1000, 999999)
        print(f"\n{BOLD}{CYAN}================================================={RESET}")
        print(f"{BOLD}{CYAN}===             MATCH {m_idx + 1:02d} / {num_matches:02d}                  ==={RESET}")
        print(f"{BOLD}{CYAN}================================================={RESET}")
        
        # Initialize match
        state = AceEngine.create_match(
            match_id=m_idx + 1,
            num_players=4,
            num_rounds=num_rounds,
            match_seed=match_seed
        )

        # Instantiate bots
        bots = {
            seat: cls(player_id=seat, seed=match_seed + 100 + seat)
            for seat, cls in bots_classes.items()
        }

        # Start first round
        state, events = AceEngine.advance(state)
        for ev in events:
            if ev.event_type == "ROUND_STARTED":
                print(f"\n{GREEN}[Round 1 Started]{RESET}")
            elif ev.event_type == "CARDS_DEALT" and ev.payload["player_id"] == 0:
                hand_s = " ".join(color_card(c) for c in ev.payload["hand"])
                print(f"Your starting hand: [ {hand_s} ] ({len(ev.payload['hand'])} cards)\n")

        # Main game loop
        while not AceEngine.is_terminal(state):
            phase = AceEngine.get_game_phase(state)

            if isinstance(phase, RoundStarting):
                print(f"\n{BOLD}{YELLOW}--- Starting Round {phase.round_number} ---{RESET}")
                state, events = AceEngine.advance(state)
                for ev in events:
                    if ev.event_type == "CARDS_DEALT" and ev.payload["player_id"] == 0:
                        hand_s = " ".join(color_card(c) for c in ev.payload["hand"])
                        print(f"Your starting hand: [ {hand_s} ] ({len(ev.payload['hand'])} cards)\n")
                continue

            player_id = state.runtime_state.current_player_id
            if player_id is None:
                state, events = AceEngine.advance(state)
                # Print trick summaries and points updates
                for ev in events:
                    if ev.event_type == "TRICK_COMPLETED":
                        outcome = ev.payload["outcome"]
                        collector = ev.payload.get("collector_id")
                        coll_s = f"Player {collector} ({'You' if collector == 0 else bots_meta[collector]})" if collector is not None else "None"
                        print(f"{YELLOW}[Trick Completed] Outcome: {outcome} | Collector: {coll_s}{RESET}")
                    elif ev.event_type == "ROUND_COMPLETED":
                        loser = ev.payload["loser_id"]
                        loser_s = f"Player {loser} ({'You' if loser == 0 else bots_meta[loser]})"
                        print(f"\n{RED}{BOLD}[Round Over] {loser_s} lost this round and collected points!{RESET}")
                continue

            legal_actions = AceEngine.get_legal_actions(state)

            # 1. Human Decision Phase (Player 0)
            if player_id == 0:
                print(f"{BOLD}{BLUE}Your turn!{RESET}")
                print(f"Your hand: [ " + " ".join(color_card(c) for c in state.round_state.players[0].hand) + " ]")
                
                # Print active trick plays
                curr_trick = state.round_state.current_trick
                if curr_trick and curr_trick.plays:
                    trick_s = ", ".join(f"P{p.player_id}: {color_card(p.card)}" for p in curr_trick.plays)
                    print(f"Current trick plays: [ {trick_s} ]")

                # Display options
                print("Legal actions:")
                for idx, act in enumerate(legal_actions):
                    if isinstance(act, PlayCardAction):
                        print(f"  [{idx}] Play {color_card(act.card)}")
                    elif isinstance(act, StealAction):
                        print(f"  [{idx}] Steal from leader")
                    elif isinstance(act, DeclineStealAction):
                        print(f"  [{idx}] Decline Steal")

                # Input Loop
                while True:
                    try:
                        choice = input(f"Select action index [0-{len(legal_actions)-1}]: ").strip()
                        choice_idx = int(choice)
                        if 0 <= choice_idx < len(legal_actions):
                            chosen_action = legal_actions[choice_idx]
                            break
                        print("Index out of bounds.")
                    except ValueError:
                        print("Invalid input. Please enter a valid integer index.")

                print(f"You played: {chosen_action}\n")
                res = AceEngine.apply_action(state, chosen_action)
                if isinstance(res, Success):
                    state = res.new_state
                else:
                    print(f"Error applying action: {res.message}")

            # 2. Bot Decision Phase (Players 1, 2, 3)
            else:
                bot = bots[player_id]
                player_view = get_player_view(state, player_id)
                
                # Bot selects action
                chosen_action = bot.select_action(player_view, legal_actions)
                
                # Print user-friendly bot action notifications
                role_s = bots_meta[player_id]
                if isinstance(chosen_action, PlayCardAction):
                    print(f"Player {player_id} ({role_s}) plays {color_card(chosen_action.card)}")
                elif isinstance(chosen_action, StealAction):
                    print(f"Player {player_id} ({role_s}) steals the trick!")
                elif isinstance(chosen_action, DeclineStealAction):
                    print(f"Player {player_id} ({role_s}) declines to steal")
                    
                res = AceEngine.apply_action(state, chosen_action)
                if isinstance(res, Success):
                    state = res.new_state
                else:
                    print(f"Error applying bot action: {res.message}")

        # Match Over
        print_scoreboard(state, bots_meta)
        
    print(f"\n{BOLD}{GREEN}=== Tournament Over! Thank you for playing! ==={RESET}\n")

if __name__ == "__main__":
    main()
