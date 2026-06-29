# scripts/play_against_rl.py
"""Interactive CLI interface to play against the trained RLAgent.

Allows a human player (Player 0) to play a full match against three RLAgent bots.
"""

import sys
import random
from typing import List, Sequence
sys.path.append('.')

from engine.rules import AceEngine, Success, Error
from engine.types import (
    Action, StealAction, DeclineStealAction, PlayCardAction, RoundStarting,
    AwaitingStealDecision, AwaitingCardPlay, EngineState
)
from engine.card import card_to_str, get_suit, get_rank
from engine.rl_agent import RLAgent
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

def print_scoreboard(state: EngineState):
    """Print a clean match scoreboard."""
    print(f"\n{BOLD}{MAGENTA}=== MATCH SCOREBOARD ==={RESET}")
    for p in state.match_state.players:
        role = "Human (You)" if p.player_id == 0 else f"Bot RL {p.player_id}"
        pts = p.half_points / 2.0
        print(f"  Player {p.player_id} ({role:<17}) | Points: {pts:<5} | Wins: {p.rounds_won} | Losses: {p.rounds_lost}")
    print("=" * 24 + "\n")

def main():
    print(f"{BOLD}{CYAN}================================================={RESET}")
    print(f"{BOLD}{CYAN}===   ACE CARD GAME: HUMAN VS RL CHAMPION BOTS   ==={RESET}")
    print(f"{BOLD}{CYAN}================================================={RESET}\n")
    print("Rules: You are Player 0. Players 1, 2, and 3 are RL champion bots.")
    print("The goal is to empty your hand. Don't be the last active player!\n")

    checkpoint_path = "engine/rl_champion.pt"
    if not os.path.exists(checkpoint_path):
        print(f"{YELLOW}Warning: Checkpoint '{checkpoint_path}' not found. Bots will play randomly!{RESET}")
        print("Run 'python scripts/train_self_play.py' first to train the champion.\n")

    # Game config
    num_rounds = 3
    num_players = 4
    match_seed = random.randint(1, 100000)

    # Initialize match
    state = AceEngine.create_match(
        match_id=1,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=match_seed
    )

    # Instantiate RL bots
    bots = {
        1: RLAgent(player_id=1, checkpoint_path=checkpoint_path, explore=False),
        2: RLAgent(player_id=2, checkpoint_path=checkpoint_path, explore=False),
        3: RLAgent(player_id=3, checkpoint_path=checkpoint_path, explore=False),
    }

    # Start first round
    state, events = AceEngine.advance(state)
    for ev in events:
        if ev.event_type == "ROUND_STARTED":
            print(f"{GREEN}[Round 1 Started]{RESET}")
        elif ev.event_type == "CARDS_DEALT" and ev.payload["player_id"] == 0:
            hand_s = " ".join(color_card(c) for c in ev.payload["hand"])
            print(f"Your starting hand: [ {hand_s} ] ({len(ev.payload['hand'])} cards)\n")

    # Main game loop
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)

        # Check for auto-advances
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
            # Print trick summaries and points updates from advanced events
            for ev in events:
                if ev.event_type == "TRICK_COMPLETED":
                    outcome = ev.payload["outcome"]
                    collector = ev.payload.get("collector_id")
                    coll_s = f"Player {collector}" if collector is not None else "None"
                    print(f"{YELLOW}[Trick Completed] Outcome: {outcome} | Collector: {coll_s}{RESET}")
                elif ev.event_type == "ROUND_COMPLETED":
                    loser = ev.payload["loser_id"]
                    print(f"\n{RED}{BOLD}[Round Over] Player {loser} lost this round and collected points!{RESET}")
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

            print(f"You chose action: {chosen_action}\n")
            res = AceEngine.apply_action(state, chosen_action)
            if isinstance(res, Success):
                state = res.new_state
            else:
                print(f"Error applying action: {res.message}")

        # 2. Bot Decision Phase (Players 1, 2, 3)
        else:
            bot = bots[player_id]
            player_view = get_player_view(state, player_id)
            
            # Bot selects action using the loaded RL model
            chosen_action = bot.select_action(player_view, legal_actions)
            
            # Print user-friendly bot action notifications
            if isinstance(chosen_action, PlayCardAction):
                print(f"Player {player_id} plays {color_card(chosen_action.card)}")
            elif isinstance(chosen_action, StealAction):
                print(f"Player {player_id} steals the trick!")
            elif isinstance(chosen_action, DeclineStealAction):
                print(f"Player {player_id} declines to steal")
                
            res = AceEngine.apply_action(state, chosen_action)
            if isinstance(res, Success):
                state = res.new_state
            else:
                print(f"Error applying bot action: {res.message}")

    # Game Over
    print_scoreboard(state)
    print(f"{BOLD}{GREEN}=== Game Over! Thank you for playing! ==={RESET}\n")

if __name__ == "__main__":
    import os
    main()
