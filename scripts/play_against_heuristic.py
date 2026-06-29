# scripts/play_against_heuristic.py
"""Interactive CLI interface to play against the HeuristicAgent.

Allows a human player (Player 0) to play a full match against three HeuristicAgent bots.
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
from engine.heuristic_agent import HeuristicAgent
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
        role = "Human (You)" if p.player_id == 0 else f"Bot Heuristic {p.player_id}"
        pts = p.half_points / 2.0
        print(f"  Player {p.player_id} ({role:<17}) | Points: {pts:<5} | Wins: {p.rounds_won} | Losses: {p.rounds_lost}")
    print("=" * 24 + "\n")

def main():
    print(f"{BOLD}{CYAN}================================================={RESET}")
    # Display ASCII-style game title
    print(f"{BOLD}{CYAN}===   ACE CARD GAME: HUMAN VS HEURISTIC BOTS  ==={RESET}")
    print(f"{BOLD}{CYAN}================================================={RESET}\n")
    print("Rules: You are Player 0. Players 1, 2, and 3 are Heuristic bots.")
    print("The goal is to empty your hand. Don't be the last active player!\n")

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

    # Instantiate bots
    bots = {
        1: HeuristicAgent(player_id=1, seed=match_seed + 101),
        2: HeuristicAgent(player_id=2, seed=match_seed + 102),
        3: HeuristicAgent(player_id=3, seed=match_seed + 103),
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
        assert player_id is not None

        # Display trick status before player acts
        round_st = state.round_state
        assert round_st is not None
        curr_trick = round_st.current_trick

        if player_id == 0:
            # --- HUMAN TURN ---
            print("-" * 50)
            # Display current trick plays so far
            if curr_trick and curr_trick.plays:
                plays_s = " -> ".join(f"P{p.player_id}:{color_card(p.card)}" for p in curr_trick.plays)
                print(f"Current Trick Plays: {plays_s}")
            
            # Display hand
            my_hand = round_st.players[0].hand
            hand_s = " ".join(color_card(c) for c in my_hand)
            print(f"Your Hand: [ {hand_s} ] ({len(my_hand)} cards)")

            # Describe options with explainability breakdown
            legal_actions = AceEngine.get_legal_actions(state)
            player_view = get_player_view(state, 0)
            advisor = HeuristicAgent(player_id=0, seed=match_seed)
            evals = advisor.evaluate_legal_actions(player_view, legal_actions)
            eval_map = {ev.action: ev for ev in evals}

            print(f"\n{BOLD}Your Options & Heuristic Agent Advice:{RESET}")
            for idx, act in enumerate(legal_actions):
                if isinstance(act, StealAction):
                    action_str = f"{YELLOW}Steal{RESET} from target player P{phase.steal_target if isinstance(phase, AwaitingStealDecision) else ''}"
                elif isinstance(act, DeclineStealAction):
                    action_str = "Decline Steal"
                elif isinstance(act, PlayCardAction):
                    action_str = f"Play Card {color_card(act.card)}"
                else:
                    action_str = str(act)
                
                print(f"  [{idx}] {action_str}")
                if act in eval_map:
                    ev_obj = eval_map[act]
                    bd_parts = [f"{k}: {v:+.1f}" for k, v in ev_obj.breakdown.items()]
                    bd_str = ", ".join(bd_parts) if bd_parts else "No rules active"
                    print(f"      AI Score: {BOLD}{ev_obj.total_score:+.1f}{RESET} ( {bd_str} )")

            # Loop until valid input
            while True:
                try:
                    choice = input(f"\nEnter choice index (0 to {len(legal_actions)-1}) or 'q' to quit: ").strip()
                    if choice.lower() == 'q':
                        print("Exiting game.")
                        return
                    choice_idx = int(choice)
                    if 0 <= choice_idx < len(legal_actions):
                        action = legal_actions[choice_idx]
                        break
                    print("Error: Index out of range.")
                except ValueError:
                    print("Error: Please enter a valid number.")

            # Apply human action
            res = AceEngine.apply_action(state, action)
            assert isinstance(res, Success)
            state = res.new_state

            # Print events resulting from play
            for ev in res.events:
                if ev.event_type == "CARD_PLAYED":
                    print(f"  You played {color_card(ev.payload['card'])}")
                elif ev.event_type == "TRICK_COMPLETED":
                    outcome = ev.payload['outcome']
                    collector = f"Player {ev.payload['collector_id']}" if ev.payload['collector_id'] is not None else "Discarded"
                    color = RED if outcome == "INTERRUPTED" else GREEN
                    print(f"  {color}Trick Completed: {outcome}. Collector/Result: {collector}{RESET}")
                elif ev.event_type == "STEAL_EXECUTED":
                    print(f"  {YELLOW}You stole {len(ev.payload['cards'])} cards from Player {ev.payload['victim_id']}!{RESET}")
                elif ev.event_type == "PLAYER_INACTIVE":
                    print(f"  {GREEN}Player {ev.payload['player_id']} went Inactive (Reason: {ev.payload['reason']}){RESET}")
                elif ev.event_type == "ROUND_ENDED":
                    result_s = f"Loser: Player {ev.payload['loser_id']}" if not ev.payload['is_draw'] else "Draw"
                    print(f"\n{BOLD}{RED}[Round Ended] {result_s}{RESET}")
                    print_scoreboard(state)

        else:
            # --- BOT TURN ---
            player_view = get_player_view(state, player_id)
            legal_actions = AceEngine.get_legal_actions(state)
            bot = bots[player_id]

            action = bot.select_action(player_view, legal_actions)
            res = AceEngine.apply_action(state, action)
            assert isinstance(res, Success)
            state = res.new_state

            # Print what the bot did
            for ev in res.events:
                if ev.event_type == "CARD_PLAYED":
                    print(f"Bot P{player_id} played {color_card(ev.payload['card'])}")
                elif ev.event_type == "TRICK_COMPLETED":
                    outcome = ev.payload['outcome']
                    collector = f"Player {ev.payload['collector_id']}" if ev.payload['collector_id'] is not None else "Discarded"
                    color = RED if outcome == "INTERRUPTED" else GREEN
                    print(f"{color}Trick Completed: {outcome}. Collector/Result: {collector}{RESET}")
                elif ev.event_type == "STEAL_EXECUTED":
                    role = "you" if ev.payload['victim_id'] == 0 else f"Bot P{ev.payload['victim_id']}"
                    print(f"{YELLOW}Bot P{player_id} stole from {role}!{RESET}")
                elif ev.event_type == "PLAYER_INACTIVE":
                    role = "you" if ev.payload['player_id'] == 0 else f"Bot P{ev.payload['player_id']}"
                    print(f"{GREEN}Player {role} went Inactive (Reason: {ev.payload['reason']}){RESET}")
                elif ev.event_type == "ROUND_ENDED":
                    result_s = f"Loser: Player {ev.payload['loser_id']}" if not ev.payload['is_draw'] else "Draw"
                    print(f"\n{BOLD}{RED}[Round Ended] {result_s}{RESET}")
                    print_scoreboard(state)

    # Match over
    print(f"\n{BOLD}{MAGENTA}====================================={RESET}")
    print(f"{BOLD}{MAGENTA}=== MATCH COMPLETED - FINAL RANKINGS ==={RESET}")
    print(f"{BOLD}{MAGENTA}====================================={RESET}")
    result = AceEngine.get_result(state)
    assert result is not None
    for r in result.rankings:
        role = "Human (You)" if r.player_id == 0 else f"Bot P{r.player_id}"
        pts = r.half_points / 2.0
        print(f"Rank {r.rank}: {role:<15} | Score: {pts:<5} | Record (W/L/D): {r.rounds_won}/{r.rounds_lost}/{r.rounds_drawn}")

if __name__ == '__main__':
    main()
