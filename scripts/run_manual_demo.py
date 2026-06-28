# scripts/run_manual_demo.py
"""Interactive CLI demo for manual simulation of the Ace Engine.

Allows step-by-step game execution with custom player counts, round counts, seeds,
and custom card hand rigging for manual testing of specific rules and suit-breaks.
"""

import sys
import random
from dataclasses import replace
sys.path.append('.')

from engine.rules import AceEngine, Success, Error, get_immediate_active_left
from engine.types import (
    StealAction, DeclineStealAction, PlayCardAction, RoundStarting,
    AwaitingStealDecision, AwaitingCardPlay, EngineState, RoundState,
    RoundPlayerState, TrickState, Event
)
from engine.card import card_to_str, str_to_card, sort_cards


def get_input_in_range(prompt: str, min_val: int, max_val: int, default: int) -> int:
    """Helper to get integer input within a specified range."""
    while True:
        try:
            val_str = input(f"{prompt} (default {default}): ").strip()
            if not val_str:
                return default
            val = int(val_str)
            if min_val <= val <= max_val:
                return val
            print(f"Error: Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Error: Please enter a valid integer.")


def parse_card_list(input_str: str, assigned: set) -> list[int]:
    """Parse a comma/space separated list of card names (e.g. 'A♠, 7♦') or card IDs (0-51)."""
    if not input_str.strip():
        return []
    cards = []
    # Replace commas with spaces and split
    parts = input_str.replace(",", " ").split()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.isdigit():
            card_id = int(p)
            if 0 <= card_id < 52:
                if card_id in assigned:
                    print(f"Warning: Card ID {card_id} is already assigned. Ignored.")
                else:
                    cards.append(card_id)
            else:
                print(f"Warning: Card ID {card_id} is out of range (0-51). Ignored.")
        else:
            try:
                card_id = str_to_card(p)
                if card_id in assigned:
                    print(f"Warning: Card {p} (ID: {card_id}) is already assigned. Ignored.")
                else:
                    cards.append(card_id)
            except ValueError:
                print(f"Warning: Could not parse card string '{p}'. Ignored.")
    return cards


def setup_custom_deal(state: EngineState, round_num: int, num_players: int) -> tuple[EngineState, list[Event]]:
    """Prompt the user to assign custom hands to players, filling the rest randomly."""
    print("\n--- Custom Hand Configuration ---")
    print("Enter cards for each player. Format: 'A♠, 10♣, 2♦' or IDs '0, 17, 51'.")
    print("Press Enter to skip card assignment (remaining cards will be dealt randomly).")

    assigned = set()
    custom_hands = [[] for _ in range(num_players)]

    for i in range(num_players):
        while True:
            cards_str = input(f"  Enter cards for Player {i}: ").strip()
            parsed = parse_card_list(cards_str, assigned)
            for c in parsed:
                custom_hands[i].append(c)
                assigned.add(c)
            break

    # Calculate default deal hand sizes
    base_size = 52 // num_players
    extra = 52 % num_players
    target_sizes = [base_size + (1 if idx < extra else 0) for idx in range(num_players)]

    # Fill remaining unassigned cards
    remaining = list(set(range(52)) - assigned)
    random.shuffle(remaining)

    for i in range(num_players):
        needed = target_sizes[i] - len(custom_hands[i])
        if needed > 0:
            for _ in range(needed):
                if remaining:
                    c = remaining.pop()
                    custom_hands[i].append(c)
                    assigned.add(c)

    # Distribute any leftovers if players entered more than their target size
    while remaining:
        c = remaining.pop()
        min_p = min(range(num_players), key=lambda idx: len(custom_hands[idx]))
        custom_hands[min_p].append(c)
        assigned.add(c)

    # Sort all hands canonically
    for i in range(num_players):
        custom_hands[i] = sort_cards(custom_hands[i])

    # Lead player is whoever holds A♠ (card 0)
    lead_player_id = None
    for i in range(num_players):
        if 0 in custom_hands[i]:
            lead_player_id = i
            break
    if lead_player_id is None:
        lead_player_id = 0

    # Build Events
    events = []
    seq = state.runtime_state.action_sequence_number
    ts = seq

    if round_num == 1:
        events.append(Event(
            sequence=len(events) + seq,
            event_type="MATCH_STARTED",
            round_number=None,
            trick_number=None,
            payload={
                "match_id": state.match_state.match_id,
                "num_players": num_players,
                "num_rounds": state.match_state.num_rounds,
                "match_seed": state.match_state.match_seed
            },
            timestamp=ts
        ))

    events.append(Event(
        sequence=len(events) + seq,
        event_type="ROUND_STARTED",
        round_number=round_num,
        trick_number=None,
        payload={
            "round_number": round_num,
            "round_seed": 0
        },
        timestamp=ts
    ))

    for i in range(num_players):
        events.append(Event(
            sequence=len(events) + seq,
            event_type="CARDS_DEALT",
            round_number=round_num,
            trick_number=None,
            payload={
                "player_id": i,
                "hand": custom_hands[i],
                "hand_size": len(custom_hands[i])
            },
            timestamp=ts
        ))

    events.append(Event(
        sequence=len(events) + seq,
        event_type="TRICK_STARTED",
        round_number=round_num,
        trick_number=1,
        payload={
            "trick_number": 1,
            "lead_player_id": lead_player_id
        },
        timestamp=ts
    ))

    # Initialize States
    round_players = [
        RoundPlayerState(
            player_id=i,
            hand=custom_hands[i],
            is_active=True,
            is_round_winner=False,
            is_round_loser=False
        ) for i in range(num_players)
    ]

    trick = TrickState(
        trick_number=1,
        lead_player_id=lead_player_id,
        lead_suit=None,
        plays=[],
        status="STEAL_PHASE",
        steals=[]
    )

    round_state = RoundState(
        round_number=round_num,
        round_seed=0,
        players=round_players,
        active_player_ids=list(range(num_players)),
        current_trick=trick,
        trick_history=[],
        lead_player_id=lead_player_id,
        discard_pile=[],
        status="IN_PROGRESS"
    )

    new_match_status = "IN_PROGRESS" if state.match_state.status == "INIT" else state.match_state.status
    new_match_state = replace(state.match_state, status=new_match_status)

    steal_target = get_immediate_active_left(lead_player_id, round_state.active_player_ids, num_players)
    next_phase = AwaitingStealDecision(player_id=lead_player_id, steal_target=steal_target)
    legal_actions: list[Action] = [
        StealAction(player_id=lead_player_id),
        DeclineStealAction(player_id=lead_player_id)
    ]

    new_runtime_state = replace(
        state.runtime_state,
        current_phase=next_phase,
        current_player_id=lead_player_id,
        pending_legal_actions=legal_actions
    )

    new_state = EngineState(
        match_state=new_match_state,
        round_state=round_state,
        runtime_state=new_runtime_state
    )

    # Invariants Validation
    from engine.invariants import validate_invariants
    violations = validate_invariants(new_state)
    if violations:
        raise RuntimeError(f"Custom configurations violated invariants: {violations}")

    return new_state, events


def main():
    print("=============================================")
    print("=== Welcome to the Ace Engine Manual Demo ===")
    print("=============================================\n")

    # Get match configurations
    num_players = get_input_in_range("Enter number of players (3-6)", 3, 6, 4)
    num_rounds = get_input_in_range("Enter number of rounds (1-100)", 1, 100, 2)

    seed_str = input("Enter match seed (integer, or press Enter for random): ").strip()
    if seed_str:
        try:
            match_seed = int(seed_str)
        except ValueError:
            print("Invalid seed. Using default 42.")
            match_seed = 42
    else:
        match_seed = random.randint(1, 100000)

    # Ask if the user wants custom card configurations
    custom_choice = input("\nDo you want to customize player hands for this match? (y/n): ").strip().lower() == 'y'

    print(f"\nInitializing a {num_players}-player match for {num_rounds} rounds with seed {match_seed}...")

    # Create the match state
    state = AceEngine.create_match(
        match_id=1,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=match_seed
    )

    # Initialize Round 1 (Custom vs Standard dealing)
    if custom_choice:
        state, events = setup_custom_deal(state, 1, num_players)
    else:
        state, events = AceEngine.advance(state)

    for ev in events:
        if ev.event_type == "MATCH_STARTED":
            print(f"\n[Match Start] ID: {ev.payload['match_id']} | Players: {ev.payload['num_players']} | Seed: {ev.payload['match_seed']}")
        elif ev.event_type == "ROUND_STARTED":
            print(f"\n[Round 1 Start] Seed: {ev.payload['round_seed']}")
        elif ev.event_type == "CARDS_DEALT":
            hands_str = ", ".join(card_to_str(c) for c in ev.payload['hand'])
            print(f"  Dealt Player {ev.payload['player_id']} hand: [{hands_str}]")
        else:
            print(f"  [Event] {ev.event_type}: {ev.payload}")

    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)

        # Check if the engine needs an auto-advance (e.g. starting a new round)
        if isinstance(phase, RoundStarting):
            print(f"\n" + "-" * 60)
            print(f"--- Starting Round {phase.round_number} ---")
            print("-" * 60)
            
            if custom_choice:
                state, events = setup_custom_deal(state, phase.round_number, num_players)
            else:
                state, events = AceEngine.advance(state)
                
            for ev in events:
                if ev.event_type == "ROUND_STARTED":
                    print(f"\n[Round {phase.round_number} Start] Seed: {ev.payload['round_seed']}")
                elif ev.event_type == "ACES_RESERVED":
                    print(f"  [Reserved Aces] Player {ev.payload['player_id']} holds {ev.payload['count']} reserved aces")
                elif ev.event_type == "CARDS_DEALT":
                    hands_str = ", ".join(card_to_str(c) for c in ev.payload['hand'])
                    print(f"  Dealt Player {ev.payload['player_id']} hand: [{hands_str}]")
                else:
                    print(f"  [Event] {ev.event_type}: {ev.payload}")
            continue

        print("\n" + "=" * 60)
        # Display player hands and active statuses
        print("Current Hands:")
        assert state.round_state is not None
        for p in state.round_state.players:
            hand_str = ", ".join(card_to_str(c) for c in p.hand)
            active_str = "Active" if p.is_active else "Inactive (Winner)"
            m_player = next(mp for mp in state.match_state.players if mp.player_id == p.player_id)
            losses_str = f"Losses: {m_player.consecutive_loss_count}"
            print(f"  Player {p.player_id} ({active_str:^17}) [{losses_str}]: [{hand_str}]")

        # Describe the decision requirements
        if isinstance(phase, AwaitingStealDecision):
            victim_hand_size = len(state.round_state.players[phase.steal_target].hand)
            print(f"\nSteal Phase: Player {phase.player_id} is lead.")
            print(f"Target: Player {phase.steal_target} (holds {victim_hand_size} cards)")
        elif isinstance(phase, AwaitingCardPlay):
            suit_names = ["Spade ♠", "Club ♣", "Heart ♥", "Diamond ♦"]
            lead_suit_str = suit_names[phase.lead_suit] if phase.lead_suit is not None else "Any"
            print(f"\nPlay Phase: Player {phase.player_id}'s turn.")
            print(f"Lead Suit: {lead_suit_str:<10} | Must follow: {phase.must_follow}")

        # Gather and list all legal actions
        legal = AceEngine.get_legal_actions(state)
        print("\nLegal Actions:")
        for idx, act in enumerate(legal):
            action_str = ""
            if isinstance(act, StealAction):
                assert isinstance(phase, AwaitingStealDecision)
                action_str = f"Steal from Player {phase.steal_target}"
            elif isinstance(act, DeclineStealAction):
                action_str = "Decline Steal"
            elif isinstance(act, PlayCardAction):
                action_str = f"Play Card {card_to_str(act.card)} (ID: {act.card})"
            else:
                action_str = str(act)
            print(f"  [{idx}] {action_str}")

        # Get choice input
        try:
            choice = input(f"\nChoose action index (0 to {len(legal)-1}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Exiting demo.")
                break
            choice_idx = int(choice)
            if not 0 <= choice_idx < len(legal):
                print("Invalid index. Try again.")
                continue
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid index number.")
            continue

        # Apply the chosen action
        action = legal[choice_idx]
        res = AceEngine.apply_action(state, action)

        if isinstance(res, Error):
            print(f"\n[Error] {res.message}")
        else:
            state = res.new_state
            print(f"\nAction Applied: {action}")
            for ev in res.events:
                if ev.event_type == "CARD_PLAYED":
                    print(f"  [Event] Player {ev.payload['player_id']} played {card_to_str(ev.payload['card'])}")
                elif ev.event_type == "TRICK_COMPLETED":
                    outcome = ev.payload['outcome']
                    collector = f"Player {ev.payload['collector_id']}" if ev.payload['collector_id'] is not None else "Discarded"
                    print(f"  [Event] Trick Completed: {outcome}. Result: {collector}")
                elif ev.event_type == "STEAL_EXECUTED":
                    cards_str = ", ".join(card_to_str(c) for c in ev.payload['cards'])
                    print(f"  [Event] Player {ev.payload['stealer_id']} stole cards [{cards_str}] from Player {ev.payload['victim_id']}")
                elif ev.event_type == "PLAYER_INACTIVE":
                    print(f"  [Event] Player {ev.payload['player_id']} went Inactive (Reason: {ev.payload['reason']})")
                elif ev.event_type == "ROUND_ENDED":
                    result_str = f"Loser: Player {ev.payload['loser_id']}" if not ev.payload['is_draw'] else "Draw"
                    print(f"  [Event] Round {ev.payload['round_number']} Ended. Result: {result_str}")
                else:
                    print(f"  [Event] {ev.event_type}: {ev.payload}")

    # Display final scoreboard on match completion
    if AceEngine.is_terminal(state):
        print("\n" + "=" * 60)
        print("=== Match Completed ===")
        result = AceEngine.get_result(state)
        assert result is not None
        for r in result.rankings:
            print(f"Rank {r.rank}: Player {r.player_id} | Points: {r.half_points / 2:<5} | Win/Loss/Draw: {r.rounds_won}/{r.rounds_lost}/{r.rounds_drawn}")


if __name__ == '__main__':
    main()
