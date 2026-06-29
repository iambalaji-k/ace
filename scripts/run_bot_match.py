# scripts/run_bot_match.py
"""Automated match viewer for bot vs. bot simulations.

Enables watching 4 random bots play a game of Ace step-by-step in the terminal,
with custom parameters and execution delay.
"""

import argparse
import time

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, DeclineStealAction, StealAction, PlayCardAction
from engine.agent import RandomAgent
from engine.events import get_player_view
from engine.card import card_to_str, sort_cards


import sys


def main() -> None:
    # Reconfigure stdout to support UTF-8 card symbols on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        import typing
        typing.cast(typing.Any, sys.stdout).reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Watch automated bots play Ace.")
    parser.add_argument("--players", type=int, default=4, help="Number of players (3-6)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Match seed")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between turns")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Starting Bot Match: {args.players} Players, {args.rounds} Rounds (Seed {args.seed})")
    print("=" * 60)

    # 1. Initialize match state
    state = AceEngine.create_match(
        match_id=1,
        num_players=args.players,
        num_rounds=args.rounds,
        match_seed=args.seed
    )

    # 2. Setup agents
    agents = [
        RandomAgent(player_id=i, seed=args.seed + 100 + i)
        for i in range(args.players)
    ]

    # Start Round 1
    state, _ = AceEngine.advance(state)
    round_counter = 1
    print(f"\n--- Starting Round {round_counter} ---")
    assert state.round_state is not None
    for p in state.round_state.players:
        hand_str = " ".join(card_to_str(c) for c in sort_cards(list(p.hand)))
        reserved_aces = state.match_state.players[p.player_id].consecutive_loss_count
        loss_suffix = f" (holding {reserved_aces} reserved aces)" if reserved_aces > 0 else ""
        print(f"Player {p.player_id}: [{hand_str}]{loss_suffix}")
    print("-" * 40)

    # 3. Main execution loop
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)

        # Handle Round transitions
        if isinstance(phase, RoundStarting):
            round_counter += 1
            print(f"\n--- Starting Round {round_counter} ---")
            state, _ = AceEngine.advance(state)
            
            # Print initial hands
            assert state.round_state is not None
            for p in state.round_state.players:
                hand_str = " ".join(card_to_str(c) for c in sort_cards(list(p.hand)))
                reserved_aces = state.match_state.players[p.player_id].consecutive_loss_count
                loss_suffix = f" (holding {reserved_aces} reserved aces)" if reserved_aces > 0 else ""
                print(f"Player {p.player_id}: [{hand_str}]{loss_suffix}")
            print("-" * 40)
            time.sleep(args.delay * 2)
            continue

        player_id = state.runtime_state.current_player_id
        assert player_id is not None

        # Mask state details for the current active player view
        player_view = get_player_view(state, player_id)
        legal_actions = AceEngine.get_legal_actions(state)

        # Get choice from agent
        agent = agents[player_id]
        action = agent.select_action(player_view, legal_actions)

        # Print action description
        if isinstance(action, DeclineStealAction):
            print(f"Player {player_id} declines to steal.")
        elif isinstance(action, StealAction):
            # Target is the immediate active player to the left
            assert state.round_state is not None
            from engine.rules import get_immediate_active_left
            target = get_immediate_active_left(player_id, state.round_state.active_player_ids, args.players)
            print(f"Player {player_id} steals from Player {target}!")
        elif isinstance(action, PlayCardAction):
            print(f"Player {player_id} plays {card_to_str(action.card)}")

        # Apply choice
        res = AceEngine.apply_action(state, action)
        assert isinstance(res, Success)
        state = res.new_state

        # Log events (e.g. Interrupted trick, round completes, etc.)
        for ev in res.events:
            if ev.event_type == "TRICK_INTERRUPTED":
                collector = ev.payload.get("collector_id")
                cards = [card_to_str(c) for c in ev.payload.get("collected_cards", [])]
                print(f"  ⚡ Interruption! Player {collector} collects: {' '.join(cards)}")
            elif ev.event_type == "PLAYER_RE_ENTERED":
                p_re = ev.payload.get("player_id")
                print(f"  🔄 Player {p_re} has re-entered the round and is ineligible to win.")
            elif ev.event_type == "ROUND_COMPLETE":
                loser = ev.payload.get("loser_id")
                loser_str = f"Player {loser}" if loser is not None else "None (Draw)"
                print(f"\nRound {round_counter} Complete! Loser: {loser_str}")

        time.sleep(args.delay)

    # 4. Match completes, print rankings scoreboard
    result = AceEngine.get_result(state)
    assert result is not None

    print("\n" + "=" * 60)
    print("MATCH RESULTS SCOREBOARD")
    print("=" * 60)
    print(f"{'Rank':<6}{'Player':<10}{'Points':<10}{'Won':<8}{'Lost':<8}{'Drawn':<8}")
    for p in result.rankings:
        print(f"{p.rank:<6}Player {p.player_id:<10}{p.half_points/2.0:<10.1f}{p.rounds_won:<8}{p.rounds_lost:<8}{p.rounds_drawn:<8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
