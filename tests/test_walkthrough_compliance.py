# tests/test_walkthrough_compliance.py
"""Walkthrough compliance unit tests.

Verifies that the Ace Engine transitions and logic conform exactly to the
2-round narrated match walkthrough in example_game_walkthrough.md.
"""

from dataclasses import replace
from engine.rules import AceEngine, Success, get_immediate_active_left
from engine.types import (
    DeclineStealAction, StealAction, PlayCardAction, RoundStarting,
    AwaitingStealDecision, RoundPlayerState, TrickState,
    RoundState, EngineState
)
from engine.card import str_to_card, sort_cards


def test_walkthrough_round_1_tricks_1_to_5():
    # Setup initial match
    state = AceEngine.create_match(
        match_id=1,
        num_players=4,
        num_rounds=2,
        match_seed=42
    )

    # Rig the hands for Round 1
    # P0: A♠ 10♠ 8♠ 5♠ | 10♣ 6♣ 2♣ | J♥ 7♥ 3♥ | 9♦ 5♦ 2♦
    p0_cards = [str_to_card(c) for c in [
        "A♠", "10♠", "8♠", "5♠", "10♣", "6♣", "2♣", "J♥", "7♥", "3♥", "9♦", "5♦", "2♦"
    ]]
    # P1: K♠ 9♠ 6♠ 3♠ | A♣ 8♣ 4♣ | K♥ 9♥ 5♥ | Q♦ 7♦ 3♦
    p1_cards = [str_to_card(c) for c in [
        "K♠", "9♠", "6♠", "3♠", "A♣", "8♣", "4♣", "K♥", "9♥", "5♥", "Q♦", "7♦", "3♦"
    ]]
    # P2: Q♠ 7♠ 4♠ | K♣ J♣ 7♣ 3♣ | Q♥ 8♥ 4♥ | A♦ 8♦ 4♦
    p2_cards = [str_to_card(c) for c in [
        "Q♠", "7♠", "4♠", "K♣", "J♣", "7♣", "3♣", "Q♥", "8♥", "4♥", "A♦", "8♦", "4♦"
    ]]
    # P3: J♠ 2♠ | Q♣ 9♣ 5♣ | A♥ 10♥ 6♥ 2♥ | K♦ J♦ 10♦ 6♦
    p3_cards = [str_to_card(c) for c in [
        "J♠", "2♠", "Q♣", "9♣", "5♣", "A♥", "10♥", "6♥", "2♥", "K♦", "J♦", "10♦", "6♦"
    ]]

    # Ensure hands are canonically sorted
    p0_cards = sort_cards(p0_cards)
    p1_cards = sort_cards(p1_cards)
    p2_cards = sort_cards(p2_cards)
    p3_cards = sort_cards(p3_cards)

    # Initialize a custom round 1 state matching the deal
    round_players = [
        RoundPlayerState(player_id=0, hand=tuple(p0_cards), is_active=True, is_round_winner=False, is_round_loser=False),
        RoundPlayerState(player_id=1, hand=tuple(p1_cards), is_active=True, is_round_winner=False, is_round_loser=False),
        RoundPlayerState(player_id=2, hand=tuple(p2_cards), is_active=True, is_round_winner=False, is_round_loser=False),
        RoundPlayerState(player_id=3, hand=tuple(p3_cards), is_active=True, is_round_winner=False, is_round_loser=False),
    ]

    trick = TrickState(
        trick_number=1,
        lead_player_id=0,
        lead_suit=None,
        plays=(),
        status="STEAL_PHASE",
        steals=()
    )

    round_state = RoundState(
        round_number=1,
        round_seed=0,
        players=tuple(round_players),
        active_player_ids=(0, 1, 2, 3),
        current_trick=trick,
        trick_history=(),
        lead_player_id=0,
        discard_pile=(),
        status="IN_PROGRESS"
    )

    steal_target = get_immediate_active_left(0, [0, 1, 2, 3], 4)
    next_phase = AwaitingStealDecision(player_id=0, steal_target=steal_target)
    legal_actions = [
        StealAction(player_id=0),
        DeclineStealAction(player_id=0)
    ]

    runtime_state = replace(
        state.runtime_state,
        current_phase=next_phase,
        current_player_id=0,
        pending_legal_actions=legal_actions
    )

    state = EngineState(
        match_state=replace(state.match_state, status="IN_PROGRESS"),
        round_state=round_state,
        runtime_state=runtime_state
    )

    # Validate initial deal setup
    assert state.round_state is not None
    assert state.round_state.lead_player_id == 0

    # --- Trick 1 ---
    # P0 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=0))
    assert isinstance(res, Success)
    state = res.new_state

    # Plays: P0 plays A♠, P1 plays K♠, P2 plays Q♠, P3 plays J♠
    plays = [(0, "A♠"), (1, "K♠"), (2, "Q♠"), (3, "J♠")]
    for p_id, card_name in plays:
        res = AceEngine.apply_action(state, PlayCardAction(player_id=p_id, card=str_to_card(card_name)))
        assert isinstance(res, Success)
        state = res.new_state

    # Verify Trick 1 completed successfully, winner is P0
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 2
    assert state.round_state.current_trick.lead_player_id == 0
    assert len(state.round_state.discard_pile) == 4

    # --- Trick 2 ---
    # P0 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=0))
    assert isinstance(res, Success)
    state = res.new_state

    # Plays: P0 plays 9♦, P1 plays Q♦, P2 plays A♦, P3 plays K♦
    plays = [(0, "9♦"), (1, "Q♦"), (2, "A♦"), (3, "K♦")]
    for p_id, card_name in plays:
        res = AceEngine.apply_action(state, PlayCardAction(player_id=p_id, card=str_to_card(card_name)))
        assert isinstance(res, Success)
        state = res.new_state

    # Verify Trick 2 completed, winner is P2 (played A♦)
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 3
    assert state.round_state.current_trick.lead_player_id == 2
    assert len(state.round_state.discard_pile) == 8

    # --- Trick 3 ---
    # P2 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=2))
    assert isinstance(res, Success)
    state = res.new_state

    # Plays: P2 plays 7♠, P3 plays 2♠, P0 plays 10♠, P1 plays 9♠
    plays = [(2, "7♠"), (3, "2♠"), (0, "10♠"), (1, "9♠")]
    for p_id, card_name in plays:
        res = AceEngine.apply_action(state, PlayCardAction(player_id=p_id, card=str_to_card(card_name)))
        assert isinstance(res, Success)
        state = res.new_state

    # Verify Trick 3 completed, winner is P0 (played 10♠)
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 4
    assert state.round_state.current_trick.lead_player_id == 0
    assert len(state.round_state.discard_pile) == 12

    # --- Trick 4 (Interrupted Trick) ---
    # P0 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=0))
    assert isinstance(res, Success)
    state = res.new_state

    # Plays: P0 plays 8♠, P1 plays 6♠, P2 plays 4♠, P3 plays Q♣ (breaks suit)
    plays = [(0, "8♠"), (1, "6♠"), (2, "4♠"), (3, "Q♣")]
    for p_id, card_name in plays:
        res = AceEngine.apply_action(state, PlayCardAction(player_id=p_id, card=str_to_card(card_name)))
        assert isinstance(res, Success)
        state = res.new_state

    # Verify Trick 4 completed via interruption. P0 is the collector (holds highest spade: 8♠)
    # Cards collected: 8♠, 6♠, 4♠, Q♣. Discard pile remains 12 cards. Lead remains P0.
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 5
    assert state.round_state.current_trick.lead_player_id == 0
    assert len(state.round_state.discard_pile) == 12
    # P0 has 13 cards now after picking them up
    p0_state = next(p for p in state.round_state.players if p.player_id == 0)
    assert len(p0_state.hand) == 13

    # --- Trick 5 (Demonstrating Steal) ---
    # P0 is lead. P0 steals from P1
    res = AceEngine.apply_action(state, StealAction(player_id=0))
    assert isinstance(res, Success)
    state = res.new_state
    assert state.round_state is not None

    # Verify P1 went inactive (became round winner) and holds 0 cards
    p1_state = next(p for p in state.round_state.players if p.player_id == 1)
    assert len(p1_state.hand) == 0
    assert not p1_state.is_active

    # P0 declines further stealing
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=0))
    assert isinstance(res, Success)
    state = res.new_state

    # Plays: P0 plays 5♠. P2 has no spades, breaks suit and plays K♣
    res = AceEngine.apply_action(state, PlayCardAction(player_id=0, card=str_to_card("5♠")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=2, card=str_to_card("K♣")))
    assert isinstance(res, Success)
    state = res.new_state

    # Verify Trick 5 completed via interruption. P0 collects 5♠ and K♣. Lead remains P0.
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 6
    assert state.round_state.current_trick.lead_player_id == 0


def test_walkthrough_round_1_endgame_and_round_2():
    # Mocking the state exactly at the beginning of Round 1 Trick 15 (Revised)
    state = AceEngine.create_match(
        match_id=1,
        num_players=4,
        num_rounds=2,
        match_seed=42
    )

    # P0: 6♠ │ 6♣ │ 3♥ │ 2♦
    p0_cards = [str_to_card(c) for c in ["6♠", "6♣", "3♥", "2♦"]]
    # P2: 7♣ │ 8♦
    p2_cards = [str_to_card(c) for c in ["7♣", "8♦"]]
    # P3: 6♥
    p3_cards = [str_to_card(c) for c in ["6♥"]]

    p0_cards = sort_cards(p0_cards)
    p2_cards = sort_cards(p2_cards)
    p3_cards = sort_cards(p3_cards)

    # P1 is inactive since Trick 5
    round_players = [
        RoundPlayerState(player_id=0, hand=tuple(p0_cards), is_active=True, is_round_winner=False, is_round_loser=False),
        RoundPlayerState(player_id=1, hand=(), is_active=False, is_round_winner=True, is_round_loser=False),
        RoundPlayerState(player_id=2, hand=tuple(p2_cards), is_active=True, is_round_winner=False, is_round_loser=False),
        RoundPlayerState(player_id=3, hand=tuple(p3_cards), is_active=True, is_round_winner=False, is_round_loser=False),
    ]

    trick = TrickState(
        trick_number=15,
        lead_player_id=3,
        lead_suit=None,
        plays=(),
        status="STEAL_PHASE",
        steals=()
    )

    round_state = RoundState(
        round_number=1,
        round_seed=0,
        players=tuple(round_players),
        active_player_ids=(0, 2, 3),
        current_trick=trick,
        trick_history=(),
        lead_player_id=3,
        # Rest of the cards are in discard pile
        discard_pile=tuple(sorted(list(set(range(52)) - set(p0_cards) - set(p2_cards) - set(p3_cards)))),
        status="IN_PROGRESS"
    )

    steal_target = get_immediate_active_left(3, [0, 2, 3], 4)
    next_phase = AwaitingStealDecision(player_id=3, steal_target=steal_target)
    legal_actions = [
        StealAction(player_id=3),
        DeclineStealAction(player_id=3)
    ]

    runtime_state = replace(
        state.runtime_state,
        current_phase=next_phase,
        current_player_id=3,
        pending_legal_actions=legal_actions
    )

    state = EngineState(
        match_state=replace(state.match_state, status="IN_PROGRESS"),
        round_state=round_state,
        runtime_state=runtime_state
    )

    # --- Trick 15 (Revised) ---
    # P3 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=3))
    assert isinstance(res, Success)
    state = res.new_state

    # P3 plays 6♥, P0 plays 3♥, P2 plays 8♦ (breaks suit)
    res = AceEngine.apply_action(state, PlayCardAction(player_id=3, card=str_to_card("6♥")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=0, card=str_to_card("3♥")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=2, card=str_to_card("8♦")))
    assert isinstance(res, Success)
    state = res.new_state

    # Trick is interrupted, P3 collects: 6♥, 3♥, 8♦. P3 has re-entered. Next lead is P3.
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 16
    assert state.round_state.current_trick.lead_player_id == 3
    p3_state = next(p for p in state.round_state.players if p.player_id == 3)
    assert len(p3_state.hand) == 3

    # --- Trick 16 ---
    # P3 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=3))
    assert isinstance(res, Success)
    state = res.new_state

    # P3 plays 6♥, P0 plays 2♦ (breaks suit)
    res = AceEngine.apply_action(state, PlayCardAction(player_id=3, card=str_to_card("6♥")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=0, card=str_to_card("2♦")))
    assert isinstance(res, Success)
    state = res.new_state

    # Trick is interrupted. P3 collects 6♥ and 2♦. P3 leads next.
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 17
    assert state.round_state.current_trick.lead_player_id == 3

    # --- Trick 17 ---
    # P3 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=3))
    assert isinstance(res, Success)
    state = res.new_state

    # P3 plays 3♥, P0 plays 6♣ (breaks suit)
    res = AceEngine.apply_action(state, PlayCardAction(player_id=3, card=str_to_card("3♥")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=0, card=str_to_card("6♣")))
    assert isinstance(res, Success)
    state = res.new_state

    # Trick interrupted. P3 collects 3♥ and 6♣. P3 leads.
    assert state.round_state is not None
    assert state.round_state.current_trick is not None
    assert state.round_state.current_trick.trick_number == 18
    assert state.round_state.current_trick.lead_player_id == 3

    # --- Trick 18 ---
    # P3 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=3))
    assert isinstance(res, Success)
    state = res.new_state

    # P3 plays 8♦, P0 plays 6♠ (breaks suit)
    res = AceEngine.apply_action(state, PlayCardAction(player_id=3, card=str_to_card("8♦")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=0, card=str_to_card("6♠")))
    assert isinstance(res, Success)
    state = res.new_state

    # Trick interrupted. P3 collects. P0's hand size becomes 0.
    # At trick boundary, P0 becomes inactive (Round Winner).
    assert state.round_state is not None
    p0_state = next(p for p in state.round_state.players if p.player_id == 0)
    assert len(p0_state.hand) == 0
    assert not p0_state.is_active

    # --- Trick 19 ---
    # P3 declines steal
    res = AceEngine.apply_action(state, DeclineStealAction(player_id=3))
    assert isinstance(res, Success)
    state = res.new_state

    # Plays: P3 plays 6♣, P2 plays 7♣
    res = AceEngine.apply_action(state, PlayCardAction(player_id=3, card=str_to_card("6♣")))
    assert isinstance(res, Success)
    state = res.new_state

    res = AceEngine.apply_action(state, PlayCardAction(player_id=2, card=str_to_card("7♣")))
    assert isinstance(res, Success)
    state = res.new_state

    # Successful trick. P2 has 0 cards. Trick winner is P2 (played 7♣).
    # Since P2 is now inactive, round ends because exactly 1 active player remains (P3).
    # Verification of round completion
    assert state.round_state is not None
    assert state.round_state.status == "COMPLETE"
    assert next(p for p in state.round_state.players if p.player_id == 3).is_round_loser
    assert next(p for p in state.round_state.players if p.player_id == 0).is_round_winner
    assert next(p for p in state.round_state.players if p.player_id == 1).is_round_winner
    assert next(p for p in state.round_state.players if p.player_id == 2).is_round_winner

    # Transition to Round 2 starting
    assert isinstance(state.runtime_state.current_phase, RoundStarting)
    state, _ = AceEngine.advance(state)

    # Verify Round 2 start state: P3 has consecutive_loss_count = 1 -> receives A♠
    assert state.match_state.players[3].consecutive_loss_count == 1
    assert state.round_state is not None
    assert state.round_state.round_number == 2
    # Verify P3 holds A♠ (card 0)
    p3_round_state = next(p for p in state.round_state.players if p.player_id == 3)
    assert 0 in p3_round_state.hand
    assert state.round_state.lead_player_id == 3
