# tests/test_foundations.py
from engine.card import get_suit, get_rank, card_to_str, str_to_card, sort_cards
from engine.prng import pcg_seed, pcg_next, derive_round_seed
from engine.deck import get_reserved_aces, build_and_deal_deck


def test_card_utilities():
    # A♠ should be 0
    assert get_suit(0) == 0
    assert get_rank(0) == 0
    assert card_to_str(0) == "A♠"
    assert str_to_card("A♠") == 0

    # 2♦ should be 51
    assert get_suit(51) == 3
    assert get_rank(51) == 12
    assert card_to_str(51) == "2♦"
    assert str_to_card("2♦") == 51

    # 10♣ is card 17
    # suit = 1 (Club), rank = 4 (Ten) -> 1 * 13 + 4 = 17
    assert get_suit(17) == 1
    assert get_rank(17) == 4
    assert card_to_str(17) == "10♣"
    assert str_to_card("10♣") == 17

    # Sorting cards
    unsorted = [51, 0, 17, 26]  # 2♦, A♠, 10♣, A♥
    # Expected order: A♠ (0), 10♣ (17), A♥ (26), 2♦ (51)
    assert sort_cards(unsorted) == [0, 17, 26, 51]


def test_prng_determinism():
    # Seed generator with 42
    state = pcg_seed(42)
    state, val1 = pcg_next(state)
    state, val2 = pcg_next(state)
    
    # Re-seed and check it generates the exact same values
    state_check = pcg_seed(42)
    state_check, val1_check = pcg_next(state_check)
    state_check, val2_check = pcg_next(state_check)
    
    assert val1 == val1_check
    assert val2 == val2_check
    assert state == state_check


def test_reserved_aces():
    # 0 consecutive loss
    assert get_reserved_aces(0) == []
    # 1 consecutive loss -> A♠ (0)
    assert get_reserved_aces(1) == [0]
    # 2 consecutive loss -> A♠ (0), A♣ (13)
    assert get_reserved_aces(2) == [0, 13]
    # 3 consecutive loss -> A♠ (0), A♣ (13), A♥ (26)
    assert get_reserved_aces(3) == [0, 13, 26]
    # 4 consecutive loss -> A♠ (0), A♣ (13), A♥ (26), A♦ (39)
    assert get_reserved_aces(4) == [0, 13, 26, 39]
    # 5 consecutive loss -> caps at 4 aces
    assert get_reserved_aces(5) == [0, 13, 26, 39]


def test_deal_skip_rotation():
    # 4 players, recipient P3 (index 3), 1 loss (A♠ = card 0), seed 12345
    round_seed = derive_round_seed(12345, 1)
    hands, remaining_deck = build_and_deal_deck(
        num_players=4,
        recipient_id=3,
        consecutive_loss_count=1,
        round_seed=round_seed
    )

    # All hands must have exactly 13 cards (52 cards total)
    assert len(hands) == 4
    for i in range(4):
        assert len(hands[i]) == 13

    # P3 must hold A♠ (0)
    assert 0 in hands[3]
    # All cards must be unique and sum to 52 cards
    all_cards = set()
    for hand in hands:
        for card in hand:
            all_cards.add(card)
    assert len(all_cards) == 52
    assert all_cards == set(range(52))
