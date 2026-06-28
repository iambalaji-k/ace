# engine/deck.py
"""Deck construction, reserved ace allocation, and dealing logic.

Ensures deterministic card dealing and skip rotations in accordance with the specification.
"""

from typing import List, Tuple, Optional
from engine.card import str_to_card, sort_cards
from engine.prng import deterministic_shuffle

# Reserved Aces list in order of assignment
RESERVED_ACE_STRS = ["A♠", "A♣", "A♥", "A♦"]


def get_reserved_aces(consecutive_loss_count: int) -> List[int]:
    """Get the reserved aces for a player based on their consecutive loss count.

    Formula: count = min(consecutive_loss_count, 4)
    Aces are returned in order: A♠, A♣, A♥, A♦.
    """
    count = min(consecutive_loss_count, 4)
    if count <= 0:
        return []
    return [str_to_card(RESERVED_ACE_STRS[i]) for i in range(count)]


def build_and_deal_deck(
    num_players: int,
    recipient_id: Optional[int],
    consecutive_loss_count: int,
    round_seed: int
) -> Tuple[List[List[int]], List[int]]:
    """Build the deck, remove reserved aces, shuffle the rest, and deal.

    Returns:
    - A list of hands (one list of card integers per player).
    - The shuffled remaining deck (excluding reserved aces).
    """
    # 1. Build full deck (0 to 51)
    full_deck = list(range(52))

    # 2. Get and remove reserved aces if recipient exists
    reserved_aces = []
    if recipient_id is not None:
        reserved_aces = get_reserved_aces(consecutive_loss_count)
        for ace in reserved_aces:
            full_deck.remove(ace)

    # 3. Shuffle remaining cards
    shuffled_deck = deterministic_shuffle(full_deck, round_seed)

    # 4. Give reserved aces to the recipient's hand
    hands = [[] for _ in range(num_players)]
    if recipient_id is not None:
        for ace in reserved_aces:
            hands[recipient_id].append(ace)

    # 5. Deal remaining cards clockwise starting from P0
    skip_count = len(reserved_aces)
    current_player = 0

    for card in shuffled_deck:
        if recipient_id is not None and current_player == recipient_id and skip_count > 0:
            # Skip recipient, decrement skip count, move to next player clockwise
            skip_count -= 1
            current_player = (current_player + 1) % num_players

        hands[current_player].append(card)
        current_player = (current_player + 1) % num_players

    # 6. Sort all hands canonically
    for i in range(num_players):
        hands[i] = sort_cards(hands[i])

    return hands, shuffled_deck
