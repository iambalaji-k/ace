# engine/card.py
"""Card utility functions for the Ace Engine.

Cards are represented as integers from 0 to 51.
- Suit: card // 13 (0 = Spade, 1 = Club, 2 = Heart, 3 = Diamond)
- Rank: card % 13 (0 = Ace [highest], 12 = Two [lowest])
"""

from typing import List

# Constants for suits
SUIT_SPADE = 0
SUIT_CLUB = 1
SUIT_HEART = 2
SUIT_DIAMOND = 3

SUIT_NAMES = ["Spade", "Club", "Heart", "Diamond"]
SUIT_SYMBOLS = ["♠", "♣", "♥", "♦"]

# Constants for ranks
RANK_ACE = 0
RANK_KING = 1
RANK_QUEEN = 2
RANK_JACK = 3
RANK_TEN = 4
RANK_NINE = 5
RANK_EIGHT = 6
RANK_SEVEN = 7
RANK_SIX = 8
RANK_FIVE = 9
RANK_FOUR = 10
RANK_THREE = 11
RANK_TWO = 12

RANK_NAMES = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]


def get_suit(card: int) -> int:
    """Get the suit of the card (0-3)."""
    if not 0 <= card < 52:
        raise ValueError(f"Invalid card ID: {card}")
    return card // 13


def get_rank(card: int) -> int:
    """Get the rank of the card (0-12)."""
    if not 0 <= card < 52:
        raise ValueError(f"Invalid card ID: {card}")
    return card % 13


def card_to_str(card: int) -> str:
    """Convert card integer representation (0-51) to human-readable string (e.g. 'A♠')."""
    if not 0 <= card < 52:
        raise ValueError(f"Invalid card ID: {card}")
    suit = get_suit(card)
    rank = get_rank(card)
    return f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}"


def str_to_card(card_str: str) -> int:
    """Convert human-readable string (e.g. 'A♠' or '10♦') to card integer representation."""
    card_str = card_str.strip()
    if len(card_str) < 2:
        raise ValueError(f"Invalid card string: '{card_str}'")

    # The rank part could be 1 or 2 characters (e.g., '10' vs 'A')
    rank_str = card_str[:-1]
    suit_sym = card_str[-1]

    if rank_str not in RANK_NAMES:
        raise ValueError(f"Invalid rank in card string: '{rank_str}'")
    if suit_sym not in SUIT_SYMBOLS:
        raise ValueError(f"Invalid suit symbol in card string: '{suit_sym}'")

    rank = RANK_NAMES.index(rank_str)
    suit = SUIT_SYMBOLS.index(suit_sym)

    return suit * 13 + rank


def sort_cards(cards: List[int]) -> List[int]:
    """Sort a list of cards in canonical order (Spade > Club > Heart > Diamond, descending rank).

    Since the card IDs (0-51) are naturally structured such that:
    - Spades are 0-12 (A♠=0 to 2♠=12)
    - Clubs are 13-25 (A♣=13 to 2♣=25)
    - Hearts are 26-38 (A♥=26 to 2♥=38)
    - Diamonds are 39-51 (A♦=39 to 2♦=51)
    An ascending sort of card IDs naturally matches the canonical ordering!
    """
    return sorted(cards)
