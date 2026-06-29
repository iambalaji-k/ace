# engine/action_encoding.py
"""Action Encoding API for the Ace Engine.

Maps between high-level Action objects and discrete integer action indices (0 to 53)
suitable for reinforcement learning policies and agents.
"""

from typing import Tuple, Sequence
from engine.types import Action, DeclineStealAction, StealAction, PlayCardAction


def action_to_index(action: Action) -> int:
    """Map an Action object to its unique discrete index (0 to 53)."""
    if isinstance(action, DeclineStealAction):
        return 0
    elif isinstance(action, StealAction):
        return 1
    elif isinstance(action, PlayCardAction):
        card_id = action.card
        if not 0 <= card_id < 52:
            raise ValueError(f"Invalid card ID in PlayCardAction: {card_id}")
        return 2 + card_id
    else:
        raise TypeError(f"Unknown action class: {action.__class__.__name__}")


def index_to_action(player_id: int, action_index: int) -> Action:
    """Map a discrete index (0 to 53) back to an Action object for the given player."""
    if action_index == 0:
        return DeclineStealAction(player_id=player_id)
    elif action_index == 1:
        return StealAction(player_id=player_id)
    elif 2 <= action_index <= 53:
        card_id = action_index - 2
        return PlayCardAction(player_id=player_id, card=card_id)
    else:
        raise ValueError(f"Invalid action index: {action_index}. Must be 0 to 53.")


def legal_action_mask(legal_actions: Sequence[Action]) -> Tuple[bool, ...]:
    """Generate a boolean mask of size 54 indicating which action indices are legal."""
    mask = [False] * 54
    for action in legal_actions:
        idx = action_to_index(action)
        if 0 <= idx < 54:
            mask[idx] = True
    return tuple(mask)
