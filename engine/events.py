# engine/events.py
"""Event helper and projection functions for the Ace Engine.

Handles creation and visibility projections (Public, Player, Observer) of events.
"""

from typing import Optional, Dict, Any
from dataclasses import replace
from engine.types import Event


def create_event(
    sequence: int,
    event_type: str,
    round_number: Optional[int],
    trick_number: Optional[int],
    payload: Dict[str, Any],
    timestamp: int
) -> Event:
    """Create a new Event instance."""
    return Event(
        sequence=sequence,
        event_type=event_type,
        round_number=round_number,
        trick_number=trick_number,
        payload=payload,
        timestamp=timestamp
    )


def project_event_public(event: Event) -> Event:
    """Project the event to the public view, stripping all private data."""
    payload = dict(event.payload)
    if event.event_type == "ACES_RESERVED":
        if "ace_cards" in payload:
            payload.pop("ace_cards")
    elif event.event_type == "CARDS_DEALT":
        if "hand" in payload:
            payload.pop("hand")
    elif event.event_type == "STEAL_EXECUTED":
        if "cards" in payload:
            payload.pop("cards")
    return replace(event, payload=payload)


def project_event_for_player(event: Event, viewer_id: int) -> Event:
    """Project the event for a specific player (viewer_id), showing their private data but hiding others'."""
    payload = dict(event.payload)
    if event.event_type == "ACES_RESERVED":
        if payload.get("player_id") != viewer_id:
            if "ace_cards" in payload:
                payload.pop("ace_cards")
    elif event.event_type == "CARDS_DEALT":
        if payload.get("player_id") != viewer_id:
            if "hand" in payload:
                payload.pop("hand")
    elif event.event_type == "STEAL_EXECUTED":
        stealer = payload.get("stealer_id")
        victim = payload.get("victim_id")
        if viewer_id not in (stealer, victim):
            if "cards" in payload:
                payload.pop("cards")
    return replace(event, payload=payload)


def project_event_observer(event: Event) -> Event:
    """Observer sees everything. Returns the event unchanged."""
    return event
