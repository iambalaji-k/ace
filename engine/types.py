# engine/types.py
"""Data model definitions for the Ace Engine.

Implements all match, round, and runtime state structures as frozen (immutable) dataclasses.
Also contains action types, game phases, and event representations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# --- Seating & Match Levels ---

@dataclass(frozen=True)
class PlayerState:
    player_id: int  # seat index 0 to N-1
    consecutive_loss_count: int
    rounds_won: int
    rounds_lost: int
    rounds_drawn: int
    half_points: int  # Win = 2, Draw = 1, Loss = 0


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    loser_id: Optional[int]  # None if draw
    winner_ids: List[int]
    is_draw: bool


@dataclass(frozen=True)
class PlayerRanking:
    player_id: int
    rank: int  # 1-indexed
    half_points: int
    rounds_won: int
    rounds_lost: int
    rounds_drawn: int


@dataclass(frozen=True)
class MatchResult:
    rankings: List[PlayerRanking]
    total_rounds: int
    draws: int


@dataclass(frozen=True)
class MatchState:
    match_id: int
    num_rounds: int
    current_round: int  # 1-indexed
    match_seed: int
    players: List[PlayerState]  # ordered by seat index
    seating_order: List[int]  # seat indices, clockwise
    round_results: List[RoundResult]  # one per completed round
    status: str  # "INIT", "IN_PROGRESS", "COMPLETE"


# --- Round Levels ---

@dataclass(frozen=True)
class RoundPlayerState:
    player_id: int
    hand: List[int]  # card IDs (0-51) in canonical sort order
    is_active: bool
    is_round_winner: bool
    is_round_loser: bool


@dataclass(frozen=True)
class TrickPlay:
    player_id: int
    card: int  # card ID (0-51)


@dataclass(frozen=True)
class StealEvent:
    stealer_id: int
    victim_id: int
    cards_taken: List[int]  # card IDs


@dataclass(frozen=True)
class TrickState:
    trick_number: int
    lead_player_id: int
    lead_suit: Optional[int]  # suit code 0-3 (None before lead card is played)
    plays: List[TrickPlay]  # cards played, in sequence
    status: str  # "STEAL_PHASE", "PLAY_PHASE", "RESOLVED"
    steals: List[StealEvent]  # steals occurring *before* or during this trick


@dataclass(frozen=True)
class CompletedTrick:
    trick_number: int
    plays: List[TrickPlay]
    outcome: str  # "DISCARDED" | "INTERRUPTED"
    collector_id: Optional[int]
    collected_cards: List[int]  # empty if discarded


@dataclass(frozen=True)
class RoundState:
    round_number: int
    round_seed: int
    players: List[RoundPlayerState]  # ordered by seat index
    active_player_ids: List[int]  # remaining active player IDs, in seat order
    current_trick: Optional[TrickState]
    trick_history: List[CompletedTrick]
    lead_player_id: int
    discard_pile: List[int]  # card IDs, face-down
    status: str  # "INIT", "IN_PROGRESS", "COMPLETE"


# --- Actions ---

@dataclass(frozen=True)
class Action:
    player_id: int


@dataclass(frozen=True)
class StealAction(Action):
    pass


@dataclass(frozen=True)
class DeclineStealAction(Action):
    pass


@dataclass(frozen=True)
class PlayCardAction(Action):
    card: int


# --- Game Phases ---

@dataclass(frozen=True)
class GamePhase:
    pass


@dataclass(frozen=True)
class RoundStarting(GamePhase):
    round_number: int


@dataclass(frozen=True)
class AwaitingStealDecision(GamePhase):
    player_id: int
    steal_target: int


@dataclass(frozen=True)
class AwaitingCardPlay(GamePhase):
    player_id: int
    lead_suit: Optional[int]
    must_follow: bool


@dataclass(frozen=True)
class MatchComplete(GamePhase):
    result: MatchResult


# --- Runtime State & Engine State ---

@dataclass(frozen=True)
class RuntimeState:
    action_sequence_number: int
    current_phase: GamePhase
    current_player_id: Optional[int]
    pending_legal_actions: List[Action]
    prng_state: int


@dataclass(frozen=True)
class EngineState:
    match_state: MatchState
    round_state: Optional[RoundState]
    runtime_state: RuntimeState

    def get_state_hash(self) -> str:
        """Compute a SHA-256 hash of the entire state deterministically."""
        import hashlib
        import json

        # Helper to recursively convert objects to deterministic dictionary structures
        def to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dict__"):
                # Handle dataclasses
                d = {}
                for key, val in obj.__dict__.items():
                    # Skip values that are dynamic or redundant if any
                    d[key] = to_dict(val)
                # Include class name to ensure different type states have different hashes
                d["__type__"] = obj.__class__.__name__
                return d
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [to_dict(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in sorted(obj.items())}
            else:
                return obj

        serialized = json.dumps(to_dict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# --- Event System ---

@dataclass(frozen=True)
class Event:
    sequence: int
    event_type: str
    round_number: Optional[int]
    trick_number: Optional[int]
    payload: Dict[str, Any]
    timestamp: int
