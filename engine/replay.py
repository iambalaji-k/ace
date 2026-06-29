# engine/replay.py
"""Serialization schema and utilities for match replays.

Implements versioned, JSON-serializable structures to record match parameters
and sequence of player actions.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass(frozen=True)
class ReplayAction:
    sequence: int
    action_type: str  # "DeclineSteal", "Steal", "PlayCard"
    player_id: int
    card: Optional[int] = None


@dataclass(frozen=True)
class Replay:
    version: str
    match_id: int
    num_players: int
    num_rounds: int
    match_seed: int
    actions: List[ReplayAction]


def serialize_replay(replay: Replay) -> str:
    """Serialize Replay dataclass into a deterministic JSON string."""
    return json.dumps(asdict(replay), indent=2)


def deserialize_replay(json_str: str) -> Replay:
    """Deserialize JSON string into a Replay dataclass."""
    data = json.loads(json_str)
    actions = [
        ReplayAction(
            sequence=int(act["sequence"]),
            action_type=str(act["action_type"]),
            player_id=int(act["player_id"]),
            card=int(act["card"]) if act.get("card") is not None else None
        )
        for act in data["actions"]
    ]
    return Replay(
        version=str(data["version"]),
        match_id=int(data["match_id"]),
        num_players=int(data["num_players"]),
        num_rounds=int(data["num_rounds"]),
        match_seed=int(data["match_seed"]),
        actions=actions
    )


def export_replay(replay: Replay, filepath: str) -> None:
    """Save Replay dataclass into a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(serialize_replay(replay))


def import_replay(filepath: str) -> Replay:
    """Load Replay dataclass from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return deserialize_replay(f.read())
