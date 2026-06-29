# Phase 5 — Replay System Implementation Plan

This document details the design and implementation strategy for the **Replay System** (AETS Chapter 9) to capture, store, playback, and branch match histories.

---

## 1. Serialization Schema (`engine/replay.py`)

A replay must store all the initial parameters and applied inputs to perfectly reconstruct a match. We will define a `Replay` object in a new module:

```python
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
```

### Serialization Utilities
- `export_replay(replay: Replay, filepath: str) -> None`: Exports the replay to a JSON file.
- `import_replay(filepath: str) -> Replay`: Loads a replay from a JSON file.

---

## 2. Replay Playback Manager (`engine/replay_player.py`)

The player will wrap the engine transitions and handle moving backward and forward in time.

```python
class ReplayPlayer:
    def __init__(self, replay: Replay):
        self.replay = replay
        self.current_index = 0  # Points to the next action index in self.replay.actions
        self.state = None       # Current active EngineState
        self.history = []       # List of EngineState at each step index to make undo O(1)
        self._initialize()

    def _initialize(self):
        """Initializes the match and sets up the starting state."""
        # 1. Create match
        # 2. Call advance() to start round 1
        # 3. Store initial state in self.history[0]
        pass

    def step(self) -> bool:
        """Applies the next action in the replay. Returns True if successful, False if complete."""
        # 1. Get action at self.current_index
        # 2. Call AceEngine.apply_action()
        # 3. Auto-advance if state transitions to new round
        # 4. Save state to self.history, increment current_index
        pass

    def undo(self) -> bool:
        """Goes back 1 action. Returns True if successful."""
        # 1. Decrement current_index
        # 2. Load state from self.history[self.current_index]
        pass

    def jump_to(self, action_index: int) -> None:
        """Jumps directly to action index K (efficiently using history caching)."""
        # If K < current_index, load from history cache.
        # If K > current_index, step forward sequentially.
        pass

    def fork_at(self, action_index: int, new_action: Action) -> Replay:
        """Truncates the match history at action_index, applies new_action, and returns a new Replay."""
        # 1. Jump to action_index
        # 2. Apply new_action
        # 3. Create a new Replay object with truncated action list + new_action
        pass
```

---

## 3. Conformance Unit Testing (`tests/test_replays.py`)

 we will add unit tests verifying:
- **Fidelity**: Replaying a match matches the final `EngineState` and SHA-256 hash of the live-played match byte-for-byte.
- **Traversal**: Stepping forward, backward (undo), and jumping around the timeline yields correct states and counts.
- **Branching**: Forking a match at Trick 4 with a suit break creates a valid, divergent game path with its own history.
