# engine/replay_player.py
"""Playback manager for match replays.

Implements step, undo, jump-to, and branching (fork) operations on top of
staged match actions using state history caching.
"""

from typing import List, Optional

from engine.rules import AceEngine, Success, Error
from engine.types import (
    DeclineStealAction, StealAction, PlayCardAction, Action, RoundStarting,
    EngineState
)
from engine.replay import Replay, ReplayAction


class ReplayPlayer:
    """Playback controller managing game states and history for a given Replay."""

    def __init__(self, replay: Replay) -> None:
        self.replay = replay
        self.current_index: int = 0
        self.state: EngineState = self._initialize()
        self.history: List[EngineState] = [self.state]

    def _initialize(self) -> EngineState:
        """Create the initial match state and start round 1."""
        state = AceEngine.create_match(
            match_id=self.replay.match_id,
            num_players=self.replay.num_players,
            num_rounds=self.replay.num_rounds,
            match_seed=self.replay.match_seed
        )
        # Start Round 1
        state, _ = AceEngine.advance(state)
        return state

    def step(self) -> bool:
        """Step forward by applying the next action in the replay.

        Returns:
            True if action was applied successfully, False if no actions remain.
        """
        if self.current_index >= len(self.replay.actions):
            return False

        # Apply auto-advances (e.g. RoundStarting transitions) before taking player action
        while isinstance(self.state.runtime_state.current_phase, RoundStarting) and not AceEngine.is_terminal(self.state):
            self.state, _ = AceEngine.advance(self.state)

        rep_act = self.replay.actions[self.current_index]

        # Map ReplayAction to engine Action object
        action: Action
        if rep_act.action_type == "DeclineSteal":
            action = DeclineStealAction(player_id=rep_act.player_id)
        elif rep_act.action_type == "Steal":
            action = StealAction(player_id=rep_act.player_id)
        elif rep_act.action_type == "PlayCard":
            assert rep_act.card is not None
            action = PlayCardAction(player_id=rep_act.player_id, card=rep_act.card)
        else:
            raise ValueError(f"Unknown replay action type: {rep_act.action_type}")

        res = AceEngine.apply_action(self.state, action)
        if isinstance(res, Error):
            raise RuntimeError(f"Replay action at index {self.current_index} failed to apply: {res.message}")

        assert isinstance(res, Success)
        self.state = res.new_state
        self.current_index += 1

        # Keep history list aligned to current index
        self.history = self.history[:self.current_index]
        self.history.append(self.state)

        # Handle post-action auto-advances if game ends or goes to new round starting phase
        while isinstance(self.state.runtime_state.current_phase, RoundStarting) and not AceEngine.is_terminal(self.state):
            self.state, _ = AceEngine.advance(self.state)

        return True

    def undo(self) -> bool:
        """Step backward by 1 action.

        Returns:
            True if undo was successful, False if already at index 0.
        """
        if self.current_index <= 0:
            return False

        self.current_index -= 1
        self.state = self.history[self.current_index]
        return True

    def jump_to(self, target_index: int) -> None:
        """Jump directly to the specified action sequence index.

        Args:
            target_index: Index number from 0 to len(actions).
        """
        if target_index < 0 or target_index > len(self.replay.actions):
            raise ValueError(f"Target index {target_index} out of bounds")

        # Use history cache if index is already calculated
        if target_index < len(self.history):
            self.current_index = target_index
            self.state = self.history[target_index]
            return

        # Otherwise step forward to reach the index
        while self.current_index < target_index:
            success = self.step()
            if not success:
                break

    def fork_at(self, action_index: int, new_action: Action) -> Replay:
        """Diverge the match at action_index, applying a new action to fork history.

        Args:
            action_index: The sequence index where divergence occurs.
            new_action: The new card or steal action to apply.

        Returns:
            A new Replay dataclass with the branched action list.
        """
        # 1. Travel to the point of fork
        self.jump_to(action_index)

        # 2. Check and advance start of round if needed
        while isinstance(self.state.runtime_state.current_phase, RoundStarting) and not AceEngine.is_terminal(self.state):
            self.state, _ = AceEngine.advance(self.state)

        # 3. Apply the custom branching action
        res = AceEngine.apply_action(self.state, new_action)
        if isinstance(res, Error):
            raise RuntimeError(f"Fork action failed to apply: {res.message}")

        # 4. Map Action back to ReplayAction
        act_type = ""
        card_val: Optional[int] = None
        if isinstance(new_action, DeclineStealAction):
            act_type = "DeclineSteal"
        elif isinstance(new_action, StealAction):
            act_type = "Steal"
        elif isinstance(new_action, PlayCardAction):
            act_type = "PlayCard"
            card_val = new_action.card
        else:
            raise ValueError(f"Unknown action class: {new_action}")

        new_rep_act = ReplayAction(
            sequence=action_index,
            action_type=act_type,
            player_id=new_action.player_id,
            card=card_val
        )

        # 5. Build new branched action list
        new_actions = self.replay.actions[:action_index] + [new_rep_act]

        return Replay(
            version=self.replay.version,
            match_id=self.replay.match_id,
            num_players=self.replay.num_players,
            num_rounds=self.replay.num_rounds,
            match_seed=self.replay.match_seed,
            actions=new_actions
        )
