# engine/agent.py
"""Abstract base agent and baseline implementations.

Provides the default interface for bot integrations and a random baseline
agent utilizing local deterministic PRNG instances.
"""

import random
from typing import Sequence, Optional
from engine.types import Action, EngineState


class BaseAgent:
    """Base class for all simulator playing agents."""

    def __init__(self, player_id: int, seed: Optional[int] = None) -> None:
        self.player_id = player_id
        self.rng = random.Random(seed)

    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        """Select an action from the list of legal options.

        Args:
            player_view: The player-restricted state projection.
            legal_actions: Non-empty list of legally allowed Actions.

        Returns:
            The selected Action.
        """
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Baseline agent that makes uniformly random legal plays."""

    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        return self.rng.choice(legal_actions)
