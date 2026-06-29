# engine/observation.py
"""AI Observation Layer for the Ace Engine.

Provides clean, decoupled, and vectorized representations of the game state for AI models.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
from engine.types import (
    EngineState, Action, PlayCardAction, StealAction, DeclineStealAction,
    AwaitingStealDecision, AwaitingCardPlay, RoundStarting, MatchComplete
)
from engine.card import get_suit


@dataclass(frozen=True)
class PlayerObservation:
    """Decoupled, imperfect-information view of the game state for a specific player."""
    player_id: int
    hand: Tuple[int, ...]
    opponent_hand_sizes: Tuple[int, ...]  # Length equal to total players in match
    active_players: Tuple[bool, ...]       # Length equal to total players in match
    discard_count: int
    lead_player_id: Optional[int]
    lead_suit: Optional[int]
    current_trick_plays: Tuple[Tuple[int, int], ...]  # (player_id, card_id) in sequence
    legal_actions: Tuple[Action, ...]
    game_phase_type: str  # "ROUND_STARTING", "STEAL_PHASE", "PLAY_PHASE", "MATCH_COMPLETE"
    is_your_turn: bool


@dataclass(frozen=True)
class EncodedObservation:
    """Flat float vector representation along with action mask for neural network input."""
    vector: Tuple[float, ...]
    action_mask: Tuple[bool, ...]  # Boolean mask of length 54


def build_player_observation(state: EngineState, player_id: int) -> PlayerObservation:
    """Build a PlayerObservation from the current EngineState for a given player."""
    match = state.match_state
    num_players = len(match.players)
    
    # Defaults if round hasn't started or is complete
    hand: Tuple[int, ...] = ()
    opponent_hand_sizes = tuple(0 for _ in range(num_players))
    active_players = tuple(p.player_id in match.seating_order for p in match.players)
    discard_count = 0
    lead_player_id = None
    lead_suit = None
    current_trick_plays: Tuple[Tuple[int, int], ...] = ()
    
    phase = state.runtime_state.current_phase
    game_phase_type = "ROUND_STARTING"
    if isinstance(phase, AwaitingStealDecision):
        game_phase_type = "STEAL_PHASE"
    elif isinstance(phase, AwaitingCardPlay):
        game_phase_type = "PLAY_PHASE"
    elif isinstance(phase, MatchComplete):
        game_phase_type = "MATCH_COMPLETE"
        
    is_your_turn = (state.runtime_state.current_player_id == player_id)
    
    # If round is in progress, populate round-specific details
    if state.round_state is not None:
        round_st = state.round_state
        # Find player's hand
        p_state = next((p for p in round_st.players if p.player_id == player_id), None)
        if p_state is not None:
            hand = p_state.hand
            
        # Get opponent hand sizes
        hand_sizes = []
        for p in round_st.players:
            hand_sizes.append(len(p.hand))
        opponent_hand_sizes = tuple(hand_sizes)
        
        # Get active players
        active_players = tuple(p.is_active for p in round_st.players)
        
        # Discard pile count
        discard_count = len(round_st.discard_pile)
        
        # Trick details
        if round_st.current_trick is not None:
            trick = round_st.current_trick
            lead_player_id = trick.lead_player_id
            lead_suit = trick.lead_suit
            current_trick_plays = tuple((p.player_id, p.card) for p in trick.plays)
            
    # Filter pending legal actions for this player
    legal: Sequence[Action] = ()
    if is_your_turn:
        legal = state.runtime_state.pending_legal_actions
        
    return PlayerObservation(
        player_id=player_id,
        hand=hand,
        opponent_hand_sizes=opponent_hand_sizes,
        active_players=active_players,
        discard_count=discard_count,
        lead_player_id=lead_player_id,
        lead_suit=lead_suit,
        current_trick_plays=current_trick_plays,
        legal_actions=tuple(legal),
        game_phase_type=game_phase_type,
        is_your_turn=is_your_turn
    )


def encode_observation(obs: PlayerObservation) -> EncodedObservation:
    """Encode a PlayerObservation into a flat float vector and action mask."""
    # Vector length design:
    # - 52 floats: Hand binary indicators (1.0 if card present, 0.0 otherwise)
    # - 6 floats: Opponent hand sizes (normalized by 52.0, padded to max 6 players)
    # - 6 floats: Active players indicators (1.0 active, 0.0 inactive, padded)
    # - 1 float: Normalized discard pile count (count / 52.0)
    # - 52 floats: Cards played in current trick (1.0 if card played, 0.0 otherwise)
    # - 4 floats: Lead suit one-hot (Clubs, Diamonds, Hearts, Spades)
    # - 4 floats: Phase type one-hot ("ROUND_STARTING", "STEAL_PHASE", "PLAY_PHASE", "MATCH_COMPLETE")
    # - 1 float: Is your turn indicator (1.0 if True, 0.0 if False)
    # Total vector size: 52 + 6 + 6 + 1 + 52 + 4 + 4 + 1 = 126 floats.
    
    vec = [0.0] * 126
    
    # 1. Own Hand
    for card_id in obs.hand:
        if 0 <= card_id < 52:
            vec[card_id] = 1.0
            
    # 2. Opponent Hand Sizes (starting index 52)
    for idx, size in enumerate(obs.opponent_hand_sizes):
        if idx < 6:
            vec[52 + idx] = size / 52.0
            
    # 3. Active Players (starting index 58)
    for idx, active in enumerate(obs.active_players):
        if idx < 6:
            vec[58 + idx] = 1.0 if active else 0.0
            
    # 4. Discard Count (starting index 64)
    vec[64] = obs.discard_count / 52.0
    
    # 5. Cards played in current trick (starting index 65)
    for _, card_id in obs.current_trick_plays:
        if 0 <= card_id < 52:
            vec[65 + card_id] = 1.0
            
    # 6. Lead Suit (starting index 117)
    if obs.lead_suit is not None and 0 <= obs.lead_suit < 4:
        vec[117 + obs.lead_suit] = 1.0
        
    # 7. Game Phase (starting index 121)
    phase_map = {
        "ROUND_STARTING": 0,
        "STEAL_PHASE": 1,
        "PLAY_PHASE": 2,
        "MATCH_COMPLETE": 3
    }
    phase_idx = phase_map.get(obs.game_phase_type, 0)
    vec[121 + phase_idx] = 1.0
    
    # 8. Is your turn (starting index 125)
    vec[125] = 1.0 if obs.is_your_turn else 0.0
    
    # --- Action Mask generation ---
    # Size 54:
    # Index 0: DeclineSteal
    # Index 1: Steal
    # Index 2..53: PlayCardAction for card_id = index - 2
    mask = [False] * 54
    for action in obs.legal_actions:
        if isinstance(action, DeclineStealAction):
            mask[0] = True
        elif isinstance(action, StealAction):
            mask[1] = True
        elif isinstance(action, PlayCardAction):
            card_id = action.card
            if 0 <= card_id < 52:
                mask[2 + card_id] = True
                
    return EncodedObservation(
        vector=tuple(vec),
        action_mask=tuple(mask)
    )
