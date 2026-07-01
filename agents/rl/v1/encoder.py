# agents/rl/v1/encoder.py
"""Encodes the EngineState from a player's perspective into flat NumPy arrays for RL models."""

import numpy as np
from engine.types import EngineState
from engine.card import get_suit
from agents.heuristic.v1.heuristic_agent import CardTracker

def encode_state(state: EngineState, player_id: int) -> np.ndarray:
    """Encodes EngineState into a flat float32 NumPy array of size 343."""
    num_players = len(state.round_state.players)
    
    # Reconstruct tracker to get void suits and known cards
    tracker = CardTracker(num_players=num_players)
    tracker.reconstruct(
        viewer_id=player_id,
        round_state=state.round_state,
        match_state=state.match_state
    )
    
    # 1. Own Hand: 52 binary features
    own_hand_feats = np.zeros(52, dtype=np.float32)
    for card in state.round_state.players[player_id].hand:
        if card >= 0:
            own_hand_feats[card] = 1.0
            
    # 2. Discards: 52 binary features
    discard_feats = np.zeros(52, dtype=np.float32)
    for suit in range(4):
        for card in tracker.discards[suit]:
            discard_feats[card] = 1.0
            
    # 3. Opponent Voids: 4 players * 4 suits = 16 binary features
    void_feats = np.zeros((4, 4), dtype=np.float32)
    for p in range(num_players):
        for s in range(4):
            if tracker.is_void[p][s]:
                void_feats[p, s] = 1.0
                
    # 4. Known Opponent Cards: 4 players * 52 cards = 208 binary features
    known_feats = np.zeros((4, 52), dtype=np.float32)
    for p in range(num_players):
        if p == player_id:
            # Own hand is already known
            for card in state.round_state.players[player_id].hand:
                if card >= 0:
                    known_feats[p, card] = 1.0
        else:
            for card in tracker.player_known_cards[p]:
                known_feats[p, card] = 1.0
                
    # 5. Point scores & round counts: 8 features (normalized)
    score_feats = np.zeros(8, dtype=np.float32)
    for p in range(num_players):
        match_p = state.match_state.players[p]
        score_feats[p] = match_p.half_points / 500.0
        score_feats[4 + p] = match_p.rounds_won / 10.0
        
    # 6. Current Lead Suit: 4 features (one-hot)
    lead_feats = np.zeros(4, dtype=np.float32)
    curr_trick = state.round_state.current_trick
    if curr_trick and curr_trick.plays:
        lead_suit = get_suit(curr_trick.plays[0].card)
        lead_feats[lead_suit] = 1.0
        
    # 7. Game Phase: 3 features (one-hot)
    phase_feats = np.zeros(3, dtype=np.float32)
    card_count = sum(len(p.hand) for p in state.round_state.players if p.is_active)
    if card_count > 36:
        phase_feats[0] = 1.0  # Opening
    elif card_count > 16:
        phase_feats[1] = 1.0  # Middle
    else:
        phase_feats[2] = 1.0  # Endgame
        
    # Concatenate all into a single flat vector of size 343
    return np.concatenate([
        own_hand_feats,
        discard_feats,
        void_feats.flatten(),
        known_feats.flatten(),
        score_feats,
        lead_feats,
        phase_feats
    ])
