# engine/heuristic_agent_v2.py
"""Evolved Heuristic Agent (V2) using weights and parameters optimized via Genetic Algorithm."""

import json
import os
from typing import Optional, Dict
from engine.card import get_suit, get_rank, RANK_ACE
from engine.types import PlayCardAction, Action, StealAction
from engine.heuristic_agent import (
    HeuristicAgent, CardTracker, EvaluationContext, get_power_rank,
    H203_OpponentVoidAvoidance,
    H221_AvoidLeadingOnlyTwoPlayers,
    H223_PostCollectionSafeLead,
    H303_SuspectedInterruptionUnderplay,
    H104_SuitIntersectionStealFilter,
    H112_StealForFutureBenefit,
    H210_SafeLeaderWithKnownCollector,
    H211_SuitHoardingAceExposer,
    H304_InterruptionRiskUnderplay
)

class CustomH203(H203_OpponentVoidAvoidance):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            danger = ctx.suit_dangers[s]
            if danger > self.threshold:
                return -danger
        return 0.0

class CustomH221(H221_AvoidLeadingOnlyTwoPlayers):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and len(ctx.view.round_state.active_player_ids) == 2:
            s = get_suit(action.card)
            opponent = ctx.subsequent_players[0]
            opp_void_prob = ctx.void_probs[opponent][s]
            if opp_void_prob > self.threshold:
                return -1.0
            return 0.5
        return 0.0

class CustomH223(H223_PostCollectionSafeLead):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            history = ctx.view.round_state.trick_history
            if len(history) > 0:
                last_t = history[-1]
                if last_t.outcome == "INTERRUPTED" and last_t.collector_id == ctx.viewer_id:
                    danger = ctx.suit_dangers[s]
                    if danger > self.threshold:
                        return -1.0
                    return 1.0
        return 0.0

class CustomH303(H303_SuspectedInterruptionUnderplay):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame"):
                interruption_p = ctx.interruption_probs[ctx.lead_suit]
                if interruption_p > self.threshold and r_val > ctx.highest_played_lead:
                    return -1.0
        return 0.0

class CustomH104(H104_SuitIntersectionStealFilter):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]

            has_intersection = False
            for s in range(4):
                if ctx.c_own[s] > 0 and not tracker.is_void[target][s]:
                    has_intersection = True
                    break
            
            has_known_low_ranks = False
            for card in tracker.player_known_cards[target]:
                s = get_suit(card)
                r_val = get_power_rank(card)
                if ctx.c_own[s] > 0 and r_val <= self.threshold:
                    has_known_low_ranks = True
            
            if has_intersection and has_known_low_ranks:
                return 1.0
        return 0.0

class CustomH112(H112_StealForFutureBenefit):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]
            opponents = [p for p in ctx.view.round_state.active_player_ids if p != ctx.viewer_id]

            score = 0.0
            for card in tracker.player_known_cards[target]:
                s = get_suit(card)
                r_val = get_power_rank(card)
                if not any(tracker.is_void[opp][s] for opp in opponents):
                    score += 0.3
                if r_val <= self.threshold:
                    score += 0.2
                if ctx.c_own[s] == 0 and r_val >= 13:
                    score -= 0.4
            return max(-1.0, min(1.0, score))
        return 0.0

class CustomH210(H210_SafeLeaderWithKnownCollector):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if ctx.highest_known_subsequent_ranks[s] > r_val and r_val <= self.threshold:
                return 1.0
        return 0.0

class CustomH211(H211_SuitHoardingAceExposer):
    def __init__(self, hoard_size: float, rank_threshold: float):
        super().__init__()
        self.hoard_size = hoard_size
        self.rank_threshold = rank_threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Opening", "Middle"):
            s = get_suit(action.card)
            if ctx.c_own[s] >= self.hoard_size:
                if get_rank(action.card) == RANK_ACE:
                    return 0.9
                elif get_power_rank(action.card) >= self.rank_threshold:
                    return -0.8
        return 0.0

class CustomH304(H304_InterruptionRiskUnderplay):
    def __init__(self, u_suit_threshold: float, rank_threshold: float):
        super().__init__()
        self.u_suit_threshold = u_suit_threshold
        self.rank_threshold = rank_threshold
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and ctx.u_suit[ctx.lead_suit] <= self.u_suit_threshold:
                if r_val >= self.rank_threshold:
                    return -1.0
        return 0.0

class HeuristicAgentV2(HeuristicAgent):
    """Evolved Heuristic Agent using optimized weights and internal threshold parameters."""

    def __init__(self, player_id: int, seed: Optional[int] = None, weights_config: Optional[Dict[str, float]] = None) -> None:
        super().__init__(player_id, seed)
        
        # Default strategic parameter thresholds to optimize along with weights
        self.parameters = {
            "P_H203_danger_threshold": 0.5,
            "P_H221_opp_void_prob": 0.5,
            "P_H223_danger_threshold": 0.5,
            "P_H303_interruption_p": 0.5,
            "P_H104_rank_threshold": 5.0,
            "P_H112_rank_threshold": 5.0,
            "P_H210_rank_threshold": 7.0,
            "P_H211_hoard_size": 5.0,
            "P_H211_rank_threshold": 10.0,
            "P_H304_u_suit_threshold": 2.0,
            "P_H304_rank_threshold": 10.0,
        }
        
        config = {}
        if weights_config is not None:
            config.update(weights_config)
        else:
            default_path = os.path.join(os.path.dirname(__file__), "heuristic_v2_weights.json")
            if os.path.exists(default_path):
                try:
                    with open(default_path, "r") as f:
                        config = json.load(f)
                except Exception:
                    pass

        # Update weights and parameter thresholds from configuration
        if config:
            self.weights.update({k: v for k, v in config.items() if not k.startswith("P_") and k != "H114"})
            self.parameters.update({k: v for k, v in config.items() if k.startswith("P_")})
            
        # Override rules with parameterized ones
        p_rules = [
            CustomH203(self.parameters["P_H203_danger_threshold"]),
            CustomH221(self.parameters["P_H221_opp_void_prob"]),
            CustomH223(self.parameters["P_H223_danger_threshold"]),
            CustomH303(self.parameters["P_H303_interruption_p"]),
            CustomH104(self.parameters["P_H104_rank_threshold"]),
            CustomH112(self.parameters["P_H112_rank_threshold"]),
            CustomH210(self.parameters["P_H210_rank_threshold"]),
            CustomH211(self.parameters["P_H211_hoard_size"], self.parameters["P_H211_rank_threshold"]),
            CustomH304(self.parameters["P_H304_u_suit_threshold"], self.parameters["P_H304_rank_threshold"]),
        ]
        
        rule_map = {r.id: r for r in p_rules}
        self.rules = [rule_map.get(r.id, r) for r in self.rules]
