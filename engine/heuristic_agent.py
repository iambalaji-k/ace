# engine/heuristic_agent.py
"""Modular and Explainable Heuristic AI Agent for the Ace Engine.

Implements clean separation of facts (CardTracker), estimations (ProbabilityEstimator), 
lazy pre-extracted features (FeatureExtractor / EvaluationContext), and ABC rules.
"""

import threading
from typing import Sequence, Tuple, List, Optional, Dict, Set, Any
from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from functools import cached_property

from engine.agent import BaseAgent
from engine.types import Action, EngineState, PlayCardAction, StealAction, DeclineStealAction
from engine.card import get_suit, get_rank, card_to_str, sort_cards, RANK_ACE


# =====================================================================
# 0. Global Card Utilities
# =====================================================================

def get_power_rank(card: int) -> int:
    """Helper to convert raw card ID to power rank (2-14) where Ace is 14."""
    return 14 - get_rank(card)


# =====================================================================
# 1. Facts, Estimations & Derived Metrics
# =====================================================================


class CardTracker:
    """Tracks direct public observations and deterministic reconstructions (100% Facts)."""

    # Thread-local cache for prefix-based incremental reconstruction
    _thread_local = threading.local()

    @classmethod
    def _get_cache(cls):
        if not hasattr(cls._thread_local, 'cache'):
            cls._thread_local.cache = {}
        return cls._thread_local.cache

    def __init__(self, num_players: int):
        self.num_players = num_players
        # Card locations: "self", "played", "known_P" (held by player index P), or "unknown"
        self.card_locations = ["unknown"] * 52
        # Confirmed void flags: is_void[player_id][suit] = bool
        self.is_void = [[False] * 4 for _ in range(num_players)]
        # Track specific card IDs known to be in each player's hand
        self.player_known_cards = [set() for _ in range(num_players)]
        # Track discards per suit
        self.discards = [set() for _ in range(4)]
        # Track active/inactive player status trick-by-trick
        self.active_players_state = [True] * num_players

    def clone(self) -> 'CardTracker':
        """Deep copies the factual card tracker state."""
        cloned = CardTracker(self.num_players)
        cloned.card_locations = list(self.card_locations)
        cloned.is_void = [list(v) for v in self.is_void]
        cloned.player_known_cards = [set(k) for k in self.player_known_cards]
        cloned.discards = [set(d) for d in self.discards]
        return cloned

    def record_own_hand(self, viewer_id: int, hand: Tuple[int, ...]):
        for card in hand:
            self.card_locations[card] = "self"

    def apply_reserved_aces(self, recipient_id: int, loss_count: int):
        from engine.deck import get_reserved_aces
        aces = get_reserved_aces(loss_count)
        for ace in aces:
            if self.card_locations[ace] != "self":
                self.card_locations[ace] = f"known_{recipient_id}"
                self.player_known_cards[recipient_id].add(ace)

    def is_player_active_after_trick(self, player_id: int, t_idx: int, round_state) -> bool:
        if round_state.players[player_id].is_active:
            return True
        for future_t in round_state.trick_history[t_idx + 1:]:
            if any(play.player_id == player_id for play in future_t.plays):
                return True
        return False

    def reconstruct(self, viewer_id: int, round_state, match_state):
        num_players = self.num_players
        cache = self._get_cache()
        match_seed = match_state.match_seed

        # Find the longest prefix of trick history cached for this viewer in this round
        cached_tracker = None
        matched_prefix_len = 0

        for prefix_len in range(len(round_state.trick_history), -1, -1):
            prefix_plays = []
            for t in round_state.trick_history[:prefix_len]:
                prefix_plays.append((
                    t.trick_number,
                    tuple((play.player_id, play.card) for play in t.plays),
                    t.outcome,
                    t.collector_id
                ))
            prefix_tuple = tuple(prefix_plays)

            key = (num_players, match_seed, viewer_id, round_state.round_number, prefix_tuple)
            if key in cache:
                cached_tracker = cache[key]
                matched_prefix_len = prefix_len
                break

        if cached_tracker is not None:
            # Load cached snapshot
            self.card_locations = list(cached_tracker.card_locations)
            self.is_void = [list(v) for v in cached_tracker.is_void]
            self.player_known_cards = [set(k) for k in cached_tracker.player_known_cards]
            self.discards = [set(d) for d in cached_tracker.discards]
        else:
            # Reconstruct starting state
            self.card_locations = ["unknown"] * 52
            self.is_void = [[False] * 4 for _ in range(num_players)]
            self.player_known_cards = [set() for _ in range(num_players)]
            self.discards = [set() for _ in range(4)]

            self.record_own_hand(viewer_id, round_state.players[viewer_id].hand)

            # Reserved aces assignment at round start
            r = round_state.round_number
            recipient_id = None
            consecutive_loss_count = 0
            if r > 1:
                prev_result = match_state.round_results[-1]
                if not prev_result.is_draw:
                    recipient_id = prev_result.loser_id
                    if recipient_id is not None:
                        consecutive_loss_count = match_state.players[recipient_id].consecutive_loss_count
            if recipient_id is not None:
                self.apply_reserved_aces(recipient_id, consecutive_loss_count)

            # Cache starting state
            start_key = (num_players, match_seed, viewer_id, round_state.round_number, ())
            cache[start_key] = self.clone()

        # Trace tricks from matched prefix to current trick
        for t_idx in range(matched_prefix_len, len(round_state.trick_history)):
            t = round_state.trick_history[t_idx]
            played_players = {play.player_id for play in t.plays}

            # Detect steals using dynamic activity checks
            for p in range(num_players):
                was_active_before = self.is_player_active_after_trick(p, t_idx - 1, round_state) if t_idx > 0 else True
                is_active_after = self.is_player_active_after_trick(p, t_idx, round_state)

                if was_active_before and not is_active_after:
                    if p not in played_players:
                        stealer = t.plays[0].player_id if t.plays else t.lead_player_id
                        stolen_cards = list(self.player_known_cards[p])
                        for card in stolen_cards:
                            self.card_locations[card] = f"known_{stealer}"
                            self.player_known_cards[stealer].add(card)
                        self.player_known_cards[p].clear()
                        for s in range(4):
                            self.is_void[stealer][s] = self.is_void[p][s]
                            self.is_void[p][s] = True

            # Process trick plays
            if t.plays:
                lead_suit = get_suit(t.plays[0].card)
                for play in t.plays:
                    self.card_locations[play.card] = "played"
                    if play.card in self.player_known_cards[play.player_id]:
                        self.player_known_cards[play.player_id].remove(play.card)
                    if get_suit(play.card) != lead_suit:
                        self.is_void[play.player_id][lead_suit] = True

                # Discard vs Interrupted Resolution
                if t.outcome == "DISCARDED":
                    for play in t.plays:
                        self.discards[get_suit(play.card)].add(play.card)
                elif t.outcome == "INTERRUPTED":
                    collector = t.collector_id
                    if collector is not None and collector != viewer_id:
                        for play in t.plays:
                            self.card_locations[play.card] = f"known_{collector}"
                            self.player_known_cards[collector].add(play.card)
                            self.is_void[collector][get_suit(play.card)] = False

            # Cache the completed trick state
            prefix_plays = []
            for t_prev in round_state.trick_history[:t_idx + 1]:
                prefix_plays.append((
                    t_prev.trick_number,
                    tuple((play.player_id, play.card) for play in t_prev.plays),
                    t_prev.outcome,
                    t_prev.collector_id
                ))
            prefix_tuple = tuple(prefix_plays)
            key = (num_players, match_seed, viewer_id, round_state.round_number, prefix_tuple)
            cache[key] = self.clone()

        # Process current trick steals and plays (not cached since in progress)
        curr_t = round_state.current_trick
        if curr_t is not None:
            for steal in curr_t.steals:
                stolen_cards = list(self.player_known_cards[steal.victim_id])
                for card in stolen_cards:
                    self.card_locations[card] = f"known_{steal.stealer_id}"
                    self.player_known_cards[steal.stealer_id].add(card)
                self.player_known_cards[steal.victim_id].clear()
                for s in range(4):
                    self.is_void[steal.stealer_id][s] = self.is_void[steal.victim_id][s]
                    self.is_void[steal.victim_id][s] = True

            if curr_t.plays:
                lead_suit = get_suit(curr_t.plays[0].card)
                for play in curr_t.plays:
                    self.card_locations[play.card] = "played"
                    if play.card in self.player_known_cards[play.player_id]:
                        self.player_known_cards[play.player_id].remove(play.card)
                    if get_suit(play.card) != lead_suit:
                        self.is_void[play.player_id][lead_suit] = True

        # Re-apply viewer hand locations at the end to ensure they are marked "self", not "played"
        self.record_own_hand(viewer_id, round_state.players[viewer_id].hand)

        # Evict cache entries if cache size grows too large (prevent memory leak)
        if len(cache) > 1000:
            keys = list(cache.keys())
            for k in keys[:200]:
                cache.pop(k, None)

    def get_suit_counts(self, viewer_id: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Compute C_own, C_discard, C_known, and U for all suits."""
        c_own = [0] * 4
        c_discard = [len(self.discards[s]) for s in range(4)]
        c_known = [0] * 4
        
        for card_id in range(52):
            loc = self.card_locations[card_id]
            s = get_suit(card_id)
            if loc == "self":
                c_own[s] += 1
            elif loc.startswith("known_"):
                p_id = int(loc.split("_")[1])
                if p_id != viewer_id:
                    c_known[s] += 1

        u_suit = [0] * 4
        for s in range(4):
            u_suit[s] = 13 - c_own[s] - c_discard[s] - c_known[s]

        return c_own, c_discard, c_known, u_suit


class ProbabilityEstimator:
    """Calculates probabilities and expected distributions (Confidence/Estimations)."""

    def __init__(self, tracker: CardTracker):
        self.tracker = tracker

    def get_void_probability(self, player_id: int, suit: int, round_state) -> float:
        """Estimate the probability that player_id is void in suit."""
        if self.tracker.is_void[player_id][suit]:
            return 1.0
        
        c_own, c_discard, c_known, u_suit = self.tracker.get_suit_counts(player_id)
        u_s = max(0, u_suit[suit])
        if u_s == 0:
            return 1.0

        opponents = [p for p in round_state.active_player_ids if p != player_id]
        if not opponents:
            return 1.0

        h_unknown_target = max(0, len(round_state.players[player_id].hand) - len(self.tracker.player_known_cards[player_id]))
        t_unknown = sum(max(0, len(round_state.players[q].hand) - len(self.tracker.player_known_cards[q])) for q in opponents) + h_unknown_target

        if t_unknown == 0:
            return 1.0

        p_not_target = 1.0 - (h_unknown_target / t_unknown)
        p_not_target = max(0.0, min(1.0, p_not_target))
        return p_not_target ** u_s

    def interruption_probability(self, suit: int, subsequent_players: List[int], round_state) -> float:
        """Calculate the probability that at least one subsequent player breaks suit."""
        prob_no_interruption = 1.0
        for opp in subsequent_players:
            prob_no_interruption *= (1.0 - self.get_void_probability(opp, suit, round_state))
        return 1.0 - prob_no_interruption

    def expected_remaining_cards(self, player_id: int, suit: int, round_state) -> float:
        """Expected count of cards of suit S held by player_id."""
        known_count = sum(1 for c in self.tracker.player_known_cards[player_id] if get_suit(c) == suit)
        _, _, _, u_suit = self.tracker.get_suit_counts(player_id)
        u_s = u_suit[suit]
        
        opponents = [p for p in round_state.active_player_ids if p != player_id]
        h_unknown_target = max(0, len(round_state.players[player_id].hand) - len(self.tracker.player_known_cards[player_id]))
        t_unknown = sum(max(0, len(round_state.players[q].hand) - len(self.tracker.player_known_cards[q])) for q in opponents) + h_unknown_target
        
        if t_unknown == 0:
            return float(known_count)
            
        prob_allocated = h_unknown_target / t_unknown
        return known_count + (u_s * prob_allocated)


class EvaluationMetrics:
    """Computes derived strategic metrics (SuitDanger, SuitControl, CollectorProbability)."""

    def __init__(self, tracker: CardTracker, estimator: ProbabilityEstimator):
        self.tracker = tracker
        self.estimator = estimator

    def suit_danger(self, suit: int, subsequent_players: List[int], round_state) -> float:
        """Danger score representing likelihood and impact of interruption [0.0, 2.0]."""
        _, _, _, u_suit = self.tracker.get_suit_counts(self.tracker.num_players)
        u_s = u_suit[suit]
        
        interruption_p = self.estimator.interruption_probability(suit, subsequent_players, round_state)
        depletion_factor = 1.0 if u_s <= 2 else 0.0
        if u_s == 0:
            depletion_factor = 2.0
            
        return interruption_p + 0.5 * depletion_factor

    def suit_control(self, player_id: int, suit: int, round_state) -> float:
        """Control coefficient of suit held by player [0.0, 1.0]."""
        hand = round_state.players[player_id].hand
        own_count = sum(1 for c in hand if c != -1 and get_suit(c) == suit)
        
        total_in_play = 13 - len(self.tracker.discards[suit])
        if total_in_play <= 0:
            return 0.0
        return own_count / total_in_play


# =====================================================================
# 2. Lazy Evaluation Context (Cached Property Registry)
# =====================================================================

@dataclass
class EvaluationContext:
    """Precomputed parameters and lazy pre-extracted features passed to heuristics."""
    view: EngineState
    viewer_id: int
    num_players: int
    phase: str
    phase_weights: Dict[str, float]
    is_lead_play: bool
    is_last_player: bool
    lead_suit: Optional[int]
    highest_played_lead: int
    subsequent_players: List[int]
    my_turn_idx: int
    trick_order: List[int]
    consecutive_losses: int
    consecutive_loss_multiplier: float
    my_hand_size: int
    opponent_hand_sizes: Dict[int, int]
    avg_opp_hand_size: float

    # Core engines passed for lazy properties evaluation
    tracker: CardTracker
    estimator: ProbabilityEstimator
    metrics: EvaluationMetrics

    # --- Lazy precomputed properties (Concern 4) ---

    @cached_property
    def c_own(self) -> List[int]:
        c_own, _, _, _ = self.tracker.get_suit_counts(self.viewer_id)
        return c_own

    @cached_property
    def c_discard(self) -> List[int]:
        _, c_discard, _, _ = self.tracker.get_suit_counts(self.viewer_id)
        return c_discard

    @cached_property
    def c_known(self) -> List[int]:
        _, _, c_known, _ = self.tracker.get_suit_counts(self.viewer_id)
        return c_known

    @cached_property
    def u_suit(self) -> List[int]:
        _, _, _, u_suit = self.tracker.get_suit_counts(self.viewer_id)
        return u_suit

    @cached_property
    def known_cards_count(self) -> Dict[int, List[int]]:
        counts = {}
        for p in self.view.round_state.active_player_ids:
            counts[p] = [sum(1 for c in self.tracker.player_known_cards[p] if get_suit(c) == s) for s in range(4)]
        return counts

    @cached_property
    def void_probs(self) -> Dict[int, List[float]]:
        probs = {}
        for p in self.view.round_state.active_player_ids:
            probs[p] = [self.estimator.get_void_probability(p, s, self.view.round_state) for s in range(4)]
        return probs

    @cached_property
    def suit_controls(self) -> Dict[int, List[float]]:
        controls = {}
        for p in self.view.round_state.active_player_ids:
            controls[p] = [self.metrics.suit_control(p, s, self.view.round_state) for s in range(4)]
        return controls

    @cached_property
    def expected_cards(self) -> Dict[int, List[float]]:
        expected = {}
        for p in self.view.round_state.active_player_ids:
            expected[p] = [self.estimator.expected_remaining_cards(p, s, self.view.round_state) for s in range(4)]
        return expected

    @cached_property
    def interruption_probs(self) -> List[float]:
        return [self.estimator.interruption_probability(s, self.subsequent_players, self.view.round_state) for s in range(4)]

    @cached_property
    def suit_dangers(self) -> List[float]:
        return [self.metrics.suit_danger(s, self.subsequent_players, self.view.round_state) for s in range(4)]

    @cached_property
    def highest_known_opp_ranks(self) -> List[int]:
        ranks = [0] * 4
        opponents = [opp for opp in self.view.round_state.active_player_ids if opp != self.viewer_id]
        for s in range(4):
            for opp in opponents:
                for c in self.tracker.player_known_cards[opp]:
                    if get_suit(c) == s:
                        r_c = get_power_rank(c)
                        if r_c > ranks[s]:
                            ranks[s] = r_c
        return ranks

    @cached_property
    def highest_known_subsequent_ranks(self) -> List[int]:
        ranks = [0] * 4
        for s in range(4):
            for opp in self.subsequent_players:
                for c in self.tracker.player_known_cards[opp]:
                    if get_suit(c) == s:
                        r_c = get_power_rank(c)
                        if r_c > ranks[s]:
                            ranks[s] = r_c
        return ranks

    @cached_property
    def any_subsequent_void(self) -> List[bool]:
        voids = [False] * 4
        for s in range(4):
            for opp in self.subsequent_players:
                if self.tracker.is_void[opp][s]:
                    voids[s] = True
                    break
        return voids


class FeatureExtractor:
    """Precomputes and organizes all strategic features once per decision turn (Concern 5)."""

    @staticmethod
    def extract_features(view: EngineState, tracker: CardTracker, estimator: ProbabilityEstimator, metrics: EvaluationMetrics, viewer_id: int, phase: str, phase_weights: Dict[str, float]) -> EvaluationContext:
        round_st = view.round_state
        match_st = view.match_state

        play_seq = FeatureExtractor._build_play_sequence(round_st, viewer_id)
        match_ctx = FeatureExtractor._calculate_match_context(round_st, match_st, viewer_id)

        return EvaluationContext(
            view=view,
            viewer_id=viewer_id,
            num_players=len(round_st.players),
            phase=phase,
            phase_weights=phase_weights,
            is_lead_play=play_seq["is_lead_play"],
            is_last_player=play_seq["is_last_player"],
            lead_suit=play_seq["lead_suit"],
            highest_played_lead=play_seq["highest_played_lead"],
            subsequent_players=play_seq["subsequent_players"],
            my_turn_idx=play_seq["my_turn_idx"],
            trick_order=play_seq["trick_order"],
            consecutive_losses=match_ctx["consecutive_losses"],
            consecutive_loss_multiplier=match_ctx["consecutive_loss_multiplier"],
            my_hand_size=match_ctx["my_hand_size"],
            opponent_hand_sizes=match_ctx["opponent_hand_sizes"],
            avg_opp_hand_size=match_ctx["avg_opp_hand_size"],
            tracker=tracker,
            estimator=estimator,
            metrics=metrics
        )

    @staticmethod
    def _build_play_sequence(round_st, viewer_id: int) -> Dict[str, Any]:
        curr_t = round_st.current_trick
        is_lead_play = (curr_t is None or len(curr_t.plays) == 0)

        is_last_player = False
        highest_played_lead = -1
        lead_suit = None
        subsequent_players = []
        my_turn_idx = 0
        trick_order = []

        if not is_lead_play and curr_t is not None:
            lead_suit = curr_t.lead_suit
            if lead_suit is not None:
                for play in curr_t.plays:
                    if get_suit(play.card) == lead_suit:
                        r_p = get_power_rank(play.card)
                        if r_p > highest_played_lead:
                            highest_played_lead = r_p
            
            if curr_t.lead_player_id in round_st.active_player_ids:
                active_rotation = list(round_st.active_player_ids)
                lead_idx = active_rotation.index(curr_t.lead_player_id)
                trick_order = active_rotation[lead_idx:] + active_rotation[:lead_idx]
                my_turn_idx = len(curr_t.plays)
                is_last_player = (my_turn_idx == len(trick_order) - 1)
                subsequent_players = trick_order[my_turn_idx + 1:]
        else:
            active_rotation = list(round_st.active_player_ids)
            if viewer_id in active_rotation:
                my_idx = active_rotation.index(viewer_id)
                subsequent_players = active_rotation[my_idx + 1:] + active_rotation[:my_idx]

        return {
            "is_lead_play": is_lead_play,
            "is_last_player": is_last_player,
            "lead_suit": lead_suit,
            "highest_played_lead": highest_played_lead,
            "subsequent_players": subsequent_players,
            "my_turn_idx": my_turn_idx,
            "trick_order": trick_order
        }

    @staticmethod
    def _calculate_match_context(round_st, match_st, viewer_id: int) -> Dict[str, Any]:
        consecutive_losses = match_st.players[viewer_id].consecutive_loss_count
        consecutive_loss_multiplier = 1.5 if consecutive_losses > 0 else 1.0

        my_hand_size = len(round_st.players[viewer_id].hand)
        opponent_hand_sizes = {p: len(round_st.players[p].hand) for p in round_st.active_player_ids if p != viewer_id}
        avg_opp_hand_size = sum(opponent_hand_sizes.values()) / len(opponent_hand_sizes) if opponent_hand_sizes else 0.0

        return {
            "consecutive_losses": consecutive_losses,
            "consecutive_loss_multiplier": consecutive_loss_multiplier,
            "my_hand_size": my_hand_size,
            "opponent_hand_sizes": opponent_hand_sizes,
            "avg_opp_hand_size": avg_opp_hand_size
        }


@dataclass(frozen=True)
class ActionEvaluation:
    """Explainable result containing the scoring breakdown of an action."""
    action: Action
    total_score: float
    breakdown: Dict[str, float]


# =====================================================================
# 3. Abstract Base Class Rule Interface
# =====================================================================

class HeuristicRule(ABC):
    """Interface for a modular heuristic evaluator returning normalized scores [-1.0, 1.0] (Concern 1)."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Heuristic catalogue unique ID (e.g. 'H203')."""
        pass

    @property
    def default_weight(self) -> float:
        """Default score weight multiplier configured on decentralized class (Concern 2)."""
        return 100.0

    @abstractmethod
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        """Returns a normalized score between -1.0 and 1.0 representing activation strength."""
        pass


# --- Steal Heuristics ---

class H102_ExpectedSuitConcentration(HeuristicRule):
    @property
    def id(self) -> str: return "H102"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]
            
            current_concentration = max(ctx.c_own) / ctx.my_hand_size if ctx.my_hand_size > 0 else 0.0

            merged_counts = list(ctx.c_own)
            for card in tracker.player_known_cards[target]:
                merged_counts[get_suit(card)] += 1
            
            victim_size = ctx.opponent_hand_sizes[target]
            merged_size = ctx.my_hand_size + victim_size
            merged_concentration = max(merged_counts) / merged_size if merged_size > 0 else 0.0
            
            suspected_low_ranks = sum(1 for c in tracker.player_known_cards[target] if get_power_rank(c) <= 5)
            improvement = (merged_concentration - current_concentration)
            low_rank_bonus = min(1.0, suspected_low_ranks / 5.0)
            return 0.5 * improvement + 0.5 * low_rank_bonus
        return 0.0

class H103_StealCardCountPenalty(HeuristicRule):
    @property
    def id(self) -> str: return "H103"
    @property
    def default_weight(self) -> float: return 40.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]
            victim_size = ctx.opponent_hand_sizes[target]
            return -min(1.0, victim_size / 15.0)
        return 0.0

class H104_SuitIntersectionStealFilter(HeuristicRule):
    @property
    def id(self) -> str: return "H104"
    @property
    def default_weight(self) -> float: return 100.0
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
                if ctx.c_own[s] > 0 and r_val <= 5:
                    has_known_low_ranks = True
            
            if has_intersection and not has_known_low_ranks:
                return -1.0
            return 1.0
        return 0.0

class H105_EndgameTargetedSteal(HeuristicRule):
    @property
    def id(self) -> str: return "H105"
    @property
    def default_weight(self) -> float: return 150.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction) and ctx.phase == "Endgame":
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]
            victim_size = ctx.opponent_hand_sizes[target]

            if victim_size <= 2:
                low_ranks = sum(1 for c in tracker.player_known_cards[target] if get_power_rank(c) <= 8)
                void_penalty = sum(1 for s in range(4) if ctx.c_own[s] == 0 and not tracker.is_void[target][s])
                high_ranks = sum(1 for c in tracker.player_known_cards[target] if get_power_rank(c) >= 12)
                
                score = 0.5 + (0.25 * low_ranks) - (0.3 * void_penalty) - (0.2 * high_ranks)
                return max(-1.0, min(1.0, score))
        return 0.0

class H107_StealToMitigateBadLeads(HeuristicRule):
    @property
    def id(self) -> str: return "H107"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction) and ctx.viewer_id == ctx.view.round_state.current_trick.lead_player_id:
            all_leads_unsafe = True
            opponents = [p for p in ctx.view.round_state.active_player_ids if p != ctx.viewer_id]
            for card in ctx.view.round_state.players[ctx.viewer_id].hand:
                s = get_suit(card)
                if not any(tracker.is_void[opp][s] for opp in opponents):
                    all_leads_unsafe = False
                    break
            if all_leads_unsafe:
                return 0.9
        return 0.0

class H109_DeclineStealBalanced(HeuristicRule):
    @property
    def id(self) -> str: return "H109"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, DeclineStealAction):
            if all(ctx.c_own[s] >= 2 or ctx.c_own[s] == 0 for s in range(4)):
                return 0.7
        return 0.0

class H110_StealDisadvantageAssessment(HeuristicRule):
    @property
    def id(self) -> str: return "H110"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        curr_trick = ctx.view.round_state.current_trick
        target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]

        if isinstance(action, DeclineStealAction):
            score = 0.0
            for s in range(4):
                if ctx.c_own[s] == 0 and ctx.known_cards_count[target][s] > 0:
                    score -= 0.4
            for s in range(4):
                if ctx.known_cards_count[target][s] == 1:
                    score -= 0.3
            return max(-1.0, score)
        
        elif isinstance(action, StealAction):
            threats = 0
            for s in range(4):
                if ctx.c_own[s] > 0 and tracker.is_void[target][s]:
                    threats += 1
            return min(1.0, threats * 0.4)

        return 0.0

class H111_StealTimingOptimizer(HeuristicRule):
    @property
    def id(self) -> str: return "H111"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            if ctx.phase == "Opening":
                return -0.5
            elif ctx.phase == "Endgame":
                return 0.3
        return 0.0

class H112_StealForFutureBenefit(HeuristicRule):
    @property
    def id(self) -> str: return "H112"
    @property
    def default_weight(self) -> float: return 100.0
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
                if r_val <= 5:
                    score += 0.2
                if ctx.c_own[s] == 0 and r_val >= 13:
                    score -= 0.4
            return max(-1.0, min(1.0, score))
        return 0.0

class H113_ChainStealEvaluation(HeuristicRule):
    @property
    def id(self) -> str: return "H113"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if len(ctx.view.round_state.current_trick.steals) > 0:
            if isinstance(action, StealAction):
                return 0.4
            elif isinstance(action, DeclineStealAction):
                if any(ctx.c_own[s] == 0 or ctx.c_own[s] >= 5 for s in range(4)):
                    return 1.0
        return 0.0

class H115_StealToDenyVoid(HeuristicRule):
    @property
    def id(self) -> str: return "H115"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]

            has_singleton = False
            for s in range(4):
                if ctx.known_cards_count[target][s] == 1 and ctx.c_own[s] > 0:
                    has_singleton = True
            if has_singleton:
                return 0.6
        return 0.0

class H117_ReservedAceStealConsideration(HeuristicRule):
    @property
    def id(self) -> str: return "H117"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, StealAction):
            curr_trick = ctx.view.round_state.current_trick
            target = curr_trick.plays[0].player_id if curr_trick and curr_trick.plays else ctx.subsequent_players[0]
            
            r = ctx.view.round_state.round_number
            recipient_id = None
            consecutive_loss_count = 0
            if r > 1:
                prev_result = ctx.view.match_state.round_results[-1]
                if not prev_result.is_draw:
                    recipient_id = prev_result.loser_id
                    if recipient_id is not None:
                        consecutive_loss_count = ctx.view.match_state.players[recipient_id].consecutive_loss_count

            if recipient_id == target and consecutive_loss_count > 0:
                from engine.deck import get_reserved_aces
                aces = get_reserved_aces(consecutive_loss_count)
                score = 0.0
                for ace in aces:
                    if tracker.card_locations[ace] == f"known_{target}":
                        s = get_suit(ace)
                        if ctx.c_own[s] > 0:
                            score += 0.4
                        elif ctx.c_own[s] == 0:
                            score -= 0.3
                return max(-1.0, min(1.0, score))
        return 0.0


# --- Lead Heuristics ---

class H202_HighRankLeadOpening(HeuristicRule):
    @property
    def id(self) -> str: return "H202"
    @property
    def default_weight(self) -> float: return 120.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase == "Opening":
            r_val = get_power_rank(action.card)
            return (r_val - 2) / 12.0
        return 0.0

class H203_OpponentVoidAvoidance(HeuristicRule):
    @property
    def id(self) -> str: return "H203"
    @property
    def default_weight(self) -> float: return 150.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            danger = ctx.suit_dangers[s]
            if danger > 0.5:
                return -danger
        return 0.0

class H204_LowRankLeadMiddleEndgame(HeuristicRule):
    @property
    def id(self) -> str: return "H204"
    @property
    def default_weight(self) -> float: return 120.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            r_val = get_power_rank(action.card)
            return (14 - r_val) / 12.0
        return 0.0

class H205_SuitDepletionRiskLead(HeuristicRule):
    @property
    def id(self) -> str: return "H205"
    @property
    def default_weight(self) -> float: return 250.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            if ctx.u_suit[s] <= 2:
                if ctx.u_suit[s] == 0:
                    return -1.0
                return -0.5
        return 0.0

class H206_LureLead(HeuristicRule):
    @property
    def id(self) -> str: return "H206"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            has_following = any(not tracker.is_void[opp][s] for opp in ctx.subsequent_players)
            
            if ctx.any_subsequent_void[s] and has_following and r_val <= 5:
                score = 0.5
                if ctx.highest_known_subsequent_ranks[s] > r_val:
                    return score + 0.3
                return score
        return 0.0

class H208_VoidPromotingLead(HeuristicRule):
    @property
    def id(self) -> str: return "H208"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Opening", "Middle"):
            s = get_suit(action.card)
            if ctx.c_own[s] == 1:
                return 0.8
            elif ctx.c_own[s] == 2:
                return 0.4
        return 0.0

class H209_TargetSpecificVoidProbe(HeuristicRule):
    @property
    def id(self) -> str: return "H209"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if ctx.subsequent_players:
                threat_opp = min(ctx.subsequent_players, key=lambda p: len(ctx.view.round_state.players[p].hand))
                if tracker.is_void[threat_opp][s]:
                    return 0.9 if r_val <= 8 else -0.5
        return 0.0

class H210_SafeLeaderWithKnownCollector(HeuristicRule):
    @property
    def id(self) -> str: return "H210"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if ctx.highest_known_subsequent_ranks[s] > r_val and r_val <= 7:
                return 1.0
        return 0.0

class H211_SuitHoardingAceExposer(HeuristicRule):
    @property
    def id(self) -> str: return "H211"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Opening", "Middle"):
            s = get_suit(action.card)
            if ctx.c_own[s] >= 5:
                if get_rank(action.card) == RANK_ACE:
                    return 0.9
                elif get_power_rank(action.card) >= 10:
                    return -0.8
        return 0.0

class H213_ShortSuitLeadOpening(HeuristicRule):
    @property
    def id(self) -> str: return "H213"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase == "Opening":
            s = get_suit(action.card)
            if ctx.c_own[s] == 1:
                return 0.85
            elif ctx.c_own[s] == 2:
                return 0.5
        return 0.0

class H214_AvoidLeadingOpponentCollectedSuit(HeuristicRule):
    @property
    def id(self) -> str: return "H214"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            if any(ctx.known_cards_count[opp][s] >= 3 for opp in ctx.subsequent_players):
                return -1.0
        return 0.0

class H215_SafeExitLead(HeuristicRule):
    @property
    def id(self) -> str: return "H215"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase in ("Middle", "Endgame"):
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if ctx.highest_known_subsequent_ranks[s] > r_val:
                return 0.9
        return 0.0

class H216_LeadSuitWithMostCardsOpening(HeuristicRule):
    @property
    def id(self) -> str: return "H216"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and ctx.phase == "Opening":
            s = get_suit(action.card)
            if ctx.c_own[s] >= 7:
                return 0.6
            elif ctx.c_own[s] >= 5:
                return 0.4
        return 0.0

class H217_AvoidLeadingSuitWeJustCollected(HeuristicRule):
    @property
    def id(self) -> str: return "H217"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            history = ctx.view.round_state.trick_history
            if len(history) > 0:
                last_t = history[-1]
                if last_t.outcome == "INTERRUPTED" and last_t.collector_id == ctx.viewer_id:
                    if any(get_suit(c) == s for c in last_t.collected_cards):
                        return -0.8
        return 0.0

class H218_LeadSuitAllCanFollow(HeuristicRule):
    @property
    def id(self) -> str: return "H218"
    @property
    def default_weight(self) -> float: return 130.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            if not ctx.any_subsequent_void[s]:
                return 1.0
        return 0.0

class H219_AceLeadAllCanFollow(HeuristicRule):
    @property
    def id(self) -> str: return "H219"
    @property
    def default_weight(self) -> float: return 140.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and get_rank(action.card) == RANK_ACE:
            if ctx.phase in ("Opening", "Middle"):
                s = get_suit(action.card)
                if not ctx.any_subsequent_void[s]:
                    return 1.0
        return 0.0

class H220_LeadToForceOpponentCollection(HeuristicRule):
    @property
    def id(self) -> str: return "H220"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if ctx.highest_known_subsequent_ranks[s] >= 10 and r_val <= 5:
                return 0.85
        return 0.0

class H221_AvoidLeadingOnlyTwoPlayers(HeuristicRule):
    @property
    def id(self) -> str: return "H221"
    @property
    def default_weight(self) -> float: return 200.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play and len(ctx.view.round_state.active_player_ids) == 2:
            s = get_suit(action.card)
            opponent = ctx.subsequent_players[0]
            opp_void_prob = ctx.void_probs[opponent][s]
            if opp_void_prob > 0.5:
                return -1.0
            return 0.5
        return 0.0

class H222_LeadCommonRemainingSuit(HeuristicRule):
    @property
    def id(self) -> str: return "H222"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            if ctx.u_suit[s] == max(ctx.u_suit):
                return 0.5
        return 0.0

class H223_PostCollectionSafeLead(HeuristicRule):
    @property
    def id(self) -> str: return "H223"
    @property
    def default_weight(self) -> float: return 150.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.is_lead_play:
            s = get_suit(action.card)
            history = ctx.view.round_state.trick_history
            if len(history) > 0:
                last_t = history[-1]
                if last_t.outcome == "INTERRUPTED" and last_t.collector_id == ctx.viewer_id:
                    danger = ctx.suit_dangers[s]
                    if danger > 0.5:
                        return -1.0
                    return 1.0
        return 0.0


# --- Follow Heuristics ---

class H301_SafeDiscardDumpLastPlayer(HeuristicRule):
    @property
    def id(self) -> str: return "H301"
    @property
    def default_weight(self) -> float: return 225.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.is_last_player:
                has_anyone_broken = any(get_suit(p.card) != ctx.lead_suit for p in ctx.view.round_state.current_trick.plays)
                if not has_anyone_broken:
                    return (r_val - 2) / 12.0
        return 0.0

class H303_SuspectedInterruptionUnderplay(HeuristicRule):
    @property
    def id(self) -> str: return "H303"
    @property
    def default_weight(self) -> float: return 250.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame"):
                interruption_p = ctx.interruption_probs[ctx.lead_suit]
                if interruption_p > 0.5 and r_val > ctx.highest_played_lead:
                    return -1.0
        return 0.0

class H304_InterruptionRiskUnderplay(HeuristicRule):
    @property
    def id(self) -> str: return "H304"
    @property
    def default_weight(self) -> float: return 200.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and ctx.u_suit[ctx.lead_suit] <= 2:
                if r_val >= 10:
                    return -1.0
        return 0.0

class H305_MiddleEndgameCaution(HeuristicRule):
    @property
    def id(self) -> str: return "H305"
    @property
    def default_weight(self) -> float: return 125.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            
            has_anyone_broken = any(get_suit(p.card) != ctx.lead_suit for p in ctx.view.round_state.current_trick.plays)
            is_safe_discard = ctx.is_last_player and not has_anyone_broken
            
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and not is_safe_discard:
                if r_val > ctx.highest_played_lead:
                    return -0.8
        return 0.0

class H306_SpadeFollowConservation(HeuristicRule):
    @property
    def id(self) -> str: return "H306"
    @property
    def default_weight(self) -> float: return 180.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit == 0:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == 0:
                return (14 - r_val) / 12.0
        return 0.0

class H307_SequentialUnderplay(HeuristicRule):
    @property
    def id(self) -> str: return "H307"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and r_val < ctx.highest_played_lead:
                return 0.8
        return 0.0

class H308_HandBalancingFollow(HeuristicRule):
    @property
    def id(self) -> str: return "H308"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s == ctx.lead_suit:
                if ctx.c_own[s] == 1:
                    return 0.9
                elif ctx.c_own[s] >= 4:
                    return 0.5
        return 0.0

class H310_FollowKnownCardsFirst(HeuristicRule):
    @property
    def id(self) -> str: return "H310"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s == ctx.lead_suit and action.card in tracker.player_known_cards[ctx.viewer_id]:
                return 0.7
        return 0.0

class H311_UnderplaySubsequentPlayerHigh(HeuristicRule):
    @property
    def id(self) -> str: return "H311"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame"):
                if ctx.highest_known_subsequent_ranks[ctx.lead_suit] > r_val:
                    return 0.8
        return 0.0

class H313_RankEstimationFollow(HeuristicRule):
    @property
    def id(self) -> str: return "H313"
    @property
    def default_weight(self) -> float: return 130.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase in ("Middle", "Endgame"):
                r_est_max = ctx.highest_known_subsequent_ranks[ctx.lead_suit]
                if r_est_max > 0:
                    if r_val < r_est_max:
                        return 0.6
                    else:
                        return -0.9
        return 0.0

class H314_EndgameKnownCardExploitation(HeuristicRule):
    @property
    def id(self) -> str: return "H314"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit and ctx.phase == "Endgame":
                any_can_beat = (ctx.highest_known_subsequent_ranks[ctx.lead_suit] > r_val)
                any_void = ctx.any_subsequent_void[ctx.lead_suit]
                
                if not any_can_beat and not any_void:
                    return 1.0
                elif any_void:
                    my_lead_cards = [c for c in ctx.view.round_state.players[ctx.viewer_id].hand if get_suit(c) == ctx.lead_suit]
                    if my_lead_cards and action.card == min(my_lead_cards, key=lambda c: get_power_rank(c)):
                        return 0.5
        return 0.0

class H315_EndgamePerfectInformationFollow(HeuristicRule):
    @property
    def id(self) -> str: return "H315"
    @property
    def default_weight(self) -> float: return 200.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            
            if s == ctx.lead_suit and ctx.phase == "Endgame" and ctx.u_suit[ctx.lead_suit] == 0:
                any_void = ctx.any_subsequent_void[ctx.lead_suit]
                is_safe = True
                if any_void:
                    if r_val > ctx.highest_played_lead:
                        if ctx.highest_known_subsequent_ranks[ctx.lead_suit] > r_val:
                            is_safe = False
                if is_safe:
                    return 1.0
                return -1.0
        return 0.0

class H316_FollowHighWhenSafe(HeuristicRule):
    @property
    def id(self) -> str: return "H316"
    @property
    def default_weight(self) -> float: return 125.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s == ctx.lead_suit:
                if not ctx.any_subsequent_void[ctx.lead_suit]:
                    my_lead_cards = [c for c in ctx.view.round_state.players[ctx.viewer_id].hand if get_suit(c) == ctx.lead_suit]
                    if my_lead_cards and action.card == max(my_lead_cards, key=lambda c: get_power_rank(c)):
                        return 0.95
        return 0.0

class H317_PositionAwareFollow(HeuristicRule):
    @property
    def id(self) -> str: return "H317"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s == ctx.lead_suit:
                if ctx.my_turn_idx == 1:
                    if r_val >= 10:
                        return -0.3
                elif ctx.is_last_player:
                    my_lead_cards = [c for c in ctx.view.round_state.players[ctx.viewer_id].hand if get_suit(c) == ctx.lead_suit]
                    if my_lead_cards and action.card == max(my_lead_cards, key=lambda c: get_power_rank(c)):
                        return 0.3
        return 0.0


# --- Break Heuristics ---

class H401_HighRankOffSuitDump(HeuristicRule):
    @property
    def id(self) -> str: return "H401"
    @property
    def default_weight(self) -> float: return 280.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s != ctx.lead_suit and s != 0:
                return r_val / 14.0
        return 0.0

class H402_PreserveFutureVoids(HeuristicRule):
    @property
    def id(self) -> str: return "H402"
    @property
    def default_weight(self) -> float: return 150.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s != ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and ctx.c_own[s] == 1:
                return 0.8
        return 0.0

class H403_CollectorVoidDisruption(HeuristicRule):
    @property
    def id(self) -> str: return "H403"
    @property
    def default_weight(self) -> float: return 125.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            
            collector_id = None
            highest_lead_val = -1
            for play in ctx.view.round_state.current_trick.plays:
                if get_suit(play.card) == ctx.lead_suit:
                    r_c = get_power_rank(play.card)
                    if r_c > highest_lead_val:
                        highest_lead_val = r_c
                        collector_id = play.player_id
            
            if s != ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and collector_id is not None:
                if tracker.is_void[collector_id][s]:
                    return 0.8
        return 0.0

class H405_FirstBreakSuitSelection(HeuristicRule):
    @property
    def id(self) -> str: return "H405"
    @property
    def default_weight(self) -> float: return 200.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            
            has_prior_broken = any(get_suit(p.card) != ctx.lead_suit for p in ctx.view.round_state.current_trick.plays)
            
            if s != ctx.lead_suit and not has_prior_broken:
                # We are the first to break suit!
                score = 0.0
                if r_val >= 10 and s != 0:
                    score += 0.40
                
                collector_id = None
                highest_lead_val = -1
                for play in ctx.view.round_state.current_trick.plays:
                    if get_suit(play.card) == ctx.lead_suit:
                        r_c = get_power_rank(play.card)
                        if r_c > highest_lead_val:
                            highest_lead_val = r_c
                            collector_id = play.player_id
                
                if collector_id is not None and tracker.is_void[collector_id][s]:
                    score += 0.50
                
                if ctx.c_own[s] == 2:
                    score += 0.45
                if s == 0:
                    score -= 0.30
                return max(-1.0, min(1.0, score))
        return 0.0

class H406_SpecificSuitDrainingDiscard(HeuristicRule):
    @property
    def id(self) -> str: return "H406"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s != ctx.lead_suit and ctx.phase in ("Opening", "Middle"):
                if ctx.c_own[s] == 2:
                    return 0.7
                elif ctx.c_own[s] == 1:
                    return 1.0
        return 0.0

class H408_SacrificeDiscard(HeuristicRule):
    @property
    def id(self) -> str: return "H408"
    @property
    def default_weight(self) -> float: return 140.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            
            collector_id = None
            highest_lead_val = -1
            for play in ctx.view.round_state.current_trick.plays:
                if get_suit(play.card) == ctx.lead_suit:
                    r_c = get_power_rank(play.card)
                    if r_c > highest_lead_val:
                        highest_lead_val = r_c
                        collector_id = play.player_id

            if s != ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and collector_id is not None and collector_id != ctx.viewer_id:
                if r_val >= 13 and s != 0:
                    return 0.8
        return 0.0

class H410_CreateMultiSuitVoids(HeuristicRule):
    @property
    def id(self) -> str: return "H410"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s != ctx.lead_suit and ctx.phase in ("Middle", "Endgame"):
                near_empty_count = sum(1 for suit_idx in range(4) if ctx.c_own[suit_idx] <= 2 and ctx.c_own[suit_idx] > 0)
                if near_empty_count >= 2:
                    if ctx.c_own[s] == 1:
                        return 0.9
                    elif ctx.c_own[s] == 2:
                        return 0.5
        return 0.0

class H411_LowRankDrainVoidAcceleration(HeuristicRule):
    @property
    def id(self) -> str: return "H411"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            if s != ctx.lead_suit and ctx.phase in ("Middle", "Endgame"):
                if ctx.c_own[s] <= 2 and r_val <= 6:
                    return 0.75
        return 0.0

class H412_BreakWithCollectorMissingSuit(HeuristicRule):
    @property
    def id(self) -> str: return "H412"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            
            collector_id = None
            highest_lead_val = -1
            for play in ctx.view.round_state.current_trick.plays:
                if get_suit(play.card) == ctx.lead_suit:
                    r_c = get_power_rank(play.card)
                    if r_c > highest_lead_val:
                        highest_lead_val = r_c
                        collector_id = play.player_id

            if s != ctx.lead_suit and ctx.phase in ("Middle", "Endgame") and collector_id is not None:
                if tracker.is_void[collector_id][s]:
                    score = 0.5
                    if r_val >= 10:
                        score += 0.3
                    return min(1.0, score)
        return 0.0

class H413_AvoidBreakingLongestSuit(HeuristicRule):
    @property
    def id(self) -> str: return "H413"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and not ctx.is_lead_play and ctx.lead_suit is not None:
            s = get_suit(action.card)
            if s != ctx.lead_suit:
                if ctx.c_own[s] == max(ctx.c_own):
                    return -0.4
        return 0.0


# --- Positional, Match-level & Meta Heuristics ---

class H502_ActivePlayerCountAdjustment(HeuristicRule):
    @property
    def id(self) -> str: return "H502"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction):
            r_val = get_power_rank(action.card)
            if len(ctx.view.round_state.active_player_ids) == 3 and r_val >= 10:
                return -0.15
        return 0.0

class H503_HandSizeRelativeAssessment(HeuristicRule):
    @property
    def id(self) -> str: return "H503"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction):
            r_val = get_power_rank(action.card)
            if ctx.my_hand_size > 1.5 * ctx.avg_opp_hand_size:
                if r_val >= 10:
                    return -0.2
            elif ctx.my_hand_size < 0.7 * ctx.avg_opp_hand_size:
                if not ctx.is_lead_play and not ctx.is_last_player and r_val >= 10:
                    return 0.3
        return 0.0

class H504_WinProximityBonus(HeuristicRule):
    @property
    def id(self) -> str: return "H504"
    @property
    def default_weight(self) -> float: return 150.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.my_hand_size <= 3:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            
            play_might_collect = False
            if ctx.is_lead_play:
                if any(tracker.is_void[opp][s] for opp in ctx.subsequent_players):
                    play_might_collect = True
            else:
                if not ctx.is_last_player and r_val > ctx.highest_played_lead:
                    play_might_collect = True
            
            if play_might_collect:
                return -1.0
            return 0.5
        return 0.0

class H508_RoundDrawAwareness(HeuristicRule):
    @property
    def id(self) -> str: return "H508"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction):
            r_val = get_power_rank(action.card)
            if all(size <= 2 for size in ctx.opponent_hand_sizes.values()):
                if r_val <= 6:
                    return 0.4
        return 0.0

class H702_FinalRoundPressure(HeuristicRule):
    @property
    def id(self) -> str: return "H702"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.view.match_state.current_round == ctx.view.match_state.num_rounds:
            my_points = ctx.view.match_state.players[ctx.viewer_id].half_points
            opponents = [p for p in range(ctx.num_players) if p != ctx.viewer_id]
            max_opp_points = max(ctx.view.match_state.players[p].half_points for p in opponents)
            r_val = get_power_rank(action.card)

            if my_points < max_opp_points:
                if r_val >= 12:
                    return 0.2
            else:
                if r_val <= 6:
                    return 0.3
        return 0.0

class H703_TargetTrailingPlayer(HeuristicRule):
    @property
    def id(self) -> str: return "H703"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction):
            opponents = [p for p in ctx.view.round_state.active_player_ids if p != ctx.viewer_id]
            trailing_opps = [p for p in opponents if ctx.view.match_state.players[p].consecutive_loss_count > 0]
            if trailing_opps:
                r_val = get_power_rank(action.card)
                return 0.3 if r_val <= 6 else -0.3
        return 0.0

class H705_OpponentNearVictoryThreat(HeuristicRule):
    @property
    def id(self) -> str: return "H705"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction):
            victory_threshold = int(ctx.view.match_state.num_rounds * 0.6)
            dangerous = [p for p in range(ctx.num_players) if p != ctx.viewer_id and ctx.view.match_state.players[p].rounds_won >= victory_threshold]
            if dangerous:
                r_val = get_power_rank(action.card)
                return 0.25 if r_val <= 6 else -0.25
        return 0.0

class H706_EarlyRoundStealExploitation(HeuristicRule):
    @property
    def id(self) -> str: return "H706"
    @property
    def default_weight(self) -> float: return 100.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if len(ctx.view.round_state.trick_history) == 0:
            if isinstance(action, StealAction):
                return 0.5
            elif isinstance(action, DeclineStealAction):
                singletons = sum(1 for s in range(4) if ctx.c_own[s] == 1)
                voids = sum(1 for s in range(4) if ctx.c_own[s] == 0)
                if singletons >= 3 or voids >= 1:
                    return 0.8
        return 0.0

class H707_LossSpiralRecoveryPlay(HeuristicRule):
    @property
    def id(self) -> str: return "H707"
    @property
    def default_weight(self) -> float: return 130.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and ctx.consecutive_losses >= 2:
            from engine.deck import get_reserved_aces
            aces = get_reserved_aces(ctx.consecutive_losses)
            if action.card in aces:
                s = get_suit(action.card)
                if not ctx.any_subsequent_void[s]:
                    return 1.0
        return 0.0

class H708_TwoPlayerEndgamePerfectPlay(HeuristicRule):
    @property
    def id(self) -> str: return "H708"
    @property
    def default_weight(self) -> float: return 250.0
    def evaluate(self, action: Action, tracker: CardTracker, ctx: EvaluationContext) -> float:
        if isinstance(action, PlayCardAction) and len(ctx.view.round_state.active_player_ids) == 2:
            s = get_suit(action.card)
            r_val = get_power_rank(action.card)
            opponent = next(p for p in ctx.view.round_state.active_player_ids if p != ctx.viewer_id)

            expected_growth = 0.0
            if ctx.is_lead_play:
                opp_void_prob = ctx.void_probs[opponent][s]
                expected_growth = opp_void_prob * 1.0
            else:
                assert ctx.lead_suit is not None
                if s == ctx.lead_suit:
                    if r_val > ctx.highest_played_lead:
                        expected_growth = 1.0
                    else:
                        expected_growth = 0.0
                else:
                    expected_growth = -1.0

            if expected_growth <= 0.0:
                return 1.0
            return -1.0
        return 0.0


# =====================================================================
# 4. Master Heuristic Agent & Registry
# =====================================================================

class HeuristicAgent(BaseAgent):
    """Refactored HeuristicAgent using modular rule classes and explainability."""

    def __init__(self, player_id: int, seed: Optional[int] = None) -> None:
        super().__init__(player_id, seed)
        
        self.rules: List[HeuristicRule] = [
            H102_ExpectedSuitConcentration(),
            H103_StealCardCountPenalty(),
            H104_SuitIntersectionStealFilter(),
            H105_EndgameTargetedSteal(),
            H107_StealToMitigateBadLeads(),
            H109_DeclineStealBalanced(),
            H110_StealDisadvantageAssessment(),
            H111_StealTimingOptimizer(),
            H112_StealForFutureBenefit(),
            H113_ChainStealEvaluation(),
            H115_StealToDenyVoid(),
            H117_ReservedAceStealConsideration(),
            H202_HighRankLeadOpening(),
            H203_OpponentVoidAvoidance(),
            H204_LowRankLeadMiddleEndgame(),
            H205_SuitDepletionRiskLead(),
            H206_LureLead(),
            H208_VoidPromotingLead(),
            H209_TargetSpecificVoidProbe(),
            H210_SafeLeaderWithKnownCollector(),
            H211_SuitHoardingAceExposer(),
            H213_ShortSuitLeadOpening(),
            H214_AvoidLeadingOpponentCollectedSuit(),
            H215_SafeExitLead(),
            H216_LeadSuitWithMostCardsOpening(),
            H217_AvoidLeadingSuitWeJustCollected(),
            H218_LeadSuitAllCanFollow(),
            H219_AceLeadAllCanFollow(),
            H220_LeadToForceOpponentCollection(),
            H221_AvoidLeadingOnlyTwoPlayers(),
            H222_LeadCommonRemainingSuit(),
            H223_PostCollectionSafeLead(),
            H301_SafeDiscardDumpLastPlayer(),
            H303_SuspectedInterruptionUnderplay(),
            H304_InterruptionRiskUnderplay(),
            H305_MiddleEndgameCaution(),
            H306_SpadeFollowConservation(),
            H307_SequentialUnderplay(),
            H308_HandBalancingFollow(),
            H310_FollowKnownCardsFirst(),
            H311_UnderplaySubsequentPlayerHigh(),
            H313_RankEstimationFollow(),
            H314_EndgameKnownCardExploitation(),
            H315_EndgamePerfectInformationFollow(),
            H316_FollowHighWhenSafe(),
            H317_PositionAwareFollow(),
            H401_HighRankOffSuitDump(),
            H402_PreserveFutureVoids(),
            H403_CollectorVoidDisruption(),
            H405_FirstBreakSuitSelection(),
            H406_SpecificSuitDrainingDiscard(),
            H408_SacrificeDiscard(),
            H410_CreateMultiSuitVoids(),
            H411_LowRankDrainVoidAcceleration(),
            H412_BreakWithCollectorMissingSuit(),
            H413_AvoidBreakingLongestSuit(),
            H502_ActivePlayerCountAdjustment(),
            H503_HandSizeRelativeAssessment(),
            H504_WinProximityBonus(),
            H508_RoundDrawAwareness(),
            H702_FinalRoundPressure(),
            H703_TargetTrailingPlayer(),
            H705_OpponentNearVictoryThreat(),
            H706_EarlyRoundStealExploitation(),
            H707_LossSpiralRecoveryPlay(),
            H708_TwoPlayerEndgamePerfectPlay()
        ]

        # Decentralized weights dictionary setup (Concern 2)
        self.weights = {rule.id: rule.default_weight for rule in self.rules}
        # Hard constraints and compatibility weights
        self.weights["H114"] = 9999.0

    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        evals = self.evaluate_legal_actions(player_view, legal_actions)
        return evals[0].action

    def evaluate_legal_actions(self, player_view: EngineState, legal_actions: Sequence[Action]) -> List[ActionEvaluation]:
        """Detailed scoring and tie-breaking returning explainable ActionEvaluations."""
        round_st = player_view.round_state
        if round_st is None:
            return [ActionEvaluation(a, 0.0, {}) for a in legal_actions]

        # 1. Initialize Tracker & Estimators
        tracker = CardTracker(len(round_st.players))
        tracker.reconstruct(self.player_id, round_st, player_view.match_state)
        estimator = ProbabilityEstimator(tracker)
        metrics = EvaluationMetrics(tracker, estimator)

        # 2. Extract context-wide features
        phase, phase_weights = self.estimate_phase_blended(round_st)
        ctx = FeatureExtractor.extract_features(player_view, tracker, estimator, metrics, self.player_id, phase, phase_weights)

        # 3. Score all legal actions using modular rules
        scored_actions: List[ActionEvaluation] = []
        for action in legal_actions:
            # Check hard constraints first
            is_valid, hard_score = self.check_hard_constraints(action, round_st)
            if not is_valid:
                scored_actions.append(ActionEvaluation(action, hard_score, {"HARD_CONSTRAINT": hard_score}))
                continue

            breakdown = {}
            total = 0.0
            for rule in self.rules:
                val = rule.evaluate(action, tracker, ctx)
                if abs(val) > 1e-5:
                    w = self.weights.get(rule.id, 100.0)
                    if ctx.consecutive_losses > 0 and val < 0.0:
                        val *= ctx.consecutive_loss_multiplier
                    
                    weighted_val = val * w
                    total += weighted_val
                    breakdown[rule.id] = weighted_val

            scored_actions.append(ActionEvaluation(action, total, breakdown))

        # 4. Tie-breaking sorting
        def tie_breaker_key(eval_obj: ActionEvaluation) -> Tuple[float, int, int]:
            size = self.get_remaining_hand_size_after_play(eval_obj.action, round_st)
            div = self.get_suit_diversity_after_play(eval_obj.action, round_st)
            return (eval_obj.total_score, -size, div)

        scored_actions.sort(key=tie_breaker_key, reverse=True)
        return scored_actions

    def check_hard_constraints(self, action: Action, round_state) -> Tuple[bool, float]:
        """Hard constraint validator (Auto-Loss Prevention)."""
        if isinstance(action, StealAction) and len(round_state.active_player_ids) == 2:
            return False, -9999.0
        return True, 0.0

    def estimate_phase_blended(self, round_state) -> Tuple[str, Dict[str, float]]:
        """Data-driven soft-blended game phase estimator."""
        t_num = len(round_state.trick_history) + 1
        active_count = len(round_state.active_player_ids)
        total_players = len(round_state.players)
        avg_hand_size = sum(len(p.hand) for p in round_state.players) / total_players

        opening_score = max(0.0, 1.0 - (t_num / 6.0)) if avg_hand_size >= 9 else 0.0
        endgame_score = 0.0
        if active_count <= 2:
            endgame_score = 1.0
        elif avg_hand_size <= 4:
            endgame_score = min(1.0, (5.0 - avg_hand_size) / 2.0)
        elif t_num >= 12:
            endgame_score = min(1.0, (t_num - 11.0) / 4.0)

        middle_score = max(0.0, 1.0 - opening_score - endgame_score)
        
        total = opening_score + middle_score + endgame_score
        weights = {
            "Opening": opening_score / total,
            "Middle": middle_score / total,
            "Endgame": endgame_score / total
        }

        primary_phase = max(weights, key=weights.get)
        return primary_phase, weights

    def get_remaining_hand_size_after_play(self, action: Action, round_state) -> int:
        if isinstance(action, PlayCardAction):
            return len(round_state.players[self.player_id].hand) - 1
        return len(round_state.players[self.player_id].hand)

    def get_suit_diversity_after_play(self, action: Action, round_state) -> int:
        if isinstance(action, PlayCardAction):
            suits = set()
            for card in round_state.players[self.player_id].hand:
                if card != action.card:
                    suits.add(get_suit(card))
            return len(suits)
        return len({get_suit(c) for c in round_state.players[self.player_id].hand})
