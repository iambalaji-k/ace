# engine/heuristic_agent.py
"""Modular Heuristic Evaluation Agent for the Ace Engine.

Implements the HeuristicAgent using a state tracker and a modular weighted heuristic framework
based on Ace_Heuristic_Catalogue_v1.0.md.
"""

from typing import Sequence, Tuple, List, Optional, Dict, Set
import random
from dataclasses import replace

from engine.agent import BaseAgent
from engine.types import Action, EngineState, PlayCardAction, StealAction, DeclineStealAction
from engine.card import get_suit, get_rank, card_to_str, sort_cards, RANK_ACE


class CardTracker:
    """Core Utility for card counting, void tracking, and probability estimation.

    Processes the round history from the player's perspective to reconstruct
    publicly known card states and active/inactive player tracking.
    """

    def __init__(self, num_players: int):
        self.num_players = num_players
        # Card locations: "self", "played", "known_P" (where P is player ID index), or "unknown"
        self.card_locations = ["unknown"] * 52
        # Dynamic void flags: is_void[player_id][suit] = bool
        self.is_void = [[False] * 4 for _ in range(num_players)]
        # Reconstruct cards known to be in each player's hand
        self.player_known_cards = [set() for _ in range(num_players)]
        # Track discards per suit
        self.discards = [set() for _ in range(4)]
        # Track active players state trick-by-trick
        self.active_players_state = [True] * num_players

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
        # If active in the current round state, they are active
        if round_state.players[player_id].is_active:
            return True
        # If they play in any trick after t_idx, they must have been active
        for future_t in round_state.trick_history[t_idx + 1:]:
            if any(play.player_id == player_id for play in future_t.plays):
                return True
        return False

    def reconstruct(self, viewer_id: int, round_state, match_state):
        num_players = self.num_players
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

        # Chronological reconstruction of completed tricks
        for t_idx, t in enumerate(round_state.trick_history):
            played_players = {play.player_id for play in t.plays}
            
            # Detect steals: active players before trick T who did not play but became inactive after
            for p in range(num_players):
                if self.active_players_state[p] and not self.is_player_active_after_trick(p, t_idx, round_state):
                    if p not in played_players:
                        # Steal detected! Stealer is lead player of trick T
                        stealer = t.plays[0].player_id if t.plays else t.lead_player_id
                        # Transfer B's known cards to A
                        stolen_cards = list(self.player_known_cards[p])
                        for card in stolen_cards:
                            self.card_locations[card] = f"known_{stealer}"
                            self.player_known_cards[stealer].add(card)
                        self.player_known_cards[p].clear()
                        # Transfer void flags
                        for s in range(4):
                            self.is_void[stealer][s] = self.is_void[p][s]
                            self.is_void[p][s] = True
                
                self.active_players_state[p] = self.is_player_active_after_trick(p, t_idx, round_state)

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

        # Process current trick steals and plays
        curr_t = round_state.current_trick
        if curr_t is not None:
            # Steals
            for steal in curr_t.steals:
                stolen_cards = list(self.player_known_cards[steal.victim_id])
                for card in stolen_cards:
                    self.card_locations[card] = f"known_{steal.stealer_id}"
                    self.player_known_cards[steal.stealer_id].add(card)
                self.player_known_cards[steal.victim_id].clear()
                for s in range(4):
                    self.is_void[steal.stealer_id][s] = self.is_void[steal.victim_id][s]
                    self.is_void[steal.victim_id][s] = True

            # Plays
            if curr_t.plays:
                lead_suit = get_suit(curr_t.plays[0].card)
                for play in curr_t.plays:
                    self.card_locations[play.card] = "played"
                    if play.card in self.player_known_cards[play.player_id]:
                        self.player_known_cards[play.player_id].remove(play.card)
                    if get_suit(play.card) != lead_suit:
                        self.is_void[play.player_id][lead_suit] = True

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
                # Make sure it's not self
                p_id = int(loc.split("_")[1])
                if p_id != viewer_id:
                    c_known[s] += 1

        u_suit = [0] * 4
        for s in range(4):
            u_suit[s] = 13 - c_own[s] - c_discard[s] - c_known[s]

        return c_own, c_discard, c_known, u_suit

    def get_void_probability(self, player_id: int, suit: int, round_state) -> float:
        """Estimate the probability that player_id is void in suit."""
        if self.is_void[player_id][suit]:
            return 1.0
        
        _, _, _, u_suit = self.get_suit_counts(self.num_players)  # dummy viewer
        c_own, c_discard, c_known, u_suit = self.get_suit_counts(player_id)
        u_s = u_suit[suit]
        if u_s == 0:
            return 1.0

        # Total unknown cards in play across all active opponents
        opponents = [p for p in round_state.active_player_ids if p != player_id]
        if not opponents:
            return 1.0

        h_unknown_target = max(0, len(round_state.players[player_id].hand) - len(self.player_known_cards[player_id]))
        t_unknown = sum(max(0, len(round_state.players[q].hand) - len(self.player_known_cards[q])) for q in opponents) + h_unknown_target

        if t_unknown == 0:
            return 1.0

        p_not_target = 1.0 - (h_unknown_target / t_unknown)
        if p_not_target < 0.0:
            p_not_target = 0.0

        return p_not_target ** u_s


class HeuristicAgent(BaseAgent):
    """Playing agent driven by the modular Heuristic Evaluation Framework."""

    def __init__(self, player_id: int, seed: Optional[int] = None) -> None:
        super().__init__(player_id, seed)
        # Default configuration weights
        self.weights = {
            "H102": 50.0,
            "H103": 40.0,
            "H104": 100.0,
            "H105": 150.0,
            "H107": 90.0,
            "H109": 70.0,
            "H110": 80.0,
            "H111": 50.0,
            "H112": 30.0,
            "H113": 100.0,
            "H114": 9999.0,
            "H115": 60.0,
            "H117": 40.0,
            "H202": 12.0,
            "H203": 200.0,
            "H204": 10.0,
            "H205": 250.0,
            "H206": 70.0,
            "H208": 80.0,
            "H209": 90.0,
            "H210": 100.0,
            "H211": 115.0,
            "H213": 85.0,
            "H214": 100.0,
            "H215": 90.0,
            "H216": 60.0,
            "H217": 80.0,
            "H218": 130.0,
            "H219": 140.0,
            "H220": 85.0,
            "H221": 200.0,
            "H222": 50.0,
            "H223": 150.0,
            "H301": 15.0,
            "H303": 250.0,
            "H304": 200.0,
            "H305": 100.0,
            "H306": 15.0,
            "H307": 80.0,
            "H308": 50.0,
            "H310": 70.0,
            "H311": 80.0,
            "H313": 120.0,
            "H314": 100.0,
            "H315": 200.0,
            "H316": 120.0,
            "H317": 30.0,
            "H401": 20.0,
            "H402": 120.0,
            "H403": 100.0,
            "H405": 100.0,
            "H406": 100.0,
            "H408": 110.0,
            "H410": 90.0,
            "H411": 75.0,
            "H412": 130.0,
            "H413": 40.0,
            "H502": 15.0,
            "H503": 30.0,
            "H504": 150.0,
            "H507": 50.0,
            "H508": 40.0,
            "H701": 1.0,
            "H702": 30.0,
            "H703": 30.0,
            "H704": 1.0,
            "H705": 25.0,
            "H706": 80.0,
            "H707": 130.0,
            "H708": 250.0,
        }

    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        """Select highest scoring legal action by evaluating active heuristics."""
        if not legal_actions:
            raise IndexError("Empty legal actions list.")

        round_st = player_view.round_state
        if round_st is None:
            return self.rng.choice(legal_actions)

        # 1. Initialize tracker
        tracker = CardTracker(len(round_st.players))
        tracker.reconstruct(self.player_id, round_st, player_view.match_state)

        # 2. Determine game phase (H510: Trick Count Phase Estimator)
        phase = self.estimate_game_phase(round_st)

        # 3. Score all legal actions
        scored_actions = []
        for action in legal_actions:
            score = self.evaluate_action(action, tracker, player_view, phase)
            scored_actions.append((score, action))

        # 4. Tie-breaking rule (H903)
        scored_actions.sort(key=lambda x: x[0], reverse=True)
        best_score = scored_actions[0][0]
        candidates = [x for x in scored_actions if abs(x[0] - best_score) < 1e-5]

        if len(candidates) == 1:
            return candidates[0][1]

        # Apply deterministic ties breakers
        # 1st tie-breaker: minimize hand size (if play action)
        if isinstance(candidates[0][1], PlayCardAction):
            candidates.sort(key=lambda x: self.get_remaining_hand_size_after_play(x[1], round_st))
            best_size = self.get_remaining_hand_size_after_play(candidates[0][1], round_st)
            candidates = [x for x in candidates if self.get_remaining_hand_size_after_play(x[1], round_st) == best_size]

        if len(candidates) == 1:
            return candidates[0][1]

        # 2nd tie-breaker: maintain most suit diversity
        if isinstance(candidates[0][1], PlayCardAction):
            candidates.sort(key=lambda x: self.get_suit_diversity_after_play(x[1], round_st), reverse=True)
            best_diversity = self.get_suit_diversity_after_play(candidates[0][1], round_st)
            candidates = [x for x in candidates if self.get_suit_diversity_after_play(x[1], round_st) == best_diversity]

        # 3rd tie-breaker: RNG
        return self.rng.choice(candidates)[1]

    def estimate_game_phase(self, round_state) -> str:
        """H510: Dynamic phase estimator."""
        t_num = len(round_state.trick_history) + 1
        active_count = len(round_state.active_player_ids)
        total_players = len(round_state.players)
        
        # Estimate total tricks based on average starting hand size of ~13-17 cards
        avg_hand_size = sum(len(p.hand) for p in round_state.players) / total_players
        
        if t_num <= 3 and avg_hand_size >= 10:
            return "Opening"
        elif active_count <= 2 or avg_hand_size <= 4 or t_num >= 15:
            return "Endgame"
        else:
            return "Middle"

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

    def evaluate_action(self, action: Action, tracker: CardTracker, view: EngineState, phase: str) -> float:
        """Score a single candidate action by aggregating active heuristics."""
        score = 0.0
        round_st = view.round_state
        match_st = view.match_state
        viewer_id = self.player_id
        num_players = len(round_st.players)

        c_own, c_discard, c_known, u_suit = tracker.get_suit_counts(viewer_id)

        # Precompute consecutive loss parameters
        consecutive_losses = match_st.players[viewer_id].consecutive_loss_count
        consecutive_loss_multiplier = 1.5 if consecutive_losses > 0 else 1.0

        # Determine reserved aces recipient at round start
        r = round_st.round_number
        recipient_id = None
        consecutive_loss_count = 0
        if r > 1:
            prev_result = match_st.round_results[-1]
            if not prev_result.is_draw:
                recipient_id = prev_result.loser_id
                if recipient_id is not None:
                    consecutive_loss_count = match_st.players[recipient_id].consecutive_loss_count

        # --- Part 1: Steal Phase Heuristics ---
        if isinstance(action, StealAction) or isinstance(action, DeclineStealAction):
            target = round_st.current_trick.plays[0].player_id if round_st.current_trick.plays else None
            # Fallback target if none yet in current trick plays
            if target is None:
                # Target is the immediate clockwise active player from viewer
                from engine.rules import get_immediate_active_left
                target = get_immediate_active_left(viewer_id, round_st.active_player_ids, num_players)

            victim_hand_size = len(round_st.players[target].hand)

            if isinstance(action, StealAction):
                # H102: Expected Suit Concentration
                my_hand_size = len(round_st.players[viewer_id].hand)
                if my_hand_size > 0:
                    current_concentration = max(c_own) / my_hand_size
                else:
                    current_concentration = 0.0

                # Merged counts
                merged_counts = list(c_own)
                for card in tracker.player_known_cards[target]:
                    merged_counts[get_suit(card)] += 1
                merged_hand_size = my_hand_size + victim_hand_size
                merged_concentration = max(merged_counts) / merged_hand_size if merged_hand_size > 0 else 0.0
                
                suspected_low_ranks_factor = sum(1 for c in tracker.player_known_cards[target] if (14 - get_rank(c)) <= 5)
                score += ((merged_concentration - current_concentration) * self.weights["H102"] +
                          suspected_low_ranks_factor * 40.0)

                # H103: Steal Card Count Penalty
                score -= victim_hand_size * self.weights["H103"]

                # H104: Suit Intersection Steal Filter
                has_intersection = False
                for s in range(4):
                    if c_own[s] > 0 and not tracker.is_void[target][s]:
                        has_intersection = True
                        break
                
                has_known_low_ranks_in_our_suits = False
                for card in tracker.player_known_cards[target]:
                    s = get_suit(card)
                    r_val = 14 - get_rank(card)
                    if c_own[s] > 0 and r_val <= 5:
                        has_known_low_ranks_in_our_suits = True
                
                if has_intersection and not has_known_low_ranks_in_our_suits:
                    score -= self.weights["H104"]
                else:
                    score += self.weights["H104"]

                # H105: Endgame Targeted Steal
                if phase == "Endgame" and victim_hand_size <= 2:
                    score += self.weights["H105"]
                    # Bonus low ranks
                    low_ranks_count = sum(1 for c in tracker.player_known_cards[target] if (14 - get_rank(c)) <= 8)
                    score += low_ranks_count * 50.0
                    # Penalty void suits
                    void_suits_count = sum(1 for s in range(4) if c_own[s] == 0 and not tracker.is_void[target][s])
                    score -= void_suits_count * 60.0
                    # Penalty high ranks
                    high_ranks_count = sum(1 for c in tracker.player_known_cards[target] if (14 - get_rank(c)) >= 12)
                    score -= high_ranks_count * 40.0

                # H107: Steal to Mitigate Bad Leads
                if viewer_id == round_st.current_trick.lead_player_id:
                    # Check if all our suits have void opponents
                    all_leads_unsafe = True
                    for card in round_st.players[viewer_id].hand:
                        s = get_suit(card)
                        opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                        if not any(tracker.is_void[opp][s] for opp in opponents):
                            all_leads_unsafe = False
                            break
                    if all_leads_unsafe:
                        score += self.weights["H107"]

                # H111: Steal Timing Optimizer
                if phase == "Opening":
                    score -= self.weights["H111"]
                elif phase == "Endgame":
                    score += 30.0

                # H110: Steal Disadvantage Assessment (proportional threats)
                threats = 0
                for s in range(4):
                    if c_own[s] > 0 and tracker.is_void[target][s]:
                        threats += 1
                score += threats * 40.0

                # H112: Steal for Future Trick Benefit
                for card in tracker.player_known_cards[target]:
                    s = get_suit(card)
                    r_val = 14 - get_rank(card)
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    if not any(tracker.is_void[opp][s] for opp in opponents):
                        score += self.weights["H112"]  # safe lead card
                    if r_val <= 5:
                        score += 20.0
                    if c_own[s] == 0 and r_val >= 13:
                        score -= 40.0

                # H113: Chain Steal Evaluation
                if len(round_st.current_trick.steals) > 0:
                    score += 40.0

                # H114: Auto-Loss Prevention
                if len(round_st.active_player_ids) == 2:
                    score -= self.weights["H114"]

                # H115: Steal to Deny Void
                has_singleton = False
                for s in range(4):
                    opp_s_count = sum(1 for c in tracker.player_known_cards[target] if get_suit(c) == s)
                    if opp_s_count == 1 and c_own[s] > 0:
                        has_singleton = True
                if has_singleton:
                    score += self.weights["H115"]

                # H117: Reserved Ace Steal Consideration
                if recipient_id == target and consecutive_loss_count > 0:
                    from engine.deck import get_reserved_aces
                    aces = get_reserved_aces(consecutive_loss_count)
                    for ace in aces:
                        # Check if target has not discarded it yet
                        if tracker.card_locations[ace] == f"known_{target}":
                            s = get_suit(ace)
                            if c_own[s] > 0:
                                score += self.weights["H117"]
                            elif c_own[s] == 0:
                                score -= 30.0

                # H706: Early Round Steal Exploitation
                if len(round_st.trick_history) == 0:
                    score += 50.0

                # H507/H704: Consecutive Loss Steal Avoidance
                if consecutive_losses > 0:
                    score -= self.weights["H507"]

            elif isinstance(action, DeclineStealAction):
                # H109: Decline Steal when Hand is Balanced
                is_balanced = True
                for s in range(4):
                    if c_own[s] == 1 or c_own[s] == 0:
                        is_balanced = False
                        break
                if is_balanced:
                    score += self.weights["H109"]

                # H110: Steal Disadvantage Assessment (cost of declining)
                if target is not None:
                    # If victim is known to hold cards in our void suits
                    for s in range(4):
                        if c_own[s] == 0 and sum(1 for c in tracker.player_known_cards[target] if get_suit(c) == s) > 0:
                            score -= 80.0
                    # If victim has known singleton
                    for s in range(4):
                        if sum(1 for c in tracker.player_known_cards[target] if get_suit(c) == s) == 1:
                            score -= 60.0

                # H113: Chain Steal Evaluation (Decline)
                if len(round_st.current_trick.steals) > 0:
                    is_hand_good = False
                    for s in range(4):
                        if c_own[s] == 0 or c_own[s] >= 5:
                            is_hand_good = True
                    if is_hand_good:
                        score += self.weights["H113"]

                # H114: Auto-Loss Prevention (Decline)
                if len(round_st.active_player_ids) == 2:
                    score += 500.0

                # H706: Early Round Steal Exploitation (Decline)
                if len(round_st.trick_history) == 0:
                    singleton_count = sum(1 for s in range(4) if c_own[s] == 1)
                    void_count = sum(1 for s in range(4) if c_own[s] == 0)
                    if singleton_count >= 3 or void_count >= 1:
                        score += self.weights["H706"]

        # --- Part 2 & 3: Play Card Actions ---
        elif isinstance(action, PlayCardAction):
            card = action.card
            s = get_suit(card)
            r_val = 14 - get_rank(card)  # Ace = 14, King = 13, ..., Two = 2

            curr_t = round_st.current_trick
            is_lead_play = (curr_t is None or len(curr_t.plays) == 0)

            # Precompute trick info
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
                            r_p = 14 - get_rank(play.card)
                            if r_p > highest_played_lead:
                                highest_played_lead = r_p
                if curr_t.lead_player_id in round_st.active_player_ids:
                    active_rotation = list(round_st.active_player_ids)
                    lead_idx = active_rotation.index(curr_t.lead_player_id)
                    trick_order = active_rotation[lead_idx:] + active_rotation[:lead_idx]
                    my_turn_idx = len(curr_t.plays)
                    is_last_player = (my_turn_idx == len(trick_order) - 1)
                    subsequent_players = trick_order[my_turn_idx + 1:]

            if is_lead_play:
                # --- Trick Lead Heuristics ---
                # H202: High Rank Lead (Opening)
                if phase == "Opening":
                    score += (r_val - 2) * self.weights["H202"]

                # H203: Opponent Void Avoidance (Lead)
                if phase in ("Middle", "Endgame"):
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    any_void = any(tracker.is_void[opp][s] for opp in opponents)
                    if any_void:
                        score -= self.weights["H203"]
                        # Immediate next active player void check
                        from engine.rules import get_immediate_active_left
                        next_opp = get_immediate_active_left(viewer_id, round_st.active_player_ids, num_players)
                        if tracker.is_void[next_opp][s]:
                            score -= 50.0

                # H204: Low Rank Lead (Middle/Endgame)
                if phase in ("Middle", "Endgame"):
                    score += (14 - r_val) * self.weights["H204"]

                # H205: Suit Depletion Risk Lead
                if phase in ("Middle", "Endgame"):
                    if u_suit[s] <= 2:
                        score -= 120.0
                    if u_suit[s] == 0:
                        score -= 250.0

                # H206: Lure Lead
                if phase in ("Middle", "Endgame"):
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    has_void_opp = any(tracker.is_void[opp][s] for opp in opponents)
                    has_following_opp = any(not tracker.is_void[opp][s] for opp in opponents)
                    if has_void_opp and has_following_opp:
                        if r_val <= 5:
                            score += self.weights["H206"]
                            # If some opponent has known higher card in suit S
                            has_higher = False
                            for opp in opponents:
                                for c in tracker.player_known_cards[opp]:
                                    if get_suit(c) == s and (14 - get_rank(c)) > r_val:
                                        has_higher = True
                            if has_higher:
                                score += 40.0

                # H208: Void-Promoting Lead
                if phase in ("Opening", "Middle"):
                    if c_own[s] == 1:
                        score += self.weights["H208"]
                    elif c_own[s] == 2:
                        score += 40.0

                # H209: Target Specific Void Probe
                if phase in ("Middle", "Endgame"):
                    # Find lowest hand size active opponent (the threat player)
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    if opponents:
                        threat_opp = min(opponents, key=lambda p: len(round_st.players[p].hand))
                        if tracker.is_void[threat_opp][s]:
                            if r_val <= 8:
                                score += self.weights["H209"]
                            else:
                                score -= 50.0

                # H210: Safe Leader with Known Collector
                if phase in ("Middle", "Endgame"):
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    # Check who has the highest known card of suit S
                    highest_known_opp = None
                    highest_known_rank = -1
                    for opp in opponents:
                        for c in tracker.player_known_cards[opp]:
                            if get_suit(c) == s:
                                r_c = 14 - get_rank(c)
                                if r_c > highest_known_rank:
                                    highest_known_rank = r_c
                                    highest_known_opp = opp
                    if highest_known_opp is not None and r_val <= 7:
                        score += self.weights["H210"]

                # H211: Suit Hoarding & Ace Exposer
                if phase in ("Opening", "Middle") and c_own[s] >= 5:
                    if get_rank(card) == RANK_ACE:
                        score += self.weights["H211"]
                    elif r_val >= 10:
                        score -= 95.0

                # H213: Short-Suit Lead (Opening)
                if phase == "Opening":
                    if c_own[s] == 1:
                        score += self.weights["H213"]
                    elif c_own[s] == 2:
                        score += 50.0

                # H214: Avoid Leading Opponent's Collected Suit
                for opp in [p for p in round_st.active_player_ids if p != viewer_id]:
                    # Estimate if they collected >= 3 cards of S recently
                    # We can count their known cards in S
                    opp_s_known = sum(1 for c in tracker.player_known_cards[opp] if get_suit(c) == s)
                    if opp_s_known >= 3:
                        score -= self.weights["H214"]

                # H215: Safe Exit Lead
                if phase in ("Middle", "Endgame"):
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    has_known_higher = False
                    for opp in opponents:
                        for c in tracker.player_known_cards[opp]:
                            if get_suit(c) == s and (14 - get_rank(c)) > r_val:
                                has_known_higher = True
                    if has_known_higher:
                        score += self.weights["H215"]

                # H216: Lead Suit with Most Cards (Opening)
                if phase == "Opening":
                    if c_own[s] >= 5:
                        score += self.weights["H216"]
                    elif c_own[s] >= 7:
                        score += 60.0

                # H217: Avoid Leading Suit We Just Collected
                # Check if we collected S in the last trick in trick_history
                if len(round_st.trick_history) > 0:
                    last_t = round_st.trick_history[-1]
                    if last_t.outcome == "INTERRUPTED" and last_t.collector_id == viewer_id:
                        if any(get_suit(c) == s for c in last_t.collected_cards):
                            score -= self.weights["H217"]

                # H218: Lead Suit Where All Active Players Have Cards
                all_active_have = True
                for opp in [p for p in round_st.active_player_ids if p != viewer_id]:
                    if tracker.is_void[opp][s]:
                        all_active_have = False
                if all_active_have:
                    score += self.weights["H218"]

                # H219: Ace Lead When All Can Follow
                if phase in ("Opening", "Middle") and get_rank(card) == RANK_ACE:
                    all_can_follow = True
                    for opp in [p for p in round_st.active_player_ids if p != viewer_id]:
                        if tracker.is_void[opp][s]:
                            all_can_follow = False
                    if all_can_follow:
                        score += self.weights["H219"]

                # H220: Lead to Force Opponent Collection
                opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                has_high_known_opp = False
                for opp in opponents:
                    for c in tracker.player_known_cards[opp]:
                        if get_suit(c) == s and (14 - get_rank(c)) >= 10:
                            has_high_known_opp = True
                if has_high_known_opp and r_val <= 5:
                    score += self.weights["H220"]

                # H221: Avoid Leading When Only 2 Active Players Remain
                if len(round_st.active_player_ids) == 2:
                    opponent = next(p for p in round_st.active_player_ids if p != viewer_id)
                    opp_void_prob = tracker.get_void_probability(opponent, s, round_st)
                    if opp_void_prob > 0.5:
                        score -= self.weights["H221"]
                    else:
                        score += 50.0

                # H222: Lead Most Common Remaining Suit
                max_u = max(u_suit)
                if u_suit[s] == max_u:
                    score += self.weights["H222"]

                # H223: Post-Collection Safe Lead
                if len(round_st.trick_history) > 0:
                    last_t = round_st.trick_history[-1]
                    if last_t.outcome == "INTERRUPTED" and last_t.collector_id == viewer_id:
                        any_void_opp = any(tracker.is_void[opp][s] for opp in [p for p in round_st.active_player_ids if p != viewer_id])
                        if any_void_opp:
                            score -= 200.0
                        else:
                            score += self.weights["H223"]

                # H707: Loss Spiral Recovery Play
                if consecutive_losses >= 2:
                    from engine.deck import get_reserved_aces
                    aces = get_reserved_aces(consecutive_losses)
                    if card in aces:
                        all_follow = True
                        for opp in [p for p in round_st.active_player_ids if p != viewer_id]:
                            if tracker.is_void[opp][s]:
                                all_follow = False
                        if all_follow:
                            score += self.weights["H707"]

            else:
                # --- Playing in a non-lead spot ---
                assert lead_suit is not None

                if s == lead_suit:
                    # --- Follow Suit Heuristics ---
                    # H301: Safe Discard Dump (Trick Will Be Discarded)
                    # We are the last active player to play in trick rotation
                    my_turn_idx = len(curr_t.plays)
                    
                    has_anyone_broken = any(get_suit(p.card) != lead_suit for p in curr_t.plays)

                    if is_last_player and not has_anyone_broken:
                        score += (r_val - 2) * self.weights["H301"]

                    # H303: Suspected Interruption Underplay
                    if phase in ("Middle", "Endgame"):
                        subsequent_players = trick_order[my_turn_idx + 1:]
                        any_sub_void = any(tracker.get_void_probability(p, lead_suit, round_st) > 0.5 for p in subsequent_players)
                        if any_sub_void and r_val > highest_played_lead:
                            score -= self.weights["H303"]

                    # H304: Interruption Risk Underplay
                    if phase in ("Middle", "Endgame") and u_suit[lead_suit] <= 2:
                        if r_val >= 10:
                            score -= self.weights["H304"]

                    # H305: Middle/Endgame Caution (Avoid Playing Highest)
                    if phase in ("Middle", "Endgame") and not (is_last_player and not has_anyone_broken):
                        if r_val > highest_played_lead:
                            score -= self.weights["H305"]

                    # H306: Spade Follow Conservation
                    if lead_suit == 0:  # Spade
                        score += (14 - r_val) * self.weights["H306"]

                    # H307: Sequential Underplay
                    if phase in ("Middle", "Endgame") and r_val < highest_played_lead:
                        score += self.weights["H307"]  # Base bonus for underplay

                    # H308: Hand Balancing Follow
                    if c_own[s] == 1:
                        score += 90.0
                    elif c_own[s] >= 4:
                        score += self.weights["H308"]

                    # H310: Follow with Known Cards First
                    if card in tracker.player_known_cards[viewer_id]:
                        score += self.weights["H310"]

                    # H311: Underplay Subsequent Player's High Card
                    if phase in ("Middle", "Endgame"):
                        has_high_known_opp = False
                        for opp in subsequent_players:
                            for c in tracker.player_known_cards[opp]:
                                if get_suit(c) == lead_suit and (14 - get_rank(c)) > r_val:
                                    has_high_known_opp = True
                        if has_high_known_opp:
                            score += self.weights["H311"]

                    # H313: Rank Estimation Follow
                    if phase in ("Middle", "Endgame"):
                        # Find estimated highest remaining rank of lead suit among subsequent active players
                        r_est_max = 0
                        for opp in subsequent_players:
                            for c in tracker.player_known_cards[opp]:
                                if get_suit(c) == lead_suit:
                                    r_c = 14 - get_rank(c)
                                    if r_c > r_est_max:
                                        r_est_max = r_c
                        if r_est_max > 0:
                            if r_val < r_est_max:
                                score += 60.0
                            else:
                                score -= self.weights["H313"]

                    # H314: Endgame Known-Card Exploitation
                    if phase == "Endgame":
                        # Check if no subsequent active player can beat card
                        any_can_beat = False
                        for opp in subsequent_players:
                            opp_s_cards = [c for c in tracker.player_known_cards[opp] if get_suit(c) == lead_suit]
                            if opp_s_cards and max(14 - get_rank(c) for c in opp_s_cards) > r_val:
                                any_can_beat = True
                        any_sub_void = any(tracker.is_void[opp][lead_suit] for opp in subsequent_players)
                        if not any_can_beat and not any_sub_void:
                            # It is the highest remaining card in trick
                            score += self.weights["H314"]
                        elif any_sub_void:
                            # Play lowest card
                            # Find our lowest in lead suit
                            my_s_cards = [c for c in round_st.players[viewer_id].hand if get_suit(c) == lead_suit]
                            if my_s_cards and card == min(my_s_cards, key=lambda c: 14 - get_rank(c)):
                                score += 50.0

                    # H315: Endgame Perfect Information Follow
                    # If all cards are accounted for
                    total_unknown_s = u_suit[lead_suit]
                    if phase == "Endgame" and total_unknown_s == 0:
                        # Simple simulation: who has higher cards?
                        any_void_opp = any(tracker.is_void[opp][lead_suit] for opp in subsequent_players)
                        # If someone breaks, the highest card collector collects.
                        # If we play the card, do we collect?
                        is_safe = True
                        if any_void_opp:
                            # Someone breaks!
                            # So the highest played card of lead suit collects.
                            # Is our card the highest?
                            if r_val > highest_played_lead:
                                # We might collect if no subsequent player plays higher lead-suit card
                                has_sub_higher = False
                                for opp in subsequent_players:
                                    opp_lead_cards = [c for c in tracker.player_known_cards[opp] if get_suit(c) == lead_suit]
                                    if opp_lead_cards and max(14 - get_rank(c) for c in opp_lead_cards) > r_val:
                                        has_sub_higher = True
                                if not has_sub_higher:
                                    is_safe = False
                        if is_safe:
                            score += self.weights["H315"]
                        else:
                            score -= self.weights["H315"]

                    # H316: Follow High When Trick is Safe
                    all_sub_have = all(not tracker.is_void[opp][lead_suit] for opp in subsequent_players)
                    if all_sub_have:
                        my_s_cards = [c for c in round_st.players[viewer_id].hand if get_suit(c) == lead_suit]
                        if my_s_cards and card == max(my_s_cards, key=lambda c: 14 - get_rank(c)):
                            score += self.weights["H316"]

                    # H317: Position-Aware Follow
                    if my_turn_idx == 1:  # second player
                        if r_val >= 10:
                            score -= self.weights["H317"]
                    elif is_last_player:
                        my_s_cards = [c for c in round_st.players[viewer_id].hand if get_suit(c) == lead_suit]
                        if my_s_cards and card == max(my_s_cards, key=lambda c: 14 - get_rank(c)):
                            score += 30.0

                else:
                    # --- Break Suit Heuristics ---
                    # H401: High Rank Off-Suit Dump
                    if s != 0:  # non-Spade
                        score += r_val * self.weights["H401"]

                    # H402: Preserve Future Voids
                    if phase in ("Middle", "Endgame") and c_own[s] == 1:
                        score += self.weights["H402"]

                    # H403: Collector Void Disruption
                    # Determine current collector
                    collector_id = None
                    highest_lead_val = -1
                    for play in curr_t.plays:
                        if get_suit(play.card) == lead_suit:
                            r_c = 14 - get_rank(play.card)
                            if r_c > highest_lead_val:
                                highest_lead_val = r_c
                                collector_id = play.player_id
                    if phase in ("Middle", "Endgame") and collector_id is not None:
                        if tracker.is_void[collector_id][s]:
                            score += self.weights["H403"]

                    # H405: First-Break Suit Selection
                    # Check if any prior play broke suit
                    has_prior_broken = any(get_suit(p.card) != lead_suit for p in curr_t.plays)
                    if not has_prior_broken:
                        # We are the first to break!
                        if r_val >= 10 and s != 0:
                            score += 80.0
                        if collector_id is not None and tracker.is_void[collector_id][s]:
                            score += 100.0
                        if c_own[s] == 2:
                            score += 90.0
                        if s == 0:
                            score -= 60.0

                    # H406: Specific Suit Draining Discard
                    if phase in ("Opening", "Middle"):
                        if c_own[s] == 2:
                            score += 70.0
                        elif c_own[s] == 1:
                            score += 100.0

                    # H408: King/Ace Sacrifice Discard
                    if phase in ("Middle", "Endgame") and collector_id is not None and collector_id != viewer_id:
                        if r_val >= 13 and s != 0:
                            score += self.weights["H408"]

                    # H410: Discard to Create Multi-Suit Voids
                    if phase in ("Middle", "Endgame"):
                        near_empty_suits_count = sum(1 for suit_idx in range(4) if c_own[suit_idx] <= 2 and c_own[suit_idx] > 0)
                        if near_empty_suits_count >= 2:
                            if c_own[s] == 1:
                                score += self.weights["H410"]
                            elif c_own[s] == 2:
                                score += 50.0

                    # H411: Strategic Low-Rank Drain for Void Acceleration
                    if phase in ("Middle", "Endgame") and c_own[s] <= 2:
                        if r_val <= 6:
                            score += self.weights["H411"]

                    # H412: Break with Collector's Missing Suit
                    if phase in ("Middle", "Endgame") and collector_id is not None:
                        # Check if collector previously broke suit s
                        if tracker.is_void[collector_id][s]:
                            score += 100.0
                            if r_val >= 10:
                                score += 30.0

                    # H413: Avoid Breaking with Suit We Want to Lead
                    max_own_count = max(c_own)
                    if c_own[s] == max_own_count:
                        score -= self.weights["H413"]

            # --- Modifiers (Positional, Situational, Match-Level) ---
            # H502: Active Player Count Adjustment
            if len(round_st.active_player_ids) == 3 and r_val >= 10:
                score -= self.weights["H502"]

            # H503: Hand Size Relative Assessment
            # Compute average hand size of opponents
            opponents = [p for p in round_st.active_player_ids if p != viewer_id]
            if opponents:
                avg_opp_hand = sum(len(round_st.players[p].hand) for p in opponents) / len(opponents)
                my_hand_size = len(round_st.players[viewer_id].hand)
                if my_hand_size > 1.5 * avg_opp_hand:
                    # Penalize high card plays (aggressive leads or plays)
                    if r_val >= 10:
                        score -= 20.0
                elif my_hand_size < 0.7 * avg_opp_hand:
                    # Bonus for safe dumps
                    if not is_lead_play and not is_last_player and r_val >= 10:
                        score += self.weights["H503"]

            # H504: Win Proximity Bonus
            my_hand_size = len(round_st.players[viewer_id].hand)
            if my_hand_size <= 3:
                # Is it a lead play? If so, is it void for any opponent?
                # Is it follow play? Is it higher than highest played lead card?
                play_might_collect = False
                if is_lead_play:
                    opponents = [p for p in round_st.active_player_ids if p != viewer_id]
                    if any(tracker.is_void[opp][s] for opp in opponents):
                        play_might_collect = True
                else:
                    if not is_last_player and r_val > highest_played_lead:
                        play_might_collect = True
                
                if play_might_collect:
                    score -= self.weights["H504"]
                else:
                    score += 50.0

            # H508: Round Draw Awareness
            opponents = [p for p in round_st.active_player_ids if p != viewer_id]
            if opponents and all(len(round_st.players[p].hand) <= 2 for p in opponents):
                # Play low rank to facilitate draw
                if r_val <= 6:
                    score += self.weights["H508"]

            # H702: Final Round Pressure
            if match_st.current_round == match_st.num_rounds:
                my_points = match_st.players[viewer_id].half_points
                # Find maximum points among other players
                opponents = [p for p in range(num_players) if p != viewer_id]
                max_opp_points = max(match_st.players[p].half_points for p in opponents)
                if my_points < max_opp_points:
                    # Win needed! Play aggressively (higher risk/reward)
                    if r_val >= 12:
                        score += 20.0
                else:
                    # Draw is sufficient! Play safely (lower risk)
                    if r_val <= 6:
                        score += self.weights["H702"]

            # H703: Target the Trailing Player
            opponents = [p for p in round_st.active_player_ids if p != viewer_id]
            trailing_opps = [p for p in opponents if match_st.players[p].consecutive_loss_count > 0]
            if trailing_opps:
                # Play low rank to force them to collect
                if r_val <= 6:
                    score += self.weights["H703"]
                else:
                    score -= 30.0

            # H705: Opponent Near-Victory Threat
            victory_threshold = int(match_st.num_rounds * 0.6)
            dangerous_opps = [p for p in range(num_players) if p != viewer_id and match_st.players[p].rounds_won >= victory_threshold]
            if dangerous_opps:
                # Force them to collect
                if r_val <= 6:
                    score += self.weights["H705"]
                else:
                    score -= 25.0

            # H708: Two-Player Endgame Perfect Play
            if len(round_st.active_player_ids) == 2:
                opponent = next(p for p in round_st.active_player_ids if p != viewer_id)
                # Compute exact expected hand growth for all playable cards in hand
                # Since it's 2-player follow, it's deterministic:
                # If we lead S, and they follow S: discard (growth 0)
                # If we lead S, and they break (void): we collect (growth +1 card)
                # If they lead S, and we follow S: discard (growth 0) if we are lower, else we collect (growth +1 card) if we are higher but wait!
                # If they lead S, and we break: they collect (growth +1 card), we shed (growth -1 card)
                # So we can calculate the exact growth:
                expected_growth = 0
                if is_lead_play:
                    opp_void_prob = tracker.get_void_probability(opponent, s, round_st)
                    expected_growth = opp_void_prob * 1.0  # probability of collecting
                else:
                    # We are following or breaking
                    lead_suit = curr_t.lead_suit
                    assert lead_suit is not None
                    if s == lead_suit:
                        if r_val > highest_played_lead:
                            expected_growth = 1.0  # we collect
                        else:
                            expected_growth = 0.0  # discard
                    else:
                        expected_growth = -1.0  # we break and shed
                
                # Penalize actions that cause expected hand growth
                if expected_growth <= 0.0:
                    score += self.weights["H708"]
                else:
                    score -= self.weights["H708"]

            # Apply Consecutive Loss Multipliers (H507/H704)
            if consecutive_losses > 0 and score < 0.0:
                score *= consecutive_loss_multiplier

        return score
