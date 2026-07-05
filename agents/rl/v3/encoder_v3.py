# agents/rl/v3/encoder_v3.py
"""State encoder (Encoder V3) for RL Agent 3.0/3.1 (SPRS v3.1).

Allocates feature space for up to 6 players and pads empty seats with zeros.
Total feature vector size: 1,879.
"""

import numpy as np
from engine.types import EngineState, AwaitingStealDecision, AwaitingCardPlay, PlayCardAction
from engine.card import get_suit, get_rank
from agents.heuristic.v1.heuristic_agent import CardTracker, ProbabilityEstimator


def encode_state_v3(state: EngineState, player_id: int) -> np.ndarray:
    """Encodes EngineState into a flat float32 NumPy array of size 1,879.

    Includes 489 features from V3, 242 features from Category 1-6 updates, and 1,148 new features
    representing card-player attribution, sequencing order, full round play grids, steals pedigree,
    subsequent player void tracking, interruption liability, and round dealer seats.
    """
    num_players = len(state.match_state.players)
    num_active = len(state.round_state.active_player_ids)

    # Reconstruct tracker to get void suits and known cards
    tracker = CardTracker(num_players=num_players)
    tracker.reconstruct(
        viewer_id=player_id,
        round_state=state.round_state,
        match_state=state.match_state
    )
    estimator = ProbabilityEstimator(tracker)

    # --- Baseline Features (V3 - 489 features) ---

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

    # 3. Opponent Voids: 6 players * 4 suits = 24 binary features
    void_feats = np.zeros((6, 4), dtype=np.float32)
    for p in range(num_players):
        for s in range(4):
            if tracker.is_void[p][s]:
                void_feats[p, s] = 1.0

    # 4. Known Opponent Cards: 6 players * 52 cards = 312 binary features
    known_feats = np.zeros((6, 52), dtype=np.float32)
    for p in range(num_players):
        if p == player_id:
            for card in state.round_state.players[player_id].hand:
                if card >= 0:
                    known_feats[p, card] = 1.0
        else:
            for card in tracker.player_known_cards[p]:
                known_feats[p, card] = 1.0

    # 5. Point scores & round counts: 12 features (normalized)
    score_feats = np.zeros(12, dtype=np.float32)
    for p in range(num_players):
        match_p = state.match_state.players[p]
        score_feats[p] = match_p.half_points / 500.0
        score_feats[6 + p] = match_p.rounds_won / 10.0

    # 6. Current Lead Suit: 4 features (one-hot)
    lead_feats = np.zeros(4, dtype=np.float32)
    curr_trick = state.round_state.current_trick
    if curr_trick and curr_trick.plays:
        lead_suit = get_suit(curr_trick.plays[0].card)
        lead_feats[lead_suit] = 1.0

    # 7. Game Phase: 3 features (one-hot)
    phase_feats = np.zeros(3, dtype=np.float32)
    card_count = sum(len(p.hand) for p in state.round_state.players if p.is_active)
    if card_count > (num_players * 9):
        phase_feats[0] = 1.0  # Opening
    elif card_count > (num_players * 4):
        phase_feats[1] = 1.0  # Middle
    else:
        phase_feats[2] = 1.0  # Endgame

    # 8. Trick position: normalized (0=leading, 1=all played)
    trick_pos = len(curr_trick.plays) / max(1, num_active - 1) if (curr_trick and curr_trick.plays) else 0.0

    # 9. Hand sizes per player: 6 features (normalized)
    hand_sizes = np.zeros(6, dtype=np.float32)
    for p in range(num_players):
        hand_sizes[p] = len(state.round_state.players[p].hand) / 13.0

    # 10. Rounds remaining: 1 feature (normalized)
    rounds_remaining = (state.match_state.num_rounds - state.match_state.current_round) / max(1, state.match_state.num_rounds)

    # 11. Consecutive loss count per player: 6 features (normalized)
    loss_counts = np.zeros(6, dtype=np.float32)
    for p in range(num_players):
        loss_counts[p] = state.match_state.players[p].consecutive_loss_count / 5.0

    # 12. Self void flags: 4 features
    self_voids = np.zeros(4, dtype=np.float32)
    for s in range(4):
        if tracker.is_void[player_id][s]:
            self_voids[s] = 1.0

    # 13. Am I the trick leader? (1 feature)
    is_leader = 1.0 if (curr_trick is None or len(curr_trick.plays) == 0) else 0.0

    # 14. Lead card rank: 1 feature (normalized)
    lead_rank = get_rank(curr_trick.plays[0].card) / 12.0 if (curr_trick and curr_trick.plays) else 0.0

    # 15. Total discard pile size: 1 feature (normalized)
    discard_size = len(state.round_state.discard_pile) / 52.0

    # 16. Current round number: 1 feature (normalized)
    round_num = state.round_state.round_number / max(1, state.match_state.num_rounds)

    # 17. Active player count: 1 feature (normalized)
    active_count = num_active / 6.0

    # 18. Opponent active status flags: 6 features
    active_status = np.zeros(6, dtype=np.float32)
    for p in range(num_players):
        if state.round_state.players[p].is_active:
            active_status[p] = 1.0

    # 19. Is steal phase: 1 feature
    is_steal_phase = 1.0 if isinstance(state.runtime_state.current_phase, AwaitingStealDecision) else 0.0

    # --- Category 1-6 Updates (242 features) ---

    # Category 1: Trick History & Temporal Patterns (43 features)
    trick_outcomes_history = np.zeros(20, dtype=np.float32)
    history = state.round_state.trick_history[-10:]
    for i, t in enumerate(history):
        trick_outcomes_history[2 * i] = 1.0 if t.outcome == "INTERRUPTED" else 0.0
        trick_outcomes_history[2 * i + 1] = 1.0 if t.collector_id == player_id else 0.0

    # cards_played_per_suit (4 features)
    cards_played_count = [0] * 4
    for t in state.round_state.trick_history:
        for play in t.plays:
            cards_played_count[get_suit(play.card)] += 1
    if curr_trick and curr_trick.plays:
        for play in curr_trick.plays:
            cards_played_count[get_suit(play.card)] += 1
    cards_played_per_suit = np.array([count / 13.0 for count in cards_played_count], dtype=np.float32)

    # interrupted/discarded trick counts (2 features)
    interrupted_tricks = sum(1 for t in state.round_state.trick_history if t.outcome == "INTERRUPTED")
    interrupted_trick_count = np.array([interrupted_tricks / 20.0], dtype=np.float32)

    discarded_tricks = sum(1 for t in state.round_state.trick_history if t.outcome == "DISCARDED")
    discarded_trick_count = np.array([discarded_tricks / 20.0], dtype=np.float32)

    # tricks won by player (6 features)
    tricks_won = [0] * 6
    for t in state.round_state.trick_history:
        if t.outcome == "INTERRUPTED":
            winner_id = t.collector_id
        else:
            if t.plays:
                lead_s = get_suit(t.plays[0].card)
                winner_id = None
                highest_rank = 13
                for play in t.plays:
                    if get_suit(play.card) == lead_s:
                        rk = get_rank(play.card)
                        if rk < highest_rank:
                            highest_rank = rk
                            winner_id = play.player_id
            else:
                winner_id = None
        if winner_id is not None and winner_id < 6:
            tricks_won[winner_id] += 1
    tricks_won_by_player = np.array([count / 10.0 for count in tricks_won], dtype=np.float32)

    # cards collected by player (6 features)
    cards_collected = [0] * 6
    for t in state.round_state.trick_history:
        if t.outcome == "INTERRUPTED" and t.collector_id is not None and t.collector_id < 6:
            cards_collected[t.collector_id] += len(t.collected_cards)
    cards_collected_by_player = np.array([count / 52.0 for count in cards_collected], dtype=np.float32)

    # last lead suit (4 features)
    last_lead_suit = np.zeros(4, dtype=np.float32)
    if state.round_state.trick_history:
        last_t = state.round_state.trick_history[-1]
        if last_t.plays:
            lead_s = get_suit(last_t.plays[0].card)
            last_lead_suit[lead_s] = 1.0

    # current trick number (1 feature)
    t_num = curr_trick.trick_number if curr_trick else (len(state.round_state.trick_history) + 1)
    trick_number = np.array([t_num / 20.0], dtype=np.float32)

    # Category 2: Current Trick Cards (61 features)
    current_trick_cards = np.zeros(52, dtype=np.float32)
    if curr_trick and curr_trick.plays:
        for play in curr_trick.plays:
            current_trick_cards[play.card] = 1.0

    highest_rank_val = 0.0
    if curr_trick and curr_trick.plays:
        lead_s = get_suit(curr_trick.plays[0].card)
        highest_power = 0
        for play in curr_trick.plays:
            if get_suit(play.card) == lead_s:
                power = 14 - get_rank(play.card)
                if power > highest_power:
                    highest_power = power
        if highest_power > 0:
            highest_rank_val = (highest_power - 2) / 12.0
    current_trick_highest_rank = np.array([highest_rank_val], dtype=np.float32)

    current_trick_player_who_played = np.zeros(6, dtype=np.float32)
    if curr_trick and curr_trick.plays:
        for play in curr_trick.plays:
            if play.player_id < 6:
                current_trick_player_who_played[play.player_id] = 1.0

    played_count = len(curr_trick.plays) if (curr_trick and curr_trick.plays) else 0
    is_last = 1.0 if played_count == num_active - 1 else 0.0
    am_i_last_to_play = np.array([is_last], dtype=np.float32)

    num_yet = max(0, num_active - played_count - 1)
    num_players_yet_to_play = np.array([num_yet / 6.0], dtype=np.float32)

    # Category 3: Steal Mechanics & Context (70 features)
    steal_target_id = np.zeros(6, dtype=np.float32)
    target_id = None
    if is_steal_phase:
        target_id = state.runtime_state.current_phase.steal_target
        if target_id < 6:
            steal_target_id[target_id] = 1.0

    t_hand_size = 0.0
    if is_steal_phase and target_id is not None:
        t_hand_size = len(state.round_state.players[target_id].hand) / 13.0
    steal_target_hand_size = np.array([t_hand_size], dtype=np.float32)

    steal_target_known_cards = np.zeros(52, dtype=np.float32)
    if is_steal_phase and target_id is not None:
        for card in tracker.player_known_cards[target_id]:
            steal_target_known_cards[card] = 1.0

    steal_target_void_suits = np.zeros(4, dtype=np.float32)
    if is_steal_phase and target_id is not None:
        for s in range(4):
            if tracker.is_void[target_id][s]:
                steal_target_void_suits[s] = 1.0

    steals_count = len(curr_trick.steals) if (curr_trick and curr_trick.steals) else 0
    num_steals_this_trick = np.array([steals_count / 6.0], dtype=np.float32)

    sole_active = 1.0 if (is_steal_phase and len(state.round_state.active_player_ids) == 2) else 0.0
    would_be_sole_active = np.array([sole_active], dtype=np.float32)

    ratio_val = 0.0
    if is_steal_phase and target_id is not None:
        my_hand_len = len(state.round_state.players[player_id].hand)
        target_hand_len = len(state.round_state.players[target_id].hand)
        ratio_val = my_hand_len / max(1, target_hand_len)
    hand_size_ratio = np.array([ratio_val], dtype=np.float32)

    steal_suit_overlap = np.zeros(4, dtype=np.float32)
    if is_steal_phase and target_id is not None:
        my_hand = state.round_state.players[player_id].hand
        for s in range(4):
            has_my_suit = any(get_suit(c) == s for c in my_hand if c >= 0)
            target_might_have = not tracker.is_void[target_id][s]
            if has_my_suit and target_might_have:
                steal_suit_overlap[s] = 1.0

    # Category 4: Card Probability & Suit Distribution Intelligence (34 features)
    c_own, c_discard, c_known, u_suit = tracker.get_suit_counts(player_id)
    cards_remaining_per_suit = np.array([u / 13.0 for u in u_suit], dtype=np.float32)

    my_hand = state.round_state.players[player_id].hand
    my_counts = [sum(1 for c in my_hand if c >= 0 and get_suit(c) == s) for s in range(4)]
    my_suit_control = np.zeros(4, dtype=np.float32)
    for s in range(4):
        total_rem = 13 - len(tracker.discards[s])
        my_suit_control[s] = my_counts[s] / max(1, total_rem)

    my_suit_strength = np.zeros(4, dtype=np.float32)
    for s in range(4):
        suit_cards = [c for c in my_hand if c >= 0 and get_suit(c) == s]
        if suit_cards:
            my_suit_strength[s] = sum(((14 - get_rank(c)) - 2) / 12.0 for c in suit_cards) / len(suit_cards)

    my_highest_per_suit = np.zeros(4, dtype=np.float32)
    for s in range(4):
        suit_cards = [c for c in my_hand if c >= 0 and get_suit(c) == s]
        if suit_cards:
            my_highest_per_suit[s] = max(((14 - get_rank(c)) - 2) / 12.0 for c in suit_cards)

    my_lowest_per_suit = np.zeros(4, dtype=np.float32)
    for s in range(4):
        suit_cards = [c for c in my_hand if c >= 0 and get_suit(c) == s]
        if suit_cards:
            my_lowest_per_suit[s] = min(((14 - get_rank(c)) - 2) / 12.0 for c in suit_cards)

    opponents = [p for p in state.round_state.active_player_ids if p != player_id]
    interruption_risk_per_suit = np.zeros(4, dtype=np.float32)
    for s in range(4):
        interruption_risk_per_suit[s] = estimator.interruption_probability(s, opponents, state.round_state)

    suit_dominance = np.zeros(4, dtype=np.float32)
    for s in range(4):
        for rank in range(13):
            card_id = s * 13 + rank
            loc = tracker.card_locations[card_id]
            if loc != "played" and card_id not in tracker.discards[s]:
                if loc == "self":
                    suit_dominance[s] = 1.0
                break

    aces_held = sum(1 for card in my_hand if card >= 0 and get_rank(card) == 0)
    num_aces_in_hand = np.array([aces_held / 4.0], dtype=np.float32)

    faces_held = sum(1 for card in my_hand if card >= 0 and get_rank(card) < 4)
    num_face_cards = np.array([faces_held / 16.0], dtype=np.float32)

    singleton_suits = np.zeros(4, dtype=np.float32)
    for s in range(4):
        if my_counts[s] == 1:
            singleton_suits[s] = 1.0

    # Category 5: Positional & Relational Player Features (25 features)
    relative_seat_positions = np.zeros(6, dtype=np.float32)
    for p in range(num_players):
        dist = (p - player_id) % num_players
        relative_seat_positions[p] = dist / num_players

    my_pts = state.match_state.players[player_id].half_points
    better_players = sum(1 for p in range(num_players) if state.match_state.players[p].half_points > my_pts)
    my_rank_norm = 1.0 - (better_players / max(1, num_players - 1))
    my_rank_in_match = np.array([my_rank_norm], dtype=np.float32)

    leader_pts = max(state.match_state.players[p].half_points for p in range(num_players))
    point_gap_to_leader = np.array([(leader_pts - my_pts) / 50.0], dtype=np.float32)

    last_pts = min(state.match_state.players[p].half_points for p in range(num_players))
    point_gap_to_last = np.array([(my_pts - last_pts) / 50.0], dtype=np.float32)

    leader_player_id = np.zeros(6, dtype=np.float32)
    lead_idx = int(np.argmax([state.match_state.players[p].half_points for p in range(num_players)]))
    if lead_idx < 6:
        leader_player_id[lead_idx] = 1.0

    am_i_leading_match = np.array([1.0 if better_players == 0 else 0.0], dtype=np.float32)

    won = state.match_state.players[player_id].rounds_won
    lost = state.match_state.players[player_id].rounds_lost
    win_loss_ratio = np.array([won / (won + lost + 1.0)], dtype=np.float32)

    r_played = (state.match_state.current_round - 1) / max(1, state.match_state.num_rounds)
    rounds_played = np.array([r_played], dtype=np.float32)

    reserved_aces_recipient = np.zeros(6, dtype=np.float32)
    recipient_id = None
    loss_count = 0
    r = state.round_state.round_number
    if r > 1:
        prev_result = state.match_state.round_results[-1]
        if not prev_result.is_draw:
            recipient_id = prev_result.loser_id
            if recipient_id is not None:
                loss_count = state.match_state.players[recipient_id].consecutive_loss_count
    if recipient_id is not None and recipient_id < 6:
        reserved_aces_recipient[recipient_id] = 1.0

    aces_count = min(loss_count, 4) if recipient_id is not None else 0
    reserved_aces_count = np.array([aces_count / 4.0], dtype=np.float32)

    # Category 6: Reserved Aces Mechanic (9 features)
    reserved_aces_in_play = np.zeros(4, dtype=np.float32)
    if recipient_id is not None:
        for i in range(min(loss_count, 4)):
            reserved_aces_in_play[i] = 1.0

    i_received = 1.0 if recipient_id == player_id else 0.0
    i_received_reserved_aces = np.array([i_received], dtype=np.float32)

    aces_accounted_for = np.zeros(4, dtype=np.float32)
    for s in range(4):
        ace_card = s * 13
        if tracker.card_locations[ace_card] != "unknown":
            aces_accounted_for[s] = 1.0

    # --- New Appended Features (Curing Blindness updates - 1,148 features) ---

    # 28. Current Trick Card-Player Matrix (312 features)
    curr_trick_matrix = np.zeros((6, 52), dtype=np.float32)
    if curr_trick and curr_trick.plays:
        for play in curr_trick.plays:
            if play.player_id < 6:
                curr_trick_matrix[play.player_id, play.card] = 1.0

    # 29. Last Completed Trick Matrix (312 features)
    last_trick_matrix = np.zeros((6, 52), dtype=np.float32)
    if state.round_state.trick_history:
        last_t = state.round_state.trick_history[-1]
        for play in last_t.plays:
            if play.player_id < 6:
                last_trick_matrix[play.player_id, play.card] = 1.0

    # 30. Trick Leaders (12 features)
    current_trick_leader = np.zeros(6, dtype=np.float32)
    next_trick_leader = np.zeros(6, dtype=np.float32)
    
    # Current trick leader
    if curr_trick:
        l_id = curr_trick.lead_player_id
        if l_id < 6:
            current_trick_leader[l_id] = 1.0
            
    # Next trick leader prediction
    next_leader_id = 0
    if curr_trick and curr_trick.plays:
        lead_s = get_suit(curr_trick.plays[0].card)
        winning_play = curr_trick.plays[0]
        highest_power = 14 - get_rank(winning_play.card)
        is_interrupted = False
        for play in curr_trick.plays:
            card_s = get_suit(play.card)
            if card_s != lead_s:
                is_interrupted = True
            else:
                power = 14 - get_rank(play.card)
                if power > highest_power:
                    highest_power = power
                    winning_play = play
        next_leader_id = winning_play.player_id
    elif state.round_state.trick_history:
        next_leader_id = state.round_state.lead_player_id
    else:
        next_leader_id = state.round_state.lead_player_id
    if next_leader_id < 6:
        next_trick_leader[next_leader_id] = 1.0

    # 31. Current Trick Play Order (36 features)
    curr_trick_play_order = np.zeros((6, 6), dtype=np.float32)
    if curr_trick and curr_trick.plays:
        for idx, play in enumerate(curr_trick.plays):
            if play.player_id < 6 and idx < 6:
                curr_trick_play_order[play.player_id, idx] = 1.0

    # Helper function to check player active status after trick index
    def is_active_after_trick(p_id, t_idx_check):
        if state.round_state.players[p_id].is_active:
            return True
        for future_t in state.round_state.trick_history[t_idx_check + 1:]:
            if any(play.player_id == p_id for play in future_t.plays):
                return True
        return False

    # 32. Round Steal Matrix (36 features)
    round_steal_matrix = np.zeros((6, 6), dtype=np.float32)
    # Reconstruct steal events dynamically from completed trick history
    for t_idx, t in enumerate(state.round_state.trick_history):
        played_players = {play.player_id for play in t.plays}
        for p in range(num_players):
            was_active_before = is_active_after_trick(p, t_idx - 1) if t_idx > 0 else True
            is_active_after = is_active_after_trick(p, t_idx)
            if was_active_before and not is_active_after:
                if p not in played_players:
                    stealer = t.plays[0].player_id if t.plays else t.lead_player_id
                    if stealer < 6 and p < 6:
                        round_steal_matrix[stealer, p] += 1.0
    # Add active trick steals
    if curr_trick:
        for steal in curr_trick.steals:
            if steal.stealer_id < 6 and steal.victim_id < 6:
                round_steal_matrix[steal.stealer_id, steal.victim_id] += 1.0

    # 33. Round Standing Trajectories (60 features)
    round_loser_history = np.zeros((5, 6), dtype=np.float32)
    round_winner_history = np.zeros((5, 6), dtype=np.float32)
    r_history = state.match_state.round_results[-5:]
    for idx, r_res in enumerate(r_history):
        if not r_res.is_draw:
            if r_res.loser_id is not None and r_res.loser_id < 6:
                round_loser_history[idx, r_res.loser_id] = 1.0
            for win_id in r_res.winner_ids:
                if win_id < 6:
                    round_winner_history[idx, win_id] = 1.0

    # 34. Must Follow Suit Rule Constraint (1 feature)
    must_follow_suit_feat = np.array([0.0], dtype=np.float32)
    if isinstance(state.runtime_state.current_phase, AwaitingCardPlay):
        if state.runtime_state.current_phase.must_follow:
            must_follow_suit_feat[0] = 1.0

    # 35. Opponent Void & Expected Card Probabilities (48 features)
    opp_void_probs = np.zeros((6, 4), dtype=np.float32)
    opp_expected_cards = np.zeros((6, 4), dtype=np.float32)
    for p in range(num_players):
        if p < 6:
            for s in range(4):
                opp_void_probs[p, s] = estimator.get_void_probability(p, s, state.round_state)
                opp_expected_cards[p, s] = estimator.expected_remaining_cards(p, s, state.round_state) / 13.0

    # 36. Highest Known Subsequent Ranks (4 features)
    subsequent_players = []
    curr_idx = state.round_state.active_player_ids.index(player_id)
    if curr_trick and curr_trick.plays:
        played = {play.player_id for play in curr_trick.plays}
        for i in range(1, num_active):
            idx = (curr_idx + i) % num_active
            p = state.round_state.active_player_ids[idx]
            if p not in played:
                subsequent_players.append(p)
    else:
        for i in range(1, num_active):
            idx = (curr_idx + i) % num_active
            subsequent_players.append(state.round_state.active_player_ids[idx])
            
    highest_subsequent_ranks = np.zeros(4, dtype=np.float32)
    for s in range(4):
        ranks_s = [0]
        for opp in subsequent_players:
            for c in tracker.player_known_cards[opp]:
                if get_suit(c) == s:
                    ranks_s.append(14 - get_rank(c))
        highest_subsequent_ranks[s] = (max(ranks_s) - 2) / 12.0 if max(ranks_s) > 0 else 0.0

    # 37. Highest Played Lead Suit Rank (1 feature)
    highest_played_lead_feat = np.array([0.0], dtype=np.float32)
    if curr_trick and curr_trick.plays:
        lead_s = get_suit(curr_trick.plays[0].card)
        max_power = 0
        for play in curr_trick.plays:
            if get_suit(play.card) == lead_s:
                power = 14 - get_rank(play.card)
                if power > max_power:
                    max_power = power
        if max_power > 0:
            highest_played_lead_feat[0] = (max_power - 2) / 12.0

    # 38. Any Subsequent Player Void Flags (4 features)
    any_subsequent_void_feat = np.zeros(4, dtype=np.float32)
    for s in range(4):
        if any(tracker.is_void[opp][s] for opp in subsequent_players):
            any_subsequent_void_feat[s] = 1.0

    # 39. Round Card Play Tally Grid (312 features)
    round_card_play_grid = np.zeros((6, 52), dtype=np.float32)
    for t in state.round_state.trick_history:
        for play in t.plays:
            if play.player_id < 6:
                round_card_play_grid[play.player_id, play.card] += 1.0
    if curr_trick and curr_trick.plays:
        for play in curr_trick.plays:
            if play.player_id < 6:
                round_card_play_grid[play.player_id, play.card] += 1.0

    # 40. Interruption Liability per Suit (4 features)
    interruption_liability = np.zeros(4, dtype=np.float32)
    for s in range(4):
        my_suit_cards = [c for c in my_hand if c >= 0 and get_suit(c) == s]
        if my_suit_cards and subsequent_players:
            my_max_rank = max(14 - get_rank(c) for c in my_suit_cards)
            
            sub_max_opp_rank = 0
            for opp in subsequent_players:
                for c in tracker.player_known_cards[opp]:
                    if get_suit(c) == s:
                        r_c = 14 - get_rank(c)
                        if r_c > sub_max_opp_rank:
                            sub_max_opp_rank = r_c
                            
            any_sub_void_prob = 1.0 - np.prod([1.0 - estimator.get_void_probability(opp, s, state.round_state) for opp in subsequent_players])
            
            if my_max_rank > sub_max_opp_rank:
                interruption_liability[s] = float(any_sub_void_prob)

    # 41. Round Leader / Dealer Seat (6 features)
    round_leader_seat = np.zeros(6, dtype=np.float32)
    r_lead = state.round_state.lead_player_id
    if r_lead < 6:
        round_leader_seat[r_lead] = 1.0

    # Concatenate all into a single flat vector of size:
    # 731 (V3/V4 updating baseline) + 1,148 (Curing Blindness additions) = 1,879
    return np.concatenate([
        # Baseline V3 features (731 features total)
        own_hand_feats,
        discard_feats,
        void_feats.flatten(),
        known_feats.flatten(),
        score_feats,
        lead_feats,
        phase_feats,
        np.array([trick_pos], dtype=np.float32),
        hand_sizes,
        np.array([rounds_remaining], dtype=np.float32),
        loss_counts,
        self_voids,
        np.array([is_leader], dtype=np.float32),
        np.array([lead_rank], dtype=np.float32),
        np.array([discard_size], dtype=np.float32),
        np.array([round_num], dtype=np.float32),
        np.array([active_count], dtype=np.float32),
        active_status,
        np.array([is_steal_phase], dtype=np.float32),
        trick_outcomes_history,
        cards_played_per_suit,
        interrupted_trick_count,
        discarded_trick_count,
        tricks_won_by_player,
        cards_collected_by_player,
        last_lead_suit,
        trick_number,
        current_trick_cards,
        current_trick_highest_rank,
        current_trick_player_who_played,
        am_i_last_to_play,
        num_players_yet_to_play,
        steal_target_id,
        steal_target_hand_size,
        steal_target_known_cards,
        steal_target_void_suits,
        num_steals_this_trick,
        would_be_sole_active,
        hand_size_ratio,
        steal_suit_overlap,
        cards_remaining_per_suit,
        my_suit_control,
        my_suit_strength,
        my_highest_per_suit,
        my_lowest_per_suit,
        interruption_risk_per_suit,
        suit_dominance,
        num_aces_in_hand,
        num_face_cards,
        singleton_suits,
        relative_seat_positions,
        my_rank_in_match,
        point_gap_to_leader,
        point_gap_to_last,
        leader_player_id,
        am_i_leading_match,
        win_loss_ratio,
        rounds_played,
        reserved_aces_recipient,
        reserved_aces_count,
        reserved_aces_in_play,
        i_received_reserved_aces,
        aces_accounted_for,
        # New Curing Blindness features (1,148 features)
        curr_trick_matrix.flatten(),
        last_trick_matrix.flatten(),
        current_trick_leader,
        next_trick_leader,
        curr_trick_play_order.flatten(),
        round_steal_matrix.flatten(),
        round_loser_history.flatten(),
        round_winner_history.flatten(),
        must_follow_suit_feat,
        opp_void_probs.flatten(),
        opp_expected_cards.flatten(),
        highest_subsequent_ranks,
        highest_played_lead_feat,
        any_subsequent_void_feat,
        round_card_play_grid.flatten(),
        interruption_liability,
        round_leader_seat
    ])
