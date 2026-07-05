# scripts/training/train_self_play_v4.py
"""Self-Play Reinforcement Learning (PPO + GAE) Pipeline for RL Agent 3.1 (SPRS v3.1).

Integrates:
- Trajectory transitions accumulating rewards over opponent turns.
- GAE advantage calculation with strict round boundary finalization (Fix 1).
- Event-level dense causality rewards blended with placements (Fix 2).
- Dual critic tracking for round-level and match-level returns (Improvement 3).
- Curriculum opponent strength training (Improvement 7).
- Huber Loss for value function optimization (Improvement 9).
- Phase-adaptive entropy scaling (Improvement 10).
- Shape-adaptive weight initialization.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from typing import List, Dict, Any, Tuple
sys.path.append('.')

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, EngineState, MatchComplete, PlayCardAction, StealAction, DeclineStealAction, AwaitingStealDecision
from engine.tournament import TournamentConfig, _run_single_match
from engine.events import get_player_view
from agents.rl.v3.encoder_v3 import encode_state_v3
from engine.action_encoding import action_to_index, index_to_action
from engine.card import get_suit
from agents.rl.v3.model_v3 import AceNetV3
from agents.rl.v3.rl_agent_v3 import RLAgentV3
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.mcts.v1.mcts_agent import MCTSAgent
from agents.random.agent import RandomAgent, BaseAgent
from agents.heuristic.v1.heuristic_agent import CardTracker

# Import RL v1.0 agent safely if it exists, otherwise fallback
try:
    from agents.rl.v1.rl_agent import RLAgent
    RL_V1_AVAILABLE = os.path.exists("checkpoints/rl_champion.pt")
except ImportError:
    RL_V1_AVAILABLE = False


_cached_rl_v2_model = None

def get_rl_v2_model() -> Any:
    global _cached_rl_v2_model
    if _cached_rl_v2_model is None:
        from agents.rl.v2.model_v2 import AceNetV2
        model_path = "checkpoints/rl_champion_v2.pt"
        if os.path.exists(model_path):
            try:
                _cached_rl_v2_model = AceNetV2()
                _cached_rl_v2_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
                _cached_rl_v2_model.eval()
            except Exception as e:
                print(f"Failed to load cached RL v2 model: {e}")
                _cached_rl_v2_model = None
    return _cached_rl_v2_model


class LeagueMCTSAgent(MCTSAgent):
    """MCTS Agent configured with 300 iterations and 0.8s time limit for training."""
    def __init__(self, player_id: int, num_players: int, seed=None):
        super().__init__(player_id, seed=seed, max_iterations=300, time_limit=0.80)


def sample_opponent(active_model: AceNetV3, checkpoints_dir: str, player_id: int, seed: int, num_players: int, epoch: int) -> BaseAgent:
    """Samples an opponent based on the curriculum-adapted league distribution."""
    r = random.random()

    # Determine probabilities based on epoch curriculum
    if epoch <= 40:
        # Phase 1: High heuristic, low complexity
        p_self, p_heur, p_rl_v2, p_mcts, p_rand = 0.50, 0.35, 0.10, 0.00, 0.05
    elif epoch <= 100:
        # Phase 2: Moderate complexity
        p_self, p_heur, p_rl_v2, p_mcts, p_rand = 0.55, 0.20, 0.25, 0.00, 0.00
    else:
        # Phase 3: High complexity, near-optimal play
        p_self, p_heur, p_rl_v2, p_mcts, p_rand = 0.60, 0.10, 0.30, 0.00, 0.00

    if r < p_self:  # Self-play / past checkpoints
        if os.path.exists(checkpoints_dir):
            files = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(".pt") and "epoch" in f]
            if files and random.random() < 0.4:
                chosen_ckpt = random.choice(files)
                try:
                    ckpt_model = AceNetV3()
                    ckpt_model.load_state_dict(torch.load(chosen_ckpt, map_location=torch.device('cpu'), weights_only=False))
                    return RLAgentV3(player_id=player_id, model=ckpt_model, explore=True, temperature=1.0, seed=seed)
                except Exception:
                    pass
        # Fallback to active learning agent
        return RLAgentV3(player_id=player_id, model=active_model, explore=True, temperature=1.0, seed=seed)
    elif r < p_self + p_heur:  # Heuristic V2
        return HeuristicAgentV2(player_id=player_id, seed=seed)
    elif r < p_self + p_heur + p_rl_v2:  # RL V2
        from agents.rl.v2.rl_agent_v2 import RLAgentV2
        model_v2 = get_rl_v2_model()
        if model_v2 is not None:
            return RLAgentV2(player_id=player_id, model=model_v2, explore=True, temperature=1.0, seed=seed)
        else:
            return HeuristicAgentV2(player_id=player_id, seed=seed)
    elif r < p_self + p_heur + p_rl_v2 + p_mcts:  # MCTS
        return LeagueMCTSAgent(player_id=player_id, num_players=num_players, seed=seed)
    else:  # Random
        return RandomAgent(player_id=player_id, seed=seed)


def rollout_worker(model_state_dict: dict, checkpoints_dir: str, seed: int, epoch: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Worker function for parallel rollout collection."""
    torch.set_num_threads(1)
    local_model = AceNetV3()
    local_model.load_state_dict(model_state_dict)
    local_model.eval()
    return collect_match_trajectories(local_model, checkpoints_dir, seed, epoch)


def compute_normalized_potential(state: EngineState, learning_player_id: int) -> Tuple[float, float, float]:
    """Computes normalized hand and knowledge potential values for state evaluation."""
    if state.round_state is None:
        return 0.0, 0.0, 0.0
    p_state = state.round_state.players[learning_player_id]
    if not p_state.is_active:
        return 0.0, 0.0, 0.0

    num_players = len(state.round_state.players)
    active_opps = [p for p in state.round_state.players if p.player_id != learning_player_id and p.is_active]
    num_active_opps = len(active_opps)
    
    if num_active_opps == 0:
        return 0.0, 0.0, 0.0  # Terminal win state potential is 0

    # 1. Bounded Hand Advantage (H_hat) - preserves distance magnitude
    h_me = len(p_state.hand)
    max_hand = max(len(p.hand) for p in state.round_state.players if p.is_active)
    h_hat = 1.0 - (h_me / max(1, max_hand))

    # 2. Bounded Knowledge Ratio (I_hat)
    tracker = CardTracker(num_players=num_players)
    tracker.reconstruct(
        viewer_id=learning_player_id,
        round_state=state.round_state,
        match_state=state.match_state
    )
    known_cards = sum(len(tracker.player_known_cards[p]) for p in range(num_players) if p != learning_player_id)
    known_voids = sum(sum(1 for s in range(4) if tracker.is_void[p][s]) for p in range(num_players) if p != learning_player_id)
    
    total_opp_cards = sum(len(p.hand) for p in active_opps)
    max_voids = 4 * num_active_opps
    info_hat = (known_cards + known_voids) / max(1, total_opp_cards + max_voids)

    # 98% Hand clearances priority vs 2% Knowledge acquisition ratio
    combined = 0.98 * h_hat + 0.02 * info_hat
    return combined, h_hat, info_hat


def collect_match_trajectories(model: AceNetV3, checkpoints_dir: str, seed: int, epoch: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Plays a single match, accumulating rewards across opponent turns, and finalizes round transitions."""
    num_players = random.randint(3, 6)
    num_rounds = random.randint(3, 7)

    # Seat rotation: Place active learner at a random seat
    learning_player_id = random.randint(0, num_players - 1)

    state = AceEngine.create_match(
        match_id=seed,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=seed
    )

    # Initialize agents
    agents = []
    for i in range(num_players):
        if i == learning_player_id:
            agents.append(RLAgentV3(player_id=i, model=model, explore=True, temperature=1.0, seed=seed + i))
        else:
            opp_seed = seed + 100 + i
            agents.append(sample_opponent(model, checkpoints_dir, i, opp_seed, num_players, epoch))

    trajectories = []
    accumulated_reward = 0.0
    
    reward_sums = {
        "round_placement": 0.0,
        "match_placement": 0.0,
        "shaping": 0.0,
        "phi_hand_avg": 0.0,
        "phi_info_avg": 0.0,
        "phi_total_avg": 0.0,
        "step_count": 0
    }

    # Initialize state potential tracking
    phi_old, _, _ = compute_normalized_potential(state, learning_player_id)
    last_played_card = {}

    step_limit = 2000
    steps = 0
    while not AceEngine.is_terminal(state):
        steps += 1
        if steps > step_limit:
            print(f"Warning: Match exceeded step limit of {step_limit}. Forcing termination to prevent hang.")
            break
            
        phase = AceEngine.get_game_phase(state)
        
        player_id = state.runtime_state.current_player_id
        if player_id is None:
            state, _ = AceEngine.advance(state)
            continue

        player_view = get_player_view(state, player_id)
        legal_acts = AceEngine.get_legal_actions(state)

        # Select action
        agent = agents[player_id]
        action = agent.select_action(player_view, legal_acts)

        is_learning_turn = (player_id == learning_player_id)
        learning_state_vec = None
        learning_action_idx = None
        learning_log_prob = None
        learning_legal_indices = None

        if is_learning_turn:
            # 1. Capture learner state & action info before step
            learning_state_vec = encode_state_v3(player_view, learning_player_id)
            learning_action_idx = action_to_index(action)
            learning_legal_indices = [action_to_index(a) for a in legal_acts]

            # Compute current log probability
            state_tensor = torch.tensor(learning_state_vec, dtype=torch.float32).unsqueeze(0)
            legal_tensor = torch.tensor([learning_legal_indices], dtype=torch.long)
            with torch.no_grad():
                probs = model.get_masked_policy(state_tensor, legal_tensor).squeeze(0).numpy()
            prob = np.clip(probs[learning_action_idx], 1e-8, 1.0)
            learning_log_prob = np.log(prob)

            # 2. Transition shaping: calculate shaping reward from previous state to current decision state
            phi_new, h_val, i_val = compute_normalized_potential(state, learning_player_id)
            
            # Potential shaping applies only on non-terminal transitions (round_done is False)
            shaping_reward = 0.0
            if trajectories and not trajectories[-1]['round_done']:
                shaping_reward = SHAPING_COEFF * (GAMMA * phi_new - phi_old)
                trajectories[-1]['reward_round'] += accumulated_reward + shaping_reward
                reward_sums["shaping"] += shaping_reward
                reward_sums["phi_hand_avg"] += h_val
                reward_sums["phi_info_avg"] += i_val
                reward_sums["phi_total_avg"] += phi_new
                reward_sums["step_count"] += 1
                accumulated_reward = 0.0
                
            phi_old = phi_new

        # Apply action
        res = AceEngine.apply_action(state, action)
        events = []
        if isinstance(res, Success):
            new_state = res.new_state
            events.extend(res.events)
            new_state, adv_events = AceEngine.advance(new_state)
            events.extend(adv_events)
        else:
            break

        # Track last played card if PlayCardAction
        if isinstance(action, PlayCardAction):
            last_played_card[player_id] = action.card

        # Check for round end and dense events
        for event in events:
            e_type = event.event_type
            payload = event.payload

            if e_type == "ROUND_STARTED":
                last_played_card = {}

            elif e_type == "STEAL_EXECUTED":
                if payload["stealer_id"] == learning_player_id:
                    penalty = 0.02 * len(payload["cards"])
                    accumulated_reward -= penalty
                    reward_sums["shaping"] -= penalty

            elif e_type == "TRICK_COMPLETED":
                outcome = payload["outcome"]
                if outcome == "INTERRUPTED":
                    if payload["collector_id"] == learning_player_id:
                        penalty = 0.02 * len(payload["cards_collected"])
                        accumulated_reward -= penalty
                        reward_sums["shaping"] -= penalty
                    else:
                        if learning_player_id in last_played_card and payload["cards_collected"]:
                            lead_suit = get_suit(payload["cards_collected"][0])
                            card_played = last_played_card[learning_player_id]
                            if card_played in payload["cards_collected"] and get_suit(card_played) != lead_suit:
                                accumulated_reward += 0.05
                                reward_sums["shaping"] += 0.05
                elif outcome == "DISCARDED":
                    if learning_player_id in last_played_card:
                        card_played = last_played_card[learning_player_id]
                        if card_played in payload["cards_discarded"]:
                            accumulated_reward += 0.010
                            reward_sums["shaping"] += 0.010

            elif e_type == "ROUND_ENDED":
                is_draw = payload.get("is_draw", False)
                if is_draw:
                    round_reward = 0.0
                else:
                    loser_id = payload["loser_id"]
                    winner_ids = list(payload["winner_ids"])
                    if loser_id == learning_player_id:
                        rank = num_players
                    elif learning_player_id in winner_ids:
                        rank = winner_ids.index(learning_player_id) + 1
                    else:
                        rank = num_players

                    if rank == num_players:
                        round_reward = -1.5  # Dominant penalty for losing the round
                    else:
                        # Smoothly scale from 1.2 (Rank 1) down to 1.0 (Rank num_players-1)
                        step_size = 0.2 / max(1, num_players - 2)
                        round_reward = 1.2 - (rank - 1) * step_size

                accumulated_reward += round_reward
                reward_sums["round_placement"] += round_reward

                if trajectories:
                    # Apply final potential drop to 0.0 to ensure zero-sum shaping
                    final_shaping = SHAPING_COEFF * (GAMMA * 0.0 - phi_old)
                    trajectories[-1]['reward_round'] += accumulated_reward + final_shaping
                    reward_sums["shaping"] += final_shaping
                    trajectories[-1]['round_done'] = True
                    reward_sums["step_count"] += 1  # Keep step logging statistics consistent
                    accumulated_reward = 0.0

                phi_old, _, _ = compute_normalized_potential(new_state, learning_player_id)

        # Append new step on the learner's decision turn
        if is_learning_turn:
            trajectories.append({
                'state': learning_state_vec,
                'action_idx': learning_action_idx,
                'log_prob': learning_log_prob,
                'legal_indices': learning_legal_indices,
                'reward_round': 0.0,  # Will be populated on the next learning step or at round end
                'reward_match': 0.0,
                'round_done': False,
                'done': False   # Match-level done flag
            })

        state = new_state

    # Finalize match terminal state
    if trajectories:
        # Determine match rank
        final_players = state.match_state.players
        sorted_players = sorted(final_players, key=lambda x: x.half_points, reverse=True)
        match_rank = next(rank + 1 for rank, p in enumerate(sorted_players) if p.player_id == learning_player_id)
        
        # Generalized Match Placement Formula
        if match_rank == num_players:
            match_reward = -1.5
        else:
            match_reward = 1.5 - 1.35 * ((match_rank - 1) / max(1, num_players - 2))

        trajectories[-1]['reward_round'] += accumulated_reward
        trajectories[-1]['reward_match'] = match_reward
        trajectories[-1]['done'] = True
        reward_sums["match_placement"] += match_reward

    return trajectories, reward_sums


def compute_gae(
    trajectories: List[Dict[str, Any]], 
    values_round: np.ndarray, 
    values_match: np.ndarray, 
    next_val_round: float, 
    next_val_match: float, 
    gamma: float = 0.99, 
    lam: float = 0.95
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Calculates Generalized Advantage Estimations (GAE) for both round and match values."""
    adv_round = []
    gae_r = 0.0
    for step in reversed(range(len(trajectories))):
        reward_r = trajectories[step]['reward_round']
        round_done = trajectories[step].get('round_done', False)
        done = trajectories[step]['done']
        boundary = round_done or done
        val_r = values_round[step]
        val_next_r = values_round[step + 1] if not boundary else next_val_round
        
        delta_r = reward_r + gamma * val_next_r * (1.0 - float(boundary)) - val_r
        gae_r = delta_r + gamma * lam * (1.0 - float(boundary)) * gae_r
        adv_round.insert(0, gae_r)

    returns_round = [adv + val for adv, val in zip(adv_round, values_round[:-1])]

    adv_match = []
    gae_m = 0.0
    for step in reversed(range(len(trajectories))):
        reward_m = trajectories[step]['reward_match']
        done = trajectories[step]['done']
        val_m = values_match[step]
        val_next_m = values_match[step + 1] if not done else next_val_match
        
        delta_m = reward_m + gamma * val_next_m * (1.0 - float(done)) - val_m
        gae_m = delta_m + gamma * lam * (1.0 - float(done)) * gae_m
        adv_match.insert(0, gae_m)

    returns_match = [adv + val for adv, val in zip(adv_match, values_match[:-1])]

    return adv_round, returns_round, adv_match, returns_match


def run_gating_tournament(candidate_model: AceNetV3, champion_path: str) -> Tuple[bool, float, float]:
    """Evaluates the candidate model against the incumbent champion over 102 seat-balanced matches."""
    print("\n[GATING] Running evaluation tournament against incumbent champion...")
    if not os.path.exists(champion_path):
        return True, 1.0, 0.0

    champion_model = AceNetV3()
    champion_model.load_state_dict(torch.load(champion_path, map_location=torch.device('cpu'), weights_only=False))
    champion_model.eval()

    class CandidateRLAgent(RLAgentV3):
        def __init__(self, player_id: int, seed=None):
            super().__init__(player_id, model=candidate_model, explore=False, seed=seed)

    class ChampionRLAgent(RLAgentV3):
        def __init__(self, player_id: int, seed=None):
            super().__init__(player_id, model=champion_model, explore=False, seed=seed)

    permutations = [
        [0, 1, 0, 1],  # Cand, Champ, Cand, Champ
        [1, 0, 1, 0],  # Champ, Cand, Champ, Cand
        [0, 0, 1, 1],  # Cand, Cand, Champ, Champ
        [1, 1, 0, 0],  # Champ, Champ, Cand, Cand
        [0, 1, 1, 0],  # Cand, Champ, Champ, Cand
        [1, 0, 0, 1]   # Champ, Cand, Cand, Champ
    ]

    agent_types = [CandidateRLAgent, ChampionRLAgent]
    cand_losses = 0
    champ_losses = 0

    gating_start_time = time.time()
    for cycle in range(17):
        cycle_start_time = time.time()
        seed = 45000 + cycle
        for layout in permutations:
            config = TournamentConfig(
                num_matches=1,
                num_players=4,
                num_rounds=5,
                base_seed=seed,
                agent_classes=[agent_types[val] for val in layout]
            )
            state = _run_single_match(config, 0)
            for res in state.match_state.round_results:
                if not res.is_draw and res.loser_id is not None:
                    loser_type = layout[res.loser_id]
                    if loser_type == 0:
                        cand_losses += 1
                    else:
                        champ_losses += 1
        cycle_dur = time.time() - cycle_start_time
        elapsed_gating = time.time() - gating_start_time
        print(f"  [GATING] Cycle {cycle + 1:02d}/17 completed ({(cycle + 1) * 6:3d}/102 matches) | Cycle Time: {cycle_dur:.2f}s | Total Elapsed: {elapsed_gating:.2f}s", flush=True)

    cand_surv = 1.0 - (cand_losses / (102 * 5 * 2))
    champ_surv = 1.0 - (champ_losses / (102 * 5 * 2))
    print(f"[GATING] Candidate Survival: {cand_surv*100.0:.2f}% | Champion Survival: {champ_surv*100.0:.2f}%")

    return cand_surv > champ_surv, cand_surv, champ_surv


def load_shape_adaptive(model: nn.Module, checkpoint_path: str):
    """Loads matching keys from checkpoint_path, ignoring mismatched shapes (e.g. input projection)."""
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model_dict = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                model_dict[k] = v
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)
            
    model.load_state_dict(model_dict)
    print(f"Shape-adaptive loading complete. Loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} keys.")


class Tee:
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


SHAPING_COEFF = 0.15  # Upgraded shaping coefficient (15% scaling)
GAMMA = 0.99          # Discount factor
ALPHA = 0.60          # Advantage blend ratio (60% round reward, 40% match reward)

def train_self_play_v4(epochs: int = 300, matches_per_epoch: int = 60):
    # Optimize CPU operations by limiting threads to 3
    torch.set_num_threads(3)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(log_dir, "train_self_play_v4.log"), "a")

    print("====================================================")
    print("===  RL AGENT 3.1 PPO TRAINING PIPELINE (V4)     ===")
    print("====================================================\n")

    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = AceNetV3()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    champion_path = "checkpoints/rl_champion_v4.pt"
    active_path = "checkpoints/rl_active_v4.pt"
    resume_path = "checkpoints/train_resume_v4.pt"

    start_epoch = 1

    # Load from resume checkpoint if it exists
    if os.path.exists(resume_path):
        try:
            resume_data = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(resume_data['model_state_dict'])
            optimizer.load_state_dict(resume_data['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-4
            start_epoch = resume_data['epoch'] + 1
            random.setstate(resume_data['random_state'])
            np.random.set_state(resume_data['np_random_state'])
            torch.set_rng_state(resume_data['torch_rng_state'])
            print(f"Resumed training from epoch {start_epoch} using resume file.")
        except Exception as e:
            print(f"Failed to resume from file: {e}. Falling back to default initialization.")

    # Shape-adaptive initialization from imitation champions if starting at epoch 1
    if start_epoch == 1:
        for prev_ckpt in ["checkpoints/rl_champion_v3.pt", "checkpoints/rl_champion_v2.pt"]:
            if os.path.exists(prev_ckpt):
                try:
                    load_shape_adaptive(model, prev_ckpt)
                    break
                except Exception as e:
                    print(f"Skipping initial load from {prev_ckpt}: {e}")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        # Dynamic Exponential learning rate decay (3e-4 -> 3e-5 over 200 epochs)
        decay_rate = (3e-5 / 3e-4) ** (1 / 199)
        current_lr = 3e-4 * (decay_rate ** (epoch - 1))
        
        # Apply to optimizer parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        epoch_times = {
            "rollout_collect": 0.0,
            "gae_prep": 0.0,
            "ppo_update": 0.0,
            "gating": 0.0
        }
        
        model.eval()

        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns_round = []
        all_returns_match = []
        all_advantages = []
        all_legal_indices = []

        total_match_rounds = 0
        total_rewards = 0.0

        epoch_reward_components = {
            "round_placement": 0.0,
            "match_placement": 0.0,
            "shaping": 0.0,
            "phi_hand_avg": 0.0,
            "phi_info_avg": 0.0,
            "phi_total_avg": 0.0,
            "step_count": 0
        }

        # 1. Rollout Collection phase (Parallelized with ProcessPoolExecutor)
        rollout_start = time.time()
        
        # Get model state dict CPU-mapped for pickling
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        futures = {}
        with ProcessPoolExecutor(max_workers=3) as executor:
            for m in range(matches_per_epoch):
                seed = epoch * 10000 + m
                fut = executor.submit(rollout_worker, model_state_dict, checkpoints_dir, seed, epoch)
                futures[fut] = (m, seed)
                
            for fut in as_completed(futures):
                m, seed = futures[fut]
                try:
                    trajectories, match_rewards = fut.result()
                    if not trajectories:
                        print(f"  [ROLLOUT] Match {m+1:02d}/{matches_per_epoch} (Seed: {seed}) failed (0 steps)", flush=True)
                        continue
                        
                    print(f"  [ROLLOUT] Match {m+1:02d}/{matches_per_epoch} (Seed: {seed}) completed: {len(trajectories):4d} steps", flush=True)
                    
                    for k in epoch_reward_components:
                        epoch_reward_components[k] += match_rewards[k]

                    # GAE prep and value calculation
                    prep_start = time.time()
                    states_vecs = np.array([t['state'] for t in trajectories])
                    states_tensor = torch.tensor(states_vecs, dtype=torch.float32)
                    with torch.no_grad():
                        _, values_r_tensor, values_m_tensor = model(states_tensor)
                        values_r = values_r_tensor.squeeze(1).numpy()
                        values_m = values_m_tensor.squeeze(1).numpy()

                    values_r = np.append(values_r, 0.0)
                    values_m = np.append(values_m, 0.0)

                    # GAE calculations for both round and match value functions
                    adv_r, ret_r, adv_m, ret_m = compute_gae(trajectories, values_r, values_m, 0.0, 0.0, gamma=GAMMA)

                    for step in range(len(trajectories)):
                        all_states.append(trajectories[step]['state'])
                        all_actions.append(trajectories[step]['action_idx'])
                        all_old_log_probs.append(trajectories[step]['log_prob'])
                        all_returns_round.append(ret_r[step])
                        all_returns_match.append(ret_m[step])
                        
                        # Blend advantages: ALPHA for round dynamics, (1-ALPHA) for match dynamics
                        blended_adv = ALPHA * adv_r[step] + (1.0 - ALPHA) * adv_m[step]
                        all_advantages.append(blended_adv)
                        
                        all_legal_indices.append(trajectories[step]['legal_indices'])
                        total_rewards += trajectories[step]['reward_round'] + trajectories[step]['reward_match']

                    total_match_rounds += len(trajectories)
                    epoch_times["gae_prep"] += (time.time() - prep_start)
                except Exception as e:
                    print(f"  [ROLLOUT] Match {m+1:02d}/{matches_per_epoch} (Seed: {seed}) raised exception: {e}", flush=True)
            
        epoch_times["rollout_collect"] = (time.time() - rollout_start) - epoch_times["gae_prep"]

        # Adaptive Entropy Annealing
        ent_coef = max(0.002, 0.02 - (0.02 - 0.002) * (epoch - 1) / 100.0)

        # 2. Optimization Update phase (PPO updates)
        ppo_start = time.time()
        if all_states:
            model.train()

            states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
            actions_tensor = torch.tensor(all_actions, dtype=torch.long)
            old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32)
            returns_r_tensor = torch.tensor(all_returns_round, dtype=torch.float32)
            returns_m_tensor = torch.tensor(all_returns_match, dtype=torch.float32)
            advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)

            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            batch_size = 128
            num_samples = len(all_states)
            early_stop = False

            for update_epoch in range(4):
                if early_stop:
                    break

                permutation = torch.randperm(num_samples)
                epoch_policy_losses = []
                epoch_value_losses = []
                epoch_entropy_losses = []
                epoch_raw_entropies = []
                epoch_clip_fractions = []
                epoch_kl_divergences = []

                for start_idx in range(0, num_samples, batch_size):
                    batch_indices = permutation[start_idx:start_idx + batch_size]
                    if len(batch_indices) < 32:
                        continue

                    b_states = states_tensor[batch_indices]
                    b_actions = actions_tensor[batch_indices]
                    b_old_log_probs = old_log_probs_tensor[batch_indices]
                    b_returns_r = returns_r_tensor[batch_indices]
                    b_returns_m = returns_m_tensor[batch_indices]
                    b_advantages = advantages_tensor[batch_indices]
                    
                    b_legal_indices = [all_legal_indices[idx] for idx in batch_indices.numpy()]
                    max_len = max(len(indices) for indices in b_legal_indices)
                    padded_legal_indices = np.zeros((len(batch_indices), max_len), dtype=np.int64)
                    for i, indices in enumerate(b_legal_indices):
                        padded_legal_indices[i, :len(indices)] = indices
                    b_legal_indices_tensor = torch.tensor(padded_legal_indices, dtype=torch.long)

                    optimizer.zero_grad()

                    policy_logits, values_r_pred, values_m_pred = model(b_states)
                    values_r_pred = values_r_pred.squeeze(1)
                    values_m_pred = values_m_pred.squeeze(1)

                    mask = torch.ones_like(policy_logits) * -1e9
                    for i, indices in enumerate(b_legal_indices):
                        mask[i, indices] = 0.0
                    masked_logits = policy_logits + mask
                    new_probs = F.softmax(masked_logits, dim=-1)

                    dist = torch.distributions.Categorical(new_probs)
                    new_log_probs = dist.log_prob(b_actions)

                    # Policy loss
                    ratios = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratios * b_advantages
                    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value losses (Huber Loss / smooth_l1_loss for gradient stability)
                    val_loss_r = F.smooth_l1_loss(values_r_pred, b_returns_r)
                    val_loss_m = F.smooth_l1_loss(values_m_pred, b_returns_m)
                    value_loss = 0.5 * val_loss_r + 0.5 * val_loss_m

                    # Phase-Adaptive Entropy scaling:
                    # state features vector has is_steal_phase at index 488 (the last index).
                    # Steal decisions are binary (index 0 or 1), requiring less exploration.
                    # Cards plays (indices 2 to 53) require full exploration.
                    # We scale down steal phase entropy by 50%.
                    is_steal = b_states[:, 488]
                    entropy_multiplier = 1.0 - 0.5 * is_steal
                    raw_ent = dist.entropy()
                    entropy_loss = - (ent_coef * entropy_multiplier * raw_ent).mean()

                    loss = policy_loss + 0.5 * value_loss + entropy_loss

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    # Record statistics
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_entropy_losses.append(entropy_loss.item())
                    epoch_raw_entropies.append(raw_ent.mean().item())

                    with torch.no_grad():
                        kl = (b_old_log_probs - new_log_probs).mean().item()
                        epoch_kl_divergences.append(kl)
                        clipped = (ratios < 0.8) | (ratios > 1.2)
                        clip_frac = clipped.float().mean().item()
                        epoch_clip_fractions.append(clip_frac)

                mean_kl = np.mean(epoch_kl_divergences)
                mean_pol = np.mean(epoch_policy_losses)
                mean_val = np.mean(epoch_value_losses)
                mean_raw_ent = np.mean(epoch_raw_entropies)
                mean_clip = np.mean(epoch_clip_fractions)

                print(f"  [PPO Epoch {update_epoch+1}/4] Policy Loss: {mean_pol:.4f} | Value Loss: {mean_val:.4f} | Entropy (Raw): {mean_raw_ent:.4f} | Clip Frac: {mean_clip:.4f} | KL: {mean_kl:.4f}")

                if mean_kl > 0.02:
                    print(f"  [PPO] Early stopping update at epoch {update_epoch+1} (approx KL {mean_kl:.4f} > 0.02)")
                    early_stop = True

        epoch_times["ppo_update"] = time.time() - ppo_start

        epoch_dur = time.time() - epoch_start
        avg_reward = total_rewards / max(1, total_match_rounds)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Steps: {len(all_states):4d} | Avg Step Reward: {avg_reward:.4f} | Time: {epoch_dur:.2f}s")
        print(f"  [TIME BREAKDOWN] Rollout: {epoch_times['rollout_collect']:.2f}s | GAE Prep: {epoch_times['gae_prep']:.2f}s | PPO Update: {epoch_times['ppo_update']:.2f}s | Gating: {epoch_times['gating']:.2f}s")
        
        # Log reward component averages
        r_pl = epoch_reward_components['round_placement']
        m_pl = epoch_reward_components['match_placement']
        shp = epoch_reward_components['shaping']
        steps_logged = max(1, epoch_reward_components['step_count'])
        h_pot = epoch_reward_components['phi_hand_avg'] / steps_logged
        i_pot = epoch_reward_components['phi_info_avg'] / steps_logged
        tot_pot = epoch_reward_components['phi_total_avg'] / steps_logged
        
        print(f"  [REWARDS] Round Place: {r_pl:.2f} | Match Place: {m_pl:.2f} | Shaping: {shp:.2f}")
        print(f"  [POTENTIALS] Hand Pot: {h_pot:.4f} | Know Pot: {i_pot:.4f} | Total Pot: {tot_pot:.4f}")
        print(f"  [ENTROPY] Coeff: {ent_coef:.4f} | LR: {current_lr:.2e}")

        # Periodically save active checkpoints (every 10 epochs)
        if epoch % 10 == 0:
            active_ckpt = os.path.join(checkpoints_dir, f"rl_checkpoint_epoch_{epoch}_v4.pt")
            torch.save(model.state_dict(), active_ckpt)
            torch.save(model.state_dict(), active_path)
            print(f"--> Saved active checkpoint: {active_ckpt}")

        # Champion Gating Evaluation (every 20 epochs)
        if epoch % 20 == 0:
            gating_start = time.time()
            promoted, cand_surv, champ_surv = run_gating_tournament(model, champion_path)
            epoch_times["gating"] = time.time() - gating_start
            
            # Save gating history to CSV
            history_path = os.path.join(checkpoints_dir, "gating_history_v4.csv")
            file_exists = os.path.exists(history_path)
            with open(history_path, "a") as f:
                if not file_exists:
                    f.write("epoch,candidate_survival,champion_survival,promoted\n")
                f.write(f"{epoch},{cand_surv:.4f},{champ_surv:.4f},{int(promoted)}\n")
                
            if promoted:
                torch.save(model.state_dict(), champion_path)
                print(f"[GATING] SUCCESS: Candidate model PROMOTED to Champion ({champion_path})")
            else:
                print("[GATING] REJECTED: Candidate model did not outperform incumbent champion.")

        # Save training resume checkpoint at the end of every epoch
        resume_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()
        }
        torch.save(resume_data, resume_path)

    # Save final trained active weights
    torch.save(model.state_dict(), active_path)
    print(f"\nPPO Training complete in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    train_self_play_v4()
