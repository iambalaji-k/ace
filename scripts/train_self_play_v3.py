# scripts/train_self_play_v3.py
"""Self-Play Reinforcement Learning (PPO + GAE) Pipeline for RL Agent 2.0 (SPRS v3).

Integrates:
- Trajectory transitions accumulating rewards over opponent turns.
- GAE advantage calculation with strict round boundary finalization.
- MCTS dynamic 5-6 player rollout allocation.
- Statistical Champion Evaluation Gating (candidate vs incumbent).
- Model saving including optimizer, epoch, and RNG states for training resumption.
- Detailed PPO training metrics logging and early stopping on target-KL.
- State-Potential Reward System (SPRS v3) replacing legacy heuristics.
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
from engine.types import RoundStarting, EngineState, MatchComplete, PlayCardAction, StealAction, DeclineStealAction
from engine.tournament import TournamentConfig, _run_single_match
from engine.events import get_player_view
from engine.encoder_v2 import encode_state_v2
from engine.action_encoding import action_to_index, index_to_action
from engine.card import get_suit
from engine.model_v2 import AceNetV2
from engine.rl_agent_v2 import RLAgentV2
from engine.heuristic_agent_v2 import HeuristicAgentV2
from engine.mcts_agent import MCTSAgent
from engine.agent import RandomAgent, BaseAgent
from engine.heuristic_agent import CardTracker

# Import RL v1.0 agent safely if it exists, otherwise fallback
try:
    from engine.rl_agent import RLAgent
    RL_V1_AVAILABLE = os.path.exists("engine/rl_champion.pt")
except ImportError:
    RL_V1_AVAILABLE = False


class LeagueMCTSAgent(MCTSAgent):
    """CPU-friendly MCTS Agent with lower iterations for fast training lookahead."""
    def __init__(self, player_id: int, num_players: int, seed=None):
        # Throttle iterations dynamically based on player count to prevent CPU bottlenecks
        iterations_map = {3: 60, 4: 50, 5: 40, 6: 30}
        iters = iterations_map.get(num_players, 50)
        super().__init__(player_id, seed=seed, max_iterations=iters, time_limit=0.10)


def sample_opponent(active_model: AceNetV2, checkpoints_dir: str, player_id: int, seed: int, num_players: int) -> BaseAgent:
    """Samples an opponent based on the 60% / 15% / 10% / 15% league distribution."""
    r = random.random()

    if r < 0.60:  # Self-play / past checkpoints
        if os.path.exists(checkpoints_dir):
            files = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(".pt") and "epoch" in f]
            if files and random.random() < 0.4:
                chosen_ckpt = random.choice(files)
                try:
                    ckpt_model = AceNetV2()
                    ckpt_model.load_state_dict(torch.load(chosen_ckpt, map_location=torch.device('cpu'), weights_only=False))
                    return RLAgentV2(player_id=player_id, model=ckpt_model, explore=True, temperature=1.0, seed=seed)
                except Exception:
                    pass
        # Fallback to active learning agent
        return RLAgentV2(player_id=player_id, model=active_model, explore=True, temperature=1.0, seed=seed)
    elif r < 0.75:  # Heuristic V2 (15%)
        return HeuristicAgentV2(player_id=player_id, seed=seed)
    elif r < 0.85:  # MCTS (10%)
        return LeagueMCTSAgent(player_id=player_id, num_players=num_players, seed=seed)
    else:  # RL v1.0 / Random (15%)
        # RL v1.0 is strictly hardcoded to 4 players. For other sizes, fallback to RandomAgent.
        if RL_V1_AVAILABLE and num_players == 4:
            return RLAgent(player_id=player_id, checkpoint_path="engine/rl_champion.pt", explore=True)
        else:
            return RandomAgent(player_id=player_id, seed=seed)


def compute_normalized_potential(state: EngineState, learning_player_id: int) -> Tuple[float, float, float]:
    """Computes normalized hand and knowledge potential values for state evaluation."""
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
    # Note: Reconstruct is virtually free here due to CardTracker's thread-local prefix caching
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

    # 80% Hand clearances priority vs 20% Knowledge acquisition ratio
    combined = 0.8 * h_hat + 0.2 * info_hat
    return combined, h_hat, info_hat


def collect_match_trajectories(model: AceNetV2, checkpoints_dir: str, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
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
            agents.append(RLAgentV2(player_id=i, model=model, explore=True, temperature=1.0, seed=seed + i))
        else:
            agents.append(sample_opponent(model, checkpoints_dir, i, seed + 100 + i, num_players))

    state, _ = AceEngine.advance(state)

    trajectories = []
    
    # Reward tracking buffers
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

    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)
        
        # Check round completions to finalize round boundaries
        if isinstance(phase, RoundStarting) or state.round_state.status == "COMPLETE":
            # Round completed! Apply sparse round placement reward
            prev_result = state.match_state.round_results[-1]
            if prev_result.loser_id == learning_player_id:
                rank = num_players
            elif learning_player_id in prev_result.winner_ids:
                rank = list(prev_result.winner_ids).index(learning_player_id) + 1
            else:
                rank = num_players

            # Generalized Round Placement Formula (no penalty for 3rd)
            if rank == num_players:
                round_reward = -1.0
            else:
                round_reward = 1.0 - 0.9 * ((rank - 1) / max(1, num_players - 2))

            if trajectories:
                trajectories[-1]['reward'] = accumulated_reward + round_reward
                trajectories[-1]['done'] = True
                
            reward_sums["round_placement"] += round_reward
            accumulated_reward = 0.0

            state, _ = AceEngine.advance(state)
            # Re-initialize potential for new round
            phi_old, _, _ = compute_normalized_potential(state, learning_player_id)
            continue

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
            learning_state_vec = encode_state_v2(player_view, learning_player_id)
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
            
            # Potential shaping applies only on non-terminal transitions (done is False)
            shaping_reward = 0.0
            if trajectories and not trajectories[-1]['done']:
                shaping_reward = 0.99 * phi_new - phi_old
                trajectories[-1]['reward'] += accumulated_reward + shaping_reward
                reward_sums["shaping"] += shaping_reward
                reward_sums["phi_hand_avg"] += h_val
                reward_sums["phi_info_avg"] += i_val
                reward_sums["phi_total_avg"] += phi_new
                reward_sums["step_count"] += 1
                accumulated_reward = 0.0
                
            phi_old = phi_new

        # Apply action
        res = AceEngine.apply_action(state, action)
        if isinstance(res, Success):
            new_state = res.new_state
            # Advance past automatic phases
            new_state, _ = AceEngine.advance(new_state)
        else:
            break

        # Append new step on the learner's decision turn
        if is_learning_turn:
            trajectories.append({
                'state': learning_state_vec,
                'action_idx': learning_action_idx,
                'log_prob': learning_log_prob,
                'legal_indices': learning_legal_indices,
                'reward': 0.0,  # Will be populated on the next learning step or at round end
                'done': False
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

        trajectories[-1]['reward'] += accumulated_reward + match_reward
        trajectories[-1]['done'] = True
        reward_sums["match_placement"] += match_reward

    return trajectories, reward_sums


def compute_gae(trajectories: List[Dict[str, Any]], values: np.ndarray, next_value: float, gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """Calculates Generalized Advantage Estimations (GAE) with strict round-boundary finalization."""
    advantages = []
    gae = 0.0
    for step in reversed(range(len(trajectories))):
        reward = trajectories[step]['reward']
        done = trajectories[step]['done']
        val = values[step]
        val_next = values[step + 1] if not done else next_value
        
        delta = reward + gamma * val_next - val
        gae = delta + gamma * lam * (1.0 - float(done)) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def run_gating_tournament(candidate_model: AceNetV2, champion_path: str) -> Tuple[bool, float, float]:
    """Evaluates the candidate model against the incumbent champion over 102 seat-balanced matches."""
    print("\n[GATING] Running evaluation tournament against incumbent champion...")
    if not os.path.exists(champion_path):
        return True, 1.0, 0.0

    champion_model = AceNetV2()
    champion_model.load_state_dict(torch.load(champion_path, map_location=torch.device('cpu'), weights_only=False))
    champion_model.eval()

    class CandidateRLAgent(RLAgentV2):
        def __init__(self, player_id: int, seed=None):
            super().__init__(player_id, model=candidate_model, explore=False, seed=seed)

    class ChampionRLAgent(RLAgentV2):
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
    # Run 17 cycles of permutations with seed pairing (17 * 6 = 102 matches)
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


class Tee:
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def train_self_play_v3(epochs: int = 200, matches_per_epoch: int = 20):
    # Optimize CPU operations by limiting threads to 1
    torch.set_num_threads(1)

    log_dir = "engine/logs"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(log_dir, "train_self_play_v3.log"), "a")

    print("====================================================")
    print("===   RL AGENT 3.0 PPO TRAINING PIPELINE (SPRS)  ===")
    print("====================================================\n")

    checkpoints_dir = "engine/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = AceNetV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    champion_path = "engine/rl_champion_v3.pt"
    active_path = "engine/rl_active_v3.pt"
    resume_path = "engine/checkpoints/train_resume_v3.pt"

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
            print(f"Resumed training from epoch {start_epoch} using resume file. (Learning rate set to 3e-4)")
        except Exception as e:
            print(f"Failed to resume from file: {e}. Falling back to default checkpoint.")

    # If not resumed, attempt loading imitation champion as initialization
    if start_epoch == 1:
        initializer_path = "engine/rl_champion_v2.pt"  # Use v2 imitation model as initial weights
        if os.path.exists(initializer_path):
            try:
                model.load_state_dict(torch.load(initializer_path, map_location=torch.device('cpu'), weights_only=False))
                print(f"Initialized active network with imitation champion checkpoint: {initializer_path}")
            except Exception as e:
                print(f"Failed to load champion initializer: {e}. Starting from scratch.")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
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
        all_returns = []
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

        # 1. Rollout Collection phase
        rollout_start = time.time()
        for m in range(matches_per_epoch):
            seed = epoch * 10000 + m
            print(f"  [ROLLOUT] Match {m+1:02d}/{matches_per_epoch} (Seed: {seed})... ", end="", flush=True)
            match_start = time.time()
            trajectories, match_rewards = collect_match_trajectories(model, checkpoints_dir, seed)
            match_dur = time.time() - match_start
            
            if not trajectories:
                print(f"Failed (0 steps) in {match_dur:.2f}s", flush=True)
                continue
                
            print(f"Completed: {len(trajectories):4d} steps | Time: {match_dur:.2f}s", flush=True)

            for k in epoch_reward_components:
                epoch_reward_components[k] += match_rewards[k]

            # GAE prep and value calculation
            prep_start = time.time()
            states_vecs = np.array([t['state'] for t in trajectories])
            states_tensor = torch.tensor(states_vecs, dtype=torch.float32)
            with torch.no_grad():
                _, values_tensor = model(states_tensor)
                values = values_tensor.squeeze(1).numpy()

            values = np.append(values, 0.0)

            # GAE calculations
            advantages, returns = compute_gae(trajectories, values, 0.0)

            for step in range(len(trajectories)):
                all_states.append(trajectories[step]['state'])
                all_actions.append(trajectories[step]['action_idx'])
                all_old_log_probs.append(trajectories[step]['log_prob'])
                all_returns.append(returns[step])
                all_advantages.append(advantages[step])
                all_legal_indices.append(trajectories[step]['legal_indices'])
                total_rewards += trajectories[step]['reward']

            total_match_rounds += len(trajectories)
            epoch_times["gae_prep"] += (time.time() - prep_start)
            
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
            returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
            advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)

            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            batch_size = 128
            num_samples = len(all_states)
            early_stop = False

            # PPO Updates
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
                    b_returns = returns_tensor[batch_indices]
                    b_advantages = advantages_tensor[batch_indices]
                    
                    b_legal_indices = [all_legal_indices[idx] for idx in batch_indices.numpy()]
                    max_len = max(len(indices) for indices in b_legal_indices)
                    padded_legal_indices = np.zeros((len(batch_indices), max_len), dtype=np.int64)
                    for i, indices in enumerate(b_legal_indices):
                        padded_legal_indices[i, :len(indices)] = indices
                    b_legal_indices_tensor = torch.tensor(padded_legal_indices, dtype=torch.long)

                    optimizer.zero_grad()

                    policy_logits, values_pred = model(b_states)
                    values_pred = values_pred.squeeze(1)

                    mask = torch.ones_like(policy_logits) * -1e9
                    for i, indices in enumerate(b_legal_indices):
                        mask[i, indices] = 0.0
                    masked_logits = policy_logits + mask
                    new_probs = F.softmax(masked_logits, dim=-1)

                    dist = torch.distributions.Categorical(new_probs)
                    new_log_probs = dist.log_prob(b_actions)

                    ratios = torch.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratios * b_advantages
                    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values_pred, b_returns)

                    raw_ent = dist.entropy().mean()
                    entropy_loss = -ent_coef * raw_ent

                    loss = policy_loss + 0.5 * value_loss + entropy_loss

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    # Record batch statistics
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_entropy_losses.append(entropy_loss.item())
                    epoch_raw_entropies.append(raw_ent.item())

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
        print(f"  [ENTROPY] Coeff: {ent_coef:.4f}")

        # Periodically save active checkpoints (every 10 epochs)
        if epoch % 10 == 0:
            active_ckpt = os.path.join(checkpoints_dir, f"rl_checkpoint_epoch_{epoch}_v3.pt")
            torch.save(model.state_dict(), active_ckpt)
            torch.save(model.state_dict(), active_path)
            print(f"--> Saved active checkpoint: {active_ckpt}")

        # Champion Gating Evaluation (every 20 epochs)
        if epoch % 20 == 0:
            gating_start = time.time()
            promoted, cand_surv, champ_surv = run_gating_tournament(model, champion_path)
            epoch_times["gating"] = time.time() - gating_start
            
            # Save gating history to CSV
            history_path = os.path.join(checkpoints_dir, "gating_history_v3.csv")
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
    train_self_play_v3()
