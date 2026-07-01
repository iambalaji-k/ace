# scripts/training/train_self_play_v2.py
"""Self-Play Reinforcement Learning (PPO + GAE) Pipeline for RL Agent 2.0.

Integrates:
- Trajectory transitions accumulating rewards over opponent turns.
- GAE advantage calculation with strict round boundary finalization.
- MCTS dynamic 5-6 player rollout allocation.
- Statistical Champion Evaluation Gating (candidate vs incumbent).
- Model saving including optimizer, epoch, and RNG states for training resumption.
- Detailed PPO training metrics logging and early stopping on target-KL.
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
from agents.rl.v2.encoder_v2 import encode_state_v2
from engine.action_encoding import action_to_index, index_to_action
from engine.card import get_suit
from agents.rl.v2.model_v2 import AceNetV2
from agents.rl.v2.rl_agent_v2 import RLAgentV2
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.mcts.v1.mcts_agent import MCTSAgent
from agents.random.agent import RandomAgent, BaseAgent

# Import RL v1.0 agent safely if it exists, otherwise fallback
try:
    from agents.rl.v1.rl_agent import RLAgent
    RL_V1_AVAILABLE = os.path.exists("checkpoints/rl_champion.pt")
except ImportError:
    RL_V1_AVAILABLE = False


class LeagueMCTSAgent(MCTSAgent):
    """CPU-friendly MCTS Agent with lower iterations for fast training lookahead."""
    def __init__(self, player_id: int, num_players: int, seed=None):
        # Throttle iterations dynamically based on player count to prevent CPU bottlenecks
        iterations_map = {3: 100, 4: 80, 5: 60, 6: 40}
        iters = iterations_map.get(num_players, 80)
        super().__init__(player_id, seed=seed, max_iterations=iters, time_limit=0.15)


def sample_opponent(active_model: AceNetV2, checkpoints_dir: str, player_id: int, seed: int, num_players: int) -> BaseAgent:
    """Samples an opponent based on the 55% / 15% / 15% / 15% league distribution."""
    r = random.random()

    if r < 0.55:  # Self-play / past checkpoints
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
    elif r < 0.70:  # Heuristic V2
        return HeuristicAgentV2(player_id=player_id, seed=seed)
    elif r < 0.85:  # MCTS (CPU optimized iterations)
        return LeagueMCTSAgent(player_id=player_id, num_players=num_players, seed=seed)
    else:  # RL v1.0 / Random
        # RL v1.0 is strictly hardcoded to 4 players. For other sizes, fallback to RandomAgent.
        if RL_V1_AVAILABLE and num_players == 4:
            return RLAgent(player_id=player_id, checkpoint_path="checkpoints/rl_champion.pt", explore=True)
        else:
            return RandomAgent(player_id=player_id, seed=seed)



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
    round_win_rewarded = False
    last_played_card = {}

    reward_sums = {
        "round_victory": 0.0,
        "steal_penalty": 0.0,
        "suit_break": 0.0,
        "discard": 0.0,
        "round_loss": 0.0,
        "endgame_bonus": 0.0
    }

    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)
        
        # Check round completions to finalize round boundaries
        if isinstance(phase, RoundStarting) or state.round_state.status == "COMPLETE":
            # Round completed! Finalize any pending learner transition with done=True
            if trajectories and not trajectories[-1]['done']:
                trajectories[-1]['reward'] = accumulated_reward
                trajectories[-1]['done'] = True
                accumulated_reward = 0.0

            state, _ = AceEngine.advance(state)
            round_win_rewarded = False
            last_played_card = {}
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

        # Track card plays for trick causality tracking
        if isinstance(action, PlayCardAction):
            last_played_card[player_id] = action.card

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

            # 2. Transition accumulation: assign previous turn's accumulated rewards to the last transition
            if trajectories:
                trajectories[-1]['reward'] = accumulated_reward
                accumulated_reward = 0.0

        # Apply action
        res = AceEngine.apply_action(state, action)
        if isinstance(res, Success):
            events = res.events
            new_state = res.new_state
            # Advance past automatic phases
            new_state, adv_events = AceEngine.advance(new_state)
            events.extend(adv_events)
        else:
            break

        # Calculate intermediate rewards from events (affects accumulated reward on any player's turn)
        p_state = new_state.round_state.players[learning_player_id]
        is_active = p_state.is_active

        # Non-overlapping round victory detection when hand is emptied
        if state.round_state.players[learning_player_id].is_active and not is_active and not p_state.is_round_loser:
            if not round_win_rewarded:
                accumulated_reward += 1.0
                reward_sums["round_victory"] += 1.0
                round_win_rewarded = True

        for event in events:
            e_type = event.event_type
            payload = event.payload

            if e_type == "STEAL_EXECUTED":
                if payload["victim_id"] == learning_player_id:
                    # Non-overlapping victory reward for steal escape
                    if not round_win_rewarded:
                        accumulated_reward += 1.0
                        reward_sums["round_victory"] += 1.0
                        round_win_rewarded = True
                elif payload["stealer_id"] == learning_player_id:
                    # Penalty for hand overloading
                    penalty = 0.05 * len(payload["cards"])
                    accumulated_reward -= penalty
                    reward_sums["steal_penalty"] -= penalty

            elif e_type == "TRICK_COMPLETED":
                outcome = payload["outcome"]
                if outcome == "INTERRUPTED":
                    if payload["collector_id"] == learning_player_id:
                        penalty = 0.05 * len(payload["cards_collected"])
                        accumulated_reward -= penalty
                        reward_sums["steal_penalty"] -= penalty
                    else:
                        # Interruption Causality: check if learner broke suit
                        if learning_player_id in last_played_card and payload["cards_collected"]:
                            lead_suit = get_suit(payload["cards_collected"][0])
                            card_played = last_played_card[learning_player_id]
                            # If card was played in this trick and suit didn't match lead suit
                            if card_played in payload["cards_collected"] and get_suit(card_played) != lead_suit:
                                accumulated_reward += 0.10  # Reward for forcing opponent collection
                                reward_sums["suit_break"] += 0.10
                elif outcome == "DISCARDED":
                    # Discard reward: verify card played by learner was discarded
                    if learning_player_id in last_played_card:
                        card_played = last_played_card[learning_player_id]
                        if card_played in payload["cards_discarded"]:
                            accumulated_reward += 0.015
                            reward_sums["discard"] += 0.015

            elif e_type == "ROUND_ENDED":
                if payload["loser_id"] == learning_player_id:
                    accumulated_reward -= 1.0
                    reward_sums["round_loss"] -= 1.0
                elif learning_player_id in payload["winner_ids"]:
                    if not round_win_rewarded:
                        accumulated_reward += 1.0
                        reward_sums["round_victory"] += 1.0
                        round_win_rewarded = True

        # Endgame risk mitigation bonus (small weight, potential-based representation)
        total_cards = sum(len(p.hand) for p in new_state.round_state.players if p.is_active)
        num_active = len(new_state.round_state.active_player_ids)
        avg_size = total_cards / max(1, num_active)
        if avg_size <= 4.0 and is_active:
            own_hand = new_state.round_state.players[learning_player_id].hand
            low_cards = sum(1 for c in own_hand if (c % 13) >= 9)
            bonus = 0.005 * low_cards
            accumulated_reward += bonus
            reward_sums["endgame_bonus"] += bonus

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
        trajectories[-1]['reward'] = accumulated_reward
        trajectories[-1]['done'] = True

    return trajectories, reward_sums




def compute_gae(trajectories: List[Dict[str, Any]], values: np.ndarray, next_value: float, gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """Calculates GAE advantages and discounted returns."""
    advantages = []
    gae = 0.0
    for step in reversed(range(len(trajectories))):
        reward = trajectories[step]['reward']
        done = trajectories[step]['done']
        val = values[step]
        val_next = values[step + 1] if step + 1 < len(trajectories) else next_value

        delta = reward + gamma * val_next * (1.0 - float(done)) - val
        gae = delta + gamma * lam * (1.0 - float(done)) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns


def run_gating_tournament(candidate_model: AceNetV2, champion_path: str) -> Tuple[bool, float, float]:
    """Evaluates the candidate model against the incumbent champion over 102 seat-balanced matches.

    Returns (promoted, candidate_survival, champion_survival).
    """
    print("\n[GATING] Running evaluation tournament against incumbent champion...")
    if not os.path.exists(champion_path):
        return True, 1.0, 0.0  # If no champion exists, promote candidate immediately

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

    # Run 17 cycles of permutations with seed pairing (17 * 6 = 102 matches)
    for cycle in range(17):
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

    cand_surv = 1.0 - (cand_losses / (102 * 5 * 2))
    champ_surv = 1.0 - (champ_losses / (102 * 5 * 2))
    print(f"[GATING] Candidate Survival: {cand_surv*100.0:.2f}% | Champion Survival: {champ_surv*100.0:.2f}%")

    # Candidate wins if its survival rate is higher
    return cand_surv > champ_surv, cand_surv, champ_surv


class Tee:
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def train_self_play_v2(epochs: int = 200, matches_per_epoch: int = 20):
    # Redirect stdout to both console and a log file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(log_dir, "train_self_play.log"), "a")

    print("====================================================")
    print("===   RL AGENT 2.0 PPO TRAINING PIPELINE (CPU)   ===")
    print("====================================================\n")

    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = AceNetV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    champion_path = "checkpoints/rl_champion_v2.pt"
    active_path = "checkpoints/rl_active_v2.pt"
    resume_path = "checkpoints/train_resume.pt"

    start_epoch = 1

    # Load from resume checkpoint if it exists
    if os.path.exists(resume_path):
        try:
            resume_data = torch.load(resume_path, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(resume_data['model_state_dict'])
            optimizer.load_state_dict(resume_data['optimizer_state_dict'])
            # Manually force the new learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-4
            start_epoch = resume_data['epoch'] + 1
            random.setstate(resume_data['random_state'])
            np.random.set_state(resume_data['np_random_state'])
            torch.set_rng_state(resume_data['torch_rng_state'])
            print(f"Resumed training from epoch {start_epoch} using resume file. (Learning rate set to 3e-4)")
        except Exception as e:
            print(f"Failed to resume from file: {e}. Falling back to default checkpoint.")

    # If not resumed, attempt loading champion as initialization
    if start_epoch == 1 and os.path.exists(champion_path):
        try:
            model.load_state_dict(torch.load(champion_path, map_location=torch.device('cpu'), weights_only=False))
            print(f"Initialized active network with champion checkpoint: {champion_path}")
        except Exception as e:
            print(f"Failed to load champion initializer: {e}. Starting from scratch.")

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
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
            "round_victory": 0.0,
            "steal_penalty": 0.0,
            "suit_break": 0.0,
            "discard": 0.0,
            "round_loss": 0.0,
            "endgame_bonus": 0.0
        }

        # 1. Rollout Collection phase
        for m in range(matches_per_epoch):
            seed = epoch * 10000 + m
            trajectories, match_rewards = collect_match_trajectories(model, checkpoints_dir, seed)
            if not trajectories:
                continue

            for k in epoch_reward_components:
                epoch_reward_components[k] += match_rewards[k]

            # Forward pass to get values for GAE
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

        # 2. Optimization Update phase (PPO updates)
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

            # Adaptive Entropy Annealing: decrease from 0.02 to 0.002 over first 100 epochs
            ent_coef = max(0.002, 0.02 - (0.02 - 0.002) * (epoch - 1) / 100.0)

            # PPO Updates
            for update_epoch in range(4):
                if early_stop:
                    break

                permutation = torch.randperm(num_samples)
                epoch_kls = []
                epoch_policy_losses = []
                epoch_value_losses = []
                epoch_entropy_losses = []
                epoch_raw_entropies = []
                epoch_clip_fractions = []

                for start_idx in range(0, num_samples, batch_size):
                    batch_indices = permutation[start_idx:start_idx + batch_size]

                    b_states = states_tensor[batch_indices]
                    b_actions = actions_tensor[batch_indices]
                    b_old_log_probs = old_log_probs_tensor[batch_indices]
                    b_returns = returns_tensor[batch_indices]
                    b_advantages = advantages_tensor[batch_indices]

                    logits, values = model(b_states)

                    log_probs = []
                    entropies = []
                    for idx, batch_idx in enumerate(batch_indices):
                        step_logits = logits[idx]
                        legal_idx = all_legal_indices[batch_idx]

                        mask = torch.full_like(step_logits, -1e9)
                        mask[legal_idx] = 0.0
                        masked_logits = step_logits + mask

                        probs = F.softmax(masked_logits, dim=-1)
                        log_prob = torch.log(probs[b_actions[idx]] + 1e-8)
                        log_probs.append(log_prob)

                        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                        entropies.append(entropy)

                    log_probs = torch.stack(log_probs)
                    entropies = torch.stack(entropies)

                    # PPO Clipped Surrogate Loss
                    ratios = torch.exp(log_probs - b_old_log_probs)
                    surr1 = ratios * b_advantages
                    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * b_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss (Critic)
                    value_loss = F.mse_loss(values.squeeze(1), b_returns)
                    # Entropy Regularization
                    raw_ent = entropies.mean()
                    entropy_loss = -ent_coef * raw_ent

                    # Total update loss
                    total_loss = policy_loss + 0.5 * value_loss + entropy_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                    # Compute batch metrics
                    with torch.no_grad():
                        log_ratios = log_probs - b_old_log_probs
                        approx_kl = ((ratios - 1) - log_ratios).mean().item()
                        epoch_kls.append(approx_kl)
                        
                        clip_frac = ((ratios < 0.8) | (ratios > 1.2)).float().mean().item()
                        epoch_clip_fractions.append(clip_frac)

                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_entropy_losses.append(entropy_loss.item())
                    epoch_raw_entropies.append(raw_ent.item())

                # End of PPO epoch: check average KL divergence
                mean_kl = np.mean(epoch_kls)
                mean_pol = np.mean(epoch_policy_losses)
                mean_val = np.mean(epoch_value_losses)
                mean_ent = np.mean(epoch_entropy_losses)
                mean_raw_ent = np.mean(epoch_raw_entropies)
                mean_clip = np.mean(epoch_clip_fractions)

                # Log metrics for debugging
                print(f"  [PPO Epoch {update_epoch+1}/4] Policy Loss: {mean_pol:.4f} | Value Loss: {mean_val:.4f} | Entropy (Raw): {mean_raw_ent:.4f} | Clip Frac: {mean_clip:.4f} | KL: {mean_kl:.4f}")

                if mean_kl > 0.02:
                    print(f"  [PPO] Early stopping update at epoch {update_epoch+1} (approx KL {mean_kl:.4f} > 0.02)")
                    early_stop = True

        epoch_dur = time.time() - epoch_start
        avg_reward = total_rewards / max(1, total_match_rounds)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Steps: {len(all_states):4d} | Avg Step Reward: {avg_reward:.4f} | Time: {epoch_dur:.2f}s")
        print(f"  [REWARDS] Win: {epoch_reward_components['round_victory']:.2f} | Loss: {epoch_reward_components['round_loss']:.2f} | Steal Pen: {epoch_reward_components['steal_penalty']:.2f} | Break: {epoch_reward_components['suit_break']:.2f} | Discard: {epoch_reward_components['discard']:.2f} | Endgm: {epoch_reward_components['endgame_bonus']:.2f}")
        print(f"  [ENTROPY] Coeff: {ent_coef:.4f}")

        # Periodically save active checkpoints (every 10 epochs)
        if epoch % 10 == 0:
            active_ckpt = os.path.join(checkpoints_dir, f"rl_checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), active_ckpt)
            torch.save(model.state_dict(), active_path)
            print(f"--> Saved active checkpoint: {active_ckpt}")

        # Champion Gating Evaluation (every 20 epochs)
        if epoch % 20 == 0:
            promoted, cand_surv, champ_surv = run_gating_tournament(model, champion_path)
            
            # Save gating history to CSV
            history_path = os.path.join(checkpoints_dir, "gating_history.csv")
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
    train_self_play_v2()
