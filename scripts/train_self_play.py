# scripts/train_self_play.py
"""Self-Play Reinforcement Learning (Actor-Critic) Pipeline for the Ace Engine.

Learns completely from scratch (zero human bias) on CPU.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

sys.path.append('.')

from engine.rules import AceEngine, Success
from engine.types import MatchComplete, RoundStarting
from engine.events import get_player_view
from engine.encoder import encode_state
from engine.model import AceNet
from engine.rl_agent import RLAgent, action_to_idx

def play_self_play_match(model: AceNet, seed: int) -> tuple[list[dict], dict]:
    """Plays a single self-play match and returns player trajectories and match stats."""
    num_players = 4
    num_rounds = 5
    
    state = AceEngine.create_match(
        match_id=seed,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=seed
    )
    state, _ = AceEngine.advance(state)
    
    # Instantiate exploration agents sharing the same model
    agents = [RLAgent(player_id=p, model=model, explore=True, temperature=1.0) for p in range(num_players)]
    
    # Store trajectories: list of (state_vector, action_idx, legal_action_indices)
    trajectories = {p: [] for p in range(num_players)}
    
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)
        if isinstance(phase, RoundStarting):
            state, _ = AceEngine.advance(state)
            continue
            
        player_id = state.runtime_state.current_player_id
        if player_id is None:
            state, _ = AceEngine.advance(state)
            continue
            
        player_view = get_player_view(state, player_id)
        legal_acts = AceEngine.get_legal_actions(state)
        
        # Select action via RLAgent (which includes exploration sampling)
        action = agents[player_id].select_action(player_view, legal_acts)
        
        # Record trajectory entry
        state_vec = encode_state(player_view, player_id)
        action_idx = action_to_idx(action)
        legal_idx_list = [action_to_idx(a) for a in legal_acts]
        
        trajectories[player_id].append({
            'state': state_vec,
            'action_idx': action_idx,
            'legal_indices': legal_idx_list
        })
        
        res = AceEngine.apply_action(state, action)
        if isinstance(res, Success):
            state = res.new_state
        else:
            break
            
    # Calculate returns (z) for each player at match end
    returns = {}
    rounds_lost = [0] * num_players
    for p in range(num_players):
        won_rounds = 0
        lost_rounds = 0
        for res in state.match_state.round_results:
            if not res.is_draw:
                if res.loser_id != p:
                    won_rounds += 1
                else:
                    lost_rounds += 1
                    rounds_lost[p] += 1
            else:
                won_rounds += 1
                
        rank = 4
        phase = AceEngine.get_game_phase(state)
        if isinstance(phase, MatchComplete):
            for r in phase.result.rankings:
                if r.player_id == p:
                    rank = r.rank
                    break
                    
        placement_scores = {1: 1.0, 2: 0.4, 3: 0.0, 4: -1.0}
        total_r = len(state.match_state.round_results)
        survival_rate = (won_rounds - lost_rounds) / total_r if total_r > 0 else 0.0
        
        # Return R = survival_rate + placement_score
        returns[p] = survival_rate + placement_scores.get(rank, -1.0)
        
    # Collate training samples
    samples = []
    for p in range(num_players):
        ret = returns[p]
        for step in trajectories[p]:
            samples.append({
                'state': step['state'],
                'action_idx': step['action_idx'],
                'legal_indices': step['legal_indices'],
                'return': ret
            })
            
    return samples, {'rounds_lost': rounds_lost, 'total_rounds': len(state.match_state.round_results)}

def train_self_play(epochs: int = 300, matches_per_epoch: int = 15):
    """Executes the self-play reinforcement learning loop on CPU."""
    print("====================================================")
    print("===   SELF-PLAY RL TRAINING PIPELINE (CPU)       ===")
    print("====================================================\n")
    
    model = AceNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    champion_path = "engine/rl_champion.pt"
    if os.path.exists(champion_path):
        try:
            model.load_state_dict(torch.load(champion_path, map_location=torch.device('cpu')))
            print(f"Loaded existing model checkpoint: {champion_path}")
        except Exception:
            pass
            
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.eval()
        
        all_samples = []
        epoch_losses = []
        epoch_returns = []
        total_lost_rounds = 0
        total_simulated_rounds = 0
        
        # 1. Self-Play Match Generation (Sequential for CPU cache efficiency)
        for m in range(matches_per_epoch):
            seed = epoch * 1000 + m
            samples, stats = play_self_play_match(model, seed)
            all_samples.extend(samples)
            total_lost_rounds += sum(stats['rounds_lost'])
            total_simulated_rounds += stats['total_rounds']
            for s in samples:
                epoch_returns.append(s['return'])
                
        # 2. Update Model Weights (Gradient Descent step)
        if all_samples:
            model.train()
            
            # Prepare tensor data batch
            states = torch.tensor(np.array([s['state'] for s in all_samples]), dtype=torch.float32)
            actions = torch.tensor([s['action_idx'] for s in all_samples], dtype=torch.long)
            returns = torch.tensor([s['return'] for s in all_samples], dtype=torch.float32)
            
            # Forward pass
            logits, values = model(states)
            
            # Calculate log probabilities of chosen actions with legal masks
            log_probs = []
            for i, s in enumerate(all_samples):
                # Create mask for this step's legal actions
                step_logits = logits[i]
                mask = torch.full_like(step_logits, -1e9)
                mask[s['legal_indices']] = 0.0
                masked_logits = step_logits + mask
                probs = F.softmax(masked_logits, dim=-1)
                
                # Prevent log(0) with clamping
                log_prob = torch.log(probs[s['action_idx']] + 1e-8)
                log_probs.append(log_prob)
                
            log_probs = torch.stack(log_probs)
            
            # Policy gradient loss with Advantage Baseline: Advantage = Return - Value
            advantage = returns - values.squeeze(1).detach()
            policy_loss = -(log_probs * advantage).mean()
            
            # Value loss (Mean Squared Error prediction of returns)
            value_loss = F.mse_loss(values.squeeze(1), returns)
            
            # Total Loss
            loss = policy_loss + 0.5 * value_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        # Calculate stats
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_return = np.mean(epoch_returns) if epoch_returns else 0.0
        avg_survival = 100.0 - (total_lost_rounds / (4.0 * total_simulated_rounds) * 100.0)
        
        epoch_dur = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Avg Return: {avg_return:.2f} | Survival Rate: {avg_survival:.2f}% | Time: {epoch_dur:.2f}s")
        
    # Save the updated champion checkpoint
    torch.save(model.state_dict(), champion_path)
    print(f"\nTraining complete in {time.time() - start_time:.2f} seconds!")
    print(f"Trained champion model successfully saved to: {champion_path}")

if __name__ == "__main__":
    # Import Tuple locally to prevent any scope overlap
    from typing import Tuple
    train_self_play()
