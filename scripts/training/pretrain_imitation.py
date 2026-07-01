# scripts/training/pretrain_imitation.py
"""Supervised Imitation Learning Pretraining for RL Agent 2.0.

Collects expert demonstration trajectories by running matches with HeuristicAgentV2,
then trains AceNetV2 to clone the heuristic policy using masked cross-entropy loss.
"""

import sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any
sys.path.append('.')

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, EngineState
from engine.events import get_player_view
from agents.rl.v2.encoder_v2 import encode_state_v2
from engine.action_encoding import action_to_index
from agents.heuristic.v2.heuristic_agent_v2 import HeuristicAgentV2
from agents.rl.v2.model_v2 import AceNetV2


def collect_demonstration_data(num_matches: int = 200, base_seed: int = 5000) -> List[List[Dict[str, Any]]]:
    """Runs HeuristicAgentV2 self-play matches and collects state-action pairs grouped by match."""
    print(f"Collecting demonstration trajectories from {num_matches} matches...")
    num_players = 4
    num_rounds = 5
    matches_dataset = []

    for m_idx in range(num_matches):
        match_seed = base_seed + m_idx
        state = AceEngine.create_match(
            match_id=m_idx,
            num_players=num_players,
            num_rounds=num_rounds,
            match_seed=match_seed
        )

        agents = [HeuristicAgentV2(player_id=i, seed=match_seed + 100 + i) for i in range(num_players)]
        state, _ = AceEngine.advance(state)
        match_data = []

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

            # Get heuristic action choice
            agent = agents[player_id]
            action = agent.select_action(player_view, legal_acts)

            # Encode features
            state_vec = encode_state_v2(player_view, player_id)
            action_idx = action_to_index(action)
            legal_indices = [action_to_index(a) for a in legal_acts]

            match_data.append({
                'state': state_vec,
                'action_idx': action_idx,
                'legal_indices': legal_indices
            })

            # Apply action
            res = AceEngine.apply_action(state, action)
            if isinstance(res, Success):
                state = res.new_state
            else:
                break
        
        matches_dataset.append(match_data)

    total_transitions = sum(len(m) for m in matches_dataset)
    print(f"Collected {len(matches_dataset)} matches containing {total_transitions} state-action pairs.")
    return matches_dataset


def train_imitation(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    print("====================================================")
    print("===      IMITATION LEARNING PRETRAINING LOOP     ===")
    print("====================================================\n")

    # 1. Collect dataset
    matches_dataset = collect_demonstration_data(num_matches=150, base_seed=7000)
    if not matches_dataset:
        print("Error: No data collected.")
        return

    # Shuffle matches to ensure random seat/seed splits
    random.shuffle(matches_dataset)

    # 2. Split matches into train/validation sets (90/10) to prevent data leakage
    split = int(len(matches_dataset) * 0.9)
    train_matches = matches_dataset[:split]
    val_matches = matches_dataset[split:]

    # Flatten matches into lists of individual transitions
    train_data = [item for match in train_matches for item in match]
    val_data = [item for match in val_matches for item in match]

    print(f"Dataset split completed:")
    print(f"  Training samples:   {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")

    # 3. Instantiate model and optimizer
    model = AceNetV2()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_path = "checkpoints/rl_champion_v2.pt"

    best_val_loss = float('inf')

    # 4. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_correct = 0

        # Shuffle train data each epoch
        random.shuffle(train_data)

        # Minibatch iteration
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            if not batch:
                continue

            states = torch.tensor(np.array([item['state'] for item in batch]), dtype=torch.float32)
            targets = torch.tensor([item['action_idx'] for item in batch], dtype=torch.long)

            # Compute forward pass
            logits, _ = model(states)

            # Construct masked log-probabilities to avoid computing loss over illegal actions
            log_probs = []
            for j, item in enumerate(batch):
                step_logits = logits[j]
                mask = torch.full_like(step_logits, -1e9)
                mask[item['legal_indices']] = 0.0
                masked_logits = step_logits + mask
                log_prob = F.log_softmax(masked_logits, dim=-1)
                log_probs.append(log_prob)

            log_probs = torch.stack(log_probs)

            # Compute Negative Log-Likelihood loss
            loss = F.nll_loss(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_losses.append(loss.item())

            # Track training accuracy
            preds = torch.argmax(log_probs, dim=-1)
            train_correct += (preds == targets).sum().item()

        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                if not batch:
                    continue

                states = torch.tensor(np.array([item['state'] for item in batch]), dtype=torch.float32)
                targets = torch.tensor([item['action_idx'] for item in batch], dtype=torch.long)

                logits, _ = model(states)

                log_probs = []
                for j, item in enumerate(batch):
                    step_logits = logits[j]
                    mask = torch.full_like(step_logits, -1e9)
                    mask[item['legal_indices']] = 0.0
                    masked_logits = step_logits + mask
                    log_prob = F.log_softmax(masked_logits, dim=-1)
                    log_probs.append(log_prob)

                log_probs = torch.stack(log_probs)
                loss = F.nll_loss(log_probs, targets)

                val_losses.append(loss.item())
                preds = torch.argmax(log_probs, dim=-1)
                val_correct += (preds == targets).sum().item()

        avg_train_loss = np.mean(train_losses)
        train_acc = (train_correct / len(train_data)) * 100.0
        avg_val_loss = np.mean(val_losses)
        val_acc = (val_correct / len(val_data)) * 100.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nImitation pretraining complete. Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    train_imitation()
