# scripts/train_genetic_weights.py
"""Genetic Algorithm Weight Optimization Pipeline for HeuristicAgentV2.

Plays matches against dynamic opponent pools to evolve heuristic weights and internal threshold parameters.
Includes parallelized multiprocessing, larger evaluation sets, unseen validation seed sets,
adaptive mutation scheduling, hybrid block crossovers, early stopping, co-evolution,
diversity novelty penalties, feature-level group mutations, and anchored Hall of Fame validation.
"""

import sys
import random
import csv
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

sys.path.append('.')

from engine.rules import AceEngine, Success
from engine.types import EngineState, RoundStarting, PlayCardAction, StealAction, DeclineStealAction, MatchComplete
from engine.events import get_player_view
from engine.agent import RandomAgent
from engine.heuristic_agent import HeuristicAgent
from engine.heuristic_agent_v2 import HeuristicAgentV2

def run_match_for_eval(
    candidate_weights: Dict[str, float],
    match_seed: int,
    opponent_type: str,
    opponents_weights: Optional[List[Dict[str, float]]] = None
) -> Tuple[int, int, int]:
    """Runs a single match of 3 rounds and returns (rounds_won, rounds_lost, rank_sum) for Player 0."""
    num_players = 4
    num_rounds = 3
    
    state = AceEngine.create_match(
        match_id=1,
        num_players=num_players,
        num_rounds=num_rounds,
        match_seed=match_seed
    )
    
    # Player 0 is HeuristicAgentV2 using mutated weights and parameters
    candidate = HeuristicAgentV2(player_id=0, seed=match_seed + 100, weights_config=candidate_weights)
    
    # Define agents pool
    agents = [candidate]
    
    # Opponents setup
    if opponents_weights and len(opponents_weights) >= 3:
        for idx in range(1, num_players):
            agents.append(HeuristicAgentV2(player_id=idx, seed=match_seed + 100 + idx, weights_config=opponents_weights[idx - 1]))
    else:
        for idx in range(1, num_players):
            if opponent_type == "mix1":
                # 1 Baseline V1, 2 Random Agents
                if idx == 1:
                    agents.append(HeuristicAgent(player_id=idx, seed=match_seed + 100 + idx))
                else:
                    agents.append(RandomAgent(player_id=idx, seed=match_seed + 100 + idx))
            elif opponent_type == "mix2":
                # 2 Baseline V1, 1 Random Agent
                if idx in (1, 2):
                    agents.append(HeuristicAgent(player_id=idx, seed=match_seed + 100 + idx))
                else:
                    agents.append(RandomAgent(player_id=idx, seed=match_seed + 100 + idx))
            else:
                # 3 Baseline V1 (high strategy table)
                agents.append(HeuristicAgent(player_id=idx, seed=match_seed + 100 + idx))
            
    # Start Match
    state, _ = AceEngine.advance(state)
    
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
        legal_actions = AceEngine.get_legal_actions(state)
        
        agent = agents[player_id]
        action = agent.select_action(player_view, legal_actions)
        
        res = AceEngine.apply_action(state, action)
        if isinstance(res, Success):
            state = res.new_state
        else:
            break
            
    # Calculate round metrics
    rounds_won = 0
    rounds_lost = 0
    for res in state.match_state.round_results:
        if not res.is_draw:
            if res.loser_id != 0:
                rounds_won += 1
            else:
                rounds_lost += 1
        else:
            rounds_won += 1  # Draw counts as survival
            
    # Find final placement (rank) of Player 0 in the match
    rank = 4
    phase = AceEngine.get_game_phase(state)
    if isinstance(phase, MatchComplete):
        for r in phase.result.rankings:
            if r.player_id == 0:
                rank = r.rank
                break
                
    return rounds_won, rounds_lost, rank

def evaluate_candidate(
    candidate_weights: Dict[str, float],
    seeds: List[int],
    opponent_type: str,
    opponents_weights: Optional[List[Dict[str, float]]] = None
) -> float:
    """Evaluates candidate chromosome over a batch of seeds and returns fitness."""
    total_won = 0
    total_lost = 0
    total_placement_score = 0.0
    
    # Non-linear placement weighting: reward 1st, penalize 4th heavily
    placement_scores = {1: 150.0, 2: 60.0, 3: 0.0, 4: -150.0}
    
    for seed in seeds:
        won, lost, rank = run_match_for_eval(candidate_weights, seed, opponent_type, opponents_weights)
        total_won += won
        total_lost += lost
        total_placement_score += placement_scores.get(rank, -150.0)
        
    total_rounds = total_won + total_lost
    if total_rounds == 0:
        return -300.0
        
    win_rate = total_won / total_rounds
    loss_rate = total_lost / total_rounds
    avg_placement_score = total_placement_score / len(seeds)
    
    # Fitness = (100 * WinRate) - (200 * LossRate) + avg_placement_score
    return (100.0 * win_rate) - (200.0 * loss_rate) + avg_placement_score

def worker_eval_candidate(args: Tuple[Dict[str, float], List[int], str, Optional[List[Dict[str, float]]]]) -> float:
    """Windows-safe multiprocessing worker function."""
    candidate_weights, seeds, opponent_type, opponents_weights = args
    return evaluate_candidate(candidate_weights, seeds, opponent_type, opponents_weights)

def clamp_parameter(k: str, val: float) -> float:
    """Clamps evolved parameters to their specific valid logical domains."""
    if "rank_threshold" in k:
        return max(2.0, min(14.0, val))
    elif "hoard_size" in k:
        return max(2.0, min(13.0, val))
    elif "u_suit_threshold" in k:
        return max(0.0, min(13.0, val))
    else:
        return max(0.0, min(1.0, val))

def compute_euclidean_distance(c1: Dict[str, float], c2: Dict[str, float]) -> float:
    """Computes the Euclidean distance between two candidate weight vectors."""
    diff_sum = 0.0
    for k in c1.keys():
        if k != "H114" and not k.startswith("P_"):
            diff_sum += (c1.get(k, 100.0) - c2.get(k, 100.0)) ** 2
    return diff_sum ** 0.5

def draw_ascii_chart(history: List[float]):
    """Renders a beautiful ASCII scatter chart of maximum fitness trend."""
    if not history:
        return
    print("\n" + "="*50)
    print("=== MAX FITNESS PROGRESS TRAJECTORY ===")
    print("="*50)
    min_val = min(history)
    max_val = max(history)
    val_range = max_val - min_val if max_val != min_val else 1.0
    
    height = 10
    width = len(history)
    
    scaled = [int((val - min_val) / val_range * (height - 1)) for val in history]
    
    for r in range(height - 1, -1, -1):
        line = ""
        for val in scaled:
            if val == r:
                line += "#"
            else:
                line += " "
        val_at_r = min_val + (r / (height - 1)) * val_range
        print(f"{val_at_r:6.1f} | {line}")
    print("       +" + "-" * width)
    print("Gen:    " + "".join(str(i % 10) for i in range(1, width + 1)))
    print("="*50 + "\n")

def main():
    print("====================================================")
    print("===    GENETIC WEIGHT OPTIMIZATION PIPELINE     ===")
    print("====================================================\n")

    random.seed(42)
    
    # 1. Initialize Registry
    base_agent = HeuristicAgent(0)
    baseline_weights = dict(base_agent.weights)

    # 2. Add Strategy Parameters to baseline weights list
    default_parameters = {
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
    baseline_weights.update(default_parameters)

    # Population setup
    pop_size = 60
    generations = 40
    population: List[Dict[str, float]] = []
    
    # Load baseline (V1) as individual #0
    population.append(dict(baseline_weights))
    
    # Perturb the rest of the starting generation (both weights and thresholds)
    for _ in range(pop_size - 1):
        ind = {}
        for k, v in baseline_weights.items():
            if k == "H114":
                ind[k] = v
            elif k.startswith("P_"):
                # Perturb parameter threshold
                if "rank_threshold" in k or "hoard_size" in k or "u_suit_threshold" in k:
                    ind[k] = clamp_parameter(k, v + random.gauss(0, 1.0))
                else:
                    ind[k] = clamp_parameter(k, v + random.gauss(0, 0.1))
            else:
                # Perturb weight
                ind[k] = max(0.0, min(500.0, v + random.gauss(0, 30.0)))
        population.append(ind)

    # Validation Set (Static, unseen seeds, scaled to 100 to reduce noise)
    val_seeds = [4000 + idx for idx in range(100)]

    # Hall of Fame (HoF) tracking - starts with the baseline V1 weights
    hall_of_fame = [dict(baseline_weights)]
    best_overall_weights = dict(baseline_weights)
    
    # Initialize elite pool (starts as copies of baseline weights)
    elite_pool = [dict(baseline_weights) for _ in range(10)]

    # Logging file setup
    log_file = "scripts/genetic_training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "max_fitness", "avg_fitness", "min_fitness", "val_fitness", "val_hof_fitness", "best_weights"])

    history_max = []
    history_val = []
    start_time = time.time()

    # Early stopping trackers
    best_val_score = -9999.0
    no_improvement_count = 0

    # Dynamic opponent pool phases (including Co-Evolution self-play in Phase 3)
    def get_opponent_type(gen: int) -> str:
        if gen <= 12:
            return "mix1"      # 1 V1, 2 Random
        elif gen <= 24:
            return "mix2"      # 2 V1, 1 Random
        else:
            return "co_evolve" # Co-Evolution (opponents are selected from best population members)

    for gen in range(1, generations + 1):
        opp_type = get_opponent_type(gen)
        
        # 50 training seeds for high statistical confidence per candidate
        eval_seeds = [3000 + gen * 100 + idx for idx in range(50)]
        
        # Select opponents weights if co_evolve is active
        co_evolve_opps = None
        if opp_type == "co_evolve":
            # Opponents are selected from best 10 population members of the prior generation
            co_evolve_opps = random.choices(elite_pool, k=3)
        
        # Parallel evaluation using ProcessPoolExecutor
        tasks = [(ind, eval_seeds, opp_type, co_evolve_opps) for ind in population]
        with ProcessPoolExecutor() as executor:
            raw_fitnesses = list(executor.map(worker_eval_candidate, tasks))
            
        # 2. Diversity Preservation: Fitness Sharing / Novelty Penalty
        # Adjust fitness by adding a bonus for novelty (distance from rest of the population)
        fitnesses = []
        for i, ind in enumerate(population):
            distances = [compute_euclidean_distance(ind, other) for j, other in enumerate(population) if i != j]
            avg_dist = sum(distances) / len(distances) if distances else 0.0
            # Novelty bonus adds up to +15.0 for high diversity
            novelty_bonus = min(15.0, 0.02 * avg_dist)
            fitnesses.append(raw_fitnesses[i] + novelty_bonus)

        # Zip and sort by adjusted fitness descending
        sorted_pop = [p for _, p in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
        sorted_fit = sorted(fitnesses, reverse=True)
        
        max_fit = sorted_fit[0]
        avg_fit = sum(sorted_fit) / len(sorted_fit)
        min_fit = sorted_fit[-1]
        history_max.append(max_fit)
        
        # Validate the best chromosome on the unseen validation seeds against baseline V1
        val_score = evaluate_candidate(sorted_pop[0], val_seeds, "v1_only")
        history_val.append(val_score)

        if val_score > best_val_score:
            best_overall_weights = dict(sorted_pop[0])

        # Anchor HoF Validation sampling:
        # Seat 1: The original baseline V1
        # Seat 2: The best overall weights found so far
        # Seat 3: A random champion from the historical HoF pool
        anchor_opps = [
            dict(baseline_weights),
            dict(best_overall_weights),
            random.choice(hall_of_fame)
        ]
        val_hof_score = evaluate_candidate(sorted_pop[0], val_seeds, "hof", anchor_opps)
        
        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, max_fit, avg_fit, min_fit, val_score, val_hof_score, json.dumps(sorted_pop[0])])
            
        # Live Progress Display
        progress = int((gen / generations) * 20)
        bar = "#" * progress + "-" * (20 - progress)
        print(f"Gen {gen:02d}/40 [{bar}] | Max: {max_fit:6.1f} | Val: {val_score:6.1f} | HoF: {val_hof_score:6.1f} | Pool: {opp_type:<7}")

        # Update Hall of Fame periodically (every 6 generations) or if validation fit is high
        if gen % 6 == 0 or val_score > 120.0:
            hall_of_fame.append(dict(sorted_pop[0]))

        # Reset patience counter on environment transitions
        if gen in (13, 25):
            no_improvement_count = 0
            best_val_score = -9999.0  # Reset best score benchmark for the new environment

        # Early stopping plateau detector (only enabled in co-evolution phase, gen >= 25)
        if val_score > best_val_score + 0.1:
            best_val_score = val_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if gen >= 25 and no_improvement_count >= 20:
            print(f"\n[EARLY STOPPING] Validation score plateaued for 20 generations at {best_val_score:.1f}. Terminating training.")
            break

        # Adaptive Mutation Scheduling
        mut_rate = 0.25 - 0.20 * (gen / generations)
        mut_sigma = 30.0 - 25.0 * (gen / generations)

        # Elitism: keep top 2 chromosomes intact
        new_pop = [sorted_pop[0], sorted_pop[1]]

        # Selection, Crossover & Mutation to fill population
        while len(new_pop) < pop_size:
            # Tournament selection (size 3)
            def select_parent():
                candidates = random.sample(list(zip(fitnesses, population)), 3)
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][1]
                
            p_a = select_parent()
            p_b = select_parent()
            
            # Hybrid Crossover Strategy: 50% Grouped Block Crossover, 50% Arithmetic Recombination
            c1, c2 = {}, {}
            if random.random() < 0.5:
                # Grouped Block Crossover (preserves functional relationships between H1xx, H2xx, etc.)
                for grp in ["H1", "H2", "H3", "H4", "H5", "H7"]:
                    use_p_a = random.random() < 0.5
                    for k in baseline_weights.keys():
                        if k.startswith(grp):
                            if use_p_a:
                                c1[k] = p_a[k]
                                c2[k] = p_b[k]
                            else:
                                c1[k] = p_b[k]
                                c2[k] = p_a[k]
                c1["H114"] = baseline_weights["H114"]
                c2["H114"] = baseline_weights["H114"]
                # Inherit parameters starting with "P_"
                for k in baseline_weights.keys():
                    if k.startswith("P_"):
                        if random.random() < 0.5:
                            c1[k] = p_a[k]
                            c2[k] = p_b[k]
                        else:
                            c1[k] = p_b[k]
                            c2[k] = p_a[k]
            else:
                # Arithmetic Crossover (blends continuous weights and thresholds)
                alpha = random.uniform(0.1, 0.9)
                for k in baseline_weights.keys():
                    if k == "H114":
                        c1[k] = baseline_weights[k]
                        c2[k] = baseline_weights[k]
                    elif k.startswith("P_"):
                        c1[k] = clamp_parameter(k, alpha * p_a[k] + (1.0 - alpha) * p_b[k])
                        c2[k] = clamp_parameter(k, (1.0 - alpha) * p_a[k] + alpha * p_b[k])
                    else:
                        c1[k] = max(0.0, min(500.0, alpha * p_a[k] + (1.0 - alpha) * p_b[k]))
                        c2[k] = max(0.0, min(500.0, (1.0 - alpha) * p_a[k] + alpha * p_b[k]))
                    
            # Mutation (Vectorized Adaptive Gaussian Mutation + Coordinated Feature Group Mutation)
            for child in (c1, c2):
                mutated = {}
                # 30% chance of coordinated group mutation (mutating related heuristics in unison)
                shift_group = None
                shift_delta = 0.0
                if random.random() < 0.30:
                    shift_group = random.choice(["H1", "H2", "H3", "H4", "H5", "H7"])
                    shift_delta = random.gauss(0, mut_sigma)

                for k, v in child.items():
                    if k == "H114":
                        mutated[k] = v
                    elif k.startswith("P_"):
                        # Mutate parameters
                        if random.random() < mut_rate:
                            if "rank_threshold" in k or "hoard_size" in k or "u_suit_threshold" in k:
                                mutated[k] = clamp_parameter(k, v + random.gauss(0, mut_sigma * 0.1))
                            else:
                                mutated[k] = clamp_parameter(k, v + random.gauss(0, mut_sigma * 0.01))
                        else:
                            mutated[k] = v
                    else:
                        # Mutate weights
                        if shift_group and k.startswith(shift_group):
                            mutated[k] = max(0.0, min(500.0, v + shift_delta))
                        elif random.random() < mut_rate:
                            mutated[k] = max(0.0, min(500.0, v + random.gauss(0, mut_sigma)))
                        else:
                            mutated[k] = v
                new_pop.append(mutated)
                
        # Trim population to exactly pop_size
        population = new_pop[:pop_size]
        
        # Update elite pool with the best sorted chromosomes of this generation
        elite_pool = sorted_pop[:10]

    # Save the evolved weights & parameters
    best_weights = sorted_pop[0]
    weights_dest = "engine/heuristic_v2_weights.json"
    with open(weights_dest, "w") as f:
        json.dump(best_weights, f, indent=4)
        
    duration = time.time() - start_time
    print(f"\nTraining complete in {duration:.2f} seconds!")
    print(f"Evolved weights successfully written to: {weights_dest}")
    
    # Render progress chart
    draw_ascii_chart(history_max)

if __name__ == "__main__":
    main()
