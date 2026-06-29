# engine/tournament.py
"""Deterministic parallel tournament runner.

Orchestrates execution of multiple matches in parallel threads, calculates
statistical metrics (mean, std-dev, confidence intervals), and exports results to CSV files.
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import List, Type, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from engine.rules import AceEngine, Success
from engine.types import RoundStarting, EngineState
from engine.agent import BaseAgent
from engine.events import get_player_view


@dataclass(frozen=True)
class TournamentConfig:
    num_matches: int
    num_players: int
    num_rounds: int
    base_seed: int
    agent_classes: List[Type[BaseAgent]]


@dataclass(frozen=True)
class PlayerStats:
    player_id: int
    agent_name: str
    mean_points: float
    std_dev_points: float
    ci_95_lower: float
    ci_95_upper: float
    win_ratio: float
    loss_ratio: float
    draw_ratio: float
    max_consecutive_losses: int


@dataclass(frozen=True)
class TournamentResults:
    config: TournamentConfig
    player_stats: Dict[int, PlayerStats]
    match_records: List[Dict[str, Any]]
    execution_time: float


def _run_single_match(config: TournamentConfig, match_index: int) -> EngineState:
    """Run a single match simulation and return the final EngineState."""
    match_seed = config.base_seed + match_index

    # 1. Create match state
    state = AceEngine.create_match(
        match_id=match_index,
        num_players=config.num_players,
        num_rounds=config.num_rounds,
        match_seed=match_seed
    )

    # 2. Instantiate agents
    agents = [
        config.agent_classes[i](player_id=i, seed=match_seed + 1000 + i)
        for i in range(config.num_players)
    ]

    # Start Round 1
    state, _ = AceEngine.advance(state)

    # 3. Match execution loop
    while not AceEngine.is_terminal(state):
        phase = AceEngine.get_game_phase(state)

        # Check for auto-advances (e.g. RoundStarting transition)
        if isinstance(phase, RoundStarting):
            state, _ = AceEngine.advance(state)
            continue

        player_id = state.runtime_state.current_player_id
        assert player_id is not None

        # Mask state details for the current active player view
        player_view = get_player_view(state, player_id)
        legal_actions = AceEngine.get_legal_actions(state)

        # Get choice from agent
        agent = agents[player_id]
        action = agent.select_action(player_view, legal_actions)

        # Apply choice
        res = AceEngine.apply_action(state, action)
        if not isinstance(res, Success):
            # Fallback in case of invalid bot choice: pick first legal action
            res = AceEngine.apply_action(state, legal_actions[0])

        assert isinstance(res, Success)
        state = res.new_state

    return state


class TournamentRunner:
    """Orchestrator for managing multi-threaded tournament matches and statistics."""

    def __init__(self, config: TournamentConfig) -> None:
        self.config = config

    def run(self) -> TournamentResults:
        import time
        start_time = time.time()

        # Run matches in parallel using ThreadPoolExecutor
        states: List[EngineState] = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_run_single_match, self.config, idx)
                for idx in range(self.config.num_matches)
            ]
            for fut in futures:
                states.append(fut.result())

        end_time = time.time()
        exec_duration = end_time - start_time

        # Aggregate statistics
        records = []
        player_points: Dict[int, List[float]] = {i: [] for i in range(self.config.num_players)}
        player_wins: Dict[int, int] = {i: 0 for i in range(self.config.num_players)}
        player_losses: Dict[int, int] = {i: 0 for i in range(self.config.num_players)}
        player_draws: Dict[int, int] = {i: 0 for i in range(self.config.num_players)}
        player_max_losses: Dict[int, int] = {i: 0 for i in range(self.config.num_players)}

        for idx, s in enumerate(states):
            result = AceEngine.get_result(s)
            assert result is not None
            for player in result.rankings:
                p_id = player.player_id
                points = player.half_points / 2.0
                player_points[p_id].append(points)
                player_wins[p_id] += player.rounds_won
                player_losses[p_id] += player.rounds_lost
                player_draws[p_id] += player.rounds_drawn
                
                # Retrieve final consecutive losses from state
                m_player = s.match_state.players[p_id]
                loss_count = m_player.consecutive_loss_count
                if loss_count > player_max_losses[p_id]:
                    player_max_losses[p_id] = loss_count

                records.append({
                    "match_id": idx,
                    "player_id": p_id,
                    "rank": player.rank,
                    "points": points,
                    "rounds_won": player.rounds_won,
                    "rounds_lost": player.rounds_lost,
                    "rounds_drawn": player.rounds_drawn,
                    "consecutive_losses": loss_count
                })

        player_stats = {}
        total_rounds = self.config.num_matches * self.config.num_rounds

        for p_id in range(self.config.num_players):
            points_list = player_points[p_id]
            mean_pt = sum(points_list) / self.config.num_matches
            
            # Standard Deviation
            variance = sum((x - mean_pt) ** 2 for x in points_list) / max(1, self.config.num_matches - 1)
            std_dev = math.sqrt(variance)

            # 95% Confidence Interval for the mean points
            ci_margin = 1.96 * (std_dev / math.sqrt(self.config.num_matches))
            
            win_r = player_wins[p_id] / total_rounds
            loss_r = player_losses[p_id] / total_rounds
            draw_r = player_draws[p_id] / total_rounds

            agent_name = self.config.agent_classes[p_id].__name__

            player_stats[p_id] = PlayerStats(
                player_id=p_id,
                agent_name=agent_name,
                mean_points=mean_pt,
                std_dev_points=std_dev,
                ci_95_lower=mean_pt - ci_margin,
                ci_95_upper=mean_pt + ci_margin,
                win_ratio=win_r,
                loss_ratio=loss_r,
                draw_ratio=draw_r,
                max_consecutive_losses=player_max_losses[p_id]
            )

        return TournamentResults(
            config=self.config,
            player_stats=player_stats,
            match_records=records,
            execution_time=exec_duration
        )

    def export_to_csv(self, results: TournamentResults, output_dir: str) -> None:
        """Export tournament match records and player summary statistics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Summary CSV
        summary_path = os.path.join(output_dir, "tournament_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "player_id", "agent_name", "mean_points", "std_dev_points",
                "ci_95_lower", "ci_95_upper", "win_ratio", "loss_ratio",
                "draw_ratio", "max_consecutive_losses"
            ])
            for p_id, stats in sorted(results.player_stats.items()):
                writer.writerow([
                    stats.player_id, stats.agent_name,
                    f"{stats.mean_points:.4f}", f"{stats.std_dev_points:.4f}",
                    f"{stats.ci_95_lower:.4f}", f"{stats.ci_95_upper:.4f}",
                    f"{stats.win_ratio:.4f}", f"{stats.loss_ratio:.4f}",
                    f"{stats.draw_ratio:.4f}", stats.max_consecutive_losses
                ])

        # 2. Detailed Match Records CSV
        records_path = os.path.join(output_dir, "tournament_records.csv")
        with open(records_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "match_id", "player_id", "rank", "points", "rounds_won",
                "rounds_lost", "rounds_drawn", "consecutive_losses"
            ])
            for rec in results.match_records:
                writer.writerow([
                    rec["match_id"], rec["player_id"], rec["rank"],
                    f"{rec['points']:.1f}", rec["rounds_won"],
                    rec["rounds_lost"], rec["rounds_drawn"],
                    rec["consecutive_losses"]
                ])
