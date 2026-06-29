# tests/test_tournament.py
"""Unit tests for the Tournament Runner.

Verifies thread-safety, absolute determinism of parallel match simulations,
and correct format exports of summary and detailed records to CSV files.
"""

import os
import shutil
from engine.agent import RandomAgent
from engine.tournament import TournamentConfig, TournamentRunner


def test_tournament_runner_execution_and_determinism():
    # 1. Configuration setup (4 players, 2 rounds, 5 matches)
    config = TournamentConfig(
        num_matches=5,
        num_players=4,
        num_rounds=2,
        base_seed=100,
        agent_classes=[RandomAgent, RandomAgent, RandomAgent, RandomAgent]
    )

    # 2. Run first tournament instance
    runner_1 = TournamentRunner(config)
    results_1 = runner_1.run()

    assert len(results_1.match_records) == 20  # 5 matches * 4 players
    assert results_1.execution_time > 0.0

    # Verify that values calculated for player 0 exist
    assert 0 in results_1.player_stats
    p0_stats = results_1.player_stats[0]
    assert p0_stats.win_ratio >= 0.0

    # 3. Run second tournament instance with identical seed and configuration
    runner_2 = TournamentRunner(config)
    results_2 = runner_2.run()

    # Verify perfect determinism across both parallel runs
    for p_id in range(4):
        stats_1 = results_1.player_stats[p_id]
        stats_2 = results_2.player_stats[p_id]
        assert stats_1.mean_points == stats_2.mean_points
        assert stats_1.std_dev_points == stats_2.std_dev_points
        assert stats_1.win_ratio == stats_2.win_ratio
        assert stats_1.loss_ratio == stats_2.loss_ratio
        assert stats_1.max_consecutive_losses == stats_2.max_consecutive_losses


def test_tournament_csv_exports():
    config = TournamentConfig(
        num_matches=3,
        num_players=4,
        num_rounds=1,
        base_seed=200,
        agent_classes=[RandomAgent, RandomAgent, RandomAgent, RandomAgent]
    )

    runner = TournamentRunner(config)
    results = runner.run()

    # Define temporary output path for benchmarking exports
    temp_output_dir = "benchmarks/temp_test_run"
    
    # Export reports
    runner.export_to_csv(results, temp_output_dir)

    summary_file = os.path.join(temp_output_dir, "tournament_summary.csv")
    records_file = os.path.join(temp_output_dir, "tournament_records.csv")

    assert os.path.exists(summary_file)
    assert os.path.exists(records_file)

    # Validate content sizes (header + data rows)
    with open(summary_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 5  # 1 header + 4 players rows

    with open(records_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 13  # 1 header + 12 rows (3 matches * 4 players)

    # Clean up output directory
    shutil.rmtree(temp_output_dir)
