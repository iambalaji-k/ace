# Phase 6 — Tournament Runner Implementation Plan

This document outlines the design and implementation strategy for the **Tournament Runner** (Phase 6) to orchestrate and benchmark thousands of matches.

---

## 1. Abstract Agent Interface (`engine/agent.py`)

To allow different AI agents to participate in a tournament, we must define the standard base interface class:

```python
class BaseAgent:
    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    def select_action(self, player_view: EngineState, legal_actions: List[Action]) -> Action:
        """Select an action based on the player's view of the state."""
        raise NotImplementedError
```

*Note: For testing the runner prior to implementing Phase 7 (AI Agents), we will write a temporary `RandomAgent` baseline in `engine/agent.py` that selects randomly from `legal_actions`.*

---

## 2. Tournament Configurations & Runner (`engine/tournament.py`)

The Tournament Runner will coordinate multi-threaded execution of matches:

```python
@dataclass(frozen=True)
class TournamentConfig:
    num_matches: int
    num_players: int
    num_rounds: int
    base_seed: int
    agent_types: List[Type[BaseAgent]]  # e.g., P0: RandomAgent, P1: HeuristicAgent, etc.

class TournamentRunner:
    def __init__(self, config: TournamentConfig):
        self.config = config
        self.results = []

    def run(self) -> TournamentResults:
        """Run matches in parallel using ThreadPoolExecutor."""
        # 1. Generate N match seeds from base_seed: match_seed_i = base_seed + i
        # 2. Spawn ThreadPoolExecutor with max_workers matching CPU cores.
        # 3. Each worker:
        #    - Instantiates the N agents.
        #    - Calls AceEngine.create_match and advances.
        #    - Plays match loops: calls agent.select_action() for the active player, applies action.
        #    - Returns final MatchResult.
        # 4. Aggregates results and returns TournamentResults object.
        pass
```

---

## 3. Statistical Metrics & Exporters

The runner will track key statistics for benchmarking and analysis:

- **Point Table**: Mean, standard deviation, and 95% confidence intervals of points scored by each agent.
- **Ratios**: Win, Loss, and Draw ratios per agent.
- **Game Metrics**: Max consecutive loss counter observed, total rounds played, re-entry counts.
- **Execution Performance**: Speed metrics (matches per second).

### CSV Exporter
- Export a detailed row-by-row match record to `benchmarks/tournament_records.csv`.
- Export a summary table to `benchmarks/tournament_summary.csv`.

---

## 4. Conformance Unit Testing (`tests/test_tournament.py`)

We will add unit tests verifying:
- **Thread Safety**: Parallel execution does not leak state, crash, or violate invariants.
- **Determinism**: Running a 100-match tournament twice with the same seed yields identical scores, win ratios, and CSV lines.
- **CSV Format Integrity**: Verifies columns exist and data values serialize correctly.
