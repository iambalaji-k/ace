# Ace Engine 🃏

A professional-grade, specification-driven, mathematically deterministic card game engine in Python. Built from the ground up for high-performance simulation, reinforcement learning self-play, parallel tournaments, and byte-perfect replay execution.

---

## 🚀 Vision & Core Philosophy

The Ace Engine is designed as a foundational simulation library for imperfect information card games. It enforces absolute determinism and strict architectural boundaries:

1. **Immutable State transitions**: The entire game is modeled as a pure state machine. State is represented by deeply nested immutable dataclasses. Transitions are side-effect-free: applying an action returns a new, independent state object rather than mutating the existing one.
2. **Perfect Determinism**: All random operations (deck shuffles, deal distributions, agent choices) are powered by a customized, language-agnostic **PCG-XSH-RR-64/32** Pseudo-Random Number Generator (PRNG). Seeding a match guarantees byte-for-byte reproducibility across operating systems, CPU architectures, and languages.
3. **Imperfect Information / State Masking**: The engine enforces a strict boundary between public and private information. AI agents interact with the engine solely via a projected **Player View** that masks other players' hands and hides the face-down discard pile.
4. **Production Readiness**: Employs rigorous static analysis checks (100% type safety with `basedpyright`, static checks with `ruff`) and holds a comprehensive test suite covering spec compliance, mathematical foundations, and edge cases.

---

## 🎮 Game Rules & Mechanics Primer

The Ace game is a trick-taking card game played with a standard 52-card deck (Spades high, Diamonds low; Ace high, Two low). It introduces several distinct tactical mechanics:

* **Reserved Aces & Skip Dealing**: When a player loses a round, their consecutive loss counter increases. In the next deal, they receive a guaranteed reserved Ace for each consecutive loss, and the dealer skips their seat during standard card distribution rotations for that round.
* **The Steal Phase**: Before a trick's card play begins, the lead player has the option to steal the entire hand of the active player to their left. If a player is stolen from, they immediately go out of play (becoming inactive) and are declared a **Round Winner**. The stealer merges the stolen cards into their hand, and can continue stealing clockwise.
* **Interrupted Tricks**: Standard play follows the lead suit. If a player cannot follow suit, they *must* break suit (play a card of another suit). This immediately interrupts and ends the trick. The player who played the highest card of the lead suit collects all cards on the table, merging them into their hand.
* **Re-Entry & Ineligibility**: If a player empties their hand during a trick but is subsequently forced to collect cards because someone broke suit, they **re-enter** active play. However, they are marked as `re_entered = True` and are disqualified from being a Winner in that round.
* **Endgame & Scoring**: The round ends when exactly one active player remains (who is declared the **Round Loser**), or if all players run out of cards simultaneously (**Draw**). Winning a round yields half-points, losing yields 0, and a draw resets consecutive loss counters.

---

## 📁 Repository Directory Structure

```text
ace-engine/
├── docs/                             # Developer documentation and guides
│   ├── ARCHITECTURE.md               # Normative architecture definitions
│   ├── Developer_Guide.md            # Quick-start dev onboarding guide
│   ├── phase_2_plan.md               # Compliance catalog implementation plan
│   └── status.md                     # Active project status registry
├── spec/                             # Frozen normative specification chapters
│   ├── compliance_schema.json        # JSON Schema for compliance tests
│   ├── compliance_test_catalog.md    # Normalized catalog list (TEST-0001 to TEST-0010)
│   ├── chapter_01_foundations.md     # Chapter 1: Cards & PRNG
│   ├── chapter_02_data_model.md      # Chapter 2: State objects representation
│   ├── chapter_03_match_lifecycle.md # Chapter 3: Match configuration and seating
│   ├── chapter_04_round_lifecycle.md # Chapter 4: Deal, reserved cards, rotations
│   ├── chapter_05_rules_engine.md    # Chapter 5: Legal action validation
│   ├── chapter_06_state_machine.md   # Chapter 6: Engine state transitions
│   ├── chapter_07_event_system.md    # Chapter 7: Projection of public/private events
│   ├── chapter_08_validation.md      # Chapter 8: Invariants checks (INV-001 to INV-012)
│   ├── chapter_09_replay.md          # Chapter 9: Replay log specs
│   ├── chapter_10_ai_interface.md    # Chapter 10: Player views and actions
│   └── chapter_11_compliance.md      # Chapter 11: Compliance criteria
├── engine/                           # Core Game Engine codebase
│   ├── card.py                       # Card representations, suit/rank helpers, string mapping
│   ├── prng.py                       # PCG-64/32 PRNG and Fisher-Yates card shuffler
│   ├── deck.py                       # Reserved ace allocation and deal skip rotations
│   ├── types.py                      # Immutable state models, action classes, game events
│   ├── invariants.py                 # Enforcer checking INV-001 to INV-012 invariants
│   ├── rules.py                      # Engine transition state machine and validation
│   ├── events.py                     # Public, private, and observer event projections
│   ├── replay.py                     # JSON serialization schemas and file utilities
│   ├── replay_player.py              # Playback player (step, undo, jump, branching fork)
│   ├── agent.py                      # BaseAgent interface and RandomAgent baseline
│   └── tournament.py                 # ThreadPoolExecutor multi-threaded tournament runner
├── tests/                            # Automated Testing suites
│   ├── compliance/                   # Conformance JSON test files
│   │   ├── TEST-0001.json            # PRNG output validation vectors
│   │   └── TEST-0005.json            # Decline steal state verification
│   ├── test_compliance_harness.py    # Automated test runner loading compliance JSONs
│   ├── test_foundations.py           # Unit tests checking cards, shuffles, PRNG
│   ├── test_rules.py                 # Integration tests for match simulations
│   ├── test_walkthrough_compliance.py# Validation of the 2-round narrated match walkthrough
│   ├── test_replays.py               # Replay player fidelity, undo, and branching tests
│   └── test_tournament.py            # ThreadPoolExecutor thread-safety and determinism tests
├── scripts/                          # Developer execution scripts
│   ├── run_manual_demo.py            # Interactive CLI tool for testing rules manually
│   └── run_bot_match.py              # Progression viewer for automated bot matches
├── replays/                          # Target folder for saving JSON replay logs
└── benchmarks/                       # Target folder for tournament CSV exports
```

---

## 🛠️ Developer Setup & Execution Guide

### 1. Requirements & Setup
Ensure you have **Python 3.10+** installed. Clone the repository and initialize the virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies (pytest, ruff, basedpyright)
pip install pytest ruff basedpyright
```

### 2. Running Automated Tests
Run the entire suite of 21 tests (including unit, integration, replay, and tournament tests):

```bash
python -m pytest
```

### 3. Running Code Quality Linters & Typecheckers
Ensure your code changes remain fully type-safe and comply with repository formatting rules:

```bash
# Run Ruff code analysis
python -m ruff check engine/ tests/ scripts/

# Run Basedpyright static typechecker
python -m basedpyright engine/ tests/ scripts/
```

---

## 🕹️ Match Simulation & CLI Tools

### A. Watch Automated Bots Play (`run_bot_match.py`)
Watch 4 random agents play an automated game step-by-step with a custom turn delay. You will see hands dealt, steal attempts, card plays, suit breaks, and a final rankings scoreboard:

```bash
python -m scripts.run_bot_match --players 4 --rounds 2 --seed 42 --delay 0.3
```

### B. Manually Rig & Simulate a Match (`run_manual_demo.py`)
Run an interactive CLI simulation where you can set custom card hands for specific players and test rules (like forcing a suit break) step-by-step:

```bash
python -m scripts.run_manual_demo
```

---

## 📊 Replay Player & Tournament Runner APIs

### 1. Replay System Playback (`engine/replay_player.py`)
Record, load, step, or undo actions. You can also fork history to perform branching "what-if" analyses:

```python
from engine.replay import import_replay
from engine.replay_player import ReplayPlayer
from engine.types import PlayCardAction

# Import log
replay = import_replay("replays/match_123.json")
player = ReplayPlayer(replay)

# Traversal
player.step()      # Step 1 action forward
player.undo()      # Step 1 action backward (O(1) cached)
player.jump_to(10) # Jump straight to action index 10

# Branching / Forking analysis at action index 5
new_replay = player.fork_at(5, PlayCardAction(player_id=0, card=0))
```

### 2. Multi-Threaded Tournament Benchmarking (`engine/tournament.py`)
Run thousands of matches in parallel threads using `ThreadPoolExecutor`. The runner compiles points mean, standard deviation, 95% confidence intervals, win ratios, and exports results to CSV:

```python
from engine.tournament import TournamentConfig, TournamentRunner
from engine.agent import RandomAgent

# Config a 1,000-match tournament
config = TournamentConfig(
    num_matches=1000,
    num_players=4,
    num_rounds=3,
    base_seed=10000,
    agent_classes=[RandomAgent, RandomAgent, RandomAgent, RandomAgent]
)

runner = TournamentRunner(config)
results = runner.run()

# Export statistics to CSV files
runner.export_to_csv(results, "benchmarks/")
```

---

## 🤖 AI Agents (Heuristics, MCTS, and Self-Learning)

The Ace Engine features three distinct tiers of artificial intelligence, transitioning from structured rule-based systems to deep self-play reinforcement learning:

### 1. Evolved Heuristic Agent (`HeuristicAgentV2`)
* **Core Logic**: An expert-rule system containing **55 distinct card-play and steal heuristics** categorized by match phase (Opening, Middle, Endgame) and hand configurations.
* **Optimization (Genetic Algorithm)**: Co-evolves both the 55 heuristic weights and **11 strategic parameter thresholds** (e.g. opponent void risk limits, suit-hoarding boundaries) to maximize match placement outcomes.
* **Training Command**:
  ```bash
  python scripts/train_genetic_weights.py
  ```
  *(Optimizes weights and outputs updated parameters to `engine/heuristic_v2_weights.json`)*.

### 2. Monte Carlo Tree Search Agent (`MCTSAgent`)
* **Core Logic**: Implements **Information Set MCTS (ISMCTS)** to play under imperfect information.
* **Hand Determinization**: Features a backtracking constraint satisfaction search that generates random opponent hands matching all `CardTracker` void suits, known cards, and current trick cards.
* **Heuristic-Guided Rollouts**: Utilizes our evolved `HeuristicAgentV2` to make playout choices, producing high-fidelity rollouts and extremely fast search convergence on CPU.
* **Evaluation Command**:
  ```bash
  python scripts/evaluate_mcts_agent.py
  ```

### 3. Self-Learning RL Agent (`RLAgent`)
* **Core Logic**: A pure reinforcement learning agent trained entirely from scratch with **zero human bias**.
* **Model**: A lightweight dual Actor-Critic network (`AceNet` in PyTorch) running state vectorization (343 input features) and legal action masking.
* **Training Method**: REINFORCE Policy Gradient with Advantage baseline, highly optimized to complete thousands of training matches on a standard **CPU** in under 5 minutes.
* **Training & Evaluation Commands**:
  ```bash
  # Train the agent (saves weights to engine/rl_champion.pt)
  python scripts/train_self_play.py

  # Benchmark RL Agent against Heuristic V2
  python scripts/evaluate_rl_agent.py
  ```

---

## 🎮 Play Against the Agents (CLI Interface)

You can play interactive card matches directly against these agents in your terminal:

* **Play against Heuristic bots**:
  ```bash
  python scripts/play_against_heuristic.py
  ```
* **Play against the trained RL Champion bots**:
  ```bash
  python scripts/play_against_rl.py
  ```

---

## 🗺️ Project Roadmap & Status

- [x] **Phase 0**: Repository Setup & Core Infrastructure Documents
- [x] **Phase 1**: Chapter 1–11 Protocol Specification drafting
- [x] **Phase 2**: Compliance Test Catalog & Automated loader
- [x] **Phase 3**: Immutable State Machine Data Models
- [x] **Phase 4**: Rules Engine & Headless Simulator logic
- [x] **Phase 5**: Appendix/Replay Player (Undo, Redo, Branching)
- [x] **Phase 6**: Tournament runner & Statistical CSV exporter
- [x] **Phase 7**: Random AI Agent baseline integration
- [x] **Phase 8**: **Heuristic AI Agent** (Rule-based weights co-evolved via Genetic Algorithm)
- [x] **Phase 9**: **Monte Carlo AI Agent** (Imperfect Information MCTS with guided rollouts)
- [ ] **Phase 10**: **Neural AI Agent** (Deep learning observation tensors)
- [x] **Phase 11**: **Reinforcement Learning Pipeline** (Model-free self-play REINFORCE on CPU)
- [ ] **Phase 12**: **Optimization & Benchmarking** (1,000,000 game runs hot-path profiling)
