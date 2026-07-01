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
ace/
├── docs/                             # Developer documentation, guides, and specifications
│   ├── ARCHITECTURE.md               # Normative architecture definitions
│   ├── AETS_Project_Bootstrap.md     # Original bootstrap plan
│   ├── Ace_Heuristic_Catalogue_v1.0.md # Detailed rule-based logic catalogs
│   ├── example_game_walkthrough.md    # Multi-round narrated match walkthrough
│   └── status.md                     # Active project status registry
├── spec/                             # Frozen normative specification chapters (Compliance specs)
├── engine/                           # Core Game Engine codebase (Rules only, no AI brain code)
│   ├── rules.py                      # Engine transition state machine and validation
│   ├── types.py                      # Immutable state models, action classes, game events
│   ├── card.py                       # Card representations, suit/rank helpers
│   ├── deck.py                       # Reserved ace allocation and deal skip rotations
│   ├── events.py                     # Public, private, and observer event projections
│   ├── invariants.py                 # Enforcer checking INV-001 to INV-012 invariants
│   ├── tournament.py                 # Multi-threaded tournament match runner
│   ├── replay.py                     # Replay log saving/loading schemas
│   ├── replay_player.py              # Interactive playback player (step, undo, branch fork)
│   └── prng.py                       # PCG-XSH-RR-64/32 Pseudo-Random Number Generator
├── agents/                           # Decision-making agents separated by type and version
│   ├── random/                       # Fallback agents (BaseAgent and RandomAgent)
│   ├── heuristic/                    # Rule-based bots
│   │   ├── v1/                       # Legacy hardcoded rules (heuristic_agent.py + CardTracker)
│   │   └── v2/                       # Parameterized rules (heuristic_agent_v2.py + evolved JSONs)
│   ├── mcts/                         # Search-based bots
│   │   └── v1/                       # Information Set MCTS (mcts_agent.py)
│   └── rl/                           # Neural network bots
│       ├── v1/                       # Legacy 4-player PPO model (encoder/model/rl_agent.py)
│       └── v2/                       # Variable player count model (encoder_v2/model_v2/rl_agent_v2.py)
├── checkpoints/                      # Unified directory for saved model weights and resume states
│   ├── rl_champion.pt                # Version 1.0 Champion weights
│   ├── rl_champion_v2.pt             # Version 2.0 Imitation-Learning Baseline weights
│   ├── rl_champion_v3.pt             # Version 3.0 Gated Champion weights (created during self-play)
│   └── train_resume_v3.pt            # Epoch-by-epoch training recovery files
├── tests/                            # Automated testing suite
├── logs/                             # Target output directory for logs and standard outputs
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
pip install pytest ruff basedpyright torch numpy
```

### 2. Running Automated Tests
Run the entire suite of 38 tests (including unit, integration, replay, and tournament tests):

```bash
# Set PYTHONPATH to root on CMD (Windows)
set PYTHONPATH=. && venv\Scripts\pytest

# Set PYTHONPATH on PowerShell
$env:PYTHONPATH="." ; venv/Scripts/pytest

# Set PYTHONPATH on Linux/macOS
PYTHONPATH=. pytest
```

### 3. Running Code Quality Linters & Typecheckers
Ensure code modifications comply with repository formatting and type safety:

```bash
# Run Ruff lint check
python -m ruff check engine/ tests/ scripts/ agents/

# Run Basedpyright static typechecker
python -m basedpyright engine/ tests/ scripts/ agents/
```

---

## 🕹️ Match Simulation & CLI Tools

### A. Watch Automated Bots Play (`run_bot_match.py`)
Watch 4 random agents play an automated game step-by-step with a custom turn delay. You will see hands dealt, steal attempts, card plays, suit breaks, and a final rankings scoreboard:

```bash
python scripts/utils/run_bot_match.py --players 4 --rounds 2 --seed 42 --delay 0.3
```

### B. Manually Rig & Simulate a Match (`run_manual_demo.py`)
Run an interactive CLI simulation where you can set custom card hands for specific players and test rules (like forcing a suit break) step-by-step:

```bash
python scripts/utils/run_manual_demo.py
```

---

## 🤖 AI Agents (Heuristics, MCTS, and Self-Learning)

The Ace Engine features four distinct tiers of artificial intelligence, transitioning from structured rule-based systems to deep self-play reinforcement learning:

### 1. Evolved Heuristic Agent (`HeuristicAgentV2`)
* **Core Logic**: An expert-rule system containing **55 distinct card-play and steal heuristics** categorized by match phase (Opening, Middle, Endgame) and hand configurations.
* **Optimization (Genetic Algorithm)**: Co-evolves both the 55 heuristic weights and **11 strategic parameter thresholds** (e.g. opponent void risk limits, suit-hoarding boundaries) to maximize match placement outcomes.
* **Training Command**:
  ```bash
  python scripts/training/train_genetic_weights.py
  ```
  *(Optimizes weights and outputs updated parameters to `agents/heuristic/v2/heuristic_v2_weights.json`)*.

### 2. Monte Carlo Tree Search Agent (`MCTSAgent`)
* **Core Logic**: Implements **Information Set MCTS (ISMCTS)** to play under imperfect information.
* **Hand Determinization**: Features a backtracking constraint satisfaction search that generates random opponent hands matching all `CardTracker` void suits, known cards, and current trick cards.
* **Heuristic-Guided Rollouts**: Utilizes our evolved `HeuristicAgentV2` to make playout choices, producing high-fidelity rollouts and extremely fast search convergence on CPU.
* **Evaluation Command**:
  ```bash
  python scripts/evaluation/evaluate_mcts_agent.py
  ```

### 3. Self-Learning RL Agents
The repository supports three iterations of Reinforcement Learning:
* **RL Agent 1.0 (Legacy)**: Flat 343-feature state representation. Hardcoded strictly to 4-player matches. Trained using REINFORCE. (*Weights: `checkpoints/rl_champion.pt`*)
* **RL Agent 2.0 (Behavior Cloning)**: Dynamic 459-feature state representation supporting variable player counts (3–6 players). Bootstrapped via behavior cloning from MCTS and Heuristic V2. (*Weights: `checkpoints/rl_champion_v2.pt`*)
* **RL Agent 3.0 (SPRS v3 Self-Play)**: Advanced PPO pipeline featuring **State-Potential Reward System (SPRS v3)**. It replaces event-based rewards with distance-preserving hand potentials and knowledge ratio metrics to eliminate reward hacking.
* **RL v3.0 Training Command**:
  ```bash
  # Train the RL v3.0 agent (resumes automatically from checkpoints/train_resume_v3.pt)
  python scripts/training/train_self_play_v3.py
  ```

---

## 🎮 Play Against the Agents (CLI Interface)

You can play interactive card matches directly against these agents in your terminal. You can choose the agent (Heuristic V1/V2, MCTS, RL V1/V2), set the number of matches/rounds, and customize your seat position:

```bash
# Start the interactive CLI play suite
python scripts/play/play_interactive.py
```

---

## 📊 Universal Benchmark Tournament
Evaluate the playing strength of all agents (RL Agent V2, Heuristic V1, Heuristic V2, RL Agent V1, MCTS Agent) in a unified, seat-balanced tournament of 100 matches:

```bash
python scripts/evaluation/benchmark_all_agents.py
```

---

## 🗺️ Project Roadmap & Status

- [x] **Phase 0–7**: Core Rules Engine, state-machine transitions, event projection, and compliance tests.
- [x] **Phase 8**: **Heuristic AI Agent** (Rule-based weights co-evolved via Genetic Algorithm).
- [x] **Phase 9**: **Monte Carlo AI Agent** (Imperfect Information MCTS with guided rollouts).
- [x] **Phase 10**: **Neural AI Agent** (Imitation Learning Behavior Cloning baseline).
- [x] **Phase 11**: **Reinforcement Learning Self-Play** (PPO self-play pipeline on CPU).
- [x] **Phase 12**: **SPRS v3 Reward Redesign** (Fully integrated State-Potential Reward System to solve reward-hacking and stabilize convergence).
- [x] **Phase 13**: **Repository Reorganization** (Decoupled agents by type and version, separated weights into `checkpoints/`, and moved docs to `docs/`).
- [ ] **Phase 14**: **Long-running training runs** (Train RL Agent 3.0 for 500+ epochs under SPRS v3 to surpass human heuristic and MCTS baselines).
