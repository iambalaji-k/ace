# Ace Engine Architecture Guide

This document describes the design patterns, layer breakdown, and architectural decisions of the Ace Engine.

---

## 1. System Decomposition

The project separates the **Simulator Core** (pure game logic) from the **Game Engine** wrapper (AI models, visual UI, tournaments).

```text
  +---------------------------------------------------------+
  |                   Game Engine Wrapper                   |
  |  - Tournament Runner    - Gym-like RL Training Loop     |
  |  - Replay Player        - Network Sync & Visual Clients |
  +---------------------------------------------------------+
                               |
                               | (API: create_match / apply_action)
                               v
  +---------------------------------------------------------+
  |                     Simulator Core                      |
  |  - engine/rules.py      - engine/invariants.py          |
  |  - engine/types.py      - engine/prng.py                |
  +---------------------------------------------------------+
```

### 1.1 The Simulator Core (`engine/`)
A pure, side-effect-free library. It implements:
- Card representations & sorting rules.
- Reserved ace Dealing and skip rotations.
- Match, Round, and Trick state transitions.
- Strict invariant checks.

### 1.2 The Engine Wrapper (To be built)
The orchestrator that links the simulator to interfaces (reinforcement learning loops, visual displays, tournament schedules).

---

## 2. State Model Separation

To keep persistent state separate from transient execution state (Refinement 4), we organize `EngineState` into three distinct sub-components:

```text
EngineState
├── MatchState (Match-level persistent metrics, loss counters)
├── RoundState (Round-level hands, tricks, and discard piles)
└── RuntimeState (Transient current phase, legal moves, PRNG state)
```

- **Immutability**: All states are `@dataclass(frozen=True)`. Applying an action produces a completely new `EngineState` object.
- **State Hashing**: After every transition, the engine generates a deterministic SHA-256 state hash. This makes debugging cross-language implementations trivial.

---

## 3. Information Model & Projections

The simulator models three levels of visibility (Refinement 5):
1. **Observer Projection**: Full visibility of all hands, deck seeds, and private steals.
2. **Player Projection**: Filters state to show the player's own hand and history, while masking other players' private hands.
3. **Public Projection**: Completely strips out private cards, showing only public event histories and card counts.
