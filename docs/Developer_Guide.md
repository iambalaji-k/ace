# Developer Guide for the Ace Engine Simulator

Welcome to the Developer Guide for the **Ace Engine**. This document provides an architectural overview, describes the codebase files, details the testing infrastructure, and outlines the roadmap for future enhancements.

---

## 1. Directory Structure

```text
D:\Vibe Coding\ace\
├── engine/                   # Core engine simulator source code
│   ├── __init__.py
│   ├── card.py               # Card encoding, suit/rank helpers, sort logic
│   ├── deck.py               # Reserved Ace allocation and dealing skip rotations
│   ├── events.py             # Event creation and Public/Player/Observer projections
│   ├── invariants.py         # Enforcers of INV-001 through INV-012
│   ├── prng.py               # PCG-XSH-RR-64/32 PRNG and Fisher-Yates shuffle
│   ├── rules.py              # Game transitions, legal action generator, main game loop
│   └── types.py              # Frozen MatchState, RoundState, RuntimeState, Actions
├── tests/                    # Test suites
│   ├── test_foundations.py   # Card, PRNG, and deal tests
│   └── test_rules.py         # Transition, steal, illegal card, and full match tests
├── docs/                     # Living design and developer documentation
│   └── Developer_Guide.md    # This document
└── venv/                     # Python virtual environment
```

---

## 2. Core Modules Architecture

### 2.1 Card Utilities (`engine/card.py`)
Cards are encoded as canonical integers `0-51`.
- **Spade ♠ (0)**: `0–12` (A♠=0 to 2♠=12)
- **Club ♣ (1)**: `13–25` (A♣=13 to 2♣=25)
- **Heart ♥ (2)**: `26–38` (A♥=26 to 2♥=38)
- **Diamond ♦ (3)**: `39–51` (A♦=39 to 2♦=51)

*Complexity*: `sort_cards(hands)` operates in $O(H \log H)$ (where $H \le 17$ is hand size), and suit/rank conversion is $O(1)$.

### 2.2 PRNG and Shuffle (`engine/prng.py`)
Implements the PCG-XSH-RR-64/32 pseudo-random number generator using multiplier `6364136223846793005` and increment `1442695040888963407`. Wrapping is handled by explicit 64-bit masking.
- `deterministic_shuffle(deck, seed)` uses the Durstenfeld variant of the Fisher-Yates algorithm, iterating backward from `n-1` down to `1`.

*Complexity*: Shuffle is $O(52)$, Seed derivation is $O(R)$ (where $R$ is current round number).

### 2.3 Deck dealing and skip rotations (`engine/deck.py`)
Handles reserved ace allocation and deals remaining cards clockwise, skipping the recipient for `N` rotations (where `N` is the reserved ace count).

*Complexity*: Allocation is $O(1)$, Dealing is $O(52)$.

### 2.4 State Model (`engine/types.py`)
Uses `@dataclass(frozen=True)` for immutability. Splitting the states keeps execution clean:
- `MatchState`: Persistent match parameters and player records.
- `RoundState`: Current round cards, hands, trick history, active/inactive statuses.
- `RuntimeState`: Transient game phase, active player turn, pending legal moves, action sequences.
- `EngineState`: Container holding the above states. Includes `get_state_hash()` to compute a deterministic SHA-256 state hash.

### 2.5 Invariant Checks (`engine/invariants.py`)
Implements strict validation of all invariants `INV-001` through `INV-012` on the current state.
In accordance with Option C (hybrid error handling):
- Player action validation checks for illegal moves (returns error with retry).
- Invariant validation panics with a full state dump if internal state corruption is detected.

### 2.6 Transitions (`engine/rules.py`)
Houses the main rules machine. Implements:
- `create_match()`
- `advance()` (round starting and card dealing)
- `apply_action()` (Decline/Steal/Play Card transitions, trick pickup, round resolution, point allocation)

---

## 3. Running Tests and Checks

To run the full test suite:
```bash
venv\Scripts\python -m pytest
```

To run the linter check:
```bash
venv\Scripts\ruff check engine/
```

To run the typechecker:
```bash
venv\Scripts\basedpyright engine/
```

---

## 4. Future Enhancements

1. **AI & RL Interface**:
   - Provide a gym-like step wrapper that converts `EngineState` to neural network observation arrays.
   - Design policy and value head APIs suitable for reinforcement learning.
2. **Replay & Spectating System**:
   - Save full game event records to JSON/msgpack files.
   - Implement step-by-step playback with Public and Player projections.
3. **Tournament Runner**:
   - Spawn multi-threaded match simulators running independent match states (completely thread-safe due to immutable transitions).
