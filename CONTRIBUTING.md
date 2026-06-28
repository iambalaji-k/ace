# Contributing to the Ace Engine

Thank you for your interest in contributing to the Ace Engine project! To ensure our simulator remains professional-grade, perfectly deterministic, and easy to maintain, we follow strict development processes.

---

## 🛠️ Development Workflow

We follow a specification-driven, test-first development flow:
1. **Spec Alignment**: If proposing a new feature or change, first discuss it and update the corresponding specification chapter in `spec/`.
2. **Compliance Tests**: Write compliance test cases validating the spec change *before* writing code.
3. **Implementation**: Modify the data model (`engine/types.py`) and rules (`engine/rules.py`).
4. **Verification**: Verify that the new code is fully type-safe and format-compliant.

---

## 💅 Code Quality Standards

### 1. Code Formatting & Linting
We use **Ruff** for fast linting and formatting. Ensure your code satisfies all checks:
```bash
venv\Scripts\ruff check engine/
```

### 2. Static Typechecking
We use **Basedpyright** for strict typing checks. We aim for 0 typechecking warnings and errors:
```bash
venv\Scripts\basedpyright engine/
```

### 3. Unit Testing
We use **Pytest** for our test suite. All tests must pass, and new features must be covered:
```bash
venv\Scripts\python -m pytest
```

---

## 📐 Design Principles

- **Perfect Determinism**: Avoid any platform-dependent operations. Use the canonical `engine/prng.py` implementation for all random operations.
- **Immutability**: All states (`MatchState`, `RoundState`, `RuntimeState`) must be frozen dataclasses. State transitions must produce a new state.
- **Zero Floating-Point**: Use integer arithmetic (e.g., half-points for score metrics) to avoid rounding discrepancies across languages.
