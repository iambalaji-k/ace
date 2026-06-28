# Ace Engine

A professional-grade, specification-driven, deterministic card game engine in Python. Designed for reinforcement learning self-play, tournament evaluation, and replay-ability.

---

## 🚀 Vision

The Ace Engine is designed to support:
- **Human vs Human**, **Human vs AI**, and **AI vs AI** matches.
- **Self-play** and **Reinforcement Learning** training.
- **Tournament benchmarking** with Elo ratings.
- **Perfect determinism** across platforms and programming languages.
- **Immutable state transitions** and **structured event logs** for complete replay fidelity.

---

## 📁 Repository Structure

```text
ace-engine/
├── docs/             # Living design, status, and developer guides
├── spec/             # AETS Chapter 1–11 protocol specifications (frozen)
├── engine/           # Core simulator engine package
├── tests/            # Automated compliance and rule verification suites
├── training/         # Reinforcement learning and agent training code
├── replays/          # Saved match replay records
├── benchmarks/       # Profiling and execution speed benchmarks
├── scripts/          # Developer utility scripts
└── venv/             # Virtual environment
```

---

## 🛠️ Getting Started

### 1. Requirements
- Python 3.10+
- Virtual environment (venv)

### 2. Run Tests
Verify the simulator against compliance tests:
```bash
venv\Scripts\python -m pytest
```

### 3. Run Linter
```bash
venv\Scripts\ruff check engine/
```

### 4. Run Typechecker
```bash
venv\Scripts\basedpyright engine/
```

---

## 🗺️ Project Roadmap

- [x] **Phase 0**: Repository Setup & Core Documentation
- [x] **Phase 1**: Specification Chapters 1–11
- [ ] **Phase 2**: Compliance Test Catalog Specification
- [x] **Phase 3**: Immutable Data Model
- [x] **Phase 4**: Rules Engine & Headless Simulator
- [ ] **Phase 5**: Appendix/Replay System Implementation
- [ ] **Phase 6**: Tournament Runner
- [ ] **Phase 7**: AI Agents (Random & Heuristic)
- [ ] **Phase 8**: Optimization & Benchmarks
