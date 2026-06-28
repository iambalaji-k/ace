# Ace Engine Project Roadmap

This document outlines the engineering phases for the Ace Engine project.

---

## 🗺️ Phase Timeline

### Phase 0 · Repository Setup & Core Documentation
- [x] Initial folder structure (`spec/`, `engine/`, `tests/`, etc.)
- [x] Establish linting, typechecking, and unit testing frameworks
- [x] Create project files (`README.md`, `ARCHITECTURE.md`, `ROADMAP.md`)

### Phase 1 · Specification chapters 1–11
- [x] Consolidated spec drafted
- [x] Split and freeze chapter specifications in `spec/`

### Phase 2 · Compliance Test Catalog
- [ ] Write `spec/compliance_test_catalog.md` defining tests `TEST-0001` through `TEST-0100`
- [ ] Define Initial State, Preconditions, Actions, Expected Final State, and Rule IDs validated for each case

### Phase 3 · Data Model
- [x] Implement immutable types in `engine/types.py`
- [x] Support deterministic state hashing

### Phase 4 · Rules Engine & Headless Simulator
- [x] Implement `get_legal_actions()` and `apply_action()`
- [x] Enforce `INV-001` through `INV-012` at every turn
- [x] Run headless match flows deterministically

### Phase 5 · Replay System
- [ ] Save full match events to replay files
- [ ] Implement deterministic replay player with step-by-step playback, undo, and branching

### Phase 6 · Tournament Runner
- [ ] Execute thousands of parallel matches
- [ ] Aggregate point statistics, loss counts, and export to CSV

### Phase 7 · AI Agents (Random & Heuristic)
- [ ] Build `RandomAgent` using the public Player interface
- [ ] Build `HeuristicAgent` with configurable scoring weights

### Phase 8 · Advanced AI & Reinforcement Learning
- [ ] Implement gym-like step wrapper for the engine state
- [ ] Train self-play models (neural networks)

### Phase 9 · Optimization
- [ ] Profile engine under 1,000,000 match workloads
- [ ] Identify bottlenecks and optimize critical execution paths
