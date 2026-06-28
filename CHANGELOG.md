# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-06-28

### Added
- **Core Repository Setup**: Created folder tree (`engine/`, `tests/`, `docs/`, `spec/`, `training/`, `replays/`, `benchmarks/`, `scripts/`).
- **Foundational Documentation**: Added `README.md`, `CONTRIBUTING.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `CHANGELOG.md`, and `docs/status.md`.
- **Card Encoding & Shuffling**: Implemented canonical card codes (0-51), suit/rank utilities, Fisher-Yates shuffle, and PCG-XSH-RR-64/32 PRNG.
- **Reserved Aces skipping deal**: Implemented clockwise skip-rotation dealing for players holding reserved aces.
- **Immutable State Model**: Created `MatchState`, `RoundState`, `RuntimeState`, and `EngineState` as frozen dataclasses with deterministic SHA-256 state hashing.
- **Rules Machine**: Built `get_legal_actions()`, `apply_action()`, and `advance()` transitions supporting Stealing, card plays, trick pickups, and scoring.
- **Invariant Checker**: Implemented enforcement for `INV-001` through `INV-012` validating states after every play.
- **Pytest Suite**: Added 11 compliance-grade tests executing complete multi-round games end-to-end.
- **Linter & Typechecks**: Configured Ruff and Basedpyright with 100% clean check status.
