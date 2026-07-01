# Ace Engine Simulator — Project Status Register

> **Current Project Phase**: Phase 8 (Heuristic AI Agent) — **100% Functional**
> **Next Target Phase**: Phase 9 (Monte Carlo AI Agent)

---

## 1. Phase Status Checklist (0-12)

| Phase | Description | Status | Details |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Repository Setup | 🟢 **Done** | Core directories and core documentation files created. |
| **Phase 1** | Specification | 🟢 **Done** | Chapters 1 to 11 drafted, split, and frozen under normative RFC guidelines. |
| **Phase 2** | Compliance Tests | 🟢 **Done** | Compliance test schema, test case catalog, and compliance test harness implemented. |
| **Phase 3** | Data Model | 🟢 **Done** | Immutable types and data model implemented in `engine/types.py`. |
| **Phase 4** | Rules Engine & Simulator | 🟢 **Done** | Rules engine, state transitions, legal actions, and simulator invariants implemented in `engine/rules.py`. |
| **Phase 5** | Replay System | 🟢 **Done** | Action log serializer, replay manager, and replay playback player implemented. |
| **Phase 6** | Tournament Runner | 🟢 **Done** | Parallel ThreadPoolExecutor tournament runner, stats manager, and CSV exporter implemented. |
| **Phase 7** | Random AI Agent | 🟢 **Done** | Uniform RandomAgent baseline and visualizer scoreboard match visualizer implemented. |
| **Phase 8** | Heuristic AI Agent | 🟢 **Done** | CardTracker probability/void estimator, HeuristicAgent registry of 86 heuristics, and tie-breaking implemented. |
| **Phase 9** | Monte Carlo AI Agent | 🔴 **Not Started** | Pending. |
| **Phase 10** | Neural AI Agent | 🔴 **Not Started** | Pending. |
| **Phase 11** | Reinforcement Learning | 🔴 **Not Started** | Pending. |
| **Phase 12** | Optimization & Benchmarks | 🔴 **Not Started** | Pending. |

---

## 2. File Inventory

| Path | Purpose | Lines of Code |
| :--- | :--- | :--- |
| [engine/card.py](file:///D:/Vibe%20Coding/ace/engine/card.py) | Card representations, suits, ranks, and sorting logic | ~85 |
| [engine/prng.py](file:///D:/Vibe%20Coding/ace/engine/prng.py) | PCG-XSH-RR-64/32 implementation and Fisher-Yates shuffle | ~70 |
| [engine/deck.py](file:///D:/Vibe%20Coding/ace/engine/deck.py) | Reserved ace allocation and deal skip rotations | ~60 |
| [engine/types.py](file:///D:/Vibe%20Coding/ace/engine/types.py) | Immutable dataclasses, game phases, actions, events | ~223 |
| [engine/invariants.py](file:///D:/Vibe%20Coding/ace/engine/invariants.py) | Enforces INV-001 through INV-012 at every turn | ~160 |
| [engine/rules.py](file:///D:/Vibe%20Coding/ace/engine/rules.py) | Transition machine (Steal, PlayCard, resolution, scoring) | ~886 |
| [engine/events.py](file:///D:/Vibe%20Coding/ace/engine/events.py) | Public, Player, and Observer event projections | ~50 |
| [tests/test_foundations.py](file:///D:/Vibe%20Coding/ace/tests/test_foundations.py) | Unit tests for cards, PRNG, and skip dealings | ~70 |
| [tests/test_rules.py](file:///D:/Vibe%20Coding/ace/tests/test_rules.py) | Integration tests and 2-round headless simulator flow | ~200 |
| [docs/Developer_Guide.md](file:///D:/Vibe Coding/ace/docs/Developer_Guide.md) | Dev setup guide, complexity targets, and codebase description | ~80 |
| [engine/replay.py](file:///D:/Vibe Coding/ace/engine/replay.py) | Action log serialization schemas and file utilities | ~60 |
| [engine/replay_player.py](file:///D:/Vibe Coding/ace/engine/replay_player.py) | Playback controls, history logs, undos/redos, forks | ~120 |
| [tests/test_replays.py](file:///D:/Vibe Coding/ace/tests/test_replays.py) | Verification checks for serialization determinism, jumps, branching | ~160 |
| [tests/test_walkthrough_compliance.py](file:///D:/Vibe Coding/ace/tests/test_walkthrough_compliance.py) | Game walkthrough transitions validation suite | ~240 |
| [agents/random/agent.py](file:///D:/Vibe Coding/ace/agents/random/agent.py) | Abstract base agent and RandomAgent baseline | ~40 |
| [engine/tournament.py](file:///D:/Vibe Coding/ace/engine/tournament.py) | ThreadPoolExecutor parallel match simulator runner | ~220 |
| [tests/test_tournament.py](file:///D:/Vibe Coding/ace/tests/test_tournament.py) | Unit tests verifying parallel execution, determinism, and CSV formats | ~80 |
| [scripts/run_bot_match.py](file:///D:/Vibe Coding/ace/scripts/run_bot_match.py) | Headless visualizer demo script | ~135 |
| [tests/test_agents.py](file:///D:/Vibe Coding/ace/tests/test_agents.py) | Unit tests validating RandomAgent compliance and determinism | ~50 |
| [agents/heuristic/v1/heuristic_agent.py](file:///D:/Vibe Coding/ace/agents/heuristic/v1/heuristic_agent.py) | CardTracker void/known estimator, HeuristicAgent registry, scoring | ~1130 |
| [tests/test_heuristics.py](file:///D:/Vibe Coding/ace/tests/test_heuristics.py) | Unit tests for HeuristicAgent, CardTracker, and tie-breakers | ~50 |
| [scripts/run_heuristic_tournament.py](file:///D:/Vibe Coding/ace/scripts/run_heuristic_tournament.py) | 1,000 round Heuristic vs. Random benchmark script | ~75 |

---

## 3. Test & Linter Coverage Summary

* **Pytest**: **28/28 tests passing** (all rules, compliance walkthrough, card counting, and heuristics pass).
* **Ruff**: **100% Clean** (all imports, variables, and type comparisons resolved).
* **Basedpyright**: **0 errors** (strict type safety achieved).
