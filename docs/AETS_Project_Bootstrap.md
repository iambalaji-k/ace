# Ace Engine Technical Specification (AETS)

## Project Bootstrap Document (Living Design Document)

This document is the bootstrap specification for the Ace Engine project.

## Vision

Build a deterministic, specification-driven engine supporting: - Human
vs Human - Human vs AI - AI vs AI - Self-play - Replay - Tournament
benchmarking - Reinforcement learning

## Engineering Order

1.  Specification
2.  Compliance Tests
3.  Engine
4.  Replay
5.  Tournament Runner
6.  AI
7.  UI

## Documentation Roadmap

000 Project Charter 100 Game Specification 200 Engine Technical
Specification 300 Engine API 400 Replay Specification 500 Tournament
Specification 600 AI Interface 700 Compliance Test Suite 800 Developer
Guide 900 Future Enhancements

## Specification Style

Use RFC keywords: - MUST - MUST NOT - SHALL - SHOULD - SHOULD NOT - MAY

Every item receives a permanent identifier: - RULE-x.y.z - INV-xxx -
EVENT-xxx - TEST-xxxx - DP-xxx

## Engine Philosophy

-   Deterministic
-   Immutable state transitions
-   Event-driven
-   Replayable
-   Platform independent
-   AI-agnostic

The engine only understands Agents.

## State Layers

Match State Round State Engine State Player View Observer View

Engine State = Match State + Round State.

## Match Rules

-   Match consists of fixed number of rounds.
-   One loser per round.
-   One or more winners per round.

## Confirmed Game Decisions

Players: 3-6

Deck: Standard 52 cards

Rank: A K Q J 10 9 8 7 6 5 4 3 2

Hand: Logical unordered set. UI sorted by suit (Spade, Club, Heart,
Diamond) and descending rank.

Initial dealing: - Shuffle - Clockwise - Uneven hands allowed

Consecutive losses: - Counter increments on loss - Resets on win

Reserved aces: 1=A♠ 2=A♠ A♣ 3=A♠ A♣ A♥ 4+=A♠ A♣ A♥ A♦

Reserved Ace Algorithm: 1. Build full deck 2. Remove reserved aces 3.
Shuffle remaining cards 4. Give reserved aces 5. Skip recipient during
deal equal to reserved count 6. Resume normal dealing

Gameplay: - Lead player may play any card. - Must follow suit if
possible. - Otherwise any card. - Failure to follow suit ends trick
immediately. - Remaining players do not play. - Highest lead suit card
picks up. - Pickup order equals play order. - Cards appended then UI
sorts.

Winning: - Winners determined only between rounds. - Empty hand during
interrupted trick must still pick up if required. - Round ends when one
active player remains.

Special Rule 1: - Only next lead player. - Before round starts. -
Optional. - May steal repeatedly from immediate active left. - Hidden
transfer. - Merge hands.

Information: Public: - Played cards - Card counts - Public history

Private: - Hands - Stolen cards

Observer: Everything.

Replay: Store: - Spec version - Engine version - Match seed - Round
seeds - Events

Support: - Replay - Undo - Redo - Jump - Branch

Compliance: - 200-500 deterministic tests - Validation after every
transition

Planned Chapters: 1 Foundations 2 Canonical Data Model 3 Match Lifecycle
4 Round Lifecycle 5 Rules Engine 6 State Machine 7 Event System 8
Validation 9 Replay 10 AI Interface 11 Compliance Tests

Suggested Python Stack: - Python - dataclasses - pytest - Ruff -
basedpyright - uv

Repository: docs/ engine/ tests/ training/ replays/ tournaments/ tools/
