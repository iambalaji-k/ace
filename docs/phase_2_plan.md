# Phase 2 — Compliance Test Specification Plan

This document outlines the detailed plan to design and specify the language-agnostic **Compliance Test Catalog** for the Ace Engine Simulator.

---

## 1. Compliance Testing Methodology

To verify that any simulator implementation behaves exactly as specified, we will construct a formal, language-agnostic test suite containing test specifications. 

Each test case MUST be defined in a structured format (JSON/YAML) and include:
- **Test Identifier**: Unique code in the format `TEST-XXXX`.
- **Rule/Invariant IDs**: The specific specification chapters and invariants being tested.
- **Initial Configuration**: Players, round numbers, and match seed.
- **Preconditions**: Expected initial hands or state snapshots before actions are applied.
- **Actions Sequence**: Chronological list of player choices (`Steal`, `Decline`, `PlayCard`).
- **Expected Events**: List of events emitted by the engine.
- **Expected Final State**: Final hands, points, and seating status.

---

## 2. Test Catalog Coverage Matrix

The catalog will target 8 main test categories to guarantee full coverage of AETS Chapters 1–11:

### Category A: Foundations & PRNG (TEST-0001 to TEST-0010)
- Verify PCG-XSH-RR-64/32 output vectors for known seeds.
- Verify deterministic Fisher-Yates shuffles for a standard 52-card deck.

### Category B: Deal & Skip Rotations (TEST-0011 to TEST-0020)
- Standard deal with 3, 4, and 6 players.
- Reserved ace dealing with 1, 2, and 4 skip rotations.
- Verification of card counts and canonical hand sorting after dealing.

### Category C: Steal Phase Mechanics (TEST-0021 to TEST-0030)
- Single steal execution, victim cards merging and canonical sort order.
- Immediate active left target rotation after successive steals.
- Victim transitioning to Inactive and declared a Round Winner.
- Auto-loss trigger when the lead player steals all cards.

### Category D: Play Phase & Suit Validation (TEST-0031 to TEST-0040)
- Play card, Lead Suit establishment.
- Follow suit validation.
- Rejection of illegal moves (breaking suit when holding lead-suit cards).

### Category E: Trick Resolution (TEST-0041 to TEST-0050)
- Successful trick: played cards moved to discard pile, trick winner leads next.
- Interrupted trick: collector gathers all cards, collector leads next.
- Inactive trick-winner lead transfer: next-highest active player leads (Edge Case D).

### Category F: Lifecycle & Scoring (TEST-0051 to TEST-0060)
- Surrendered hands leading to Inactive status.
- Round ending with exactly 1 active player (Round Loser).
- Round ending with 0 active players (Draw).
- Half-points updates and consecutive loss counter accumulation.
- Match completion, final ranking, and tiebreakers.

### Category G: Invariants Fuzzing (TEST-0061 to TEST-0070)
- Fuzzing games with random seeds to ensure `INV-001` through `INV-012` hold true at every single transition.

---

## 3. Automation Harness Design

To programmatically execute these tests against our simulator without rewriting the rules:
1. **JSON Test Loader**: Write a parser in `tests/test_compliance.py` that reads the JSON test files.
2. **Setup Engine**: Call `create_match` with the configured seed and players.
3. **Execute Actions**: Iterate through the action sequence, calling `apply_action()` for each step.
4. **Assert Equality**: Compare the resulting `EngineState` and generated `Event` lists with the test case assertions.

---

## 4. Immediate Delivery Plan

| Step | Action | Deliverable |
| :--- | :--- | :--- |
| **Step 1** | Draft Compliance Schema | Create `spec/compliance_schema.json` defining the schema for tests. |
| **Step 2** | Specify Catalog | Create `spec/compliance_test_catalog.md` detailing the test scenarios `TEST-0001` to `TEST-0050`. |
| **Step 3** | Write JSON Tests | Save the test definitions in JSON format under `tests/compliance/` so they are ready for the automated test harness. |
