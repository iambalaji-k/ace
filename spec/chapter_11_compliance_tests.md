# AETS Chapter 11: Compliance Tests

## 11.1 Conformance Target
To be certified as a compliant **Ace Engine Simulator**, the implementation MUST pass the complete automated compliance test suite.

## 11.2 Categories of Tests
The compliance suite MUST contain tests across the following 5 categories:

1. **Scenario Tests**: Runs pre-scripted action sequences from a fixed seed and verifies intermediate state transitions, invariants, and events.
2. **State-Point Tests**: Constructs a custom mock state and asserts that `get_legal_actions()` returns the exact list of allowed moves.
3. **Invariant Fuzz Tests**: Simulates random valid moves for thousands of tricks on a large number of random seeds. Emits a failure if any of the 12 invariants are violated at any step.
4. **Golden File Tests**: Runs complete matches, records the event logs, and compares them byte-for-byte against reference logs to prevent regressions in determinism.
5. **PRNG Conformance Tests**: Verifies that the internal PRNG outputs match the reference test vectors given a specific seed.

## 11.3 Test Definition Schema
Compliance tests SHALL be defined in a language-agnostic format (e.g. JSON), containing:
- `test_id`: String identifier (e.g. `TEST-0001`).
- `config`: Match configurations.
- `scripted_actions`: Actions to apply.
- `assertions`: List of state checks (e.g. hand sizes, trick outcomes, events).
