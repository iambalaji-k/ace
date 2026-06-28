# engine/prng.py
"""PRNG implementation for the Ace Engine.

Implements PCG-XSH-RR-64/32 and the deterministic shuffle algorithms.
All arithmetic uses 64-bit and 32-bit unsigned integers with explicit masking.
"""

from typing import List, Tuple

# PCG-XSH-RR-64/32 Constants
MULTIPLIER = 6364136223846793005
INCREMENT = 1442695040888963407

MASK64 = 0xFFFFFFFFFFFFFFFF
MASK32 = 0xFFFFFFFF


def pcg_step(state: int) -> int:
    """Advance the LCG state of the generator by one step."""
    return (state * MULTIPLIER + INCREMENT) & MASK64


def pcg_output(state: int) -> int:
    """Apply the XSH-RR output transformation to the 64-bit state to get a 32-bit random number."""
    state = state & MASK64
    xorshifted = (((state >> 18) ^ state) >> 27) & MASK32
    rot = (state >> 59) & 31  # rot is 5 bits (0-31)
    
    # Perform rotation on 32-bit value
    right_shift = (xorshifted >> rot) & MASK32
    left_shift = (xorshifted << ((-rot) & 31)) & MASK32
    return (right_shift | left_shift) & MASK32


def pcg_next(state: int) -> Tuple[int, int]:
    """Advance the state and return the next state and output."""
    out = pcg_output(state)
    next_state = pcg_step(state)
    return next_state, out


def pcg_seed(seed_value: int) -> int:
    """Initialize the generator with a seed value, returning the initial state."""
    seed_value = seed_value & MASK64
    state = 0
    state = pcg_step(state)
    state = (state + seed_value) & MASK64
    state = pcg_step(state)
    return state


def derive_round_seed(match_seed: int, round_number: int) -> int:
    """Derive the round seed from the match seed by stepping the PCG generator."""
    if round_number < 1:
        raise ValueError(f"Round number must be >= 1, got {round_number}")
    state = pcg_seed(match_seed)
    for _ in range(round_number):
        state, _ = pcg_next(state)
    return state


def deterministic_shuffle(deck: List[int], round_seed: int) -> List[int]:
    """Shuffle a list in-place using the Fisher-Yates (Durstenfeld) algorithm driven by the round PCG seed.

    Returns the shuffled list.
    """
    shuffled = list(deck)
    state = pcg_seed(round_seed)
    n = len(shuffled)
    for i in range(n - 1, 0, -1):
        state, rand = pcg_next(state)
        j = rand % (i + 1)
        # Swap element at i and j
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled
