# AETS Chapter 1: Foundations

## 1.1 Normative Language
The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

## 1.2 Card Representation
Each card in a standard 52-card deck SHALL be encoded as a unique integer in the range `0` through `51`.
- **Suit Extraction**: The card's suit MUST be derived via integer division: `suit(card) = card // 13`.
  - `0`: Spade ♠
  - `1`: Club ♣
  - `2`: Heart ♥
  - `3`: Diamond ♦
- **Rank Extraction**: The card's rank MUST be derived via modulo: `rank(card) = card % 13`.
  - `0`: Ace (A)
  - `1`: King (K)
  - ...
  - `12`: Two (2)
- **Canonical Sorting**: Card lists SHALL be sorted in ascending order of their integer ID. This sorting naturally prioritizes Spade > Club > Heart > Diamond, and within each suit, sorts descending from Ace down to Two.

## 1.3 Random Number Generation (PRNG)
To guarantee cross-platform and cross-language determinism, the engine MUST use the **PCG-XSH-RR-64/32** pseudo-random number generator.
- **Internal State**: 64-bit unsigned integer.
- **Output Size**: 32-bit unsigned integer.
- **Multiplier**: `6364136223846793005` (uint64).
- **Increment**: `1442695040888963407` (uint64).
- **Arithmetic**: All arithmetic operations SHALL wrap around at 64 bits (modulo $2^{64}$).

### 1.3.1 PCG Seed Operation
To initialize the generator state with seed $S$ (uint64):
1. Set internal state to `0`.
2. Advance state: `state = state * Multiplier + Increment` (mod $2^{64}$).
3. Add seed: `state = state + S` (mod $2^{64}$).
4. Advance state: `state = state * Multiplier + Increment` (mod $2^{64}$).

### 1.3.2 PCG Next Operation
To generate a 32-bit random number:
1. Extract output:
   `xorshifted = uint32(((state >> 18) XOR state) >> 27)`
   `rot = uint32(state >> 59)`
   `output = (xorshifted >> rot) OR (xorshifted << ((0 - rot) AND 31))`
2. Step LCG state:
   `state = state * Multiplier + Increment` (mod $2^{64}$)
3. Return `(state, output)`.

## 1.4 Shuffling
 Shuffling a list of cards in-place MUST use the Durstenfeld variant of the Fisher-Yates shuffle:
1. Initialize the PRNG state using the round's seed.
2. Iterate index $i$ backward from `n-1` down to `1`:
   - Generate $R$: `state, R = pcg_next(state)`.
   - Calculate $j$: `j = R mod (i + 1)`.
   - Swap elements at index $i$ and $j$.
