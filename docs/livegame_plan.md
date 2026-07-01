# Live Game Plan — Human-vs-Engine Card Game Interface

This document details the design and implementation strategy for a **Live Game** mode
that allows the Ace Engine to play as one human player in a physically-shuffled card game
against other human players, for the purpose of evaluating AI agent performance in a
transparent, fair setting.

---

## 1. Requirements

### 1.1 Use Case

The user has built a card game engine and wants to **demonstrate its AI capability**
against real human players. The humans shuffle and deal physically (so the deal is
trusted), and the engine plays as one seated player. Other human players know the
engine is playing — this is a transparent evaluation, not cheating.

### 1.2 Constraints

| Constraint | Reason |
|-----------|--------|
| Physical shuffle by humans | Other players won't trust an engine-dealt deck |
| User inputs ONLY their own hand | User cannot see other players' hands |
| Engine plays as the user | Agent auto-selects actions for the user's seat |
| User relays other players' moves | User inputs what other players play (card name/ID) |
| Other players' hands are unknown | Imperfect information — same as what the agent sees |
| No engine code modifications | Wrap the existing engine, don't fork it |

### 1.3 User Experience Goals

- **Transparency**: Other players know the engine is playing — no deception
- **Simplicity**: User types what other players play; agent auto-plays for them
- **Fairness**: Agent sees only imperfect information (your hand + played cards)
- **Flexibility**: Configurable player count (3-6), seat, agent type, rounds
- **Evaluation**: Engine performance is measured against real human decisions

---

## 2. Architecture

### 2.1 Core Principle: No Engine Modification

The existing `AceEngine` is **unchanged**. A thin wrapper class (`LiveGameSession`)
handles the live game scenario by pre-processing non-user plays before passing them
to the engine.

### 2.2 Why a Wrapper Works

The `AceEngine.apply_action()` expects the played card to be in the player's hand.
In the live game, other players' hands are unknown. The wrapper solves this:

```
Non-user plays card X:
  1. Check if X is in player's virtual hand (from random deal)
  2. If YES -> call apply_action() normally
  3. If NO  -> add X to their hand, THEN call apply_action()
     (engine removes X, hand size corrects itself)
```

After `apply_action()`, the card is removed from the hand. Net effect: correct state,
engine handles all game logic (trick resolution, suit breaks, scoring).

### 2.3 State Construction

| Player | Hand Source | Known to Engine |
|--------|------------|-----------------|
| User (seat N) | **Real** (user input) | Exact cards |
| Other players | **Random** (engine PRNG) | Cards exist but don't match physical deal |
| Discard pile | **Real** (from played cards) | Exact cards |
| Played cards | **Real** (user relays) | Exact cards |

The agent's `CardTracker` and `encoder_v2` work from:
- User's real hand (correct)
- Played cards (public, correct)
- Opponent hand sizes (tracked)
- Void suit inference from play patterns

This is the **same imperfect information** a human player would have.

### 2.4 Agent Compatibility

All 5 agent types work unchanged:

| Agent | Dependencies | Recommended |
|-------|-------------|-------------|
| RandomAgent | None | Baseline only |
| HeuristicAgent V1 | None | Fast, explainable |
| HeuristicAgent V2 | JSON weights | Best heuristic |
| MCTSAgent | None (uses HeuristicV2) | Planning-capable |
| **RLAgentV2** | **PyTorch + checkpoint** | **Best performance** |

**Recommendation**: Start with **Heuristic V2** (fast, explainable, no model loading).
Use **RL Agent V2** for maximum performance demonstrations.

---

## 3. Implementation Plan

### 3.1 New File: `scripts/play/play_live_game.py` (~300 lines)

#### Section 1: CLI Setup (~30 lines)

```python
argparse arguments:
  --players N        Number of players (3-6, default 4)
  --seat N           User's seat (0 to N-1, default 0)
  --agent TYPE       Agent type: heuristic-v1, heuristic-v2, mcts, rl-v1, rl-v2
  --rounds N         Number of rounds (default 3)
  --explain          Show agent reasoning for each decision
  --seed N           Optional match seed (default: random)
```

#### Section 2: Hand Input Parser (~30 lines)

```python
def parse_card_input(text: str) -> list[int]:
    """Parse card names or IDs from user input.

    Supports:
    - Card names: "A<spade>", "K<club>", "10<heart>", "2<diamond>"
    - Card IDs: "0", "13", "51"
    - Mixed: "A<spade>, 10<club>, 2<diamond>" or "0 17 51"
    - Comma or space separated

    Returns: List of card IDs (0-51)
    """

def validate_hand(cards: list[int], num_players: int) -> bool:
    """Validate hand: correct count, unique cards, all 0-51."""
```

#### Section 3: State Builder (~50 lines)

Adapted from `setup_custom_deal()` in `run_manual_demo.py:69-244`.

```python
def build_live_game_state(
    user_hand: list[int],
    user_seat: int,
    num_players: int,
    match_seed: int,
    round_number: int = 1
) -> EngineState:
    """Build EngineState for live game.

    1. Create match via AceEngine.create_match()
    2. User's hand: real cards (from input)
    3. Other hands: randomly dealt from remaining cards (PRNG)
    4. Sort all hands canonically
    5. Find lead player (whoever holds A<spade>)
    6. Build RoundState, TrickState, RuntimeState
    7. Validate invariants
    8. Return EngineState
    """
```

#### Section 4: LiveGameSession Class (~100 lines)

```python
class LiveGameSession:
    """Wraps AceEngine for live human game mode."""

    def __init__(self, state: EngineState, user_seat: int, agent: BaseAgent):
        self.state = state
        self.user_seat = user_seat
        self.agent = agent
        self.events_log = []

    def is_user_turn(self) -> bool:
        """Check if it's the user's turn to act."""

    def get_current_player(self) -> int:
        """Get the current player ID whose turn it is."""

    def apply_agent_action(self) -> tuple[Action, list[Event]]:
        """Agent auto-plays for the user's seat.

        1. Get legal actions from engine
        2. Build player_view via get_player_view()
        3. Agent selects action
        4. Apply via AceEngine.apply_action()
        5. Return action and events
        """

    def apply_human_action(self, action: Action) -> tuple[bool, list[Event]]:
        """Apply an action reported by the user for another player.

        Pre-processing:
        1. If PlayCardAction: check if card is in player's hand
           - If NOT: add card to hand first (from public pool or swap)
        2. Call AceEngine.apply_action()
        3. Return success/failure and events
        """

    def advance_round(self) -> list[Event]:
        """Auto-advance for RoundStarting phase."""

    def is_terminal(self) -> bool:
        """Check if match is complete."""

    def get_result(self) -> MatchResult:
        """Get final match result."""
```

#### Section 5: Turn Loop (~80 lines)

```python
def run_game_loop(session: LiveGameSession, explain: bool):
    """Main game loop for live game."""

    while not session.is_terminal():
        phase = get_game_phase(session.state)

        # Auto-advance for round starts
        if isinstance(phase, RoundStarting):
            events = session.advance_round()
            display_round_start(events)
            continue

        current_player = session.get_current_player()

        if current_player == session.user_seat:
            # === AGENT TURN ===
            action, events = session.apply_agent_action()
            display_agent_action(action, events, explain)

        else:
            # === HUMAN TURN (other player) ===
            action = prompt_human_action(current_player, phase)
            success, events = session.apply_human_action(action)

            if not success:
                display_error(events)
                continue

            display_human_action(current_player, action, events)

        # Display trick/round summaries from events
        display_events(events)
```

#### Section 6: Human Input Handler (~40 lines)

```python
def prompt_human_action(player_id: int, phase: GamePhase) -> Action:
    """Prompt user to input what the human player did.

    Steal Phase:
      "Player X, steal or decline? [steal/decline]: "

    Play Phase:
      "Player X plays what card? [card name or ID]: "
      "Player X's hand has N cards, lead suit is S"
    """
```

#### Section 7: Display Helpers (~30 lines)

```python
def color_card(card_id: int) -> str:
    """ANSI-colored card string."""

def display_hand(hand: list[int], label: str = ""):
    """Display a hand with colored cards."""

def display_trick_result(events: list[Event]):
    """Display trick completion summary."""

def display_scoreboard(state: EngineState):
    """Display match scoreboard."""
```

### 3.2 Existing Files Used (No Modifications)

| File | Usage |
|------|-------|
| `engine/rules.py` | `AceEngine.create_match()`, `apply_action()`, `get_legal_actions()`, `is_terminal()`, `get_result()` |
| `engine/types.py` | All state models, action types |
| `engine/events.py` | `get_player_view()` for agent observation |
| `engine/card.py` | `card_to_str()`, `str_to_card()`, `sort_cards()` |
| `engine/observation.py` | `build_player_observation()` for agent |
| `agents/rl/v2/rl_agent_v2.py` | RL Agent V2 |
| `agents/heuristic/v2/heuristic_agent_v2.py` | Heuristic V2 |
| All other agent files | As needed |

---

## 4. Edge Cases

### 4.1 Card Not in Player's Virtual Hand

**Scenario**: Physical deal gave Player 1 a card different from their virtual hand.

**Solution**: Pre-process before `apply_action()`:
```python
if isinstance(action, PlayCardAction):
    player = find_player(state, action.player_id)
    if action.card not in player.hand:
        # Add card to their hand (from public pool)
        player.hand.append(action.card)
# Now apply_action() works normally
```

### 4.2 Steal Phase

**Scenario**: Another player steals from the user.

**Solution**: Engine handles normally — user's real cards are transferred to stealer's virtual hand. User physically gives their cards to the stealer.

**Scenario**: User steals from another player.

**Solution**: Engine adds stolen cards to user's real hand. User physically takes cards.

### 4.3 Interrupted Trick (Suit Break)

**Scenario**: Suit break causes cards to be collected.

**Solution**: Engine handles normally via `apply_action()`:
- If user collects: cards added to user's real hand
- If other collects: cards added to their virtual hand

### 4.4 Re-Entry

**Scenario**: User empties hand mid-trick but collects cards from interrupted trick.

**Solution**: Engine handles normally — user's hand goes from empty to having collected cards.

### 4.5 Suit Following (Other Players)

**Scenario**: Engine can't validate suit following for others (doesn't know their real hand).

**Solution**: User validates manually. They can see the physical cards and know if a player followed suit or broke suit. The engine just accepts the card input.

### 4.6 Player Goes Inactive

**Scenario**: A player's hand becomes empty.

**Solution**: Engine tracks via `len(hand)`. When hand size reaches 0, player is marked inactive. Works correctly for both real and virtual hands.

---

## 5. CLI Interface

### 5.1 Session Setup

```
=== ACE LIVE GAME ===
Number of players (3-6) [4]:
Your seat (0-3) [0]: 2
Agent type:
  [1] Heuristic V1 (Baseline)
  [2] Heuristic V2 (Evolved)
  [3] MCTS Agent
  [4] RL Agent V1 (Champion)
  [5] RL Agent V2 (Champion) <- recommended
Choice [5]: 5
Number of rounds [3]:
Match seed (Enter for random):
```

### 5.2 Round Start

```
=== ROUND 1 ===
Enter YOUR hand (card names like A<spade>, K<club> or IDs 0-51):
> A<spade>, K<club>, 7<heart>, 3<diamond>, J<spade>

Your hand: A<spade> K<club> 7<heart> 3<diamond> J<spade>
Engine dealing remaining cards to opponents...
Lead Player: Player 0 (holds A<spade>)
```

### 5.3 Steal Phase (Other Player)

```
=== TRICK 1 — STEAL PHASE ===
Player 0 is lead.
  Steal target: Player 1 (12 cards)
  -> What did Player 0 do? [steal/decline]: decline
```

### 5.4 Play Phase (Other Player)

```
Player 0 plays a card (Lead Suit: <spade>):
  -> What card?: K<spade>
  [Player 0 played K<spade>]
```

### 5.5 Play Phase (Agent Auto-Play)

```
=== YOUR TURN ===
Your hand: A<spade> K<club> 7<heart> 3<diamond> J<spade>
Lead Suit: <spade> | Must follow: Yes

[RL Agent V2 thinking...]
-> Agent plays: A<spade>
  (Reason: Highest card in lead suit, securing the trick)

[Applied successfully]
```

### 5.6 Trick Result

```
TRICK COMPLETED
  Outcome: All followed suit (DISCARDED)
  Winner: You (A<spade> — highest in <spade>)
  Cards discarded: A<spade> K<spade> Q<spade> J<spade>
  You lead next trick.
```

### 5.7 Round/Match End

```
=== ROUND 1 COMPLETE ===
Loser: Player 3 (last with cards)
Winners: Players 0, 1, 2

Scoreboard:
  Player 0: 2.0 pts (1W/0L/0D)
  Player 1: 2.0 pts (1W/0L/0D)
  Player 2 (You): 2.0 pts (1W/0L/0D) <- RL Agent V2
  Player 3: 0.0 pts (0W/1L/0D)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests (`tests/test_live_game.py`)

| Test | Purpose |
|------|---------|
| `test_parse_card_input` | Parse card names and IDs correctly |
| `test_build_live_game_state` | State construction with user hand + random others |
| `test_apply_human_action_with_mismatch` | Card not in virtual hand is handled |
| `test_apply_human_action_steal` | Steal from/to user works |
| `test_agent_turn` | Agent auto-plays and state updates |
| `test_full_round_simulation` | Complete round with all edge cases |
| `test_invariants_hold` | Invariants valid after every action |

### 6.2 Integration Test

Run a full 3-round match with 4 players, seed 42, using `HeuristicAgentV2` as the
agent. Verify:
- All invariants hold
- Round results are valid
- Match completes normally
- Scoreboard is correct

---

## 7. Dependencies

| Dependency | Required? | Notes |
|-----------|-----------|-------|
| PyTorch | Only for RL agents | Heuristic/MCTS agents don't need it |
| `checkpoints/rl_champion_v2.pt` | Only for RL V2 | 18.9MB model file |
| All existing engine modules | Yes | No new engine code |

---

## 8. File Summary

| Action | File | Lines |
|--------|------|-------|
| **CREATE** | `scripts/play/play_live_game.py` | ~300 |
| **CREATE** | `tests/test_live_game.py` | ~150 |
| **CREATE** | `docs/livegame_plan.md` | This file |
| **MODIFY** | None | — |

---

## 9. Implementation Order

| Step | Task | Est. Lines |
|------|------|-----------|
| 1 | CLI setup + argument parsing | 30 |
| 2 | Hand input parser + validator | 30 |
| 3 | State builder (adapted from `setup_custom_deal`) | 50 |
| 4 | `LiveGameSession` class | 100 |
| 5 | Turn loop + display | 80 |
| 6 | Edge case handling (mismatches, steal, re-entry) | — |
| 7 | Unit tests | 150 |
| 8 | Integration test (full match) | — |
| 9 | Polish (colored output, explain mode) | 30 |

**Total new code**: ~450 lines (script + tests)
