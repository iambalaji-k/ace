# AETS Chapter 7: Event System

## 7.1 Event Structure
Every state transition in the simulator MUST generate one or more structured event logs. Each event object MUST contain:
- `sequence: int` (monotonic, starting at 0)
- `event_type: str`
- `round_number: Optional[int]`
- `trick_number: Optional[int]`
- `payload: dict`
- `timestamp: int` (logical engine tick counter)

## 7.2 Event Catalog

### 7.2.1 MATCH_STARTED
- **Payload**: `{ match_id, num_players, num_rounds, match_seed }`
- **Visibility**: Public.

### 7.2.2 ROUND_STARTED
- **Payload**: `{ round_number, round_seed }`
- **Visibility**: Public.

### 7.2.3 ACES_RESERVED
- **Payload**: `{ player_id, ace_cards: list[int], count: int }`
- **Visibility**: `ace_cards` is Private/Observer only. Player ID and count are Public.

### 7.2.4 CARDS_DEALT
- **Payload**: `{ player_id, hand: list[int], hand_size: int }`
- **Visibility**: `hand` is Private/Observer only. Player ID and hand size are Public.

### 7.2.5 TRICK_STARTED
- **Payload**: `{ trick_number, lead_player_id }`
- **Visibility**: Public.

### 7.2.6 STEAL_EXECUTED
- **Payload**: `{ stealer_id, victim_id, cards: list[int], victim_new_count: int }`
- **Visibility**: `cards` is Private (stealer, victim, and observer only). Other fields are Public.

### 7.2.7 STEAL_DECLINED
- **Payload**: `{ player_id }`
- **Visibility**: Public.

### 7.2.8 CARD_PLAYED
- **Payload**: `{ player_id, card: int, is_lead: bool }`
- **Visibility**: Public.

### 7.2.9 TRICK_COMPLETED
- **Payload**: `{ trick_number, outcome: str, collector_id: Optional[int], cards_collected: list[int], cards_discarded: list[int] }`
- **Visibility**: Public.

### 7.2.10 PLAYER_INACTIVE
- **Payload**: `{ player_id, reason: str }`
- **Visibility**: Public.

### 7.2.11 ROUND_ENDED
- **Payload**: `{ round_number, loser_id: Optional[int], winner_ids: list[int], is_draw: bool }`
- **Visibility**: Public.

### 7.2.12 COUNTERS_UPDATED
- **Payload**: `{ updates: list[dict] }`
- **Visibility**: Public.

### 7.2.13 MATCH_ENDED
- **Payload**: `{ rankings: list[dict], total_rounds: int, draws: int }`
- **Visibility**: Public.

## 7.3 Projections
The engine MUST support projecting events into different visibility profiles before sending them to clients:
- **Observer Profile**: Receives all payloads completely unaltered.
- **Player Profile (Viewer V)**:
  - If event is `ACES_RESERVED` or `CARDS_DEALT` and `player_id != V`, strip `ace_cards` or `hand`.
  - If event is `STEAL_EXECUTED` and $V \notin \{\text{stealer\_id}, \text{victim\_id}\}$, strip `cards`.
- **Public Profile**: Completely strips `ace_cards`, `hand`, and `cards` from all events.
