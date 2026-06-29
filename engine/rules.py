# engine/rules.py
"""Core game engine rules, state transitions, and API implementation.

Implements the AceEngine class containing all match transitions, legal action generation,
trick resolution, and scoring logic.
"""

from typing import List, Tuple, Optional, Sequence
from dataclasses import replace

from engine.card import get_suit, get_rank
from engine.prng import pcg_seed, derive_round_seed
from engine.deck import build_and_deal_deck
from engine.types import (
    EngineState, MatchState, RoundState, RuntimeState, PlayerState, RoundPlayerState,
    TrickState, TrickPlay, StealEvent, CompletedTrick, RoundResult, PlayerRanking, MatchResult,
    Action, StealAction, DeclineStealAction, PlayCardAction,
    GamePhase, RoundStarting, AwaitingStealDecision, AwaitingCardPlay, MatchComplete,
    Event
)
from engine.invariants import validate_invariants


class ValidationError(Exception):
    """Exception raised when an invariant is violated in the engine."""
    def __init__(self, message: str, state: EngineState):
        super().__init__(message)
        self.state = state


class ActionResult:
    """Result of applying an action to the state."""
    pass


class Success(ActionResult):
    def __init__(self, new_state: EngineState, events: List[Event]):
        self.new_state = new_state
        self.events = events


class Error(ActionResult):
    def __init__(self, error_code: str, message: str, legal_actions: Sequence[Action]):
        self.error_code = error_code
        self.message = message
        self.legal_actions = legal_actions


def get_immediate_active_left(player_id: int, active_player_ids: Sequence[int], num_players: int) -> int:
    """Find the next active player clockwise from the given player_id."""
    active_set = set(active_player_ids)
    curr = (player_id + 1) % num_players
    # Safeguard against infinite loop if no active player
    for _ in range(num_players):
        if curr in active_set:
            return curr
        curr = (curr + 1) % num_players
    raise ValueError("No active players found")


class AceEngine:
    @staticmethod
    def create_match(match_id: int, num_players: int, num_rounds: int, match_seed: int) -> EngineState:
        """Initialize a new MatchState, RoundState, and RuntimeState."""
        if not 3 <= num_players <= 6:
            raise ValueError(f"Invalid player count: {num_players}. Must be 3-6.")
        if num_rounds < 1:
            raise ValueError(f"Invalid round count: {num_rounds}. Must be >= 1.")

        players = [
            PlayerState(
                player_id=i,
                consecutive_loss_count=0,
                rounds_won=0,
                rounds_lost=0,
                rounds_drawn=0,
                half_points=0
            ) for i in range(num_players)
        ]

        match_state = MatchState(
            match_id=match_id,
            num_rounds=num_rounds,
            current_round=1,
            match_seed=match_seed,
            players=tuple(players),
            seating_order=tuple(range(num_players)),
            round_results=(),
            status="INIT"
        )

        initial_prng_state = pcg_seed(match_seed)

        runtime_state = RuntimeState(
            action_sequence_number=0,
            current_phase=RoundStarting(round_number=1),
            current_player_id=None,
            pending_legal_actions=(),
            prng_state=initial_prng_state
        )

        state = EngineState(
            match_state=match_state,
            round_state=None,
            runtime_state=runtime_state
        )

        # Validate invariants immediately
        violations = validate_invariants(state)
        if violations:
            raise ValidationError(f"Invariants violated on init: {violations}", state)

        return state

    @staticmethod
    def get_game_phase(state: EngineState) -> GamePhase:
        """Get the current game phase from runtime state."""
        return state.runtime_state.current_phase

    @staticmethod
    def get_legal_actions(state: EngineState) -> Sequence[Action]:
        """Get the list of pending legal actions from runtime state."""
        return state.runtime_state.pending_legal_actions

    @staticmethod
    def is_terminal(state: EngineState) -> bool:
        """Check if the match is complete."""
        return state.match_state.status == "COMPLETE"

    @staticmethod
    def get_result(state: EngineState) -> Optional[MatchResult]:
        """Get the final MatchResult if complete, otherwise None."""
        if not AceEngine.is_terminal(state):
            return None
        phase = state.runtime_state.current_phase
        if isinstance(phase, MatchComplete):
            return phase.result
        return None

    @staticmethod
    def advance(state: EngineState) -> Tuple[EngineState, List[Event]]:
        """Advance the match state for non-player phases (e.g. RoundStarting)."""
        phase = state.runtime_state.current_phase
        if not isinstance(phase, RoundStarting):
            # No auto-advance needed
            return state, []

        r = phase.round_number
        match = state.match_state
        num_players = len(match.players)
        events = []

        # Emit MATCH_STARTED if round 1
        seq = state.runtime_state.action_sequence_number
        ts = state.runtime_state.action_sequence_number  # Use sequence number as logical timestamp
        if r == 1:
            events.append(Event(
                sequence=len(events) + seq,
                event_type="MATCH_STARTED",
                round_number=None,
                trick_number=None,
                payload={
                    "match_id": match.match_id,
                    "num_players": num_players,
                    "num_rounds": match.num_rounds,
                    "match_seed": match.match_seed
                },
                timestamp=ts
            ))

        # Emit ROUND_STARTED
        round_seed = derive_round_seed(match.match_seed, r)
        events.append(Event(
            sequence=len(events) + seq,
            event_type="ROUND_STARTED",
            round_number=r,
            trick_number=None,
            payload={
                "round_number": r,
                "round_seed": round_seed
            },
            timestamp=ts
        ))

        # Determine reserved aces recipient (if any)
        recipient_id = None
        consecutive_loss_count = 0
        if r > 1:
            prev_result = match.round_results[-1]
            if not prev_result.is_draw:
                recipient_id = prev_result.loser_id
                if recipient_id is not None:
                    consecutive_loss_count = match.players[recipient_id].consecutive_loss_count

        # Build and deal deck
        hands, shuffled_deck = build_and_deal_deck(
            num_players=num_players,
            recipient_id=recipient_id,
            consecutive_loss_count=consecutive_loss_count,
            round_seed=round_seed
        )

        # Emit ACES_RESERVED
        if recipient_id is not None:
            from engine.deck import get_reserved_aces
            aces = get_reserved_aces(consecutive_loss_count)
            events.append(Event(
                sequence=len(events) + seq,
                event_type="ACES_RESERVED",
                round_number=r,
                trick_number=None,
                payload={
                    "player_id": recipient_id,
                    "ace_cards": aces,
                    "count": len(aces)
                },
                timestamp=ts
            ))

        # Emit CARDS_DEALT
        for i in range(num_players):
            events.append(Event(
                sequence=len(events) + seq,
                event_type="CARDS_DEALT",
                round_number=r,
                trick_number=None,
                payload={
                    "player_id": i,
                    "hand": hands[i],
                    "hand_size": len(hands[i])
                },
                timestamp=ts
            ))

        # Determine Lead Player (who holds A♠ = card ID 0)
        lead_player_id = None
        for i in range(num_players):
            if 0 in hands[i]:
                lead_player_id = i
                break
        if lead_player_id is None:
            raise RuntimeError("A♠ not found in any player hand after deal")

        # Emit TRICK_STARTED for Trick 1
        events.append(Event(
            sequence=len(events) + seq,
            event_type="TRICK_STARTED",
            round_number=r,
            trick_number=1,
            payload={
                "trick_number": 1,
                "lead_player_id": lead_player_id
            },
            timestamp=ts
        ))

        # Build RoundPlayerState
        round_players = [
            RoundPlayerState(
                player_id=i,
                hand=tuple(hands[i]),
                is_active=True,
                is_round_winner=False,
                is_round_loser=False
            ) for i in range(num_players)
        ]

        # First Trick State
        trick = TrickState(
            trick_number=1,
            lead_player_id=lead_player_id,
            lead_suit=None,
            plays=(),
            status="STEAL_PHASE",
            steals=()
        )

        round_state = RoundState(
            round_number=r,
            round_seed=round_seed,
            players=tuple(round_players),
            active_player_ids=tuple(range(num_players)),
            current_trick=trick,
            trick_history=(),
            lead_player_id=lead_player_id,
            discard_pile=(),
            status="IN_PROGRESS"
        )

        # Update MatchState status if in INIT
        new_match_status = match.status
        if match.status == "INIT":
            new_match_status = "IN_PROGRESS"

        new_match_state = replace(
            match,
            status=new_match_status
        )

        # Setup RuntimeState for Trick 1 Steal Phase
        steal_target = get_immediate_active_left(lead_player_id, round_state.active_player_ids, num_players)
        next_phase = AwaitingStealDecision(player_id=lead_player_id, steal_target=steal_target)
        legal_actions = [
            StealAction(player_id=lead_player_id),
            DeclineStealAction(player_id=lead_player_id)
        ]

        new_runtime_state = replace(
            state.runtime_state,
            current_phase=next_phase,
            current_player_id=lead_player_id,
            pending_legal_actions=tuple(legal_actions)
        )

        new_state = EngineState(
            match_state=new_match_state,
            round_state=round_state,
            runtime_state=new_runtime_state
        )

        # Invariant validation
        violations = validate_invariants(new_state)
        if violations:
            raise ValidationError(f"Invariants violated on round start: {violations}", new_state)

        return new_state, events

    @staticmethod
    def apply_action(state: EngineState, action: Action) -> ActionResult:
        """Apply a player's action to the state, returning the success or error result."""
        # 1. Basic phase and turn checks
        phase = state.runtime_state.current_phase
        if isinstance(phase, MatchComplete) or isinstance(phase, RoundStarting):
            return Error("WRONG_PHASE", "No action can be taken in this phase", [])

        if action.player_id != state.runtime_state.current_player_id:
            return Error("NOT_YOUR_TURN", "It is not this player's turn", state.runtime_state.pending_legal_actions)

        # 2. Check if the action is in the pending legal list
        # We need a custom equality check since action objects are distinct instances
        def is_action_equivalent(a1: Action, a2: Action) -> bool:
            if a1.__class__ is not a2.__class__:
                return False
            if a1.player_id != a2.player_id:
                return False
            if isinstance(a1, PlayCardAction) and isinstance(a2, PlayCardAction):
                return a1.card == a2.card
            return True

        if not any(is_action_equivalent(action, legal) for legal in state.runtime_state.pending_legal_actions):
            return Error("ILLEGAL_CARD" if isinstance(action, PlayCardAction) else "INVALID_ACTION_TYPE",
                         "The proposed action is illegal in the current state",
                         state.runtime_state.pending_legal_actions)

        # 3. Apply transition
        events: List[Event] = []
        seq = state.runtime_state.action_sequence_number + 1
        ts = seq

        match = state.match_state
        round_st = state.round_state
        assert round_st is not None
        trick = round_st.current_trick
        assert trick is not None
        num_players = len(match.players)

        # Initialize transition variables to satisfy typecheck analysis
        new_trick = trick
        next_phase = phase
        legal_actions: List[Action] = []
        current_player_id: Optional[int] = None

        new_round_players = [p for p in round_st.players]
        new_active_players = list(round_st.active_player_ids)
        new_discard_pile = list(round_st.discard_pile)
        new_trick_history = list(round_st.trick_history)

        round_ended = False
        is_draw = False
        round_loser_id = None
        round_winner_ids = []

        if isinstance(action, StealAction):
            # Perform steal from immediate active left
            target_id = get_immediate_active_left(action.player_id, new_active_players, num_players)
            stealer_idx = next(i for i, p in enumerate(new_round_players) if p.player_id == action.player_id)
            victim_idx = next(i for i, p in enumerate(new_round_players) if p.player_id == target_id)

            stealer = new_round_players[stealer_idx]
            victim = new_round_players[victim_idx]

            stolen_cards = list(victim.hand)
            # Remove from victim, merge into stealer
            new_stealer_hand = tuple(sorted(list(stealer.hand) + stolen_cards))

            new_round_players[stealer_idx] = replace(stealer, hand=new_stealer_hand)
            new_round_players[victim_idx] = replace(victim, hand=(), is_active=False, is_round_winner=True)
            new_active_players.remove(target_id)

            # Record steal in trick
            new_steals = list(trick.steals) + [StealEvent(stealer_id=action.player_id, victim_id=target_id, cards_taken=tuple(stolen_cards))]
            new_trick = replace(trick, steals=tuple(new_steals))

            events.append(Event(
                sequence=len(events) + seq,
                event_type="STEAL_EXECUTED",
                round_number=round_st.round_number,
                trick_number=trick.trick_number,
                payload={
                    "stealer_id": action.player_id,
                    "victim_id": target_id,
                    "cards": stolen_cards,
                    "victim_new_count": 0
                },
                timestamp=ts
            ))

            events.append(Event(
                sequence=len(events) + seq,
                event_type="PLAYER_INACTIVE",
                round_number=round_st.round_number,
                trick_number=trick.trick_number,
                payload={
                    "player_id": target_id,
                    "reason": "STOLEN_FROM"
                },
                timestamp=ts
            ))

            # Check if stealer absorbed cards from ALL other active players
            if len(new_active_players) == 1:
                # Stealer is the sole active player, so stealer loses immediately
                sole_active_id = new_active_players[0]
                assert sole_active_id == action.player_id

                sole_player_idx = next(i for i, p in enumerate(new_round_players) if p.player_id == sole_active_id)
                new_round_players[sole_player_idx] = replace(new_round_players[sole_player_idx], is_round_loser=True)

                round_ended = True
                round_loser_id = sole_active_id
                round_winner_ids = [p.player_id for p in new_round_players if p.player_id != sole_active_id]
                new_trick = replace(new_trick, status="RESOLVED")
            else:
                # Can steal again from the new immediate active left
                new_target = get_immediate_active_left(action.player_id, new_active_players, num_players)
                next_phase = AwaitingStealDecision(player_id=action.player_id, steal_target=new_target)
                legal_actions = [
                    StealAction(player_id=action.player_id),
                    DeclineStealAction(player_id=action.player_id)
                ]

        elif isinstance(action, DeclineStealAction):
            # Proceed to Play Phase
            new_trick = replace(trick, status="PLAY_PHASE")
            events.append(Event(
                sequence=len(events) + seq,
                event_type="STEAL_DECLINED",
                round_number=round_st.round_number,
                trick_number=trick.trick_number,
                payload={"player_id": action.player_id},
                timestamp=ts
            ))

            # Lead player must play a card
            next_phase = AwaitingCardPlay(player_id=action.player_id, lead_suit=None, must_follow=False)
            lead_hand = next(p.hand for p in new_round_players if p.player_id == action.player_id)
            legal_actions = [PlayCardAction(player_id=action.player_id, card=c) for c in lead_hand]

        elif isinstance(action, PlayCardAction):
            # Play a card
            next_lead_player_id: Optional[int] = None
            card = action.card
            player_idx = next(i for i, p in enumerate(new_round_players) if p.player_id == action.player_id)
            player_state = new_round_players[player_idx]

            # Remove card from hand
            new_hand = list(player_state.hand)
            new_hand.remove(card)
            new_round_players[player_idx] = replace(player_state, hand=new_hand)

            is_lead_card = len(trick.plays) == 0
            new_plays = list(trick.plays) + [TrickPlay(player_id=action.player_id, card=card)]
            card_suit = get_suit(card)

            new_lead_suit = trick.lead_suit
            if is_lead_card:
                new_lead_suit = card_suit

            new_trick = replace(trick, plays=new_plays, lead_suit=new_lead_suit)

            events.append(Event(
                sequence=len(events) + seq,
                event_type="CARD_PLAYED",
                round_number=round_st.round_number,
                trick_number=trick.trick_number,
                payload={
                    "player_id": action.player_id,
                    "card": card,
                    "is_lead": is_lead_card
                },
                timestamp=ts
            ))

            # Check trick end
            # Suit broken or all active players played
            if not is_lead_card and card_suit != new_lead_suit:
                # 1. Interrupted trick!
                # Collector is the one who played highest card of lead suit
                collector_id = None
                highest_rank = 13  # lower than any valid rank (0-12)
                for play in new_plays:
                    if get_suit(play.card) == new_lead_suit:
                        rank = get_rank(play.card)
                        if rank < highest_rank:  # 0 is highest, 12 is lowest
                            highest_rank = rank
                            collector_id = play.player_id

                assert collector_id is not None
                collected_cards = [play.card for play in new_plays]

                # Give all cards in the plays to collector's hand
                collector_idx = next(i for i, p in enumerate(new_round_players) if p.player_id == collector_id)
                collector_state = new_round_players[collector_idx]
                merged_hand = tuple(sorted(list(collector_state.hand) + collected_cards))
                new_round_players[collector_idx] = replace(collector_state, hand=merged_hand)

                completed = CompletedTrick(
                    trick_number=trick.trick_number,
                    plays=tuple(new_plays),
                    outcome="INTERRUPTED",
                    collector_id=collector_id,
                    collected_cards=tuple(collected_cards)
                )
                new_trick_history.append(completed)

                events.append(Event(
                    sequence=len(events) + seq,
                    event_type="TRICK_COMPLETED",
                    round_number=round_st.round_number,
                    trick_number=trick.trick_number,
                    payload={
                        "trick_number": trick.trick_number,
                        "outcome": "INTERRUPTED",
                        "collector_id": collector_id,
                        "cards_collected": collected_cards,
                        "cards_discarded": []
                    },
                    timestamp=ts
                ))

                # Trick evaluation: check for inactive players
                next_lead_player_id = collector_id
                trick_resolved = True

            elif len(new_plays) == len(new_active_players):
                # 2. Successful trick! (All followed suit)
                # Discard all played cards
                collected_cards = [play.card for play in new_plays]
                new_discard_pile.extend(collected_cards)

                # Determine trick winner (highest card of lead suit)
                winner_id = None
                highest_rank = 13
                for play in new_plays:
                    if get_suit(play.card) == new_lead_suit:
                        rank = get_rank(play.card)
                        if rank < highest_rank:
                            highest_rank = rank
                            winner_id = play.player_id

                completed = CompletedTrick(
                    trick_number=trick.trick_number,
                    plays=tuple(new_plays),
                    outcome="DISCARDED",
                    collector_id=None,
                    collected_cards=()
                )
                new_trick_history.append(completed)

                events.append(Event(
                    sequence=len(events) + seq,
                    event_type="TRICK_COMPLETED",
                    round_number=round_st.round_number,
                    trick_number=trick.trick_number,
                    payload={
                        "trick_number": trick.trick_number,
                        "outcome": "DISCARDED",
                        "collector_id": None,
                        "cards_collected": [],
                        "cards_discarded": collected_cards
                    },
                    timestamp=ts
                ))

                next_lead_player_id = winner_id
                trick_resolved = True

            else:
                # Trick continues, next active player in clockwise rotation must play
                trick_resolved = False
                curr_idx = new_active_players.index(action.player_id)
                next_player_id = new_active_players[(curr_idx + 1) % len(new_active_players)]

                next_phase = AwaitingCardPlay(
                    player_id=next_player_id,
                    lead_suit=new_lead_suit,
                    must_follow=any(get_suit(c) == new_lead_suit for c in next(p.hand for p in new_round_players if p.player_id == next_player_id))
                )

                # Legal actions for next player
                next_hand = next(p.hand for p in new_round_players if p.player_id == next_player_id)
                has_suit = any(get_suit(c) == new_lead_suit for c in next_hand)
                if has_suit:
                    legal_actions = [PlayCardAction(player_id=next_player_id, card=c) for c in next_hand if get_suit(c) == new_lead_suit]
                else:
                    legal_actions = [PlayCardAction(player_id=next_player_id, card=c) for c in next_hand]

            if trick_resolved:
                # Resolve inactive transitions
                newly_inactive = []
                for p in list(new_round_players):
                    if p.player_id in new_active_players and len(p.hand) == 0:
                        p_idx = new_round_players.index(p)
                        new_round_players[p_idx] = replace(p, is_active=False, is_round_winner=True)
                        new_active_players.remove(p.player_id)
                        newly_inactive.append(p.player_id)

                for pinact in newly_inactive:
                    events.append(Event(
                        sequence=len(events) + seq,
                        event_type="PLAYER_INACTIVE",
                        round_number=round_st.round_number,
                        trick_number=trick.trick_number,
                        payload={
                            "player_id": pinact,
                            "reason": "EMPTIED_HAND"
                        },
                        timestamp=ts
                    ))

                # Check round end
                if len(new_active_players) == 1:
                    round_ended = True
                    round_loser_id = new_active_players[0]
                    loser_idx = next(i for i, p in enumerate(new_round_players) if p.player_id == round_loser_id)
                    new_round_players[loser_idx] = replace(new_round_players[loser_idx], is_round_loser=True)
                    round_winner_ids = [p.player_id for p in new_round_players if p.player_id != round_loser_id]
                elif len(new_active_players) == 0:
                    round_ended = True
                    is_draw = True
                    round_loser_id = None
                    round_winner_ids = [p.player_id for p in new_round_players]
                else:
                    # Proceed to next trick
                    next_trick_num = trick.trick_number + 1

                    # Edge Case D: Inactive trick winner lead transfer
                    assert next_lead_player_id is not None
                    winner_state = next(p for p in new_round_players if p.player_id == next_lead_player_id)
                    if not winner_state.is_active:
                        # Winner of trick is inactive, find next highest active in lead suit
                        next_lead_active = None
                        highest_active_rank = 13
                        for play in new_plays:
                            if get_suit(play.card) == new_lead_suit:
                                play_player = next(p for p in new_round_players if p.player_id == play.player_id)
                                if play_player.is_active:
                                    rank = get_rank(play.card)
                                    if rank < highest_active_rank:
                                        highest_active_rank = rank
                                        next_lead_active = play.player_id

                        if next_lead_active is not None:
                            next_lead_player_id = next_lead_active
                        else:
                            # Fallback: next clockwise active from winner
                            next_lead_player_id = get_immediate_active_left(next_lead_player_id, new_active_players, num_players)

                    new_trick = TrickState(
                        trick_number=next_trick_num,
                        lead_player_id=next_lead_player_id,
                        lead_suit=None,
                        plays=(),
                        status="STEAL_PHASE",
                        steals=()
                    )

                    events.append(Event(
                        sequence=len(events) + seq,
                        event_type="TRICK_STARTED",
                        round_number=round_st.round_number,
                        trick_number=next_trick_num,
                        payload={
                            "trick_number": next_trick_num,
                            "lead_player_id": next_lead_player_id
                        },
                        timestamp=ts
                    ))

                    steal_target = get_immediate_active_left(next_lead_player_id, new_active_players, num_players)
                    next_phase = AwaitingStealDecision(player_id=next_lead_player_id, steal_target=steal_target)
                    actions: Sequence[Action] = [
                        StealAction(player_id=next_lead_player_id),
                        DeclineStealAction(player_id=next_lead_player_id)
                    ]
                    legal_actions = actions

        # 4. Resolve round end if ended
        new_match_players = [p for p in match.players]
        new_round_results = list(match.round_results)
        new_match_status = match.status
        new_round_state = None

        if round_ended:
            new_round_state = replace(
                round_st,
                players=tuple(new_round_players),
                active_player_ids=tuple(new_active_players),
                current_trick=None,
                trick_history=tuple(new_trick_history),
                discard_pile=tuple(new_discard_pile),
                status="COMPLETE"
            )

            # RoundResult
            round_res = RoundResult(
                round_number=round_st.round_number,
                loser_id=round_loser_id,
                winner_ids=tuple(round_winner_ids),
                is_draw=is_draw
            )
            new_round_results.append(round_res)

            # Update match-level players' stats & loss counters
            counter_updates = []
            for i, p in enumerate(new_match_players):
                old_count = p.consecutive_loss_count
                if is_draw:
                    new_count = 0
                    new_match_players[i] = replace(
                        p,
                        consecutive_loss_count=new_count,
                        rounds_drawn=p.rounds_drawn + 1,
                        half_points=p.half_points + 1
                    )
                elif p.player_id == round_loser_id:
                    new_count = old_count + 1
                    new_match_players[i] = replace(
                        p,
                        consecutive_loss_count=new_count,
                        rounds_lost=p.rounds_lost + 1
                    )
                else:
                    new_count = 0
                    new_match_players[i] = replace(
                        p,
                        consecutive_loss_count=new_count,
                        rounds_won=p.rounds_won + 1,
                        half_points=p.half_points + 2
                    )
                counter_updates.append({
                    "player_id": p.player_id,
                    "old_count": old_count,
                    "new_count": new_count
                })

            events.append(Event(
                sequence=len(events) + seq,
                event_type="ROUND_ENDED",
                round_number=round_st.round_number,
                trick_number=None,
                payload={
                    "round_number": round_st.round_number,
                    "loser_id": round_loser_id,
                    "winner_ids": round_winner_ids,
                    "is_draw": is_draw
                },
                timestamp=ts
            ))

            events.append(Event(
                sequence=len(events) + seq,
                event_type="COUNTERS_UPDATED",
                round_number=round_st.round_number,
                trick_number=None,
                payload={"updates": counter_updates},
                timestamp=ts
            ))

            # Check if match ended
            if round_st.round_number == match.num_rounds:
                new_match_status = "COMPLETE"

                # Compute rankings
                # Sort first by points (descending), then by fewer losses (ascending), then by seat ID (ascending)
                # Sort uses negative points, positive losses, positive ID
                sorted_players = sorted(
                    new_match_players,
                    key=lambda p: (-p.half_points, p.rounds_lost, p.player_id)
                )

                rankings = []
                for idx, p in enumerate(sorted_players):
                    rankings.append(PlayerRanking(
                        player_id=p.player_id,
                        rank=idx + 1,
                        half_points=p.half_points,
                        rounds_won=p.rounds_won,
                        rounds_lost=p.rounds_lost,
                        rounds_drawn=p.rounds_drawn
                    ))

                total_draws = sum(1 for r in new_round_results if r.is_draw)
                match_res = MatchResult(
                    rankings=tuple(rankings),
                    total_rounds=match.num_rounds,
                    draws=total_draws
                )

                events.append(Event(
                    sequence=len(events) + seq,
                    event_type="MATCH_ENDED",
                    round_number=None,
                    trick_number=None,
                    payload={
                        "rankings": [r.__dict__ for r in rankings],
                        "total_rounds": match.num_rounds,
                        "draws": total_draws
                    },
                    timestamp=ts
                ))

                next_phase = MatchComplete(result=match_res)
                legal_actions: Sequence[Action] = []
                current_player_id = None
            else:
                # Next round setup
                next_phase = RoundStarting(round_number=round_st.round_number + 1)
                legal_actions = []
                current_player_id = None
        else:
            # Round continues, construct next RoundState
            new_round_state = replace(
                round_st,
                players=tuple(new_round_players),
                active_player_ids=tuple(new_active_players),
                current_trick=new_trick,
                trick_history=tuple(new_trick_history),
                discard_pile=tuple(new_discard_pile)
            )
            if isinstance(next_phase, (AwaitingStealDecision, AwaitingCardPlay)):
                current_player_id = next_phase.player_id
            else:
                current_player_id = None

        # Update MatchState
        new_match_state = replace(
            match,
            players=tuple(new_match_players),
            round_results=tuple(new_round_results),
            status=new_match_status
        )

        # Update RuntimeState
        new_runtime_state = replace(
            state.runtime_state,
            action_sequence_number=seq,
            current_phase=next_phase,
            current_player_id=current_player_id,
            pending_legal_actions=tuple(legal_actions)
        )

        new_state = EngineState(
            match_state=new_match_state,
            round_state=new_round_state,
            runtime_state=new_runtime_state
        )

        # 5. Invariant validation
        violations = validate_invariants(new_state)
        if violations:
            # Fatal engine bug: halt and raise ValidationError
            raise ValidationError(f"Invariants violated: {violations}", new_state)

        return Success(new_state, events)
