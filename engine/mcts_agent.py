# engine/mcts_agent.py
"""Monte Carlo Tree Search (MCTS) Agent for the Ace Engine.

Implements Information Set MCTS (ISMCTS) using CardTracker for state determinization,
and HeuristicAgentV2 for rollout simulations.
"""

import math
import random
import time
import dataclasses
from typing import Sequence, List, Dict, Any, Optional, Tuple

from engine.agent import BaseAgent
from engine.rules import AceEngine, Success
from engine.types import Action, EngineState, PlayCardAction, StealAction, DeclineStealAction, MatchComplete, RoundStarting
from engine.card import get_suit, sort_cards
from engine.heuristic_agent import CardTracker
from engine.events import get_player_view
from engine.heuristic_agent_v2 import HeuristicAgentV2

class MCTSNode:
    """A node in the Monte Carlo search tree."""
    def __init__(self, action: Optional[Action] = None, parent: Optional['MCTSNode'] = None):
        self.action = action  # Action that led to this node
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visit_count = 0  # N
        self.total_value = 0.0  # Q

def project_determinization(viewer_id: int, tracker: CardTracker, state: EngineState, rng: random.Random) -> EngineState:
    """Creates a cloned EngineState with cards distributed to opponents matching all void/known constraints."""
    num_players = len(state.round_state.players)
    round_state = state.round_state
    
    # 1. Get target hand sizes
    target_sizes = [len(round_state.players[p].hand) for p in range(num_players)]
    
    # 2. Clean up opponent known cards to avoid any inconsistency (e.g. duplicate or played cards)
    viewer_hand = set(round_state.players[viewer_id].hand)
    in_trick_cards = set()
    if round_state.current_trick:
        in_trick_cards.update(play.card for play in round_state.current_trick.plays)
        
    cleaned_known = {p: set() for p in range(num_players)}
    seen_in_knowns = set()
    
    for p in range(num_players):
        if p == viewer_id:
            continue
        for card in tracker.player_known_cards[p]:
            if (tracker.card_locations[card] != "played" and
                card not in in_trick_cards and
                card not in viewer_hand and
                card not in seen_in_knowns):
                cleaned_known[p].add(card)
                seen_in_knowns.add(card)
                
    # 3. Initialize player hands with cleaned known facts, trimmed to target sizes
    hands = {}
    for p in range(num_players):
        if p == viewer_id:
            hands[p] = list(round_state.players[p].hand)
        else:
            hands[p] = list(cleaned_known[p])[:target_sizes[p]]
            
    # 4. Collect allocated cards
    allocated = set()
    for p in range(num_players):
        allocated.update(hands[p])
        
    # 5. Collect unallocated cards (exclude viewer hand, played cards, and current trick plays)
    unallocated = []
    for c in range(52):
        if (c not in allocated and
            tracker.card_locations[c] != "played" and
            c not in in_trick_cards):
            unallocated.append(c)
            
    # Shuffle unallocated cards
    rng_list = list(unallocated)
    rng.shuffle(rng_list)
    
    # 6. Backtracking search to assign cards complying with void constraints
    def backtrack(card_idx: int) -> bool:
        if card_idx >= len(rng_list):
            return True
            
        card = rng_list[card_idx]
        suit = get_suit(card)
        
        for p in range(num_players):
            if p == viewer_id:
                continue
            if len(hands[p]) < target_sizes[p] and not tracker.is_void[p][suit]:
                hands[p].append(card)
                if backtrack(card_idx + 1):
                    return True
                hands[p].pop()
        return False
        
    success = backtrack(0)
    
    # 7. Fallback if backtracking fails (guarantees safety)
    if not success:
        # Re-initialize hands with cleaned knowns trimmed to target sizes
        for p in range(num_players):
            if p != viewer_id:
                hands[p] = list(cleaned_known[p])[:target_sizes[p]]
        allocated = set()
        for p in range(num_players):
            allocated.update(hands[p])
        fallback_unallocated = [
            c for c in range(52)
            if c not in allocated and tracker.card_locations[c] != "played" and c not in in_trick_cards
        ]
        rng.shuffle(fallback_unallocated)
        
        card_idx = 0
        for p in range(num_players):
            if p == viewer_id:
                continue
            while len(hands[p]) < target_sizes[p] and card_idx < len(fallback_unallocated):
                hands[p].append(fallback_unallocated[card_idx])
                card_idx += 1
                
    # 8. Reconstruct the discard pile from tracker locations (as player view masks it to empty)
    discard_pile_cards = []
    for c in range(52):
        if tracker.card_locations[c] == "played" and c not in in_trick_cards:
            discard_pile_cards.append(c)

    # 9. Construct and return new EngineState with determinized hands and populated discard pile
    det_players = []
    for p_state in round_state.players:
        p_id = p_state.player_id
        if p_id == viewer_id:
            det_players.append(p_state)
        else:
            new_hand = tuple(sort_cards(hands[p_id]))
            det_players.append(dataclasses.replace(p_state, hand=new_hand))
            
    det_round_state = dataclasses.replace(
        round_state,
        players=tuple(det_players),
        discard_pile=tuple(sorted(discard_pile_cards))
    )
    return dataclasses.replace(state, round_state=det_round_state)

class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search agent for Ace trick-avoidance play."""
    
    def __init__(
        self,
        player_id: int,
        seed: Optional[int] = None,
        max_iterations: int = 150,
        time_limit: float = 0.15,
        exploration_constant: float = 1.414
    ) -> None:
        super().__init__(player_id)
        self.rng = random.Random(seed if seed is not None else 42)
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        
        # Instantiate rollout agents pool (using evolved HeuristicAgentV2 parameters)
        self.rollout_agents = {p: HeuristicAgentV2(player_id=p, seed=seed) for p in range(4)}
        
    def select_action(self, player_view: EngineState, legal_actions: Sequence[Action]) -> Action:
        """Selects the best action using Information Set MCTS."""
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        # Reconstruct CardTracker facts
        tracker = CardTracker(num_players=len(player_view.round_state.players))
        tracker.reconstruct(
            viewer_id=self.player_id,
            round_state=player_view.round_state,
            match_state=player_view.match_state
        )
        
        root = MCTSNode(action=None, parent=None)
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Check time limit budget
            if time.time() - start_time >= self.time_limit:
                break
                
            # 1. Determinization
            det_state = project_determinization(self.player_id, tracker, player_view, self.rng)
            
            # 2. Selection & Expansion
            curr_node = root
            curr_state = det_state
            
            while True:
                legal_acts = AceEngine.get_legal_actions(curr_state)
                if not legal_acts or AceEngine.is_terminal(curr_state):
                    break
                    
                # Find unexpanded actions
                unexpanded = [
                    a for a in legal_acts
                    if not any(self._actions_equal(a, child.action) for child in curr_node.children)
                ]
                
                if unexpanded:
                    # Expansion
                    act = self.rng.choice(unexpanded)
                    res = AceEngine.apply_action(curr_state, act)
                    if isinstance(res, Success):
                        curr_state = res.new_state
                        curr_state, _ = AceEngine.advance(curr_state)
                        
                    new_node = MCTSNode(action=act, parent=curr_node)
                    curr_node.children.append(new_node)
                    curr_node = new_node
                    break
                else:
                    # Selection
                    curr_node = self._select_uct_child(curr_node, legal_acts)
                    res = AceEngine.apply_action(curr_state, curr_node.action)
                    if isinstance(res, Success):
                        curr_state = res.new_state
                        curr_state, _ = AceEngine.advance(curr_state)
            
            # 3. Simulation (Heuristic-Guided Rollout)
            reward = self._run_rollout(curr_state)
            
            # 4. Backpropagation (Root-Perspective utility scoring)
            temp_node = curr_node
            while temp_node is not None:
                temp_node.visit_count += 1
                temp_node.total_value += reward
                temp_node = temp_node.parent
                
        # Action Selection: choose root child with highest visit count
        if not root.children:
            return self.rng.choice(legal_actions)
            
        best_child = max(root.children, key=lambda c: c.visit_count)
        # Find matching legal action to return
        for act in legal_actions:
            if self._actions_equal(act, best_child.action):
                return act
                
        return legal_actions[0]
        
    def _select_uct_child(self, node: MCTSNode, legal_actions: Sequence[Action]) -> MCTSNode:
        """Selects UCT child matching current legal actions."""
        valid_children = [
            child for child in node.children
            if any(self._actions_equal(child.action, a) for a in legal_actions)
        ]
        
        if not valid_children:
            return node.children[0]
            
        best_score = -float('inf')
        best_child = None
        
        for child in valid_children:
            if child.visit_count == 0:
                score = float('inf')
            else:
                exploitation = child.total_value / child.visit_count
                exploration = self.exploration_constant * math.sqrt(math.log(node.visit_count) / child.visit_count)
                score = exploitation + exploration
                
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
        
    def _run_rollout(self, state: EngineState) -> float:
        """Runs rollout using HeuristicAgentV2 and returns root-perspective reward."""
        sim_state = state
        
        while not AceEngine.is_terminal(sim_state):
            phase = AceEngine.get_game_phase(sim_state)
            if isinstance(phase, RoundStarting):
                sim_state, _ = AceEngine.advance(sim_state)
                continue
                
            player_id = sim_state.runtime_state.current_player_id
            if player_id is None:
                sim_state, _ = AceEngine.advance(sim_state)
                continue
                
            player_view = get_player_view(sim_state, player_id)
            legal_acts = AceEngine.get_legal_actions(sim_state)
            
            action = self.rollout_agents[player_id].select_action(player_view, legal_acts)
            res = AceEngine.apply_action(sim_state, action)
            if isinstance(res, Success):
                sim_state = res.new_state
            else:
                break
                
        # Evaluate root utility
        rounds_won = 0
        rounds_lost = 0
        for res in sim_state.match_state.round_results:
            if not res.is_draw:
                if res.loser_id != self.player_id:
                    rounds_won += 1
                else:
                    rounds_lost += 1
            else:
                rounds_won += 1
                
        rank = 4
        phase = AceEngine.get_game_phase(sim_state)
        if isinstance(phase, MatchComplete):
            for r in phase.result.rankings:
                if r.player_id == self.player_id:
                    rank = r.rank
                    break
                    
        # Non-linear placement mapping
        placement_scores = {1: 1.0, 2: 0.4, 3: 0.0, 4: -1.0}
        total_rounds = len(sim_state.match_state.round_results)
        survival_rate = (rounds_won - rounds_lost) / total_rounds if total_rounds > 0 else 0.0
        
        return survival_rate + placement_scores.get(rank, -1.0)

    def _actions_equal(self, a1: Optional[Action], a2: Optional[Action]) -> bool:
        """Helper to compare Action objects for equality."""
        if a1 is None or a2 is None:
            return a1 is None and a2 is None
        if type(a1) != type(a2):
            return False
            
        if isinstance(a1, PlayCardAction) and isinstance(a2, PlayCardAction):
            return a1.player_id == a2.player_id and a1.card == a2.card
        elif isinstance(a1, StealAction) and isinstance(a2, StealAction):
            return a1.player_id == a2.player_id
        elif isinstance(a1, DeclineStealAction) and isinstance(a2, DeclineStealAction):
            return a1.player_id == a2.player_id
            
        return a1.player_id == a2.player_id
