# AETS Chapter 10: AI Interface

## 10.1 API Accessibility
The engine API MUST expose a programmatic interface for bot and agent execution:
- `get_legal_actions(state) -> list[Action]`
- `apply_action(state, action) -> ActionResult`

## 10.2 Public Agent Interface
AI agents MUST interact with the engine only through the public **Player View** projection of the state. Agents SHALL NOT have access to the raw `EngineState` or internal variables containing hidden information (such as other players' hands or the discard pile).

## 10.3 Reinforcement Learning (RL) Gym Adapter
The engine SHALL support an adapter layer conforming to reinforcement learning gym environments (e.g. OpenAI Gym / Farama Gymnasium):
- `reset(seed)`: Initializes a match and returns the initial player observation.
- `step(action)`: Applies the action and returns:
  - `observation`: The Player View projection.
  - `reward`: The incremental reward (e.g. `+2` for round win, `0` for draw, `-2` for round loss).
  - `terminated`: True if the match is complete.
  - `info`: Auxiliary logging data.
