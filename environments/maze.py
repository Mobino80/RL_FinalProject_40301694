"""
maze.py
-------
Dynamic maze environment (MDP) for the RL final project.

RL Final Project - Student ID: 40301694

State  : s = (row, col, k, energy)
           row, col : agent position
           k        : 0/1, whether the key has been collected
           energy   : remaining energy (LIMITED ENERGY extra feature)
         Energy is part of the state, so the Markov property holds:
         the next-step behaviour of the environment depends only on
         (s, a), never on the history.

Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
         The intended action executes with prob 0.8; with prob 0.1 the
         agent slips to each of the two perpendicular directions.
         Hitting a wall / the border / the locked door keeps the agent
         in place and costs a penalty. Every step consumes 1 energy.

Rewards: two modes, selected with reward_mode:
           "sparse" - big reward only on key pickup / goal,
                      small cost per step, penalties for bad events.
           "shaped" - sparse rewards PLUS potential-based shaping
                      F(s,s') = gamma*Phi(s') - Phi(s), where
                      Phi = -(BFS distance to the current sub-goal).
                      Potential-based shaping provably preserves the
                      optimal policy (Ng, Harada, Russell 1999), which
                      makes the "did shaping change the policy?"
                      analysis in the report well grounded.

Episode termination:
  - goal reached                      (event GOAL_REACHED, terminal)
  - energy exhausted                  (event ENERGY_DEPLETED, terminal)
  - step count reached max_steps      (event STEP_LIMIT, truncation)
    max_steps = 3 * number of walkable cells (per project spec).
    NOTE: the step cap is a safety truncation and is intentionally NOT
    part of the MDP state; the energy budget (< max_steps) already
    guarantees finite episodes, so the model stays Markovian.

Model access:
  transitions(state, action) returns [(prob, next_state, reward, done)]
  and is used ONLY by Value Iteration (model-based). Q-Learning and
  SARSA(lambda) must interact exclusively through reset()/step().
"""

import csv
import random
from collections import deque
from pathlib import Path

# generator.py is the single source of truth for cell codes and the map
try:  # works both as package (from project root) and as a script
    from environments.generator import (FREE, WALL, PENALTY, START, KEY,
                                        DOOR, GOAL, CELL_CHARS, load_map)
except ImportError:
    from generator import (FREE, WALL, PENALTY, START, KEY, DOOR, GOAL,
                           CELL_CHARS, load_map)

# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
# perpendicular slip directions for each intended action
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT),
                 LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

P_INTENDED, P_SLIP = 0.8, 0.1

# --------------------------------------------------------------------------
# Events (all events required by the project spec)
# --------------------------------------------------------------------------
EV_MOVE           = "MOVE"                  # normal move
EV_WALL_HIT       = "WALL_HIT"              # bumped into wall / border
EV_PENALTY_CELL   = "PENALTY_CELL"          # entered a penalty cell
EV_KEY_PICKUP     = "KEY_PICKUP"            # collected the key
EV_DOOR_LOCKED    = "DOOR_LOCKED_ATTEMPT"   # tried to pass locked door
EV_DOOR_PASS      = "DOOR_PASS"             # passed the door with key
EV_GOAL           = "GOAL_REACHED"          # terminal success
EV_ENERGY_OUT     = "ENERGY_DEPLETED"       # terminal failure (feature)
EV_STEP_LIMIT     = "STEP_LIMIT"            # truncation by step cap

# --------------------------------------------------------------------------
# Reward design (documented choices - justify these values in the report)
# --------------------------------------------------------------------------
DEFAULT_REWARDS = {
    "step":           -1.0,    # small cost per action -> shorter paths
    "wall":           -4.0,    # extra, total -5 on a wall bump
    "locked_door":    -4.0,    # extra, total -5 on a locked-door attempt
    "penalty_cell":   -9.0,    # extra, total -10 when entering a trap cell
    "key":           +50.0,    # sub-goal reward
    "goal":         +100.0,    # main terminal reward
    "energy_out":    -20.0,    # terminal failure penalty
}
SHAPING_COEF   = 1.0           # weight of the potential-based term
SHAPING_GAMMA  = 0.99          # gamma used inside F = g*Phi(s')-Phi(s);
                               # keep equal to the training gamma


class MazeEnv:
    """Stochastic key-door maze with limited energy."""

    def __init__(self, map_data=None, map_filename="source_map.json",
                 reward_mode="sparse", rewards=None,
                 shaping_coef=SHAPING_COEF, shaping_gamma=SHAPING_GAMMA,
                 max_steps=None, seed=0):
        if map_data is None:
            map_data = load_map(map_filename)
        self.map_data = map_data
        self.grid = [row[:] for row in map_data["grid"]]
        self.n = map_data["size"]
        self.start_pos = tuple(map_data["start"])
        self.key_pos = tuple(map_data["key"])
        self.door_pos = tuple(map_data["door"])
        self.goal_pos = tuple(map_data["goal"])
        self.initial_energy = map_data["initial_energy"]

        assert reward_mode in ("sparse", "shaped")
        self.reward_mode = reward_mode
        self.rewards = dict(DEFAULT_REWARDS if rewards is None else rewards)
        self.shaping_coef = shaping_coef
        self.shaping_gamma = shaping_gamma

        self.walkable = [(r, c) for r in range(self.n) for c in range(self.n)
                         if self.grid[r][c] != WALL]
        # step cap = 3 * walkable cells (project spec suggestion);
        # record the final value in experiments/configs/.
        self.max_steps = max_steps or 3 * len(self.walkable)

        # BFS distance maps used only by the shaping potential
        self._dist_to_key = self._bfs_from(self.key_pos, door_open=False)
        self._dist_to_goal = self._bfs_from(self.goal_pos, door_open=True)

        self.rng = random.Random(seed)
        self.state = None
        self.steps = 0
        self.done = False

    # ------------------------------------------------------------------ #
    # Core API (used by ALL algorithms)                                   #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = (*self.start_pos, 0, self.initial_energy)
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action):
        """Sample one stochastic transition. Returns (s', r, done, info)."""
        if self.done:
            raise RuntimeError("Episode finished - call reset() first.")
        assert action in ACTIONS

        # sample the effective direction: 0.8 intended, 0.1 / 0.1 slips
        u = self.rng.random()
        if u < P_INTENDED:
            effective = action
        elif u < P_INTENDED + P_SLIP:
            effective = PERPENDICULAR[action][0]
        else:
            effective = PERPENDICULAR[action][1]

        next_state, reward, done, event = self._transition(self.state,
                                                           effective)
        self.steps += 1
        truncated = False
        if not done and self.steps >= self.max_steps:
            done, truncated, event = True, True, EV_STEP_LIMIT

        info = {"event": event, "effective_action": effective,
                "intended_action": action, "truncated": truncated,
                "steps": self.steps}
        self.state, self.done = next_state, done
        return next_state, reward, done, info

    # ------------------------------------------------------------------ #
    # Model access (used ONLY by Value Iteration)                          #
    # ------------------------------------------------------------------ #
    def transitions(self, state, action):
        """Full transition model: list of (prob, next_state, reward, done).

        Encodes the 0.8 / 0.1 / 0.1 stochasticity explicitly, as required
        by the spec ("the uncertainty must be part of the transition
        model"). Identical outcomes are merged.
        """
        outcomes = {}
        branches = [(P_INTENDED, action),
                    (P_SLIP, PERPENDICULAR[action][0]),
                    (P_SLIP, PERPENDICULAR[action][1])]
        for prob, eff in branches:
            ns, r, done, _ = self._transition(state, eff)
            key = (ns, r, done)
            outcomes[key] = outcomes.get(key, 0.0) + prob
        return [(p, ns, r, done) for (ns, r, done), p in outcomes.items()]

    def all_states(self):
        """Iterate over every non-terminal state (for Value Iteration)."""
        for (r, c) in self.walkable:
            if (r, c) == self.goal_pos:
                continue                      # terminal
            for k in (0, 1):
                if (r, c) == self.door_pos and k == 0:
                    continue                  # unreachable: on door w/o key
                for e in range(1, self.initial_energy + 1):
                    yield (r, c, k, e)

    def is_terminal(self, state):
        r, c, _, e = state
        return (r, c) == self.goal_pos or e <= 0

    # ------------------------------------------------------------------ #
    # Deterministic one-branch dynamics                                    #
    # ------------------------------------------------------------------ #
    def _transition(self, state, effective_action):
        """Apply ONE effective direction deterministically."""
        r, c, k, e = state
        dr, dc = ACTION_DELTAS[effective_action]
        nr, nc = r + dr, c + dc

        event = EV_MOVE
        base = self.rewards["step"]

        blocked = (not (0 <= nr < self.n and 0 <= nc < self.n)
                   or self.grid[nr][nc] == WALL)
        locked = ((nr, nc) == self.door_pos and k == 0) if not blocked \
                 else False

        if blocked:
            nr, nc = r, c
            event = EV_WALL_HIT
            base += self.rewards["wall"]
        elif locked:
            nr, nc = r, c
            event = EV_DOOR_LOCKED
            base += self.rewards["locked_door"]

        nk = k
        if (nr, nc) == self.key_pos and k == 0:
            nk = 1
            event = EV_KEY_PICKUP
            base += self.rewards["key"]
        elif (nr, nc) == self.door_pos and k == 1 and (nr, nc) != (r, c):
            event = EV_DOOR_PASS
        elif self.grid[nr][nc] == PENALTY and (nr, nc) != (r, c):
            event = EV_PENALTY_CELL
            base += self.rewards["penalty_cell"]

        ne = e - 1                              # every step costs energy
        done = False
        if (nr, nc) == self.goal_pos:
            event = EV_GOAL
            base += self.rewards["goal"]
            done = True
        elif ne <= 0:
            event = EV_ENERGY_OUT
            base += self.rewards["energy_out"]
            done = True

        next_state = (nr, nc, nk, ne)
        if self.reward_mode == "shaped":
            base += self.shaping_coef * self._shaping(state, next_state,
                                                      done)
        return next_state, base, done, event

    # ------------------------------------------------------------------ #
    # Potential-based reward shaping                                       #
    # ------------------------------------------------------------------ #
    def _potential(self, state):
        r, c, k, _ = state
        dist = self._dist_to_key if k == 0 else self._dist_to_goal
        d = dist.get((r, c))
        if d is None:                       # unreachable pocket: neutral
            d = self.n * self.n
        return -float(d)

    def _shaping(self, s, s_next, done):
        """F = gamma * Phi(s') - Phi(s); Phi(terminal) := 0."""
        phi_s = self._potential(s)
        phi_next = 0.0 if done else self._potential(s_next)
        return self.shaping_gamma * phi_next - phi_s

    def _bfs_from(self, source, door_open):
        """Distance from every walkable cell TO `source` (reverse BFS)."""
        blocked = {WALL} if door_open else {WALL, DOOR}
        dist = {source: 0}
        queue = deque([source])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.n and 0 <= nc < self.n
                        and (nr, nc) not in dist
                        and self.grid[nr][nc] not in blocked):
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    queue.append((nr, nc))
        return dist

    # ------------------------------------------------------------------ #
    # Rendering (quick debugging; the real GUI comes in phase 6)           #
    # ------------------------------------------------------------------ #
    def render_ascii(self, state=None):
        state = state or self.state
        lines = []
        for r in range(self.n):
            row = []
            for c in range(self.n):
                if state and (r, c) == (state[0], state[1]):
                    row.append("A")
                else:
                    row.append(CELL_CHARS[self.grid[r][c]])
            lines.append(" ".join(row))
        header = ""
        if state:
            header = (f"pos=({state[0]},{state[1]})  key={state[2]}  "
                      f"energy={state[3]}\n")
        return header + "\n".join(lines)


# --------------------------------------------------------------------------
# CSV event logger (project spec: all events must be stored in a log file)
# --------------------------------------------------------------------------
class MazeLogger:
    """Appends one row per step to a CSV log under results/raw_data/."""

    FIELDS = ["episode", "step", "row", "col", "key", "energy",
              "intended_action", "effective_action",
              "next_row", "next_col", "next_key", "next_energy",
              "reward", "event", "done"]

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.filepath.exists()
        self._fh = open(self.filepath, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(self.FIELDS)

    def log(self, episode, step, s, a, s_next, reward, info, done):
        self._writer.writerow([
            episode, step, s[0], s[1], s[2], s[3],
            ACTION_NAMES[info["intended_action"]],
            ACTION_NAMES[info["effective_action"]],
            s_next[0], s_next[1], s_next[2], s_next[3],
            round(reward, 4), info["event"], int(done)])

    def close(self):
        self._fh.close()


# --------------------------------------------------------------------------
# Smoke test:  python environments/maze.py
# Runs a few random-policy episodes and prints event statistics.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from collections import Counter

    env = MazeEnv(reward_mode="sparse", seed=9)
    print(f"[maze] size={env.n}x{env.n}, walkable cells={len(env.walkable)},"
          f" max_steps={env.max_steps}, energy={env.initial_energy}")

    logger = MazeLogger("results/raw_data/env_smoke_test.csv")
    events, returns = Counter(), []
    N_EPISODES = 20
    for ep in range(N_EPISODES):
        s = env.reset(seed=1000 + ep)
        total, t = 0.0, 0
        while True:
            a = env.rng.choice(ACTIONS)          # random policy
            s_next, r, done, info = env.step(a)
            logger.log(ep, t, s, a, s_next, r, info, done)
            events[info["event"]] += 1
            total += r
            s, t = s_next, t + 1
            if done:
                break
        returns.append(total)
    logger.close()

    print(f"[maze] random policy over {N_EPISODES} episodes:")
    print(f"        mean return = {sum(returns)/len(returns):.1f}")
    for ev, cnt in events.most_common():
        print(f"        {ev:<22} {cnt}")
    print("\n[maze] initial state preview:")
    env.reset()
    print(env.render_ascii())