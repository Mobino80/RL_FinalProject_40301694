"""
generator.py
------------
Map generator for the dynamic maze environment.

RL Final Project - Student ID: 40301694
    base_seed = int(StudentID[-2]) = 9
    maze_size = 15 + (9 % 4)      = 16

Responsibilities of this module:
  1. Deterministically generate a 16x16 maze from the student-specific seed.
  2. Guarantee the map contains: start, key, locked door, goal,
     >= 15% wall cells and >= 5 penalty cells.
  3. Enclose the goal in a small room whose ONLY entrance is the door,
     so the key/door mechanic is truly meaningful.
  4. Validate the map with BFS:
        (a) start -> key   is reachable while the door is locked,
        (b) key   -> goal  is reachable once the door is passable,
        (c) goal is NOT reachable from start while the door is locked.
  5. If a candidate map is invalid, regenerate it in a *deterministic,
     reproducible* way (attempt index is mixed into the seed).
  6. Save the final map (grid + metadata) as JSON in environments/maps/
     so that all three algorithms run on the exact same environment.

Extra feature chosen for the project: LIMITED ENERGY.
The initial energy budget is stored in the map file so the environment
(maze.py) and all experiments read it from a single source of truth.
"""

import json
import math
import random
from collections import deque
from pathlib import Path

# --------------------------------------------------------------------------
# Cell type codes (single source of truth - maze.py must import these)
# --------------------------------------------------------------------------
FREE    = 0   # normal walkable cell
WALL    = 1   # obstacle, agent bounces back with a penalty
PENALTY = 2   # walkable but gives an extra negative reward
START   = 3   # agent's starting cell
KEY     = 4   # key location (k: 0 -> 1)
DOOR    = 5   # locked door; passable only when k == 1
GOAL    = 6   # terminal goal cell (inside the door-gated room)

CELL_CHARS = {FREE: ".", WALL: "#", PENALTY: "*",
              START: "S", KEY: "K", DOOR: "D", GOAL: "G"}

# --------------------------------------------------------------------------
# Project constants derived from the student ID (per project specification)
# --------------------------------------------------------------------------
STUDENT_ID = "40301694"
BASE_SEED  = int(STUDENT_ID[-2])          # second-to-last digit -> 9
MAZE_SIZE  = 15 + (BASE_SEED % 4)         # 15 + (9 % 4) = 16

# Design parameters (documented choices - see report)
WALL_RATIO      = 0.18   # target wall density (spec requires >= 0.15)
NUM_PENALTY     = 7      # spec requires >= 5 penalty cells
INITIAL_ENERGY  = 50     # limited-energy feature: starting budget
MAX_ATTEMPTS    = 500    # deterministic regeneration limit

MAPS_DIR = Path(__file__).resolve().parent / "maps"

ACTIONS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right


# --------------------------------------------------------------------------
# BFS utilities (used ONLY for map validation, never as the learning agent)
# --------------------------------------------------------------------------
def bfs_distance(grid, source, target, door_passable):
    """Shortest path length between two cells with 4-connectivity.

    Walls always block. The door blocks iff door_passable is False.
    Returns the number of steps, or None if target is unreachable.
    """
    n = len(grid)
    blocked = {WALL} if door_passable else {WALL, DOOR}
    if grid[source[0]][source[1]] in blocked:
        return None
    queue, seen = deque([(source, 0)]), {source}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == target:
            return dist
        for dr, dc in ACTIONS_4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n
                    and (nr, nc) not in seen
                    and grid[nr][nc] not in blocked):
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return None


# --------------------------------------------------------------------------
# Candidate map construction
# --------------------------------------------------------------------------
def _build_goal_room(grid, n):
    """Place the goal in a 1-cell room whose only opening is the door.

    Goal sits near the bottom-right corner. All 8 surrounding cells become
    walls except the top neighbour, which becomes the door. The cell just
    outside the door is reserved as FREE so the entrance is never sealed.
    Returns (goal, door, reserved_cells).
    """
    goal = (n - 2, n - 2)
    door = (n - 3, n - 2)              # directly above the goal
    door_outside = (n - 4, n - 2)      # must stay walkable

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            r, c = goal[0] + dr, goal[1] + dc
            if 0 <= r < n and 0 <= c < n:
                grid[r][c] = WALL
    grid[door[0]][door[1]] = DOOR
    grid[goal[0]][goal[1]] = GOAL
    grid[door_outside[0]][door_outside[1]] = FREE

    reserved = {goal, door, door_outside}
    reserved.update((goal[0] + dr, goal[1] + dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    return goal, door, reserved


def _generate_candidate(rng, n):
    """Build one candidate map (may still fail BFS validation)."""
    grid = [[FREE] * n for _ in range(n)]

    goal, door, reserved = _build_goal_room(grid, n)

    # --- start: fixed to the top-left corner, far from the goal room -----
    start = (0, 0)
    grid[start[0]][start[1]] = START
    reserved.add(start)
    # keep the two cells next to the start open so it is never boxed in
    for dr, dc in ACTIONS_4:
        r, c = start[0] + dr, start[1] + dc
        if 0 <= r < n and 0 <= c < n and grid[r][c] == FREE:
            reserved.add((r, c))

    # --- key: random free cell, reasonably far from start and goal -------
    candidates = [(r, c) for r in range(n) for c in range(n)
                  if grid[r][c] == FREE and (r, c) not in reserved
                  and abs(r - start[0]) + abs(c - start[1]) >= n // 2
                  and abs(r - goal[0]) + abs(c - goal[1]) >= 4]
    key = rng.choice(candidates)
    grid[key[0]][key[1]] = KEY
    reserved.add(key)

    # --- random walls up to the target density ---------------------------
    target_walls = max(math.ceil(0.15 * n * n) + 2, round(WALL_RATIO * n * n))
    current_walls = sum(row.count(WALL) for row in grid)
    free_cells = [(r, c) for r in range(n) for c in range(n)
                  if grid[r][c] == FREE and (r, c) not in reserved]
    rng.shuffle(free_cells)
    for r, c in free_cells:
        if current_walls >= target_walls:
            break
        grid[r][c] = WALL
        current_walls += 1

    # --- penalty cells ----------------------------------------------------
    free_cells = [(r, c) for r in range(n) for c in range(n)
                  if grid[r][c] == FREE and (r, c) not in reserved]
    for r, c in rng.sample(free_cells, NUM_PENALTY):
        grid[r][c] = PENALTY

    return grid, start, key, door, goal


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_map(grid, start, key, goal):
    """Run all BFS checks. Returns (is_valid, info_dict)."""
    d_start_key = bfs_distance(grid, start, key, door_passable=False)
    d_key_goal  = bfs_distance(grid, key, goal, door_passable=True)
    d_no_key    = bfs_distance(grid, start, goal, door_passable=False)

    info = {
        "dist_start_to_key": d_start_key,
        "dist_key_to_goal": d_key_goal,
        "goal_reachable_without_key": d_no_key is not None,
    }
    is_valid = (d_start_key is not None
                and d_key_goal is not None
                and d_no_key is None)
    return is_valid, info


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def generate_valid_map(base_seed=BASE_SEED, n=MAZE_SIZE,
                       initial_energy=INITIAL_ENERGY, verbose=True):
    """Deterministically generate and validate a map.

    Regeneration is reproducible: attempt i uses seed
    base_seed * 100_000 + i, so re-running always yields the same map.
    """
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(base_seed * 100_000 + attempt)
        grid, start, key, door, goal = _generate_candidate(rng, n)
        is_valid, info = validate_map(grid, start, key, goal)
        if not is_valid:
            continue

        shortest_total = info["dist_start_to_key"] + info["dist_key_to_goal"]
        energy_warning = shortest_total > 0.7 * initial_energy
        if verbose:
            print(f"[generator] valid map found on attempt {attempt}")
            print(f"[generator] shortest path start->key: "
                  f"{info['dist_start_to_key']} steps, "
                  f"key->goal: {info['dist_key_to_goal']} steps "
                  f"(total {shortest_total})")
            if energy_warning:
                print(f"[generator] WARNING: shortest total path "
                      f"({shortest_total}) exceeds 70% of the energy budget "
                      f"({initial_energy}). With slip noise (0.8/0.1/0.1) the "
                      f"agent may run out of energy too often - consider "
                      f"increasing INITIAL_ENERGY.")

        walls = sum(row.count(WALL) for row in grid)
        return {
            "student_id": STUDENT_ID,
            "base_seed": base_seed,
            "attempt": attempt,
            "size": n,
            "grid": grid,
            "start": list(start),
            "key": list(key),
            "door": list(door),
            "goal": list(goal),
            "num_walls": walls,
            "wall_ratio": round(walls / (n * n), 4),
            "num_penalty": NUM_PENALTY,
            "initial_energy": initial_energy,
            "shortest_start_to_key": info["dist_start_to_key"],
            "shortest_key_to_goal": info["dist_key_to_goal"],
            "energy_feasibility_warning": energy_warning,
        }
    raise RuntimeError(f"No valid map found in {MAX_ATTEMPTS} attempts "
                       f"for base_seed={base_seed}.")


def save_map(map_data, filename="source_map.json"):
    """Persist the map to environments/maps/ so every algorithm uses it."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    path = MAPS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, indent=2)
    return path


def load_map(filename="source_map.json"):
    """Load a previously generated map (used by maze.py and experiments)."""
    with open(MAPS_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)
    for k in ("start", "key", "door", "goal"):
        data[k] = tuple(data[k])
    return data


def render_ascii(grid):
    """Human-readable preview of the map (also handy for the report)."""
    return "\n".join(" ".join(CELL_CHARS[cell] for cell in row)
                     for row in grid)


# --------------------------------------------------------------------------
# Script entry point:  python environments/generator.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    map_data = generate_valid_map()
    path = save_map(map_data)

    print(f"[generator] map saved to: {path}")
    print(f"[generator] size: {map_data['size']}x{map_data['size']}, "
          f"walls: {map_data['num_walls']} "
          f"({map_data['wall_ratio']*100:.1f}%), "
          f"penalty cells: {map_data['num_penalty']}, "
          f"initial energy: {map_data['initial_energy']}")
    print(f"[generator] start={tuple(map_data['start'])}, "
          f"key={tuple(map_data['key'])}, "
          f"door={tuple(map_data['door'])}, "
          f"goal={tuple(map_data['goal'])}\n")
    print(render_ascii(map_data["grid"]))