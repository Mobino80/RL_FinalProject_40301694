"""
transfer/transfer_learning.py
-----------------------------
Transfer learning for Q-Learning - phase 5.

RL Final Project - Student ID: 40301694

Per the project spec, transfer is performed ONLY on Q-Learning.

This module provides:

  1. Target-environment generation
       * "similar"   : 15-20% of the obstacles are relocated;
                       start, key and goal stay fixed.
       * "different" : at least 35% of the obstacles change, the key
                       (or the goal) moves, and new penalty cells are
                       added.
     Both maps are validated with BFS exactly like the source map
     (start -> key reachable, key -> goal reachable, goal NOT reachable
     without the key) and are generated from a deterministic seed so the
     whole experiment is reproducible.

  2. The four training scenarios required by the spec
       (1) scratch    : zero-initialised Q table (baseline)
       (2) full       : Q_T = Q_S
       (3) scaled     : Q_T(s,a) = beta * Q_S(s,a), beta in {0.25,0.5,0.75}
       (4) selective  : transfer only those states whose LOCAL
                        NEIGHBOURHOOD is identical in both maps

  3. Transfer metrics that separate the three effects the report must
     distinguish:
       * jumpstart        - greedy performance BEFORE any training on
                            the target (initial performance)
       * learning speed   - area under the success curve + number of
                            episodes needed to reach a success threshold
       * final performance- greedy success/return after training

  4. Automatic detection of NEGATIVE transfer: states where the action
     that was greedy in the source environment now walks into a cell
     that became a wall or a penalty cell in the target environment.
"""

import math
import random
from collections import defaultdict
from pathlib import Path

try:
    from environments.generator import (FREE, WALL, PENALTY, START, KEY,
                                        DOOR, GOAL, bfs_distance,
                                        validate_map, load_map, save_map)
    from environments.maze import ACTIONS, ACTION_DELTAS, ACTION_NAMES
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from environments.generator import (FREE, WALL, PENALTY, START, KEY,
                                        DOOR, GOAL, bfs_distance,
                                        validate_map, load_map, save_map)
    from environments.maze import ACTIONS, ACTION_DELTAS, ACTION_NAMES

MAX_MAP_ATTEMPTS = 800


# ===========================================================================
# 1. Target-environment generation
# ===========================================================================
def _movable_wall_cells(grid, protected):
    """Walls that may be relocated (i.e. not part of the goal room)."""
    n = len(grid)
    return [(r, c) for r in range(n) for c in range(n)
            if grid[r][c] == WALL and (r, c) not in protected]


def _protected_cells(map_data):
    """Cells that must never change: the goal room and its entrance."""
    goal = tuple(map_data["goal"])
    door = tuple(map_data["door"])
    n = map_data["size"]
    prot = {goal, door, (door[0] - 1, door[1])}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = goal[0] + dr, goal[1] + dc
            if 0 <= r < n and 0 <= c < n:
                prot.add((r, c))
    return prot


def make_target_map(source_map, kind="similar", seed=0, verbose=True):
    """Build and BFS-validate a target map derived from the source map.

    kind = "similar"   -> 15-20% of obstacles moved, landmarks fixed
    kind = "different" -> >=35% of obstacles changed, key moved,
                          extra penalty cells added
    """
    assert kind in ("similar", "different")
    n = source_map["size"]
    protected = _protected_cells(source_map)

    for attempt in range(MAX_MAP_ATTEMPTS):
        rng = random.Random(seed * 100_000 + attempt)
        grid = [row[:] for row in source_map["grid"]]
        start = tuple(source_map["start"])
        key = tuple(source_map["key"])
        door = tuple(source_map["door"])
        goal = tuple(source_map["goal"])

        movable = _movable_wall_cells(grid, protected)
        frac = rng.uniform(0.15, 0.20) if kind == "similar" \
            else rng.uniform(0.35, 0.45)
        n_move = max(1, int(round(frac * len(movable))))
        to_remove = rng.sample(movable, n_move)

        # free the selected walls
        for (r, c) in to_remove:
            grid[r][c] = FREE

        # for the "different" map: move the key first, so the new wall
        # placement can not accidentally seal it off
        if kind == "different":
            candidates = [
                (r, c) for r in range(n) for c in range(n)
                if grid[r][c] == FREE and (r, c) not in protected
                and (r, c) != start
                and abs(r - start[0]) + abs(c - start[1]) >= n // 2
                and abs(r - goal[0]) + abs(c - goal[1]) >= 4]
            if not candidates:
                continue
            new_key = rng.choice(candidates)
            grid[key[0]][key[1]] = FREE
            grid[new_key[0]][new_key[1]] = KEY
            key = new_key

        # place the same number of walls in NEW positions
        free_pool = [(r, c) for r in range(n) for c in range(n)
                     if grid[r][c] == FREE and (r, c) not in protected
                     and (r, c) not in (start, key, door, goal)]
        rng.shuffle(free_pool)
        placed = 0
        for (r, c) in free_pool:
            if placed >= n_move:
                break
            grid[r][c] = WALL
            placed += 1

        # the "different" map also gains new penalty cells
        n_new_penalty = 0
        if kind == "different":
            free_pool = [(r, c) for r in range(n) for c in range(n)
                         if grid[r][c] == FREE and (r, c) not in protected
                         and (r, c) not in (start, key, door, goal)]
            n_new_penalty = 4
            for (r, c) in rng.sample(free_pool,
                                     min(n_new_penalty, len(free_pool))):
                grid[r][c] = PENALTY

        grid[start[0]][start[1]] = START
        grid[key[0]][key[1]] = KEY

        is_valid, info = validate_map(grid, start, key, goal)
        if not is_valid:
            continue

        changed = sum(1 for r in range(n) for c in range(n)
                      if grid[r][c] != source_map["grid"][r][c])
        walls = sum(row.count(WALL) for row in grid)
        target = {
            "student_id": source_map["student_id"],
            "derived_from": "source_map.json",
            "kind": kind,
            "base_seed": seed,
            "attempt": attempt,
            "size": n,
            "grid": grid,
            "start": list(start), "key": list(key),
            "door": list(door), "goal": list(goal),
            "num_walls": walls,
            "wall_ratio": round(walls / (n * n), 4),
            "num_penalty": sum(row.count(PENALTY) for row in grid),
            "obstacles_relocated": n_move,
            "obstacle_change_fraction": round(n_move / len(movable), 4),
            "new_penalty_cells": n_new_penalty,
            "key_moved": tuple(source_map["key"]) != key,
            "cells_changed": changed,
            "initial_energy": source_map["initial_energy"],
            "shortest_start_to_key": info["dist_start_to_key"],
            "shortest_key_to_goal": info["dist_key_to_goal"],
        }
        if verbose:
            print(f"[transfer] '{kind}' target map ready "
                  f"(attempt {attempt}): {n_move} obstacles relocated "
                  f"({target['obstacle_change_fraction']:.0%}), "
                  f"key_moved={target['key_moved']}, "
                  f"cells changed={changed}, "
                  f"path {info['dist_start_to_key']}+"
                  f"{info['dist_key_to_goal']} steps")
        return target

    raise RuntimeError(f"could not build a valid '{kind}' target map")


# ===========================================================================
# 2. The four transfer scenarios
# ===========================================================================
def neighbourhood_unchanged(source_grid, target_grid, r, c, radius=1):
    """True if the local patch around (r, c) is identical in both maps."""
    n = len(source_grid)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < n and 0 <= cc < n):
                continue
            if source_grid[rr][cc] != target_grid[rr][cc]:
                return False
    return True


def build_initial_q(source_Q, scenario, source_map=None, target_map=None,
                    beta=1.0, radius=1):
    """Create the initial Q table of the target agent.

    scenario:
      "scratch"   -> empty table (all zeros by default)
      "full"      -> Q_T = Q_S
      "scaled"    -> Q_T = beta * Q_S
      "selective" -> Q_T = Q_S only where the local neighbourhood of the
                     state's cell is identical in both maps
    Returns (initial_Q_dict, stats_dict).
    """
    assert scenario in ("scratch", "full", "scaled", "selective")
    if scenario == "scratch":
        return {}, {"transferred_states": 0, "source_states": len(source_Q),
                    "transfer_ratio": 0.0}

    if scenario == "selective":
        assert source_map is not None and target_map is not None
        n = source_map["size"]
        keep = {(r, c) for r in range(n) for c in range(n)
                if neighbourhood_unchanged(source_map["grid"],
                                           target_map["grid"], r, c,
                                           radius)}
    init = {}
    for state, q_values in source_Q.items():
        r, c = state[0], state[1]
        if scenario == "full":
            init[state] = list(q_values)
        elif scenario == "scaled":
            init[state] = [beta * q for q in q_values]
        else:                                   # selective
            if (r, c) in keep:
                init[state] = list(q_values)

    stats = {"transferred_states": len(init),
             "source_states": len(source_Q),
             "transfer_ratio": round(len(init) / max(1, len(source_Q)), 4)}
    if scenario == "selective":
        stats["unchanged_cells"] = len(keep)
        stats["grid_cells"] = source_map["size"] ** 2
    return init, stats


def seed_agent_with_q(agent, initial_Q):
    """Load an initial Q table into a fresh QLearningAgent."""
    agent.Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    for state, values in initial_Q.items():
        agent.Q[state] = list(values)
    return agent


# ===========================================================================
# 3. Transfer metrics
# ===========================================================================
def learning_speed_metrics(metrics, threshold=0.30, window=1000):
    """Area under the success curve + episodes needed to reach
    `threshold` success (moving average over `window` episodes)."""
    successes = [m["success"] for m in metrics]
    auc = sum(successes) / len(successes)        # mean success over run
    episodes_to_threshold = None
    running, ma = 0, []
    for i, s in enumerate(successes):
        running += s
        if i >= window:
            running -= successes[i - window]
            ma_val = running / window
            if episodes_to_threshold is None and ma_val >= threshold:
                episodes_to_threshold = i
    return {"success_auc": round(auc, 4),
            "episodes_to_threshold": episodes_to_threshold,
            "threshold": threshold}


# ===========================================================================
# 4. Negative-transfer detection
# ===========================================================================
def find_negative_transfer(source_Q, source_map, target_map,
                           initial_energy=None, max_examples=10):
    """Find states where the transferred knowledge misleads the agent.

    A state (r, c, k, e) is flagged when the action that is greedy under
    the SOURCE Q table moves the agent into a cell that
      * was walkable in the source map, but
      * is a WALL or a PENALTY cell in the target map.
    In other words, the inherited preference points straight at a newly
    introduced obstacle or trap - a textbook case of negative transfer.

    Returns a list of dicts with the Q values, the structural change and
    the neighbouring cell types, ready to be quoted in the report.
    """
    n = source_map["size"]
    src_grid, tgt_grid = source_map["grid"], target_map["grid"]
    examples = []
    for state, q_values in source_Q.items():
        r, c, k, e = state
        if initial_energy is not None and e != initial_energy:
            continue                     # keep the table small/readable
        if src_grid[r][c] == WALL or tgt_grid[r][c] == WALL:
            continue
        best_a = max(ACTIONS, key=lambda a: q_values[a])
        dr, dc = ACTION_DELTAS[best_a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        was_ok = src_grid[nr][nc] not in (WALL, PENALTY)
        now_bad = tgt_grid[nr][nc] in (WALL, PENALTY)
        if was_ok and now_bad:
            q_sorted = sorted(range(len(ACTIONS)),
                              key=lambda a: -q_values[a])
            examples.append({
                "state": state,
                "cell": (r, c), "key_flag": k, "energy": e,
                "greedy_action_source": ACTION_NAMES[best_a],
                "target_cell": (nr, nc),
                "source_cell_type": src_grid[nr][nc],
                "target_cell_type": tgt_grid[nr][nc],
                "change": ("became WALL"
                           if tgt_grid[nr][nc] == WALL
                           else "became PENALTY"),
                "q_values_source": {ACTION_NAMES[a]: round(q_values[a], 4)
                                    for a in ACTIONS},
                "q_gap_best_vs_second":
                    round(q_values[q_sorted[0]] - q_values[q_sorted[1]], 4),
            })
    # the most harmful examples are the ones with the largest confidence
    examples.sort(key=lambda x: -x["q_gap_best_vs_second"])
    return examples[:max_examples]


def q_values_after_training(agent, state):
    """Q values of a state after the target-environment training,
    used to show HOW the agent corrected the misleading knowledge."""
    if state not in agent.Q:
        return None
    return {ACTION_NAMES[a]: round(agent.Q[state][a], 4) for a in ACTIONS}