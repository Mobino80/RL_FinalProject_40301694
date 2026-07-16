"""
tests/test_environment.py
-------------------------
Unit tests for phase 1: the map generator and the maze environment.

Run from the project root with:
    pytest tests/ -v

These tests cover the project-spec requirements:
  * seed / size derived from the student ID,
  * wall and penalty-cell quotas,
  * BFS validity (start->key, key->goal, goal gated by the door),
  * full reproducibility of map generation,
  * correctness of the stochastic transition model (probs sum to 1),
  * limited-energy feature (energy in state, depletion terminates),
  * key / locked-door / goal mechanics and their rewards,
  * Markov property (transitions depend only on (s, a)),
  * potential-based shaping consistency (shaped = sparse + F),
  * step-cap truncation.
"""

import sys
from pathlib import Path

# make the project root importable regardless of where pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from environments.generator import (BASE_SEED, MAZE_SIZE, STUDENT_ID,
                                    WALL, PENALTY, generate_valid_map,
                                    bfs_distance)
from environments.maze import (MazeEnv, ACTIONS, UP, DOWN, LEFT, RIGHT,
                               EV_KEY_PICKUP, EV_DOOR_LOCKED, EV_GOAL,
                               EV_ENERGY_OUT, EV_WALL_HIT, EV_STEP_LIMIT,
                               DEFAULT_REWARDS)


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def map_data():
    return generate_valid_map(verbose=False)


@pytest.fixture()
def env(map_data):
    return MazeEnv(map_data=map_data, reward_mode="sparse", seed=123)


# --------------------------------------------------------------------------
# Generator tests
# --------------------------------------------------------------------------
def test_seed_and_size_from_student_id():
    assert STUDENT_ID == "40301694"
    assert BASE_SEED == 9                      # second-to-last digit
    assert MAZE_SIZE == 15 + (9 % 4) == 16     # spec formula


def test_wall_and_penalty_quotas(map_data):
    n = map_data["size"]
    grid = map_data["grid"]
    walls = sum(row.count(WALL) for row in grid)
    penalties = sum(row.count(PENALTY) for row in grid)
    assert walls / (n * n) >= 0.15             # >= 15% obstacles
    assert penalties >= 5                      # >= 5 penalty cells


def test_bfs_validity(map_data):
    grid = map_data["grid"]
    start, key = tuple(map_data["start"]), tuple(map_data["key"])
    goal = tuple(map_data["goal"])
    # start -> key must be reachable while the door is locked
    assert bfs_distance(grid, start, key, door_passable=False) is not None
    # key -> goal must be reachable once the door opens
    assert bfs_distance(grid, key, goal, door_passable=True) is not None
    # the goal must NOT be reachable without the key (door is meaningful)
    assert bfs_distance(grid, start, goal, door_passable=False) is None


def test_generation_is_reproducible():
    m1 = generate_valid_map(verbose=False)
    m2 = generate_valid_map(verbose=False)
    assert m1["grid"] == m2["grid"]
    assert m1["start"] == m2["start"] and m1["key"] == m2["key"]
    assert m1["attempt"] == m2["attempt"]


# --------------------------------------------------------------------------
# Environment: basic dynamics
# --------------------------------------------------------------------------
def test_reset_initial_state(env, map_data):
    s = env.reset()
    assert s == (*tuple(map_data["start"]), 0, map_data["initial_energy"])


def test_transition_probabilities_sum_to_one(env):
    checked = 0
    for state in env.all_states():
        for a in ACTIONS:
            outcomes = env.transitions(state, a)
            assert abs(sum(p for p, *_ in outcomes) - 1.0) < 1e-9
            assert all(0 < p <= 1 for p, *_ in outcomes)
        checked += 1
        if checked >= 200:                     # sampling is enough
            break


def test_energy_decreases_every_step_including_wall_hits(env):
    s = env.reset()
    energy = s[3]
    for _ in range(10):
        s, _, done, _ = env.step(env.rng.choice(ACTIONS))
        energy -= 1
        assert s[3] == energy                  # exactly -1 per step
        if done:
            break


def test_wall_hit_keeps_position_and_penalizes(env):
    # start=(0,0): moving UP deterministically bumps the border
    s = (0, 0, 0, 50)
    ns, r, done, event = env._transition(s, UP)
    assert event == EV_WALL_HIT
    assert (ns[0], ns[1]) == (0, 0)            # stayed in place
    assert ns[3] == 49                         # energy still consumed
    assert r == DEFAULT_REWARDS["step"] + DEFAULT_REWARDS["wall"]
    assert not done


# --------------------------------------------------------------------------
# Environment: key / door / goal mechanics
# --------------------------------------------------------------------------
def _adjacent_free_state(env, target, k, energy=40):
    """Build a state standing right next to `target` plus the action
    that moves onto it (used to force key/door/goal transitions)."""
    for a, (dr, dc) in ((UP, (-1, 0)), (DOWN, (1, 0)),
                        (LEFT, (0, -1)), (RIGHT, (0, 1))):
        r, c = target[0] - dr, target[1] - dc
        if (0 <= r < env.n and 0 <= c < env.n
                and env.grid[r][c] not in (WALL,)
                and (r, c) != env.door_pos
                and (r, c) != env.goal_pos):   # goal is terminal - invalid
            return (r, c, k, energy), a
    raise RuntimeError("no free neighbour found")


def test_key_pickup_sets_flag_and_rewards(env):
    s, a = _adjacent_free_state(env, env.key_pos, k=0)
    ns, r, done, event = env._transition(s, a)
    assert event == EV_KEY_PICKUP
    assert ns[2] == 1                          # k: 0 -> 1
    assert r >= DEFAULT_REWARDS["key"] + DEFAULT_REWARDS["step"]
    assert not done


def test_locked_door_blocks_without_key(env):
    s, a = _adjacent_free_state(env, env.door_pos, k=0)
    ns, r, done, event = env._transition(s, a)
    assert event == EV_DOOR_LOCKED
    assert (ns[0], ns[1]) == (s[0], s[1])      # bounced back
    assert not done


def test_door_passable_with_key(env):
    s, a = _adjacent_free_state(env, env.door_pos, k=1)
    ns, _, done, _ = env._transition(s, a)
    assert (ns[0], ns[1]) == env.door_pos      # passed through
    assert not done


def test_goal_is_terminal_with_reward(env):
    # the only neighbour of the goal inside the room is the door
    s = (*env.door_pos, 1, 40)
    dr = env.goal_pos[0] - env.door_pos[0]
    dc = env.goal_pos[1] - env.door_pos[1]
    a = {(-1, 0): UP, (1, 0): DOWN, (0, -1): LEFT, (0, 1): RIGHT}[(dr, dc)]
    ns, r, done, event = env._transition(s, a)
    assert event == EV_GOAL and done
    assert (ns[0], ns[1]) == env.goal_pos
    assert r >= DEFAULT_REWARDS["goal"] + DEFAULT_REWARDS["step"]


def test_energy_depletion_terminates(env):
    s = (0, 0, 0, 1)                           # last unit of energy
    ns, r, done, event = env._transition(s, DOWN)
    assert event == EV_ENERGY_OUT and done
    assert ns[3] == 0


# --------------------------------------------------------------------------
# Markov property, shaping consistency, truncation, reproducibility
# --------------------------------------------------------------------------
def test_markov_property_model_depends_only_on_state_action(env):
    """Calling the model twice (any history) yields identical outcomes."""
    state = (5, 5, 0, 30)
    first = sorted(env.transitions(state, RIGHT))
    env.reset()
    for _ in range(5):                         # create some 'history'
        env.step(env.rng.choice(ACTIONS))
    second = sorted(env.transitions(state, RIGHT))
    assert first == second


def test_shaped_equals_sparse_plus_potential_term(map_data):
    sparse = MazeEnv(map_data=map_data, reward_mode="sparse", seed=1)
    shaped = MazeEnv(map_data=map_data, reward_mode="shaped", seed=1)
    s = (0, 0, 0, 50)
    for a in ACTIONS:
        ns_sp, r_sp, done, _ = sparse._transition(s, a)
        ns_sh, r_sh, _, _ = shaped._transition(s, a)
        assert ns_sp == ns_sh                  # dynamics identical
        expected_f = shaped.shaping_coef * shaped._shaping(s, ns_sp, done)
        assert r_sh == pytest.approx(r_sp + expected_f)


def test_step_limit_truncates(map_data):
    env = MazeEnv(map_data=map_data, reward_mode="sparse",
                  max_steps=3, seed=7)
    env.reset()
    done, info = False, {}
    for _ in range(3):
        _, _, done, info = env.step(DOWN)
        if done:
            break
    assert done
    assert info["event"] in (EV_STEP_LIMIT, EV_ENERGY_OUT, EV_GOAL)
    assert info["truncated"] or info["event"] != EV_STEP_LIMIT


def test_episode_reproducible_with_same_seed(map_data):
    def rollout(seed):
        e = MazeEnv(map_data=map_data, reward_mode="sparse", seed=seed)
        s = e.reset(seed=seed)
        traj = []
        for _ in range(30):
            s, r, done, info = e.step(ACTIONS[len(traj) % 4])
            traj.append((s, round(r, 6), info["event"]))
            if done:
                break
        return traj
    assert rollout(42) == rollout(42)          # identical trajectories
    assert rollout(42) != rollout(43)          # seed actually matters