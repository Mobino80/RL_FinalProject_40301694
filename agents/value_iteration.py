"""
agents/value_iteration.py
-------------------------
Value Iteration (model-based) for the dynamic maze MDP - phase 2.

RL Final Project - Student ID: 40301694

Implemented from scratch (no ready-made RL libraries, per project rules).
The agent receives the FULL model of the environment through
MazeEnv.transitions(state, action), which encodes the 0.8 / 0.1 / 0.1
transition stochasticity, and MazeEnv.all_states().

Bellman optimality update:
    V_{k+1}(s) = max_a  sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V_k(s') ]

Convergence criterion (per spec): the maximum absolute change of the
value function over two consecutive sweeps drops below `theta`.

The resulting greedy policy is the REFERENCE policy against which
Q-Learning and SARSA(lambda) will be compared in later phases.

Note on structure: every transition consumes exactly 1 unit of energy,
so the MDP is acyclic along the energy dimension. This gives Value
Iteration a finite-horizon flavour and guarantees fast convergence -
worth discussing in the report's convergence analysis.
"""

import pickle
import time
from pathlib import Path

try:
    from environments.maze import ACTIONS
except ImportError:  # allows running from inside agents/
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from environments.maze import ACTIONS


class ValueIterationAgent:
    """Tabular Value Iteration over the (row, col, key, energy) state space."""

    def __init__(self, env, gamma=0.99, theta=1e-6, max_sweeps=500):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_sweeps = max_sweeps

        self.states = list(env.all_states())
        self.V = {s: 0.0 for s in self.states}   # terminal states -> 0
        self.policy = {}
        self.history = {"deltas": [], "sweeps": 0, "runtime_sec": None,
                        "gamma": gamma, "theta": theta,
                        "num_states": len(self.states)}

    # ------------------------------------------------------------------ #
    # Core algorithm                                                       #
    # ------------------------------------------------------------------ #
    def action_value(self, state, action, V=None):
        """Q(s,a) under the current value estimate (one Bellman backup)."""
        V = V if V is not None else self.V
        q = 0.0
        for prob, next_state, reward, done in self.env.transitions(state,
                                                                   action):
            bootstrap = 0.0 if done else V.get(next_state, 0.0)
            q += prob * (reward + self.gamma * bootstrap)
        return q

    def run(self, verbose=True):
        """Sweep until max|V_{k+1} - V_k| < theta. Returns the history."""
        t0 = time.perf_counter()
        for sweep in range(1, self.max_sweeps + 1):
            delta = 0.0
            new_V = {}
            for s in self.states:
                best = max(self.action_value(s, a) for a in ACTIONS)
                new_V[s] = best
                diff = abs(best - self.V[s])
                if diff > delta:
                    delta = diff
            self.V = new_V
            self.history["deltas"].append(delta)
            if verbose and (sweep <= 5 or sweep % 10 == 0):
                print(f"[VI gamma={self.gamma}] sweep {sweep:3d}  "
                      f"max delta = {delta:.6e}")
            if delta < self.theta:
                break
        self.history["sweeps"] = sweep
        self.history["runtime_sec"] = round(time.perf_counter() - t0, 3)
        self.extract_policy()
        if verbose:
            print(f"[VI gamma={self.gamma}] converged in {sweep} sweeps, "
                  f"{self.history['runtime_sec']} s "
                  f"(final delta {delta:.2e})")
        return self.history

    def extract_policy(self):
        """Greedy policy: pi(s) = argmax_a Q(s,a)."""
        self.policy = {}
        for s in self.states:
            q_values = [self.action_value(s, a) for a in ACTIONS]
            self.policy[s] = int(max(range(len(ACTIONS)),
                                     key=lambda i: q_values[i]))
        return self.policy

    def greedy_action(self, state):
        return self.policy[state]

    # ------------------------------------------------------------------ #
    # Persistence (models go to results/models/ per project structure)    #
    # ------------------------------------------------------------------ #
    def save(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"V": self.V, "policy": self.policy,
                         "history": self.history}, f)
        return path

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)