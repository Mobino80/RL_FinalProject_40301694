"""
agents/q_learning.py
--------------------
Tabular Q-Learning (off-policy, model-free) for the maze MDP - phase 3.

RL Final Project - Student ID: 40301694

Update rule (implemented from scratch, per project rules):
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Behaviour policy: epsilon-greedy with a DECAYING epsilon. Two schedules
are provided (linear and exponential), as required by the spec.

The agent interacts with the environment ONLY through reset()/step();
it never touches env.transitions() - that model access is reserved for
Value Iteration. This is exactly the model-free / model-based split the
report must discuss.

Q-table initialisation: zeros. Because every step reward is negative,
zero is an OPTIMISTIC initial value, which itself encourages visiting
untried actions (optimism in the face of uncertainty).

Logging (three tiers, to keep files reproducible but small):
  1. per-episode metrics CSV  - ALL episodes: reward, steps, success,
     wall hits, penalty entries, key, epsilon, final event.
  2. step-level event CSV     - only for sampled episodes (MazeLogger).
  3. Q-update CSV             - full TD decomposition (Q_before, r,
     max_a' Q(s'), TD target, TD error, Q_after, alpha, gamma) for
     designated episodes -> used to reconstruct one real update BY HAND
     in the report, as the spec demands.
"""

import csv
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

try:
    from environments.maze import (ACTIONS, ACTION_NAMES, MazeLogger,
                                   EV_GOAL, EV_WALL_HIT, EV_PENALTY_CELL,
                                   EV_KEY_PICKUP)
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from environments.maze import (ACTIONS, ACTION_NAMES, MazeLogger,
                                   EV_GOAL, EV_WALL_HIT, EV_PENALTY_CELL,
                                   EV_KEY_PICKUP)


# ---------------------------------------------------------------------------
# Epsilon decay schedules (spec: implement and compare at least two)
# ---------------------------------------------------------------------------
class LinearDecay:
    """eps(ep) = start - (start-end) * ep / decay_episodes, floored at end."""

    name = "linear"

    def __init__(self, start=1.0, end=0.05, decay_fraction=0.8,
                 total_episodes=1):
        self.start, self.end = start, end
        self.decay_episodes = max(1, int(decay_fraction * total_episodes))

    def __call__(self, episode):
        frac = min(1.0, episode / self.decay_episodes)
        return self.start - (self.start - self.end) * frac


class ExponentialDecay:
    """eps(ep) = start * rate**ep, floored at end.

    `rate` is chosen so that eps reaches `end` after
    decay_fraction * total_episodes episodes - this makes the linear and
    exponential schedules directly comparable (same start, same floor,
    same time-to-floor; only the SHAPE of the curve differs).
    """

    name = "exponential"

    def __init__(self, start=1.0, end=0.05, decay_fraction=0.8,
                 total_episodes=1):
        self.start, self.end = start, end
        horizon = max(1, int(decay_fraction * total_episodes))
        self.rate = (end / start) ** (1.0 / horizon)

    def __call__(self, episode):
        return max(self.end, self.start * (self.rate ** episode))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class QLearningAgent:

    def __init__(self, env, alpha=0.15, gamma=0.99, epsilon_schedule=None,
                 seed=0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps_schedule = epsilon_schedule
        self.rng = random.Random(seed)
        self.Q = defaultdict(lambda: [0.0] * len(ACTIONS))
        self.visit_counts = defaultdict(int)     # N(s), for visit maps
        self.train_stats = None

    # ------------------------------------------------------------------ #
    # Policy                                                              #
    # ------------------------------------------------------------------ #
    def choose_action(self, state, epsilon):
        """Epsilon-greedy over Q(state, .) with random tie-breaking."""
        if self.rng.random() < epsilon:
            return self.rng.choice(ACTIONS)
        return self.greedy_action(state)

    def greedy_action(self, state):
        q = self.Q[state]
        best = max(q)
        best_actions = [a for a in ACTIONS if q[a] == best]
        return self.rng.choice(best_actions)

    def greedy_policy(self):
        """dict state -> best action, ONLY over visited states."""
        return {s: max(ACTIONS, key=lambda a: self.Q[s][a])
                for s in list(self.Q.keys())}

    # ------------------------------------------------------------------ #
    # Learning                                                            #
    # ------------------------------------------------------------------ #
    def update(self, s, a, r, s_next, done):
        """One off-policy TD(0) update; returns the full decomposition
        so it can be logged and later reconstructed by hand."""
        q_before = self.Q[s][a]
        max_next = 0.0 if done else max(self.Q[s_next])
        td_target = r + self.gamma * max_next
        td_error = td_target - q_before
        self.Q[s][a] = q_before + self.alpha * td_error
        return q_before, max_next, td_target, td_error, self.Q[s][a]

    def train(self, episodes, run_name, raw_dir,
              step_log_episodes=None, qupdate_log_episodes=None,
              base_seed=0, verbose_every=1000):
        """Full training loop with three-tier logging. Returns metrics."""
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        step_log_episodes = step_log_episodes or set()
        qupdate_log_episodes = qupdate_log_episodes or set()

        # tier 1: per-episode metrics (all episodes)
        ep_file = open(raw_dir / f"{run_name}_episodes.csv", "w",
                       newline="")
        ep_writer = csv.writer(ep_file)
        ep_writer.writerow(["episode", "epsilon", "total_reward", "steps",
                            "success", "wall_hits", "penalty_entries",
                            "key_obtained", "energy_left", "end_event"])

        # tier 2: sampled step-level event log
        step_logger = MazeLogger(raw_dir / f"{run_name}_steps_sampled.csv")

        # tier 3: full Q-update decomposition for designated episodes
        qu_file = open(raw_dir / f"{run_name}_qupdates.csv", "w",
                       newline="")
        qu_writer = csv.writer(qu_file)
        qu_writer.writerow(["episode", "step", "r", "c", "k", "e",
                            "action", "reward", "nr", "nc", "nk", "ne",
                            "done", "q_before", "max_next_q", "td_target",
                            "td_error", "q_after", "alpha", "gamma",
                            "epsilon"])

        metrics = []
        t0 = time.perf_counter()
        for ep in range(episodes):
            eps = self.eps_schedule(ep)
            s = self.env.reset(seed=base_seed * 1_000_000 + ep)
            total_r, steps = 0.0, 0
            wall_hits = penalty_hits = key_got = success = 0
            end_event = None

            while True:
                self.visit_counts[s] += 1
                a = self.choose_action(s, eps)
                s_next, r, done, info = self.env.step(a)
                decomp = self.update(s, a, r, s_next, done)

                if ep in step_log_episodes:
                    step_logger.log(ep, steps, s, a, s_next, r, info,
                                    done)
                if ep in qupdate_log_episodes:
                    q_b, mx, tgt, err, q_a = decomp
                    qu_writer.writerow([ep, steps, *s,
                                        ACTION_NAMES[a], round(r, 4),
                                        *s_next, int(done),
                                        round(q_b, 6), round(mx, 6),
                                        round(tgt, 6), round(err, 6),
                                        round(q_a, 6), self.alpha,
                                        self.gamma, round(eps, 4)])

                ev = info["event"]
                wall_hits += ev == EV_WALL_HIT
                penalty_hits += ev == EV_PENALTY_CELL
                key_got |= ev == EV_KEY_PICKUP
                success |= ev == EV_GOAL
                total_r += r
                steps += 1
                s = s_next
                if done:
                    end_event = ev
                    break

            ep_writer.writerow([ep, round(eps, 4), round(total_r, 3),
                                steps, success, wall_hits, penalty_hits,
                                key_got, s[3], end_event])
            metrics.append({"episode": ep, "epsilon": eps,
                            "total_reward": total_r, "steps": steps,
                            "success": success})
            if verbose_every and (ep + 1) % verbose_every == 0:
                recent = metrics[-verbose_every:]
                sr = sum(m["success"] for m in recent) / len(recent)
                mr = sum(m["total_reward"] for m in recent) / len(recent)
                print(f"[QL {run_name}] ep {ep + 1:5d}/{episodes}  "
                      f"eps={eps:.3f}  success={sr:5.1%}  "
                      f"meanR={mr:7.1f}")

        runtime = round(time.perf_counter() - t0, 2)
        ep_file.close(); qu_file.close(); step_logger.close()
        self.train_stats = {"episodes": episodes, "runtime_sec": runtime,
                            "alpha": self.alpha, "gamma": self.gamma,
                            "schedule": self.eps_schedule.name,
                            "visited_states": len(self.Q)}
        print(f"[QL {run_name}] done in {runtime}s, "
              f"visited {len(self.Q)} states")
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluation (greedy, epsilon = 0)                                    #
    # ------------------------------------------------------------------ #
    def evaluate(self, episodes=200, base_seed=999):
        succ, returns, steps_l = 0, [], []
        for ep in range(episodes):
            s = self.env.reset(seed=base_seed * 1_000_000 + ep)
            total = 0.0
            while True:
                a = self.greedy_action(s)
                s, r, done, info = self.env.step(a)
                total += r
                if done:
                    succ += info["event"] == EV_GOAL
                    steps_l.append(info["steps"])
                    break
            returns.append(total)
        return {"eval_episodes": episodes,
                "success_rate": succ / episodes,
                "mean_return": sum(returns) / episodes,
                "mean_steps": sum(steps_l) / len(steps_l)}

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #
    def save(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"Q": dict(self.Q),
                         "visit_counts": dict(self.visit_counts),
                         "train_stats": self.train_stats}, f)
        return path

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)