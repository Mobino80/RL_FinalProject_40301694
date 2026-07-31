"""
agents/sarsa_lambda.py
----------------------
SARSA(lambda): on-policy TD control with eligibility traces - phase 4.

RL Final Project - Student ID: 40301694

Update equations (implemented from scratch, per project rules):

    delta_t = r_{t+1} + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
    E_t(s,a) = gamma * lambda * E_{t-1}(s,a) + 1{s = s_t, a = a_t}
    Q(s,a) <- Q(s,a) + alpha * delta_t * E_t(s,a)     for all (s,a)

Key difference from Q-Learning: the bootstrap uses Q(s_{t+1}, a_{t+1})
where a_{t+1} is the action the BEHAVIOUR policy actually takes
(epsilon-greedy), not max_a'. That makes the method ON-POLICY: it
evaluates the policy it is really following, including its exploratory
mistakes. Near penalty cells this typically produces more conservative
("safer") behaviour than Q-Learning - the report must show this with
numbers, which is why evaluate() also counts penalty-cell entries.

Trace types (both implemented; `replacing` is the default):
  * accumulating : E(s,a) += 1        -> repeated visits pile up
  * replacing    : E(s,a)  = 1        -> capped at 1

Why replacing is the default here: with limited energy the agent often
bumps into walls and stays in the SAME cell for several consecutive
steps. With accumulating traces that repeated (s,a) pair accumulates a
large eligibility, which inflates its update and increases variance.
Replacing traces bound the eligibility at 1 and behave more stably in
exactly this situation. (Singh & Sutton, 1996.)

lambda = 0  =>  E is non-zero only for the current pair, so the update
reduces EXACTLY to one-step SARSA. Larger lambda propagates the TD
error backwards to states and actions visited earlier in the episode.
This is verified numerically in tests/test_sarsa_lambda.py.

Efficiency note: traces are kept in a sparse dict of "active" pairs and
pruned once they fall below TRACE_THRESHOLD, instead of sweeping the
whole state-action space every step.
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

TRACE_THRESHOLD = 1e-4     # prune traces below this value


class SarsaLambdaAgent:

    def __init__(self, env, alpha=0.2, gamma=0.99, lam=0.9,
                 trace_type="replacing", epsilon_schedule=None, seed=0):
        assert trace_type in ("replacing", "accumulating")
        self.env = env
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.trace_type = trace_type
        self.eps_schedule = epsilon_schedule
        self.rng = random.Random(seed)
        self.Q = defaultdict(lambda: [0.0] * len(ACTIONS))
        self.visit_counts = defaultdict(int)
        self.train_stats = None
        self.max_active_traces = 0     # memory-usage evidence

    # ------------------------------------------------------------------ #
    # Policy                                                              #
    # ------------------------------------------------------------------ #
    def choose_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.choice(ACTIONS)
        return self.greedy_action(state)

    def greedy_action(self, state):
        q = self.Q[state]
        best = max(q)
        return self.rng.choice([a for a in ACTIONS if q[a] == best])

    def greedy_policy(self):
        return {s: max(ACTIONS, key=lambda a: self.Q[s][a])
                for s in list(self.Q.keys())}

    # ------------------------------------------------------------------ #
    # Learning                                                            #
    # ------------------------------------------------------------------ #
    def train(self, episodes, run_name, raw_dir, base_seed=0,
              trace_log_episodes=None, trace_log_max_steps=25,
              step_log_episodes=None, verbose_every=25000):
        """SARSA(lambda) training loop with per-episode and trace logs."""
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        trace_log_episodes = trace_log_episodes or set()
        step_log_episodes = step_log_episodes or set()

        ep_file = open(raw_dir / f"{run_name}_episodes.csv", "w",
                       newline="")
        ep_writer = csv.writer(ep_file)
        ep_writer.writerow(["episode", "epsilon", "total_reward", "steps",
                            "success", "wall_hits", "penalty_entries",
                            "key_obtained", "energy_left", "end_event",
                            "active_traces_max"])

        # dedicated delta / E log (spec: record delta and E over several
        # consecutive steps of at least one short episode)
        tr_file = open(raw_dir / f"{run_name}_traces.csv", "w",
                       newline="")
        tr_writer = csv.writer(tr_file)
        tr_writer.writerow(["episode", "step", "r", "c", "k", "e",
                            "action", "reward", "next_action",
                            "q_sa_before", "q_next", "delta",
                            "E_current_pair", "num_active_traces",
                            "sum_traces", "q_sa_after",
                            "top3_traces(state|action|E)"])

        step_logger = MazeLogger(raw_dir / f"{run_name}_steps_sampled.csv")

        metrics = []
        t0 = time.perf_counter()
        for ep in range(episodes):
            eps = self.eps_schedule(ep)
            s = self.env.reset(seed=base_seed * 1_000_000 + ep)
            a = self.choose_action(s, eps)
            E = {}                       # eligibility traces, per episode
            total_r, steps = 0.0, 0
            wall_hits = penalty_hits = key_got = success = 0
            ep_max_traces, end_event = 0, None

            while True:
                self.visit_counts[s] += 1
                s_next, r, done, info = self.env.step(a)
                a_next = None if done else self.choose_action(s_next, eps)

                q_sa_before = self.Q[s][a]
                q_next = 0.0 if done else self.Q[s_next][a_next]
                delta = r + self.gamma * q_next - q_sa_before

                # ---- trace update for the visited pair ----------------
                if self.trace_type == "replacing":
                    E[(s, a)] = 1.0
                else:                              # accumulating
                    E[(s, a)] = E.get((s, a), 0.0) + 1.0

                # ---- apply the update to every active pair, then decay -
                decay = self.gamma * self.lam
                dead = []
                for pair, e_val in E.items():
                    st, act = pair
                    self.Q[st][act] += self.alpha * delta * e_val
                    new_e = e_val * decay
                    if new_e < TRACE_THRESHOLD:
                        dead.append(pair)
                    else:
                        E[pair] = new_e
                for pair in dead:
                    del E[pair]

                n_active = len(E) + len(dead)
                ep_max_traces = max(ep_max_traces, n_active)

                # ---- logging ------------------------------------------
                if ep in trace_log_episodes and steps < trace_log_max_steps:
                    top3 = sorted(E.items(), key=lambda kv: -kv[1])[:3]
                    top3_str = "; ".join(
                        f"({t[0][0][0]},{t[0][0][1]},{t[0][0][2]},"
                        f"{t[0][0][3]})|{ACTION_NAMES[t[0][1]]}|"
                        f"{t[1]:.4f}" for t in top3)
                    tr_writer.writerow([
                        ep, steps, *s, ACTION_NAMES[a], round(r, 4),
                        "TERMINAL" if a_next is None
                        else ACTION_NAMES[a_next],
                        round(q_sa_before, 6), round(q_next, 6),
                        round(delta, 6),
                        round(E.get((s, a), 0.0) / decay
                              if decay > 0 else 1.0, 6),
                        n_active, round(sum(E.values()), 4),
                        round(self.Q[s][a], 6), top3_str])
                if ep in step_log_episodes:
                    step_logger.log(ep, steps, s, a, s_next, r, info, done)

                ev = info["event"]
                wall_hits += ev == EV_WALL_HIT
                penalty_hits += ev == EV_PENALTY_CELL
                key_got |= ev == EV_KEY_PICKUP
                success |= ev == EV_GOAL
                total_r += r
                steps += 1
                s, a = s_next, a_next
                if done:
                    end_event = ev
                    break

            self.max_active_traces = max(self.max_active_traces,
                                         ep_max_traces)
            ep_writer.writerow([ep, round(eps, 4), round(total_r, 3),
                                steps, success, wall_hits, penalty_hits,
                                key_got, s[3], end_event, ep_max_traces])
            metrics.append({"episode": ep, "epsilon": eps,
                            "total_reward": total_r, "steps": steps,
                            "success": success,
                            "penalty_entries": penalty_hits})
            if verbose_every and (ep + 1) % verbose_every == 0:
                recent = metrics[-verbose_every:]
                sr = sum(m["success"] for m in recent) / len(recent)
                mr = sum(m["total_reward"] for m in recent) / len(recent)
                print(f"[SARSA({self.lam}) {run_name}] "
                      f"ep {ep + 1:6d}/{episodes}  eps={eps:.3f}  "
                      f"success={sr:5.1%}  meanR={mr:7.1f}")

        runtime = round(time.perf_counter() - t0, 2)
        ep_file.close(); tr_file.close(); step_logger.close()
        self.train_stats = {"episodes": episodes, "runtime_sec": runtime,
                            "alpha": self.alpha, "gamma": self.gamma,
                            "lambda": self.lam,
                            "trace_type": self.trace_type,
                            "visited_states": len(self.Q),
                            "max_active_traces": self.max_active_traces}
        print(f"[SARSA({self.lam}) {run_name}] done in {runtime}s, "
              f"visited {len(self.Q)} states, "
              f"max active traces {self.max_active_traces}")
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluation (greedy; also measures SAFETY, for the on/off-policy     #
    # comparison required by analytical question 2)                       #
    # ------------------------------------------------------------------ #
    def evaluate(self, episodes=300, base_seed=777):
        succ, returns = 0, []
        steps_success, penalty_total, wall_total = [], 0, 0
        for ep in range(episodes):
            s = self.env.reset(seed=base_seed * 1_000_000 + ep)
            total = 0.0
            while True:
                a = self.greedy_action(s)
                s, r, done, info = self.env.step(a)
                total += r
                penalty_total += info["event"] == EV_PENALTY_CELL
                wall_total += info["event"] == EV_WALL_HIT
                if done:
                    if info["event"] == EV_GOAL:
                        succ += 1
                        steps_success.append(info["steps"])
                    break
            returns.append(total)
        return {"eval_episodes": episodes,
                "success_rate": succ / episodes,
                "mean_return": sum(returns) / episodes,
                # steps averaged over SUCCESSFUL episodes only, otherwise
                # failures (which always run to the energy limit) dominate
                "mean_steps_success": (sum(steps_success) /
                                       len(steps_success))
                if steps_success else None,
                "penalty_entries_per_episode": penalty_total / episodes,
                "wall_hits_per_episode": wall_total / episodes}

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