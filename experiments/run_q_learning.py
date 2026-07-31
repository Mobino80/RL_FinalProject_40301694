"""
experiments/run_q_learning.py
-----------------------------
Phase-3 experiment runner.

Trains Q-Learning in FOUR configurations:
    2 epsilon schedules (linear, exponential)  x  2 reward modes
    (sparse, shaped)
so that a single script produces the evidence for BOTH mandatory
analyses: the epsilon-schedule comparison and the reward-shaping study.

Outputs:
  results/models/ql_<mode>_<schedule>_q.pkl        Q table + visit counts
  results/raw_data/ql_<...>_episodes.csv           per-episode metrics
  results/raw_data/ql_<...>_steps_sampled.csv      sampled step events
  results/raw_data/ql_<...>_qupdates.csv           TD decomposition
  results/raw_data/ql_summary.csv                  final evaluation table
  results/figures/ql_learning_curves_<metric>.png  4-run comparison
  results/figures/ql_epsilon_schedules.png
  results/figures/ql_heatmap_<best-run>_k<k>.png   max_a Q + arrows
  experiments/configs/q_learning.json

Run from the project root:
    python experiments/run_q_learning.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.generator import WALL, PENALTY
from environments.maze import MazeEnv, ACTIONS, ACTION_DELTAS
from agents.q_learning import (QLearningAgent, LinearDecay,
                               ExponentialDecay)
from agents.value_iteration import ValueIterationAgent

# ---------------------------------------------------------------------------
# Experiment configuration (persisted to configs/ for reproducibility)
# ---------------------------------------------------------------------------
CONFIG = {
    "map_filename": "source_map.json",
    "episodes": 150000,
    "alpha": 0.2,
    "gamma": 0.99,                 # identical to the VI reference
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "decay_fraction": 0.8,         # both schedules hit the floor at 80%
    "reward_modes": ["sparse", "shaped"],
    "schedules": ["linear", "exponential"],
    "eval_episodes": 300,
    "seed": 9,                     # student base seed
    "vi_reference_model": "results/models/vi_gamma_0.99.pkl",
}

MODELS_DIR = ROOT / "results" / "models"
RAW_DIR    = ROOT / "results" / "raw_data"
FIG_DIR    = ROOT / "results" / "figures"
CONFIG_DIR = ROOT / "experiments" / "configs"

MA_WINDOW = 1000        # moving-average window for learning curves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def moving_average(x, w=MA_WINDOW):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def make_schedule(name, episodes):
    cls = LinearDecay if name == "linear" else ExponentialDecay
    return cls(start=CONFIG["epsilon_start"], end=CONFIG["epsilon_end"],
               decay_fraction=CONFIG["decay_fraction"],
               total_episodes=episodes)


def plot_learning_curves(all_metrics, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, metrics in all_metrics.items():
        values = [m[metric] for m in metrics]
        ma = moving_average(values)
        ax.plot(range(len(ma)), ma, label=run_name, linewidth=1.4)
    ax.set_xlabel(f"episode (moving average, window={MA_WINDOW})")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Q-Learning: {ylabel}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_epsilon_schedules(episodes, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in CONFIG["schedules"]:
        sched = make_schedule(name, episodes)
        xs = range(0, episodes, 25)
        ax.plot(list(xs), [sched(e) for e in xs], label=name)
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.set_title("Epsilon decay schedules (same start, floor and "
                 "time-to-floor)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_q_heatmap(env, agent, k, title, out_path):
    """max_a Q(s,a) heatmap + greedy arrows, AGGREGATED over energy.

    Why aggregate instead of taking a fixed energy slice (as done for
    Value Iteration)? Energy decreases by 1 on every step, so a state
    (r, c, k, e) with a FIXED e is reachable only at one particular
    distance from the start: a fixed-e slice of a learned Q table is
    almost empty (only the start cell has e = initial_energy).
    Value Iteration does not have this problem because it sweeps over
    the whole state space including unreachable states.

    Aggregation used here:
      value  = visit-weighted mean of max_a Q(r,c,k,e) over all
               energy levels e the agent actually experienced,
      arrow  = greedy action of the MOST VISITED energy level,
               i.e. the behaviour that is typical for that cell.
    This must be stated in the report when comparing the VI and QL
    heatmaps - they are not identical objects.
    """
    n = env.n
    grid_v = np.full((n, n), np.nan)
    arrows = {}
    for r in range(n):
        for c in range(n):
            if env.grid[r][c] == WALL:
                continue
            num = den = 0.0
            best_visits, best_state = -1, None
            for e in range(1, env.initial_energy + 1):
                s = (r, c, k, e)
                if s not in agent.Q:
                    continue
                w = agent.visit_counts.get(s, 0)
                if w == 0:
                    continue
                num += w * max(agent.Q[s])
                den += w
                if w > best_visits:
                    best_visits, best_state = w, s
            if den > 0:
                grid_v[r, c] = num / den
                arrows[(r, c)] = max(
                    ACTIONS, key=lambda a: agent.Q[best_state][a])

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.viridis.copy()
    # cells with no Q data at all -> grey (walls are drawn separately below)
    cmap.set_bad(color="#8a8a8a")
    im = ax.imshow(np.ma.masked_invalid(grid_v), cmap=cmap)
    fig.colorbar(im, ax=ax, label="max_a Q(s,a)")

    # walls are drawn explicitly on top in a distinct dark red-brown,
    # so they can never be confused with "no Q data" cells
    for r in range(n):
        for c in range(n):
            if env.grid[r][c] == WALL:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor="#5c3b2e",
                                       edgecolor="#3d271e", linewidth=0.5))

    for r in range(n):
        for c in range(n):
            if (r, c) not in arrows or (r, c) == env.goal_pos:
                continue
            dr, dc = ACTION_DELTAS[arrows[(r, c)]]
            ax.annotate("", xy=(c + 0.35 * dc, r + 0.35 * dr),
                        xytext=(c - 0.25 * dc, r - 0.25 * dr),
                        arrowprops=dict(arrowstyle="->", lw=1.0,
                                        color="white"))
            if env.grid[r][c] == PENALTY:
                ax.text(c - 0.32, r - 0.22, "P", fontsize=9,
                        fontweight="bold", color="orangered")

    for pos, lab in ((env.start_pos, "S"), (env.key_pos, "K"),
                     (env.door_pos, "D"), (env.goal_pos, "G")):
        ax.text(pos[1], pos[0], lab, ha="center", va="center",
                fontsize=13, fontweight="bold", color="red")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def agreement_with_vi(agent, vi_policy, min_visits=1):
    """Policy agreement restricted to states the QL agent actually
    visited at least `min_visits` times (comparing on never-visited
    states would be meaningless - discuss this choice in the report)."""
    shared = [s for s, n in agent.visit_counts.items()
              if n >= min_visits and s in vi_policy]
    if not shared:
        return None, 0
    same = sum(1 for s in shared
               if max(ACTIONS, key=lambda a: agent.Q[s][a]) ==
               vi_policy[s])
    return same / len(shared), len(shared)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    for d in (MODELS_DIR, RAW_DIR, FIG_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    episodes = CONFIG["episodes"]
    # sampled episodes for the step-level event log (tier 2)
    step_log_eps = set(range(10)) | set(range(0, episodes, 10000)) \
        | set(range(episodes - 10, episodes))
    # designated episodes for the Q-update decomposition log (tier 3)
    qupdate_eps = {0, episodes // 2, episodes - 1}

    # optional VI reference for policy agreement
    vi_policy = None
    vi_path = ROOT / CONFIG["vi_reference_model"]
    if vi_path.exists():
        vi_policy = ValueIterationAgent.load(vi_path)["policy"]
        print(f"[QL] loaded VI reference policy from "
              f"{vi_path.relative_to(ROOT)}")
    else:
        print("[QL] WARNING: VI reference model not found - run phase 2 "
              "first for the policy-agreement analysis.")

    all_metrics, summary_rows = {}, []
    for mode in CONFIG["reward_modes"]:
        for sched_name in CONFIG["schedules"]:
            run_name = f"ql_{mode}_{sched_name}"
            print(f"\n=== {run_name} ===")
            env = MazeEnv(map_filename=CONFIG["map_filename"],
                          reward_mode=mode, seed=CONFIG["seed"])
            agent = QLearningAgent(
                env, alpha=CONFIG["alpha"], gamma=CONFIG["gamma"],
                epsilon_schedule=make_schedule(sched_name, episodes),
                seed=CONFIG["seed"])
            metrics = agent.train(
                episodes, run_name, RAW_DIR,
                step_log_episodes=step_log_eps,
                qupdate_log_episodes=qupdate_eps,
                base_seed=CONFIG["seed"], verbose_every=25000)
            all_metrics[run_name] = metrics

            ev = agent.evaluate(CONFIG["eval_episodes"],
                                base_seed=777)
            agree, n_shared = (None, 0)
            if vi_policy is not None:
                agree, n_shared = agreement_with_vi(agent, vi_policy)
            agent.save(MODELS_DIR / f"{run_name}_q.pkl")

            row = {"run": run_name, "reward_mode": mode,
                   "schedule": sched_name,
                   "train_runtime_sec":
                       agent.train_stats["runtime_sec"],
                   "visited_states": agent.train_stats["visited_states"],
                   "eval_success_rate": round(ev["success_rate"], 4),
                   "eval_mean_return": round(ev["mean_return"], 2),
                   "eval_mean_steps": round(ev["mean_steps"], 1),
                   "policy_agreement_vs_VI":
                       None if agree is None else round(agree, 4),
                   "agreement_states_compared": n_shared}
            summary_rows.append(row)
            print(f"[QL {run_name}] eval success = "
                  f"{ev['success_rate']:.1%}, "
                  f"agreement vs VI = {agree}")

            # Q heatmaps for the sparse runs (reference reward)
            if mode == "sparse":
                for k in (0, 1):
                    plot_q_heatmap(
                        env, agent, k,
                        f"Q-Learning ({sched_name})  |  key={k}  |  "
                        f"max_a Q, visit-weighted over energy",
                        FIG_DIR / f"ql_heatmap_{sched_name}_k{k}.png")

    # ---- figures ---------------------------------------------------------
    plot_epsilon_schedules(episodes,
                           FIG_DIR / "ql_epsilon_schedules.png")
    plot_learning_curves(all_metrics, "total_reward",
                         "episode return",
                         FIG_DIR / "ql_learning_curves_reward.png")
    plot_learning_curves(all_metrics, "steps", "steps per episode",
                         FIG_DIR / "ql_learning_curves_steps.png")
    plot_learning_curves(all_metrics, "success", "success rate",
                         FIG_DIR / "ql_learning_curves_success.png")

    # ---- summary + config ------------------------------------------------
    with open(RAW_DIR / "ql_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    CONFIG["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CONFIG_DIR / "q_learning.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()