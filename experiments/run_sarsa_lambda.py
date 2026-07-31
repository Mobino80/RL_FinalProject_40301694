"""
experiments/run_sarsa_lambda.py
-------------------------------
Phase-4 experiment runner.

Runs SARSA(lambda) on the SAME map, reward definition, alpha, gamma and
epsilon schedule as the Q-Learning experiments, so that any difference
is attributable to (a) on-policy vs off-policy learning and (b) the
eligibility-trace parameter lambda.

Experiments:
  1. lambda sweep : 0.0, 0.3, 0.7, 0.9   (replacing traces)  [spec]
  2. trace type   : lambda = 0.9 with accumulating traces, to justify
                    the default choice with evidence rather than words.

Outputs:
  results/models/sarsa_lam<l>_<trace>_q.pkl
  results/raw_data/sarsa_lam<l>_<trace>_episodes.csv
  results/raw_data/sarsa_lam<l>_<trace>_traces.csv     delta / E log
  results/raw_data/sarsa_lam<l>_<trace>_steps_sampled.csv
  results/raw_data/sarsa_summary.csv
  results/figures/sarsa_learning_curves_<metric>.png
  results/figures/sarsa_lambda_final_comparison.png
  results/figures/sarsa_trace_dynamics.png            delta and |E| plot
  results/figures/sarsa_heatmap_lam<best>_k<k>.png
  experiments/configs/sarsa_lambda.json

Run from the project root (after phases 2 and 3):
    python experiments/run_sarsa_lambda.py
"""

import csv
import json
import math
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
from agents.sarsa_lambda import SarsaLambdaAgent, TRACE_THRESHOLD
from agents.q_learning import ExponentialDecay
from agents.value_iteration import ValueIterationAgent

CONFIG = {
    "map_filename": "source_map.json",
    "episodes": 150000,
    "alpha": 0.2,
    "gamma": 0.99,
    "reward_mode": "sparse",
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "decay_fraction": 0.8,
    "schedule": "exponential",       # best schedule found in phase 3
    "lambdas": [0.0, 0.3, 0.7, 0.9],          # required by the spec
    # supplementary sweep: locates the point where the eligibility trace
    # outlives the episode itself (see trace_lifetime_steps in the
    # summary CSV) - this is where performance collapses.
    "supplementary_lambdas": [0.75, 0.8, 0.85],
    "trace_type": "replacing",
    "extra_accumulating_lambda": 0.9,
    "eval_episodes": 300,
    "seed": 9,
    "vi_reference_model": "results/models/vi_gamma_0.99.pkl",
}

MODELS_DIR = ROOT / "results" / "models"
RAW_DIR    = ROOT / "results" / "raw_data"
FIG_DIR    = ROOT / "results" / "figures"
CONFIG_DIR = ROOT / "experiments" / "configs"
MA_WINDOW  = 1000


def moving_average(x, w=MA_WINDOW):
    x = np.asarray(x, dtype=float)
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, "valid")


def make_schedule(episodes):
    return ExponentialDecay(start=CONFIG["epsilon_start"],
                            end=CONFIG["epsilon_end"],
                            decay_fraction=CONFIG["decay_fraction"],
                            total_episodes=episodes)


def plot_learning_curves(all_metrics, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, metrics in all_metrics.items():
        ma = moving_average([m[metric] for m in metrics])
        ax.plot(range(len(ma)), ma, label=run_name, linewidth=1.4)
    ax.set_xlabel(f"episode (moving average, window={MA_WINDOW})")
    ax.set_ylabel(ylabel)
    ax.set_title(f"SARSA(lambda): {ylabel}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_comparison(summary_rows, out_path):
    """Bar chart: final success rate and safety per lambda."""
    rows = sorted([r for r in summary_rows
                   if r["trace_type"] == CONFIG["trace_type"]],
                  key=lambda r: r["lambda"])
    labels = [f"lam={r['lambda']}" for r in rows]
    succ = [r["eval_success_rate"] for r in rows]
    pen = [r["eval_penalty_per_episode"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(labels, succ, color="#4c72b0")
    ax1.set_ylabel("greedy success rate")
    ax1.set_title("Final performance vs lambda")
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(labels, pen, color="#c44e52")
    ax2.set_ylabel("penalty-cell entries per episode")
    ax2.set_title("Safety vs lambda (lower = safer)")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trace_dynamics(trace_files, out_path):
    """Read the delta/E logs and show how lambda spreads credit.

    Left  : |TD error| per step (a short episode).
    Right : number of active eligibility traces per step - the direct
            visual proof that lambda=0 keeps exactly one active trace
            (one-step SARSA) while larger lambda keeps a growing tail.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, path in trace_files.items():
        if not Path(path).exists():
            continue
        steps, deltas, n_traces = [], [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        last_ep = rows[-1]["episode"]
        for row in rows:
            if row["episode"] != last_ep:
                continue
            steps.append(int(row["step"]))
            deltas.append(abs(float(row["delta"])))
            n_traces.append(int(row["num_active_traces"]))
        ax1.plot(steps, deltas, marker="o", markersize=3, label=label)
        ax2.plot(steps, n_traces, marker="s", markersize=3, label=label)
    ax1.set_xlabel("step"); ax1.set_ylabel("|TD error|")
    ax1.set_title("TD error along one episode")
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_xlabel("step"); ax2.set_ylabel("active eligibility traces")
    ax2.set_title("Credit spread: active traces per step")
    ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_q_heatmap(env, agent, k, title, out_path):
    """max_a Q aggregated over energy (same convention as phase 3)."""
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
                num += w * max(agent.Q[s]); den += w
                if w > best_visits:
                    best_visits, best_state = w, s
            if den > 0:
                grid_v[r, c] = num / den
                arrows[(r, c)] = max(ACTIONS,
                                     key=lambda a: agent.Q[best_state][a])

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#8a8a8a")
    im = ax.imshow(np.ma.masked_invalid(grid_v), cmap=cmap)
    fig.colorbar(im, ax=ax, label="max_a Q(s,a)")
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


def agreement_with_vi(agent, vi_policy):
    shared = [s for s, n in agent.visit_counts.items()
              if n > 0 and s in vi_policy]
    if not shared:
        return None, 0
    same = sum(1 for s in shared
               if max(ACTIONS, key=lambda a: agent.Q[s][a]) == vi_policy[s])
    return same / len(shared), len(shared)


# ---------------------------------------------------------------------------
def main():
    for d in (MODELS_DIR, RAW_DIR, FIG_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    episodes = CONFIG["episodes"]
    # the delta / E log is written for the first and the last episode:
    # episode 0 shows learning from scratch, the last one shows a short,
    # near-greedy episode - the ideal case to interpret in the report.
    trace_eps = {0, episodes - 1}
    step_eps = set(range(5)) | set(range(episodes - 5, episodes))

    vi_policy = None
    vi_path = ROOT / CONFIG["vi_reference_model"]
    if vi_path.exists():
        vi_policy = ValueIterationAgent.load(vi_path)["policy"]
        print(f"[SARSA] loaded VI reference policy")
    else:
        print("[SARSA] WARNING: VI model missing - run phase 2 first.")

    runs = [(lam, CONFIG["trace_type"]) for lam in CONFIG["lambdas"]]
    runs += [(lam, CONFIG["trace_type"])
             for lam in CONFIG["supplementary_lambdas"]]
    runs.append((CONFIG["extra_accumulating_lambda"], "accumulating"))

    all_metrics, summary_rows, trace_files = {}, {}, {}
    best_lam, best_success, best_agent, best_env = None, -1, None, None

    for lam, trace_type in runs:
        run_name = f"sarsa_lam{lam}_{trace_type}"
        print(f"\n=== {run_name} ===")
        env = MazeEnv(map_filename=CONFIG["map_filename"],
                      reward_mode=CONFIG["reward_mode"],
                      seed=CONFIG["seed"])
        agent = SarsaLambdaAgent(env, alpha=CONFIG["alpha"],
                                 gamma=CONFIG["gamma"], lam=lam,
                                 trace_type=trace_type,
                                 epsilon_schedule=make_schedule(episodes),
                                 seed=CONFIG["seed"])
        metrics = agent.train(episodes, run_name, RAW_DIR,
                              base_seed=CONFIG["seed"],
                              trace_log_episodes=trace_eps,
                              step_log_episodes=step_eps)
        all_metrics[run_name] = metrics
        trace_files[f"lambda={lam} ({trace_type})"] = \
            RAW_DIR / f"{run_name}_traces.csv"

        ev = agent.evaluate(CONFIG["eval_episodes"])
        agree, n_shared = (None, 0)
        if vi_policy is not None:
            agree, n_shared = agreement_with_vi(agent, vi_policy)
        agent.save(MODELS_DIR / f"{run_name}_q.pkl")

        # theoretical trace lifetime: number of steps before an
        # eligibility trace decays below TRACE_THRESHOLD. When this
        # exceeds the episode length (= the energy budget), credit is
        # assigned to the WHOLE trajectory, including the initial random
        # wandering - which is where learning collapses.
        decay = CONFIG["gamma"] * lam
        lifetime = (0 if decay <= 0
                    else math.log(TRACE_THRESHOLD) / math.log(decay))

        summary_rows[run_name] = {
            "run": run_name, "lambda": lam, "trace_type": trace_type,
            "trace_lifetime_steps": round(lifetime, 1),
            "energy_budget": env.initial_energy,
            "train_runtime_sec": agent.train_stats["runtime_sec"],
            "visited_states": agent.train_stats["visited_states"],
            "max_active_traces": agent.train_stats["max_active_traces"],
            "eval_success_rate": round(ev["success_rate"], 4),
            "eval_mean_return": round(ev["mean_return"], 2),
            "eval_mean_steps_success":
                None if ev["mean_steps_success"] is None
                else round(ev["mean_steps_success"], 1),
            "eval_penalty_per_episode":
                round(ev["penalty_entries_per_episode"], 3),
            "eval_wall_hits_per_episode":
                round(ev["wall_hits_per_episode"], 3),
            "policy_agreement_vs_VI":
                None if agree is None else round(agree, 4),
            "agreement_states_compared": n_shared}
        print(f"[{run_name}] success={ev['success_rate']:.1%}, "
              f"penalties/ep={ev['penalty_entries_per_episode']:.2f}, "
              f"agreement vs VI={agree}")

        if trace_type == CONFIG["trace_type"] and \
                ev["success_rate"] > best_success:
            best_lam, best_success = lam, ev["success_rate"]
            best_agent, best_env = agent, env

    # ---- figures ---------------------------------------------------------
    plot_learning_curves(all_metrics, "total_reward", "episode return",
                         FIG_DIR / "sarsa_learning_curves_reward.png")
    plot_learning_curves(all_metrics, "success", "success rate",
                         FIG_DIR / "sarsa_learning_curves_success.png")
    plot_learning_curves(all_metrics, "penalty_entries",
                         "penalty entries per episode",
                         FIG_DIR / "sarsa_learning_curves_penalty.png")
    rows = list(summary_rows.values())
    plot_final_comparison(rows,
                          FIG_DIR / "sarsa_lambda_final_comparison.png")
    plot_trace_dynamics(trace_files,
                        FIG_DIR / "sarsa_trace_dynamics.png")
    if best_agent is not None:
        for k in (0, 1):
            plot_q_heatmap(
                best_env, best_agent, k,
                f"SARSA(lambda={best_lam}) | key={k} | max_a Q, "
                f"visit-weighted over energy",
                FIG_DIR / f"sarsa_heatmap_lam{best_lam}_k{k}.png")

    # ---- summary + config ------------------------------------------------
    with open(RAW_DIR / "sarsa_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    CONFIG["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CONFIG["best_lambda_by_success"] = best_lam
    with open(CONFIG_DIR / "sarsa_lambda.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n=== Summary ===")
    for row in rows:
        print(row)
    print(f"\nBest lambda by greedy success rate: {best_lam} "
          f"({best_success:.1%})")


if __name__ == "__main__":
    main()