"""
experiments/run_stability.py
----------------------------
Run-to-run stability and memory footprint study.

RL Final Project - Student ID: 40301694

The project specification requires the three-algorithm comparison to
cover runtime, sample count, STABILITY ACROSS RUNS, MEMORY USAGE, path
quality and parameter sensitivity. The per-phase scripts already cover
runtime, samples, path quality and parameter sensitivity; this script
adds the two remaining criteria.

What varies between runs
------------------------
The MAP and the reward definition are held fixed (as the spec demands).
Only the seed changes, which drives BOTH
  * the agent's own randomness (epsilon-greedy choices, tie-breaking),
  * the environment's stochasticity (the 0.8/0.1/0.1 slips, because the
    per-episode reset seed is derived from the run seed).
Evaluation always uses the SAME fixed seed block (base_seed = 777), so
every run is graded on identical episodes and the spread that remains is
purely training variability.

Value Iteration is included as a reference: its computation is fully
deterministic given the map, so its training variance is exactly zero by
construction - only the stochastic evaluation of its policy varies.

Outputs
-------
  results/raw_data/stability_runs.csv        one row per (config, seed)
  results/raw_data/stability_aggregate.csv   mean / std / min / max
  results/raw_data/memory_footprint.csv      table sizes per algorithm
  results/figures/stability_success.png      bar chart with error bars
  results/figures/stability_curves_<cfg>.png per-seed learning curves
  experiments/configs/stability.json

Run from the project root (after phases 2-4):
    python experiments/run_stability.py
"""

import csv
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.maze import MazeEnv, ACTIONS
from agents.q_learning import QLearningAgent, ExponentialDecay
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent

CONFIG = {
    "map_filename": "source_map.json",
    "reward_mode": "sparse",
    "episodes": 150000,
    "alpha": 0.2,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "decay_fraction": 0.8,
    "seeds": [9, 17, 23, 42, 101],
    "eval_episodes": 300,
    "eval_seed": 777,
    "speed_threshold": 0.30,
    "configs": [
        {"name": "q_learning", "algo": "ql"},
        {"name": "sarsa_lam0.0", "algo": "sarsa", "lam": 0.0},
        {"name": "sarsa_lam0.3", "algo": "sarsa", "lam": 0.3},
        {"name": "sarsa_lam0.7", "algo": "sarsa", "lam": 0.7},
        {"name": "sarsa_lam0.9", "algo": "sarsa", "lam": 0.9},
    ],
}

RAW_DIR = ROOT / "results" / "raw_data"
FIG_DIR = ROOT / "results" / "figures"
CONFIG_DIR = ROOT / "experiments" / "configs"
TMP_DIR = ROOT / "results" / "raw_data" / "stability_tmp"
MA_WINDOW = 1000


# ---------------------------------------------------------------------------
def make_schedule(episodes):
    return ExponentialDecay(start=CONFIG["epsilon_start"],
                            end=CONFIG["epsilon_end"],
                            decay_fraction=CONFIG["decay_fraction"],
                            total_episodes=episodes)


def learning_speed(metrics, threshold, window=MA_WINDOW):
    succ = [m["success"] for m in metrics]
    auc = sum(succ) / len(succ)
    running, reached = 0, None
    for i, s in enumerate(succ):
        running += s
        if i >= window:
            running -= succ[i - window]
            if reached is None and running / window >= threshold:
                reached = i
    return auc, reached


def table_bytes(obj):
    """Serialised size of a learned table, in kilobytes.

    Pickle size is a reproducible, implementation-independent proxy for
    memory usage: it counts exactly the numbers that must be stored.
    """
    return len(pickle.dumps(obj)) / 1024.0


def build_agent(cfg, env, seed, episodes):
    if cfg["algo"] == "ql":
        return QLearningAgent(env, alpha=CONFIG["alpha"],
                              gamma=CONFIG["gamma"],
                              epsilon_schedule=make_schedule(episodes),
                              seed=seed)
    return SarsaLambdaAgent(env, alpha=CONFIG["alpha"],
                            gamma=CONFIG["gamma"], lam=cfg["lam"],
                            trace_type="replacing",
                            epsilon_schedule=make_schedule(episodes),
                            seed=seed)


# ---------------------------------------------------------------------------
def main():
    for d in (RAW_DIR, FIG_DIR, CONFIG_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    episodes = CONFIG["episodes"]
    rows, curves = [], {}

    for cfg in CONFIG["configs"]:
        curves[cfg["name"]] = {}
        for seed in CONFIG["seeds"]:
            t0 = time.perf_counter()
            env = MazeEnv(map_filename=CONFIG["map_filename"],
                          reward_mode=CONFIG["reward_mode"], seed=seed)
            agent = build_agent(cfg, env, seed, episodes)
            metrics = agent.train(episodes, f"stab_{cfg['name']}_s{seed}",
                                  TMP_DIR, base_seed=seed,
                                  verbose_every=0)
            ev = agent.evaluate(CONFIG["eval_episodes"],
                                base_seed=CONFIG["eval_seed"])
            auc, reached = learning_speed(metrics,
                                          CONFIG["speed_threshold"])
            wall = round(time.perf_counter() - t0, 1)

            row = {
                "config": cfg["name"], "seed": seed,
                "final_success": round(ev["success_rate"], 4),
                "final_return": round(ev["mean_return"], 2),
                "success_auc": round(auc, 4),
                "episodes_to_threshold": reached,
                "visited_states": len(agent.Q),
                "q_table_kb": round(table_bytes(dict(agent.Q)), 1),
                "train_runtime_sec": agent.train_stats["runtime_sec"],
                "wall_sec": wall,
            }
            if cfg["algo"] == "sarsa":
                row["max_active_traces"] = \
                    agent.train_stats["max_active_traces"]
            rows.append(row)
            curves[cfg["name"]][seed] = [m["success"] for m in metrics]
            print(f"[stability] {cfg['name']:14s} seed={seed:4d}  "
                  f"success={ev['success_rate']:6.1%}  "
                  f"AUC={auc:.4f}  states={len(agent.Q)}  "
                  f"({wall}s)")

    # ---- Value Iteration reference (deterministic training) -------------
    vi_path = ROOT / "results" / "models" / "vi_gamma_0.99.pkl"
    vi_row = None
    if vi_path.exists():
        vi = ValueIterationAgent.load(vi_path)
        env = MazeEnv(map_filename=CONFIG["map_filename"],
                      reward_mode=CONFIG["reward_mode"], seed=0)
        succ = []
        for seed in CONFIG["seeds"]:
            # evaluate the SAME deterministic policy under different
            # environment realisations
            wins = 0
            for ep in range(CONFIG["eval_episodes"]):
                s = env.reset(seed=seed * 1_000_000 + ep)
                while True:
                    s, r, done, info = env.step(vi["policy"].get(s, 0))
                    if done:
                        wins += info["event"] == "GOAL_REACHED"
                        break
            succ.append(wins / CONFIG["eval_episodes"])
            print(f"[stability] value_iteration  seed={seed:4d}  "
                  f"success={succ[-1]:6.1%}  (policy is identical)")
        vi_row = {
            "config": "value_iteration",
            "training_variance": 0.0,
            "eval_success_mean": round(statistics.mean(succ), 4),
            "eval_success_std": round(statistics.stdev(succ), 4),
            "table_kb": round(table_bytes({"V": vi["V"],
                                           "policy": vi["policy"]}), 1),
            "states_stored": len(vi["V"]),
            "sweeps": vi["history"]["sweeps"],
            "runtime_sec": vi["history"]["runtime_sec"],
        }

    # ---- aggregate -------------------------------------------------------
    agg_rows = []
    for cfg in CONFIG["configs"]:
        sub = [r for r in rows if r["config"] == cfg["name"]]
        succ = [r["final_success"] for r in sub]
        aucs = [r["success_auc"] for r in sub]
        thr = [r["episodes_to_threshold"] for r in sub
               if r["episodes_to_threshold"] is not None]
        agg_rows.append({
            "config": cfg["name"],
            "n_seeds": len(sub),
            "success_mean": round(statistics.mean(succ), 4),
            "success_std": round(statistics.stdev(succ), 4),
            "success_min": min(succ), "success_max": max(succ),
            "success_range": round(max(succ) - min(succ), 4),
            "coeff_of_variation":
                round(statistics.stdev(succ) / statistics.mean(succ), 4)
                if statistics.mean(succ) > 0 else None,
            "auc_mean": round(statistics.mean(aucs), 4),
            "auc_std": round(statistics.stdev(aucs), 4),
            "episodes_to_threshold_mean":
                round(statistics.mean(thr)) if thr else None,
            "runs_reaching_threshold": f"{len(thr)}/{len(sub)}",
            "q_table_kb_mean":
                round(statistics.mean([r["q_table_kb"] for r in sub]), 1),
            "runtime_sec_mean":
                round(statistics.mean(
                    [r["train_runtime_sec"] for r in sub]), 1),
        })

    with open(RAW_DIR / "stability_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    with open(RAW_DIR / "stability_aggregate.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0]))
        writer.writeheader(); writer.writerows(agg_rows)

    # ---- memory footprint table -----------------------------------------
    mem_rows = []
    for a in agg_rows:
        sub = [r for r in rows if r["config"] == a["config"]]
        mem_rows.append({
            "algorithm": a["config"],
            "stores": "Q(s,a) for visited states",
            "entries_mean": round(statistics.mean(
                [r["visited_states"] for r in sub])),
            "kilobytes_mean": a["q_table_kb_mean"],
        })
    if vi_row:
        mem_rows.append({
            "algorithm": "value_iteration",
            "stores": "V(s) + policy for ALL states",
            "entries_mean": vi_row["states_stored"],
            "kilobytes_mean": vi_row["table_kb"],
        })
    with open(RAW_DIR / "memory_footprint.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mem_rows[0]))
        writer.writeheader(); writer.writerows(mem_rows)

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [a["config"] for a in agg_rows]
    means = [a["success_mean"] for a in agg_rows]
    stds = [a["success_std"] for a in agg_rows]
    ax.bar(names, means, yerr=stds, capsize=6, color="#4c72b0",
           error_kw={"ecolor": "#c44e52", "elinewidth": 2})
    for i, a in enumerate(agg_rows):
        ax.plot([i] * len(CONFIG["seeds"]),
                [r["final_success"] for r in rows
                 if r["config"] == a["config"]],
                "o", color="black", markersize=4, alpha=.7)
    ax.set_ylabel("greedy success rate")
    ax.set_title(f"Run-to-run stability over {len(CONFIG['seeds'])} seeds "
                 f"(mean +/- std, dots = individual runs)")
    ax.grid(axis="y", alpha=.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stability_success.png", dpi=150)
    plt.close(fig)

    for name, per_seed in curves.items():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for seed, succ in per_seed.items():
            ma = np.convolve(np.asarray(succ, dtype=float),
                             np.ones(MA_WINDOW) / MA_WINDOW, "valid")
            ax.plot(range(len(ma)), ma, linewidth=1.2, label=f"seed {seed}")
        ax.set_xlabel(f"episode (MA window={MA_WINDOW})")
        ax.set_ylabel("success rate")
        ax.set_title(f"{name}: per-seed learning curves")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"stability_curves_{name}.png", dpi=150)
        plt.close(fig)

    CONFIG["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if vi_row:
        CONFIG["value_iteration_reference"] = vi_row
    with open(CONFIG_DIR / "stability.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n=== stability summary ===")
    for a in agg_rows:
        print(f"{a['config']:16s} success = {a['success_mean']:.1%} "
              f"+/- {a['success_std']:.1%}  "
              f"(range {a['success_min']:.1%}-{a['success_max']:.1%}, "
              f"CV = {a['coeff_of_variation']})")
    if vi_row:
        print(f"{'value_iteration':16s} success = "
              f"{vi_row['eval_success_mean']:.1%} +/- "
              f"{vi_row['eval_success_std']:.1%}  "
              f"(training variance is exactly 0)")
    print("\n=== memory footprint ===")
    for m in mem_rows:
        print(f"{m['algorithm']:16s} {m['entries_mean']:6d} entries, "
              f"{m['kilobytes_mean']:8.1f} KB  ({m['stores']})")


if __name__ == "__main__":
    main()