"""
experiments/run_transfer.py
---------------------------
Phase-5 experiment runner: transfer learning on Q-Learning.

Pipeline:
  1. Train (or reuse) the SOURCE Q table on the original map.
  2. Build two target maps ("similar" and "different"), BFS-validated.
  3. For each target map run six configurations:
        scratch | full | scaled(beta=0.25/0.50/0.75) | selective
  4. Report, separately for every configuration:
        * jumpstart        : greedy success/return BEFORE training
        * learning speed   : success AUC + episodes to reach threshold
        * final performance: greedy success/return AFTER training
  5. Document one concrete NEGATIVE-TRANSFER state: its source Q values,
     the structural change that made the inherited action wrong, and the
     corrected Q values after continued training.

Outputs:
  environments/maps/target_similar.json
  environments/maps/target_different.json
  results/models/transfer_<target>_<scenario>_q.pkl
  results/raw_data/transfer_<target>_<scenario>_episodes.csv
  results/raw_data/transfer_summary.csv
  results/raw_data/transfer_negative_example.csv
  results/figures/transfer_maps_comparison.png
  results/figures/transfer_curves_<target>.png
  results/figures/transfer_jumpstart_vs_final.png
  experiments/configs/transfer.json

Run from the project root (after phase 3):
    python experiments/run_transfer.py
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.generator import (FREE, WALL, PENALTY, START, KEY, DOOR,
                                    GOAL, load_map, save_map)
from environments.maze import MazeEnv
from agents.q_learning import QLearningAgent, ExponentialDecay
from transfer.transfer_learning import (make_target_map, build_initial_q,
                                        seed_agent_with_q,
                                        learning_speed_metrics,
                                        find_negative_transfer,
                                        q_values_after_training)

CONFIG = {
    "source_map": "source_map.json",
    "source_model": "results/models/ql_sparse_exponential_q.pkl",
    "reward_mode": "sparse",
    "episodes_target": 60000,     # transfer runs are shorter than the
                                  # 150k source run: the point is to
                                  # compare START and SPEED, not to push
                                  # every run to its asymptote
    "alpha": 0.2,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "decay_fraction": 0.8,
    "betas": [0.25, 0.50, 0.75],
    "selective_radius": 1,
    "eval_episodes": 300,
    "speed_threshold": 0.30,
    "seed": 9,
    "target_seed_similar": 91,
    "target_seed_different": 92,
}

MODELS_DIR = ROOT / "results" / "models"
RAW_DIR    = ROOT / "results" / "raw_data"
FIG_DIR    = ROOT / "results" / "figures"
CONFIG_DIR = ROOT / "experiments" / "configs"
MA_WINDOW  = 1000


# ---------------------------------------------------------------------------
def moving_average(x, w=MA_WINDOW):
    x = np.asarray(x, dtype=float)
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, "valid")


def make_schedule(episodes):
    return ExponentialDecay(start=CONFIG["epsilon_start"],
                            end=CONFIG["epsilon_end"],
                            decay_fraction=CONFIG["decay_fraction"],
                            total_episodes=episodes)


def plot_maps(source_map, targets, out_path):
    """Side-by-side view of the source and the two target maps, with the
    changed cells outlined so the structural difference is visible."""
    colors = ["#e8e8e8", "#5c3b2e", "#ff9b6a", "#4c9be8", "#f2c14e",
              "#9b59b6", "#2ecc71"]
    cmap = ListedColormap(colors)          # FREE..GOAL in code order
    maps = [("source", source_map)] + list(targets.items())
    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 6))
    for ax, (name, m) in zip(axes, maps):
        grid = np.array(m["grid"])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=6)
        if name != "source":
            src = np.array(source_map["grid"])
            for r in range(m["size"]):
                for c in range(m["size"]):
                    if grid[r, c] != src[r, c]:
                        ax.add_patch(plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1, fill=False,
                            edgecolor="red", linewidth=1.6))
        title = name
        if name != "source":
            title += (f"\n{m['obstacle_change_fraction']:.0%} obstacles "
                      f"moved, key_moved={m['key_moved']}")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=colors[i], label=lab) for i, lab in
               enumerate(["free", "wall", "penalty", "start", "key",
                          "door", "goal"])]
    handles.append(Patch(facecolor="none", edgecolor="red",
                         label="changed cell"))
    axes[-1].legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(target_name, all_metrics, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for run, metrics in all_metrics.items():
        ma_s = moving_average([m["success"] for m in metrics])
        ma_r = moving_average([m["total_reward"] for m in metrics])
        ax1.plot(range(len(ma_s)), ma_s, label=run, linewidth=1.3)
        ax2.plot(range(len(ma_r)), ma_r, label=run, linewidth=1.3)
    ax1.set_xlabel(f"episode (MA window={MA_WINDOW})")
    ax1.set_ylabel("success rate")
    ax1.set_title(f"Transfer to '{target_name}' target: success")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel(f"episode (MA window={MA_WINDOW})")
    ax2.set_ylabel("episode return")
    ax2.set_title(f"Transfer to '{target_name}' target: return")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_jumpstart_vs_final(rows, out_path):
    """Grouped bars: jumpstart vs final success, per target and scenario."""
    targets = sorted({r["target"] for r in rows})
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = [r for r in rows if r["target"] == tgt]
        labels = [r["scenario"] for r in sub]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, [r["jumpstart_success"] for r in sub], 0.4,
               label="jumpstart (before training)", color="#c44e52")
        ax.bar(x + 0.2, [r["final_success"] for r in sub], 0.4,
               label="final (after training)", color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("greedy success rate")
        ax.set_title(f"target: {tgt}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def get_source_q():
    """Load the phase-3 Q table, or train it if it is missing."""
    path = ROOT / CONFIG["source_model"]
    if path.exists():
        print(f"[transfer] using source Q table from "
              f"{path.relative_to(ROOT)}")
        return QLearningAgent.load(path)["Q"]
    print("[transfer] source model not found - training it now "
          "(run phase 3 first to avoid this)")
    env = MazeEnv(map_filename=CONFIG["source_map"],
                  reward_mode=CONFIG["reward_mode"], seed=CONFIG["seed"])
    agent = QLearningAgent(env, alpha=CONFIG["alpha"],
                           gamma=CONFIG["gamma"],
                           epsilon_schedule=make_schedule(150000),
                           seed=CONFIG["seed"])
    agent.train(150000, "transfer_source", RAW_DIR,
                base_seed=CONFIG["seed"], verbose_every=50000)
    agent.save(MODELS_DIR / "transfer_source_q.pkl")
    return dict(agent.Q)


def main():
    for d in (MODELS_DIR, RAW_DIR, FIG_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    source_map = load_map(CONFIG["source_map"])
    source_Q = get_source_q()

    # ---- 1. build and save the two target maps ---------------------------
    targets = {
        "similar": make_target_map(source_map, "similar",
                                   seed=CONFIG["target_seed_similar"]),
        "different": make_target_map(source_map, "different",
                                     seed=CONFIG["target_seed_different"]),
    }
    for name, tmap in targets.items():
        save_map(tmap, f"target_{name}.json")
    plot_maps(source_map, targets,
              FIG_DIR / "transfer_maps_comparison.png")

    # ---- 2. scenarios ----------------------------------------------------
    scenarios = [("scratch", None), ("full", None)]
    scenarios += [("scaled", b) for b in CONFIG["betas"]]
    scenarios += [("selective", None)]

    episodes = CONFIG["episodes_target"]
    summary_rows, negative_rows = [], []

    for tgt_name, tgt_map in targets.items():
        curves = {}
        full_agent_for_negative = None
        for scenario, beta in scenarios:
            label = scenario if beta is None else f"scaled_b{beta}"
            run_name = f"transfer_{tgt_name}_{label}"
            print(f"\n=== {run_name} ===")

            env = MazeEnv(map_data=tgt_map,
                          reward_mode=CONFIG["reward_mode"],
                          seed=CONFIG["seed"])
            agent = QLearningAgent(env, alpha=CONFIG["alpha"],
                                   gamma=CONFIG["gamma"],
                                   epsilon_schedule=make_schedule(episodes),
                                   seed=CONFIG["seed"])
            init_Q, tstats = build_initial_q(
                source_Q, scenario, source_map=source_map,
                target_map=tgt_map, beta=beta or 1.0,
                radius=CONFIG["selective_radius"])
            seed_agent_with_q(agent, init_Q)

            # ---- jumpstart: greedy evaluation BEFORE any training -------
            jump = agent.evaluate(CONFIG["eval_episodes"], base_seed=555)
            print(f"[{run_name}] jumpstart success = "
                  f"{jump['success_rate']:.1%} "
                  f"(transferred {tstats['transferred_states']} states)")

            metrics = agent.train(episodes, run_name, RAW_DIR,
                                  base_seed=CONFIG["seed"],
                                  verbose_every=20000)
            final = agent.evaluate(CONFIG["eval_episodes"], base_seed=777)
            speed = learning_speed_metrics(
                metrics, threshold=CONFIG["speed_threshold"])
            agent.save(MODELS_DIR / f"{run_name}_q.pkl")
            curves[label] = metrics

            summary_rows.append({
                "target": tgt_name, "scenario": label, "beta": beta,
                "transferred_states": tstats["transferred_states"],
                "source_states": tstats["source_states"],
                "transfer_ratio": tstats["transfer_ratio"],
                "jumpstart_success": round(jump["success_rate"], 4),
                "jumpstart_return": round(jump["mean_return"], 2),
                "success_auc": speed["success_auc"],
                "episodes_to_threshold": speed["episodes_to_threshold"],
                "final_success": round(final["success_rate"], 4),
                "final_return": round(final["mean_return"], 2),
                "train_runtime_sec": agent.train_stats["runtime_sec"],
            })
            print(f"[{run_name}] final success = "
                  f"{final['success_rate']:.1%}, "
                  f"AUC = {speed['success_auc']}, "
                  f"eps->threshold = {speed['episodes_to_threshold']}")

            if scenario == "full":
                full_agent_for_negative = agent

        plot_curves(tgt_name, curves,
                    FIG_DIR / f"transfer_curves_{tgt_name}.png")

        # ---- 3. negative-transfer evidence -------------------------------
        examples = find_negative_transfer(
            source_Q, source_map, tgt_map,
            initial_energy=None, max_examples=5)
        print(f"[transfer] '{tgt_name}': found {len(examples)} "
              f"negative-transfer candidates")
        for ex in examples:
            after = (q_values_after_training(full_agent_for_negative,
                                             ex["state"])
                     if full_agent_for_negative else None)
            negative_rows.append({
                "target": tgt_name,
                "state": str(ex["state"]),
                "cell": str(ex["cell"]),
                "key_flag": ex["key_flag"],
                "energy": ex["energy"],
                "source_greedy_action": ex["greedy_action_source"],
                "cell_it_moves_into": str(ex["target_cell"]),
                "structural_change": ex["change"],
                "q_gap_best_vs_second": ex["q_gap_best_vs_second"],
                "q_source": json.dumps(ex["q_values_source"]),
                "q_after_target_training": json.dumps(after)
                if after else "",
                "corrected": ("yes" if after and
                              max(after, key=after.get) !=
                              ex["greedy_action_source"] else "no"),
            })

    # ---- 4. persist ------------------------------------------------------
    with open(RAW_DIR / "transfer_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    if negative_rows:
        with open(RAW_DIR / "transfer_negative_example.csv", "w",
                  newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(negative_rows[0]))
            writer.writeheader()
            writer.writerows(negative_rows)

    plot_jumpstart_vs_final(summary_rows,
                            FIG_DIR / "transfer_jumpstart_vs_final.png")

    CONFIG["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    CONFIG["target_map_stats"] = {
        k: {kk: v[kk] for kk in ("obstacle_change_fraction", "key_moved",
                                 "cells_changed", "new_penalty_cells",
                                 "shortest_start_to_key",
                                 "shortest_key_to_goal")}
        for k, v in targets.items()}
    with open(CONFIG_DIR / "transfer.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()