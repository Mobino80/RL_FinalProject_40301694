"""
experiments/run_value_iteration.py
----------------------------------
Phase-2 experiment runner.

Runs Value Iteration on the fixed source map for at least three discount
factors (per spec: analyze >= 3 gamma values), then saves:

  results/models/vi_gamma_<g>.pkl          value function + policy
  results/raw_data/vi_summary.csv          sweeps, runtime, agreement
  results/raw_data/vi_convergence.csv      per-sweep max delta
  results/figures/vi_heatmap_gamma<g>_k<k>.png
                                           V heatmap + greedy arrows
  results/figures/vi_convergence.png       delta curves (log scale)
  experiments/configs/value_iteration.json experiment configuration

Run from the project root:
    python experiments/run_value_iteration.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # save figures without a display
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.generator import WALL, PENALTY, CELL_CHARS
from environments.maze import MazeEnv, ACTIONS, ACTION_DELTAS
from agents.value_iteration import ValueIterationAgent

# ---------------------------------------------------------------------------
# Experiment configuration (stored to configs/ for reproducibility)
# ---------------------------------------------------------------------------
CONFIG = {
    "map_filename": "source_map.json",
    "reward_mode": "sparse",     # reference runs use the sparse reward
    "gammas": [0.90, 0.95, 0.99],
    "theta": 1e-6,
    "max_sweeps": 500,
    "heatmap_energy_slice": None,   # None -> initial energy of the map
}

MODELS_DIR  = ROOT / "results" / "models"
RAW_DIR     = ROOT / "results" / "raw_data"
FIG_DIR     = ROOT / "results" / "figures"
CONFIG_DIR  = ROOT / "experiments" / "configs"


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def plot_value_heatmap(env, V, policy, gamma, k, energy, out_path):
    """Heatmap of V(r, c, k, e=energy) with greedy-action arrows.

    Because energy is part of the state, a 2-D heatmap is one SLICE of
    the value function; the slice (k, energy) is written in the title so
    the report can reference it precisely.
    """
    n = env.n
    grid_v = np.full((n, n), np.nan)
    for r in range(n):
        for c in range(n):
            if env.grid[r][c] == WALL:
                continue
            s = (r, c, k, energy)
            if s in V:
                grid_v[r, c] = V[s]

    fig, ax = plt.subplots(figsize=(9, 8))
    masked = np.ma.masked_invalid(grid_v)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#3a3a3a")            # walls in dark grey
    im = ax.imshow(masked, cmap=cmap)
    fig.colorbar(im, ax=ax, label="V(s)")

    # greedy-action arrows (skip terminal / special-label cells)
    for r in range(n):
        for c in range(n):
            s = (r, c, k, energy)
            if s not in policy or (r, c) == env.goal_pos:
                continue
            dr, dc = ACTION_DELTAS[policy[s]]
            ax.annotate("", xy=(c + 0.35 * dc, r + 0.35 * dr),
                        xytext=(c - 0.25 * dc, r - 0.25 * dr),
                        arrowprops=dict(arrowstyle="->", lw=1.1,
                                        color="white"))

    # landmarks + penalty cells
    for r in range(n):
        for c in range(n):
            if env.grid[r][c] == PENALTY:
                ax.text(c - 0.32, r - 0.22, "P", fontsize=9,
                        fontweight="bold", color="orangered")
    landmarks = {env.start_pos: "S", env.key_pos: "K",
                 env.door_pos: "D", env.goal_pos: "G"}
    for (r, c), label in landmarks.items():
        ax.text(c, r, label, ha="center", va="center", fontsize=13,
                fontweight="bold", color="red")

    ax.set_title(f"Value Iteration  |  gamma={gamma}  |  "
                 f"slice: key={k}, energy={energy}")
    ax.set_xticks(range(0, n, 2))
    ax.set_yticks(range(0, n, 2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(histories, out_path):
    """Max Bellman delta per sweep, log scale, one curve per gamma."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma, hist in histories.items():
        ax.plot(range(1, len(hist["deltas"]) + 1), hist["deltas"],
                marker="o", markersize=3, label=f"gamma = {gamma}")
    ax.set_yscale("log")
    ax.set_xlabel("sweep")
    ax.set_ylabel("max |V_{k+1}(s) - V_k(s)|  (log scale)")
    ax.set_title("Value Iteration convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def policy_agreement(pi_a, pi_b):
    """Fraction of states on which two policies pick the same action."""
    shared = set(pi_a) & set(pi_b)
    same = sum(1 for s in shared if pi_a[s] == pi_b[s])
    return same / len(shared)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    for d in (MODELS_DIR, RAW_DIR, FIG_DIR, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    env = MazeEnv(map_filename=CONFIG["map_filename"],
                  reward_mode=CONFIG["reward_mode"])
    e_slice = CONFIG["heatmap_energy_slice"] or env.initial_energy

    agents, histories = {}, {}
    summary_rows = []

    for gamma in CONFIG["gammas"]:
        print(f"\n=== Value Iteration, gamma = {gamma} ===")
        agent = ValueIterationAgent(env, gamma=gamma,
                                    theta=CONFIG["theta"],
                                    max_sweeps=CONFIG["max_sweeps"])
        hist = agent.run(verbose=True)
        agents[gamma], histories[gamma] = agent, hist

        model_path = MODELS_DIR / f"vi_gamma_{gamma}.pkl"
        agent.save(model_path)
        print(f"[VI] model saved -> {model_path.relative_to(ROOT)}")

        for k in (0, 1):
            # realistic slice: reaching the key itself costs energy, so
            # the k=1 layer is shown at a plausible remaining energy
            if k == 0:
                e_k = e_slice
            else:
                spent = env.map_data.get("shortest_start_to_key", 0)
                e_k = max(1, e_slice - spent)
            fig_path = FIG_DIR / f"vi_heatmap_gamma{gamma}_k{k}.png"
            plot_value_heatmap(env, agent.V, agent.policy, gamma, k,
                               e_k, fig_path)
            print(f"[VI] figure saved -> {fig_path.relative_to(ROOT)}")

        v_start = agent.V[(*env.start_pos, 0, env.initial_energy)]
        summary_rows.append({
            "gamma": gamma, "theta": CONFIG["theta"],
            "sweeps_to_converge": hist["sweeps"],
            "runtime_sec": hist["runtime_sec"],
            "num_states": hist["num_states"],
            "V_start": round(v_start, 4),
        })

    # ---- cross-gamma policy agreement (for the gamma-effect analysis) ----
    gammas = CONFIG["gammas"]
    ref = agents[gammas[-1]].policy          # highest gamma as reference
    for row, gamma in zip(summary_rows, gammas):
        row["policy_agreement_vs_gamma_" + str(gammas[-1])] = round(
            policy_agreement(agents[gamma].policy, ref), 4)

    # ---- persist raw data --------------------------------------------------
    with open(RAW_DIR / "vi_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(RAW_DIR / "vi_convergence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma", "sweep", "max_delta"])
        for gamma, hist in histories.items():
            for i, d in enumerate(hist["deltas"], start=1):
                writer.writerow([gamma, i, d])

    plot_convergence(histories, FIG_DIR / "vi_convergence.png")

    CONFIG["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CONFIG_DIR / "value_iteration.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(row)
    print(f"\nAll outputs written under results/ "
          f"and experiments/configs/.")


if __name__ == "__main__":
    main()