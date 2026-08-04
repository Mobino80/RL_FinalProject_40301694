"""
experiments/analysis.py
-----------------------
Consolidated visual analysis - phase 6.

RL Final Project - Student ID: 40301694

Produces every view required by the "minimum expected content" table of
the project specification, saving each one as an image under
results/figures/:

  | view              | content                                          |
  |-------------------|--------------------------------------------------|
  | value heatmap     | V (Value Iteration) or max_a Q (model-free) for   |
  |                   | all valid states                                  |
  | final policy      | greedy-action arrows + terminal-state markers     |
  | visit map         | how often the agent visited each state in training|
  | agent path        | the trajectory of the final greedy policy         |
  | policy difference | states that agree / disagree with the reference   |
  | transfer          | Q and policy difference before vs after transfer  |

Every figure is generated from the saved models and raw data in the
repository, never from hand-entered numbers.

Run from the project root (after phases 2-5):
    python experiments/analysis.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.generator import WALL, PENALTY, load_map
from environments.maze import MazeEnv, ACTIONS, ACTION_DELTAS, EV_GOAL
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

FIG_DIR = ROOT / "results" / "figures"
RAW_DIR = ROOT / "results" / "raw_data"
MODELS  = ROOT / "results" / "models"

VI_MODEL     = MODELS / "vi_gamma_0.99.pkl"
QL_MODEL     = MODELS / "ql_sparse_exponential_q.pkl"
SARSA_MODEL  = MODELS / "sarsa_lam0.3_replacing_q.pkl"
TRANSFER_PRE = QL_MODEL                                   # source table
TRANSFER_POST = MODELS / "transfer_different_full_q.pkl"  # after transfer

WALL_COLOR, EMPTY_COLOR = "#5c3b2e", "#8a8a8a"


# ---------------------------------------------------------------------------
# shared drawing helpers
# ---------------------------------------------------------------------------
def _draw_walls_and_marks(ax, env, mark_penalty=True):
    for r in range(env.n):
        for c in range(env.n):
            if env.grid[r][c] == WALL:
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1,
                                       facecolor=WALL_COLOR,
                                       edgecolor="#3d271e", linewidth=.5))
            elif mark_penalty and env.grid[r][c] == PENALTY:
                ax.text(c - .32, r - .22, "P", fontsize=8,
                        fontweight="bold", color="orangered")
    for pos, lab in ((env.start_pos, "S"), (env.key_pos, "K"),
                     (env.door_pos, "D"), (env.goal_pos, "G")):
        ax.text(pos[1], pos[0], lab, ha="center", va="center",
                fontsize=12, fontweight="bold", color="red")


def _aggregate_over_energy(env, Q, visits, k):
    """Visit-weighted mean of max_a Q over the energy dimension, plus the
    greedy action of the most-visited energy level (see phase 3 notes)."""
    grid = np.full((env.n, env.n), np.nan)
    arrows = {}
    for r in range(env.n):
        for c in range(env.n):
            if env.grid[r][c] == WALL:
                continue
            num = den = 0.0
            best_w, best_s = -1, None
            for e in range(1, env.initial_energy + 1):
                s = (r, c, k, e)
                if s not in Q:
                    continue
                w = visits.get(s, 0)
                if w == 0:
                    continue
                num += w * max(Q[s]); den += w
                if w > best_w:
                    best_w, best_s = w, s
            if den > 0:
                grid[r, c] = num / den
                arrows[(r, c)] = max(ACTIONS, key=lambda a: Q[best_s][a])
    return grid, arrows


def _slice_vi(env, V, policy, k, energy):
    grid = np.full((env.n, env.n), np.nan)
    arrows = {}
    for r in range(env.n):
        for c in range(env.n):
            if env.grid[r][c] == WALL:
                continue
            s = (r, c, k, energy)
            if s in V:
                grid[r, c] = V[s]
            if s in policy:
                arrows[(r, c)] = policy[s]
    return grid, arrows


# ---------------------------------------------------------------------------
# 1. value heatmap  +  2. final policy
# ---------------------------------------------------------------------------
def figure_value_and_policy(env, grid_v, arrows, title, out_path,
                            cbar_label="V(s)"):
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=EMPTY_COLOR)
    im = ax.imshow(np.ma.masked_invalid(grid_v), cmap=cmap)
    fig.colorbar(im, ax=ax, label=cbar_label)
    _draw_walls_and_marks(ax, env)
    for (r, c), a in arrows.items():
        if a is None or (r, c) == env.goal_pos:
            continue
        dr, dc = ACTION_DELTAS[a]
        ax.annotate("", xy=(c + .35 * dc, r + .35 * dr),
                    xytext=(c - .25 * dc, r - .25 * dr),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="white"))
    # terminal-state markers required by the spec
    ax.add_patch(Rectangle((env.goal_pos[1] - .5, env.goal_pos[0] - .5),
                           1, 1, fill=False, edgecolor="lime", linewidth=3))
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. visit map
# ---------------------------------------------------------------------------
def figure_visit_map(env, visits, title, out_path):
    counts = np.zeros((env.n, env.n))
    for (r, c, _, _), n in visits.items():
        counts[r, c] += n
    counts[counts == 0] = np.nan
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color=EMPTY_COLOR)
    im = ax.imshow(np.ma.masked_invalid(counts), cmap=cmap,
                   norm=matplotlib.colors.LogNorm())
    fig.colorbar(im, ax=ax, label="visits during training (log scale)")
    _draw_walls_and_marks(ax, env, mark_penalty=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 4. final greedy path
# ---------------------------------------------------------------------------
def rollout_path(env, action_fn, seed=777_000_000):
    state = env.reset(seed=seed)
    cells = [(state[0], state[1])]
    events = []
    while True:
        a = action_fn(state)
        state, r, done, info = env.step(a)
        cells.append((state[0], state[1]))
        events.append(info["event"])
        if done:
            return cells, events, info["event"] == EV_GOAL


def figure_path(env, runs, out_path):
    """runs: {label: (cells, success)} - overlays several trajectories."""
    fig, ax = plt.subplots(figsize=(9, 8))
    base = np.zeros((env.n, env.n))
    ax.imshow(base, cmap="Greys", vmin=0, vmax=1)
    _draw_walls_and_marks(ax, env)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for (label, (cells, success)), col in zip(runs.items(), colors):
        ys = [c[0] + np.random.uniform(-.08, .08) for c in cells]
        xs = [c[1] + np.random.uniform(-.08, .08) for c in cells]
        ax.plot(xs, ys, "-o", markersize=3, linewidth=1.6, color=col,
                alpha=.85,
                label=f"{label} ({'success' if success else 'failed'}, "
                      f"{len(cells) - 1} steps)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_title("Final greedy trajectories (same episode seed)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150,
                                    bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# 5. policy-difference map
# ---------------------------------------------------------------------------
def figure_policy_diff(env, Q, visits, vi_policy, label, out_path):
    """Green = same action as the Value-Iteration reference,
    red = different, grey = never visited by the model-free agent."""
    status = np.full((env.n, env.n), np.nan)
    per_cell_total, per_cell_same = defaultdict(int), defaultdict(int)
    for s, n in visits.items():
        if n == 0 or s not in vi_policy or s not in Q:
            continue
        r, c = s[0], s[1]
        per_cell_total[(r, c)] += 1
        if max(ACTIONS, key=lambda a: Q[s][a]) == vi_policy[s]:
            per_cell_same[(r, c)] += 1
    for cell, total in per_cell_total.items():
        status[cell[0], cell[1]] = per_cell_same[cell] / total

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color=EMPTY_COLOR)
    im = ax.imshow(np.ma.masked_invalid(status), cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax,
                 label="fraction of energy levels agreeing with VI")
    _draw_walls_and_marks(ax, env)
    overall = (sum(per_cell_same.values()) /
               max(1, sum(per_cell_total.values())))
    ax.set_title(f"Policy agreement with Value Iteration - {label}\n"
                 f"overall state-level agreement: {overall:.1%}")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return overall


# ---------------------------------------------------------------------------
# 6. transfer: Q and policy before vs after
# ---------------------------------------------------------------------------
def figure_transfer_diff(env, Q_before, Q_after, visits_after, out_path):
    """Left: mean change of max_a Q per cell. Right: cells whose greedy
    action changed after training in the target environment."""
    delta = np.full((env.n, env.n), np.nan)
    changed = np.full((env.n, env.n), np.nan)
    for r in range(env.n):
        for c in range(env.n):
            if env.grid[r][c] == WALL:
                continue
            diffs, flips, total = [], 0, 0
            for k in (0, 1):
                for e in range(1, env.initial_energy + 1):
                    s = (r, c, k, e)
                    if s in Q_before and s in Q_after:
                        diffs.append(max(Q_after[s]) - max(Q_before[s]))
                        total += 1
                        if (max(ACTIONS, key=lambda a: Q_after[s][a]) !=
                                max(ACTIONS, key=lambda a: Q_before[s][a])):
                            flips += 1
            if diffs:
                delta[r, c] = float(np.mean(diffs))
                changed[r, c] = flips / total

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    lim = np.nanmax(np.abs(delta)) or 1.0
    cmap1 = plt.cm.coolwarm.copy(); cmap1.set_bad(color=EMPTY_COLOR)
    im0 = axes[0].imshow(np.ma.masked_invalid(delta), cmap=cmap1,
                         vmin=-lim, vmax=lim)
    fig.colorbar(im0, ax=axes[0], label="mean change of max_a Q")
    axes[0].set_title("Q value change: after transfer training "
                      "minus source table")
    cmap2 = plt.cm.plasma.copy(); cmap2.set_bad(color=EMPTY_COLOR)
    im1 = axes[1].imshow(np.ma.masked_invalid(changed), cmap=cmap2,
                         vmin=0, vmax=1)
    fig.colorbar(im1, ax=axes[1],
                 label="fraction of states whose greedy action flipped")
    axes[1].set_title("Policy change caused by adapting to the new map")
    for ax in axes:
        _draw_walls_and_marks(ax, env)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
def _load_q(path):
    with open(path, "rb"):
        pass
    data = QLearningAgent.load(path)
    return data["Q"], data.get("visit_counts", {})


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    env = MazeEnv(map_filename="source_map.json", reward_mode="sparse",
                  seed=9)
    e_key = max(1, env.initial_energy -
                env.map_data.get("shortest_start_to_key", 0))
    produced, missing = [], []

    # ---- Value Iteration: value heatmap + final policy -------------------
    if VI_MODEL.exists():
        vi = ValueIterationAgent.load(VI_MODEL)
        for k, e in ((0, env.initial_energy), (1, e_key)):
            gv, ar = _slice_vi(env, vi["V"], vi["policy"], k, e)
            out = FIG_DIR / f"analysis_vi_value_policy_k{k}.png"
            figure_value_and_policy(
                env, gv, ar,
                f"Value Iteration: V and greedy policy "
                f"(key={k}, energy={e})", out)
            produced.append(out.name)
    else:
        missing.append("Value Iteration model (run phase 2)")

    # ---- model-free agents: max Q heatmap, visits, policy difference ----
    vi_policy = (ValueIterationAgent.load(VI_MODEL)["policy"]
                 if VI_MODEL.exists() else None)
    agreements = {}
    for label, path in (("Q-Learning", QL_MODEL),
                        ("SARSA(lambda=0.3)", SARSA_MODEL)):
        if not path.exists():
            missing.append(f"{label} model")
            continue
        Q, visits = _load_q(path)
        tag = "ql" if "Q-Learning" in label else "sarsa"
        for k in (0, 1):
            gv, ar = _aggregate_over_energy(env, Q, visits, k)
            out = FIG_DIR / f"analysis_{tag}_maxq_policy_k{k}.png"
            figure_value_and_policy(
                env, gv, ar,
                f"{label}: max_a Q and greedy policy (key={k}, "
                f"visit-weighted over energy)", out,
                cbar_label="max_a Q(s,a)")
            produced.append(out.name)
        out = FIG_DIR / f"analysis_{tag}_visit_map.png"
        figure_visit_map(env, visits, f"{label}: state visit counts", out)
        produced.append(out.name)
        if vi_policy:
            out = FIG_DIR / f"analysis_{tag}_policy_diff.png"
            agreements[label] = figure_policy_diff(
                env, Q, visits, vi_policy, label, out)
            produced.append(out.name)

    # ---- final trajectories ---------------------------------------------
    runs = {}
    if vi_policy:
        cells, _, ok = rollout_path(
            env, lambda s: vi_policy.get(s, 0))
        runs["Value Iteration"] = (cells, ok)
    for label, path in (("Q-Learning", QL_MODEL),
                        ("SARSA(0.3)", SARSA_MODEL)):
        if path.exists():
            Q, _ = _load_q(path)
            table = defaultdict(lambda: [0.0] * len(ACTIONS), Q)
            cells, _, ok = rollout_path(
                env, lambda s: max(ACTIONS, key=lambda a: table[s][a]))
            runs[label] = (cells, ok)
    if runs:
        out = FIG_DIR / "analysis_final_paths.png"
        figure_path(env, runs, out)
        produced.append(out.name)

    # ---- transfer: Q difference before vs after -------------------------
    if TRANSFER_PRE.exists() and TRANSFER_POST.exists():
        Q_before, _ = _load_q(TRANSFER_PRE)
        Q_after, visits_after = _load_q(TRANSFER_POST)
        tgt_env = MazeEnv(map_data=load_map("target_different.json"),
                          reward_mode="sparse", seed=9)
        out = FIG_DIR / "analysis_transfer_q_diff.png"
        figure_transfer_diff(tgt_env, Q_before, Q_after, visits_after, out)
        produced.append(out.name)
    else:
        missing.append("transfer models (run phase 5)")

    # ---- console summary -------------------------------------------------
    print("\n=== analysis complete ===")
    for name in produced:
        print(f"  saved results/figures/{name}")
    for label, value in agreements.items():
        print(f"  policy agreement with VI - {label}: {value:.1%}")
    if missing:
        print("\n  skipped (missing inputs):")
        for m in missing:
            print(f"    - {m}")


if __name__ == "__main__":
    main()