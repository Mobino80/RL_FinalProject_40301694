# RL Final Project — Dynamic Maze Agent

**Reinforcement Learning course — final project**
**Student ID: 40301694**

An intelligent agent in a dynamic 16 × 16 maze with stochastic movement
and a limited energy budget. The environment is modelled as an MDP and
solved with three algorithms — Value Iteration, Q-Learning and
SARSA(λ) — followed by a transfer-learning study on Q-Learning and a
standalone graphical interface.

---

## Environment at a glance

| Item | Value |
|---|---|
| Maze size | 16 × 16 (`N = 15 + (base_seed mod 4)`) |
| Base seed | 9 — the second-to-last digit of the student ID |
| State | `s = (row, col, k, energy)` |
| Actions | up, down, left, right |
| Transition noise | 0.8 intended, 0.1 / 0.1 perpendicular |
| Extra capability | **limited energy** (initial budget: 50) |
| Task | start → key → locked door → goal |

---

## Installation

Requires **Python 3.10+** (developed and tested on 3.12).

```bash
git clone https://github.com/Mobino80/RL_FinalProject_40301694.git
cd RL_FinalProject_40301694
pip install -r requirements.txt
```

`tkinter` is required by the GUI and cannot be installed with pip. It
ships with the official Python installers on Windows and macOS; on
Debian/Ubuntu run `sudo apt-get install python3-tk`.

---

## Reproducing the results

Run these commands **in order** from the repository root — each step
consumes the outputs of the previous ones. All randomness is seeded, so
re-running reproduces the committed numbers exactly.

| # | Command | What it produces | Time |
|---|---|---|---|
| 0 | `pytest tests/ -v` | 17 unit tests on the environment | 1 s |
| 1 | `python environments/generator.py` | source map + BFS validation | 1 s |
| 2 | `python experiments/run_value_iteration.py` | VI models, γ study, heatmaps | 40 s |
| 3 | `python experiments/run_q_learning.py` | 4 Q-Learning runs, logs, curves | 3 min |
| 4 | `python experiments/run_sarsa_lambda.py` | λ sweep, trace logs, curves | 10 min |
| 5 | `python experiments/run_stability.py` | multi-seed stability analysis for Q-Learning and SARSA | 24 min |
| 6 | `python experiments/run_transfer.py` | target maps, 12 transfer runs | 3 min |
| 7 | `python experiments/analysis.py` | all figures used in the report | 30 s |
| 9 | `python main.py` | graphical interface | — |

Each script writes its configuration (with a timestamp) to
`experiments/configs/*.json`, raw measurements to `results/raw_data/`,
models to `results/models/` and figures to `results/figures/`. Every
number in the report is read from these files; none was entered by hand.

---

## Repository layout

```
RL_FinalProject_40301694/
├── environments/
│   ├── generator.py         deterministic map generation + BFS validation
│   ├── maze.py              MDP: transitions, rewards, events, logging
│   └── maps/                source and target maps (JSON)
├── agents/
│   ├── value_iteration.py   Bellman sweeps, greedy policy extraction
│   ├── q_learning.py        off-policy TD(0) + epsilon schedules
│   └── sarsa_lambda.py      on-policy TD(lambda) + eligibility traces
├── transfer/
│   └── transfer_learning.py target maps, 4 transfer scenarios, metrics
├── gui/
│   ├── app.py               Tkinter application
│   └── renderer.py          canvas rendering, events, energy bar
├── experiments/
│   ├── run_*.py             one script per phase
│   ├── analysis.py          all required visual outputs
│   └── configs/             one JSON per experiment
├── results/
│   ├── raw_data/            per-episode CSVs, Q-update logs, summaries
│   ├── models/              saved value functions and Q tables
│   └── figures/             every figure used in the report
├── tests/                   unit tests for the environment
├── main.py                  entry point: launches the GUI
├── report.pdf               methodology, results and analysis
├── requirements.txt
└── README.md
```

---

## Using the graphical interface

```bash
python main.py
```

Select an **algorithm** (Value Iteration / Q-Learning / SARSA(λ)), an
**environment** (source / target-similar / target-different) and a
**mode** (training or evaluation), then use start, pause, resume, reset
and re-run to control playback. The speed slider sets 1–120 steps per
second, and the policy-arrow overlay can be toggled at any time.

* **evaluation** — loads a trained model from `results/models/` and
  follows it greedily. Run steps 1–5 above first, otherwise no model
  exists yet.
* **training** — starts from an empty Q table and learns while you
  watch. The *fast-forward* button trains 10 000 episodes without
  animation; ten presses take ε from 1.0 down to 0.05.

Value Iteration is model-based and has no episodes, so it always runs
from its computed optimal policy. If no cached model exists for the
selected map it is computed in a background thread.

---

## Implementation notes

* No ready-made RL library is used: Value Iteration, Q-Learning and
  SARSA(λ) are implemented from scratch. Only NumPy, Matplotlib, pandas
  and tkinter are required.
* BFS appears only for map validation and never replaces the learning
  agent.
* Two reward functions are available on every environment
  (`reward_mode="sparse"` or `"shaped"`); the shaped version uses
  potential-based shaping.

References: Sutton & Barto, *Reinforcement Learning: An Introduction*
(2nd ed.); Ng, Harada & Russell (1999) for potential-based shaping.

Methodology, experimental results and analysis are in **`report.pdf`**.
