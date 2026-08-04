"""
gui/app.py
----------
Standalone Tkinter interface for the dynamic maze - phase 6.

RL Final Project - Student ID: 40301694

Controls required by the project spec, all implemented here:
  * algorithm selector          : Value Iteration / Q-Learning / SARSA(lambda)
  * environment selector        : source / target-similar / target-different
  * mode selector               : training or evaluation
  * start, pause, resume, reset, re-run
  * animation-speed control     : slider (steps per second)
  * policy display on/off       : greedy arrows for the current slice
  * live readout                : episode, step, reward, epsilon,
                                  key status, energy, recent success rate

Design notes:
  * Animation uses Tk's `after()` loop rather than threads, so all canvas
    updates happen on the main thread (Tkinter is not thread-safe). The
    only exception is the Value-Iteration computation, which is slow
    (~10 s) and therefore runs in a worker thread while the UI stays
    responsive; the result is picked up by a polling `after()` callback.
  * In TRAINING mode the agent really learns while you watch: every
    animated step performs a genuine Q-Learning / SARSA(lambda) update.
    Because watching 150k episodes is impractical, a "fast-forward"
    button trains N episodes without rendering and then resumes the
    animation - this is how you can watch the policy improve.
  * In EVALUATION mode a trained model is loaded from results/models/
    and followed greedily (epsilon = 0).

Run from the project root:
    python gui/app.py          (or: python main.py)
"""

import queue
import sys
import threading
import tkinter as tk
from collections import defaultdict, deque
from pathlib import Path
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.generator import load_map, WALL
from environments.maze import MazeEnv, ACTIONS, EV_GOAL
from agents.q_learning import QLearningAgent, ExponentialDecay
from agents.sarsa_lambda import SarsaLambdaAgent, TRACE_THRESHOLD
from agents.value_iteration import ValueIterationAgent
from gui.renderer import MazeRenderer, EnergyBar

# ---------------------------------------------------------------------------
# Configuration: which map files and which trained models belong together
# ---------------------------------------------------------------------------
ENVIRONMENTS = {
    "source":            "source_map.json",
    "target (similar)":  "target_similar.json",
    "target (different)": "target_different.json",
}

ALGORITHMS = ["Value Iteration", "Q-Learning", "SARSA(lambda)"]

# pre-trained models produced by phases 2-5
MODEL_FILES = {
    ("Value Iteration", "source"):
        "results/models/vi_gamma_0.99.pkl",
    ("Q-Learning", "source"):
        "results/models/ql_sparse_exponential_q.pkl",
    ("Q-Learning", "target (similar)"):
        "results/models/transfer_similar_full_q.pkl",
    ("Q-Learning", "target (different)"):
        "results/models/transfer_different_full_q.pkl",
    ("SARSA(lambda)", "source"):
        "results/models/sarsa_lam0.3_replacing_q.pkl",
}

TRAIN_EPISODES_HINT = 150000     # horizon used by the offline experiments
DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_LAMBDA = 0.2, 0.99, 0.3
FAST_FORWARD_EPISODES = 10000
# Interactive training happens in bursts of FAST_FORWARD_EPISODES, so the
# 150k horizon of the offline scripts would keep epsilon pinned near 1.0
# for the whole session. The GUI therefore decays epsilon over a horizon
# that is reachable interactively: 10 bursts take it from 1.0 to 0.05,
# i.e. 100k episodes, comparable to the offline experiments.
GUI_EPSILON_HORIZON = 10 * FAST_FORWARD_EPISODES


class GuiEpsilonSchedule:
    """Exponential epsilon decay driven by CUMULATIVE trained episodes.

    The agents' `train()` loop calls the schedule with an episode index
    that restarts at 0 on every call. In the GUI, training is done in
    repeated bursts, so that local index must be shifted by the number of
    episodes already trained - otherwise every burst would restart at
    epsilon = 1.0 and the agent would explore randomly forever (which is
    exactly the failure mode this class fixes).

    `offset` is kept in sync with MazeApp.episodes_trained.
    """

    name = "exponential (GUI, cumulative)"

    def __init__(self, start=1.0, end=0.05, horizon=GUI_EPSILON_HORIZON):
        self.start, self.end = start, end
        self.horizon = max(1, horizon)
        self.rate = (end / start) ** (1.0 / self.horizon)
        self.offset = 0

    def __call__(self, episode_in_burst):
        n = self.offset + episode_in_burst
        return max(self.end, self.start * (self.rate ** n))


class MazeApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("RL Final Project 40301694 - Dynamic Maze Agent")
        self.resizable(False, False)

        # runtime state
        self.env = None
        self.agent = None
        self.renderer = None
        self.energy_bar = None
        self.running = False
        self.after_id = None
        self.vi_queue = queue.Queue()
        self.vi_thread = None

        self.episode = 0
        self.episodes_trained = 0     # only counts real training episodes
        self.step_in_episode = 0
        self.episode_reward = 0.0
        self.recent_results = deque(maxlen=100)
        self.sarsa_traces = {}
        self.sarsa_action = None
        self.last_cell = None

        self._build_widgets()
        self._load_environment()

    # ------------------------------------------------------------------ #
    # Widget construction                                                 #
    # ------------------------------------------------------------------ #
    def _build_widgets(self):
        outer = ttk.Frame(self, padding=8)
        outer.grid(row=0, column=0)

        # ---- left: board -------------------------------------------------
        board_frame = ttk.Frame(outer)
        board_frame.grid(row=0, column=0, sticky="n")
        self.canvas = tk.Canvas(board_frame, width=560, height=560,
                                background="white", highlightthickness=1,
                                highlightbackground="#adb5bd")
        self.canvas.grid(row=0, column=0)
        self.energy_canvas = tk.Canvas(board_frame, width=560, height=20,
                                       background="white",
                                       highlightthickness=0)
        self.energy_canvas.grid(row=1, column=0, pady=(6, 0))
        self.event_label = ttk.Label(board_frame, text="ready",
                                     font=("Segoe UI", 10, "bold"),
                                     foreground="#495057")
        self.event_label.grid(row=2, column=0, pady=(6, 0))

        # ---- right: controls --------------------------------------------
        panel = ttk.Frame(outer, padding=(14, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="n")
        row = 0

        ttk.Label(panel, text="Setup",
                  font=("Segoe UI", 11, "bold")).grid(row=row, column=0,
                                                      columnspan=2,
                                                      sticky="w")
        row += 1
        ttk.Label(panel, text="algorithm").grid(row=row, column=0,
                                                sticky="w")
        self.algo_var = tk.StringVar(value="Q-Learning")
        algo_box = ttk.Combobox(panel, textvariable=self.algo_var,
                                values=ALGORITHMS, state="readonly",
                                width=17)
        algo_box.grid(row=row, column=1, sticky="w", pady=2)
        algo_box.bind("<<ComboboxSelected>>", lambda e: self.reset())
        row += 1

        ttk.Label(panel, text="environment").grid(row=row, column=0,
                                                  sticky="w")
        self.env_var = tk.StringVar(value="source")
        env_box = ttk.Combobox(panel, textvariable=self.env_var,
                               values=list(ENVIRONMENTS), state="readonly",
                               width=17)
        env_box.grid(row=row, column=1, sticky="w", pady=2)
        env_box.bind("<<ComboboxSelected>>",
                     lambda e: self._load_environment())
        row += 1

        ttk.Label(panel, text="mode").grid(row=row, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="evaluation")
        mode_box = ttk.Combobox(panel, textvariable=self.mode_var,
                                values=["training", "evaluation"],
                                state="readonly", width=17)
        mode_box.grid(row=row, column=1, sticky="w", pady=2)
        mode_box.bind("<<ComboboxSelected>>", lambda e: self.reset())
        row += 1

        ttk.Separator(panel, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(panel, text="Playback",
                  font=("Segoe UI", 11, "bold")).grid(row=row, column=0,
                                                      columnspan=2,
                                                      sticky="w")
        row += 1
        btns = ttk.Frame(panel)
        btns.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        self.start_btn = ttk.Button(btns, text="Start", width=8,
                                    command=self.start)
        self.start_btn.grid(row=0, column=0, padx=2)
        self.pause_btn = ttk.Button(btns, text="Pause", width=8,
                                    command=self.pause)
        self.pause_btn.grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="Resume", width=8,
                   command=self.start).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(btns, text="Reset", width=8,
                   command=self.reset).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(btns, text="Re-run episode", width=18,
                   command=self.rerun_episode).grid(row=2, column=0,
                                                    columnspan=2, pady=2)
        row += 1

        self.ff_btn = ttk.Button(
            panel, text=f"Fast-forward {FAST_FORWARD_EPISODES} episodes",
            command=self.fast_forward)
        self.ff_btn.grid(row=row, column=0, columnspan=2, sticky="ew",
                         pady=2)
        row += 1

        ttk.Label(panel, text="speed (steps/sec)").grid(row=row, column=0,
                                                        sticky="w",
                                                        pady=(8, 0))
        self.speed_var = tk.DoubleVar(value=12)
        ttk.Scale(panel, from_=1, to=120, variable=self.speed_var,
                  orient="horizontal", length=150).grid(row=row, column=1,
                                                        pady=(8, 0))
        row += 1

        self.policy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="show policy arrows",
                        variable=self.policy_var,
                        command=self._refresh_policy).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row += 1
        self.policy_layer_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.policy_layer_var,
                  foreground="#7048e8",
                  font=("Segoe UI", 9)).grid(row=row, column=0,
                                             columnspan=2, sticky="w")
        row += 1

        self.trail_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="show agent trail",
                        variable=self.trail_var).grid(
            row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Separator(panel, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        ttk.Label(panel, text="Live status",
                  font=("Segoe UI", 11, "bold")).grid(row=row, column=0,
                                                      columnspan=2,
                                                      sticky="w")
        row += 1
        self.info_vars = {}
        for field in ("episode", "episodes trained", "step",
                      "episode reward", "epsilon", "key", "energy",
                      "recent success (100 ep)", "states learned"):
            ttk.Label(panel, text=field).grid(row=row, column=0,
                                              sticky="w")
            var = tk.StringVar(value="-")
            ttk.Label(panel, textvariable=var,
                      font=("Consolas", 10)).grid(row=row, column=1,
                                                  sticky="w")
            self.info_vars[field] = var
            row += 1

        ttk.Separator(panel, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        self.status_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.status_var, wraplength=200,
                  foreground="#1864ab").grid(row=row, column=0,
                                             columnspan=2, sticky="w")

    # ------------------------------------------------------------------ #
    # Environment / agent setup                                           #
    # ------------------------------------------------------------------ #
    def _load_environment(self):
        self.pause()
        name = self.env_var.get()
        try:
            map_data = load_map(ENVIRONMENTS[name])
        except FileNotFoundError:
            messagebox.showerror(
                "map not found",
                f"'{ENVIRONMENTS[name]}' is missing.\n"
                "Run environments/generator.py (source map) or "
                "experiments/run_transfer.py (target maps) first.")
            return
        self.env = MazeEnv(map_data=map_data, reward_mode="sparse", seed=9)
        cs = min(34, 560 // map_data["size"])
        self.renderer = MazeRenderer(self.canvas, map_data, cell_size=cs)
        self.renderer.draw_board()
        size = self.renderer.board_pixels()
        self.canvas.config(width=size, height=size)
        self.energy_canvas.config(width=size)
        self.energy_bar = EnergyBar(self.energy_canvas, size)
        self.reset()

    def _model_path(self):
        key = (self.algo_var.get(), self.env_var.get())
        rel = MODEL_FILES.get(key)
        return (ROOT / rel) if rel else None

    def _build_agent(self):
        """Create the agent for the current algorithm / mode / env."""
        algo, mode = self.algo_var.get(), self.mode_var.get()
        path = self._model_path()

        if algo == "Value Iteration":
            # Value Iteration is model-based and has no episodic training,
            # so the GUI always uses its computed optimal policy. If no
            # cached model exists for this map, compute it in a worker
            # thread so the interface does not freeze.
            if mode == "training":
                self.status_var.set(
                    "Value Iteration is model-based: it has no episodes. "
                    "The GUI computes its optimal policy and then runs it.")
            if path and path.exists():
                data = ValueIterationAgent.load(path)
                self.vi_policy = data["policy"]
                self.agent = "vi"
                return True
            self._start_vi_thread()
            return False

        schedule = GuiEpsilonSchedule()
        if algo == "Q-Learning":
            agent = QLearningAgent(self.env, alpha=DEFAULT_ALPHA,
                                   gamma=DEFAULT_GAMMA,
                                   epsilon_schedule=schedule, seed=9)
        else:
            agent = SarsaLambdaAgent(self.env, alpha=DEFAULT_ALPHA,
                                     gamma=DEFAULT_GAMMA,
                                     lam=DEFAULT_LAMBDA,
                                     trace_type="replacing",
                                     epsilon_schedule=schedule, seed=9)

        if mode == "evaluation":
            if path is None or not path.exists():
                messagebox.showwarning(
                    "no trained model",
                    f"No saved model for {algo} on '{self.env_var.get()}'.\n"
                    "Switch to training mode, or run the matching "
                    "experiment script first.")
                self.status_var.set("no trained model - use training mode")
                self.agent = agent
                return True
            data = type(agent).load(path)
            agent.Q = defaultdict(lambda: [0.0] * len(ACTIONS), data["Q"])
            agent.visit_counts = defaultdict(int,
                                             data.get("visit_counts", {}))
            self.status_var.set(f"loaded {path.name}")
        else:
            self.status_var.set("training from scratch - watch it learn")
        self.agent = agent
        return True

    def _start_vi_thread(self):
        """Compute Value Iteration off the UI thread."""
        if self.vi_thread and self.vi_thread.is_alive():
            return
        self.status_var.set("computing Value Iteration (about 10 s)...")
        env = self.env

        def work():
            solver = ValueIterationAgent(env, gamma=DEFAULT_GAMMA)
            solver.run(verbose=False)
            self.vi_queue.put(solver.policy)

        self.vi_thread = threading.Thread(target=work, daemon=True)
        self.vi_thread.start()
        self.after(200, self._poll_vi)

    def _poll_vi(self):
        try:
            policy = self.vi_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_vi)
            return
        self.vi_policy = policy
        self.agent = "vi"
        self.status_var.set("Value Iteration ready - press Start")
        self._refresh_policy()

    # ------------------------------------------------------------------ #
    # Policy access (uniform interface for the three algorithms)          #
    # ------------------------------------------------------------------ #
    def _greedy_action(self, state):
        if self.agent == "vi":
            return self.vi_policy.get(state, 0)
        return self.agent.greedy_action(state)

    def _refresh_policy(self):
        """Redraw the arrow overlay.

        Value Iteration has a value for every state, so a single
        (key, energy) slice is enough. A learned Q table, however, only
        contains the states the agent actually visited, and since energy
        drops by one on every step, a fixed-energy slice is almost
        empty: a cell is only ever seen at the particular energy levels
        that match its distance from the start. Therefore, for the
        model-free agents the arrow of each cell is taken from the MOST
        VISITED energy level of that cell (with the current key flag),
        which is the behaviour that is typical there.
        """
        if self.renderer is None:
            return
        if not self.policy_var.get() or self.agent is None:
            self.renderer.clear_policy()
            self.policy_layer_var.set("")
            return
        state = self.env.state or self.env.reset()
        k, e = state[2], state[3]
        self.policy_layer_var.set(
            f"   arrows: layer key={k} "
            f"({'with key -> door/goal' if k else 'no key -> find key'})")
        arrows = {}
        for r in range(self.env.n):
            for c in range(self.env.n):
                if self.env.grid[r][c] == WALL:
                    continue
                if self.agent == "vi":
                    arrows[(r, c)] = self.vi_policy.get((r, c, k, e))
                    continue
                best_w, best_s = -1, None
                for energy in range(1, self.env.initial_energy + 1):
                    s = (r, c, k, energy)
                    if s not in self.agent.Q:
                        continue
                    w = self.agent.visit_counts.get(s, 0)
                    if w > best_w:
                        best_w, best_s = w, s
                arrows[(r, c)] = (
                    max(ACTIONS, key=lambda a: self.agent.Q[best_s][a])
                    if best_s is not None else None)
        self.renderer.draw_policy(arrows)

    # ------------------------------------------------------------------ #
    # Episode control                                                     #
    # ------------------------------------------------------------------ #
    def reset(self):
        """Full reset: rebuild the agent and start a fresh episode."""
        self.pause()
        if self.env is None:
            return
        self.episode = 0
        self.episodes_trained = 0
        self.recent_results.clear()
        if not self._build_agent():
            return                      # VI still computing
        self._begin_episode()

    def rerun_episode(self):
        """Replay the current episode with the same seed."""
        self.pause()
        self._begin_episode(same_seed=True)

    def _begin_episode(self, same_seed=False):
        if not same_seed:
            self.episode_seed = 9 * 1_000_000 + self.episode
        state = self.env.reset(seed=self.episode_seed)
        self.step_in_episode = 0
        self.episode_reward = 0.0
        self.sarsa_traces = {}
        self.last_cell = (state[0], state[1])
        self.renderer.draw_board()
        self.renderer.clear_path()
        self.renderer.clear_overlay()
        self.renderer.update_key_and_door(state[2])
        self.renderer.draw_agent(state)
        self.energy_bar.update(state[3], self.env.initial_energy)
        self.event_label.config(text="episode start", foreground="#495057")

        if isinstance(self.agent, SarsaLambdaAgent):
            eps = self._current_epsilon()
            self.sarsa_action = self.agent.choose_action(state, eps)
        self._refresh_policy()
        self._update_info()

    def _current_epsilon(self):
        if self.mode_var.get() == "evaluation" or self.agent == "vi":
            return 0.0
        # keep the schedule anchored to the episodes actually trained
        schedule = self.agent.eps_schedule
        if isinstance(schedule, GuiEpsilonSchedule):
            schedule.offset = self.episodes_trained
            return schedule(0)
        return schedule(self.episodes_trained)

    def start(self):
        if self.agent is None:
            self.reset()
            return
        if self.running:
            return
        self.running = True
        self._tick()

    def pause(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    # ------------------------------------------------------------------ #
    # Animation loop - one environment step per tick                      #
    # ------------------------------------------------------------------ #
    def _tick(self):
        if not self.running:
            return
        try:
            self._do_step()
        except Exception as exc:            # keep the UI alive on error
            self.pause()
            messagebox.showerror("error during step", str(exc))
            return
        delay = max(8, int(1000 / max(1.0, self.speed_var.get())))
        self.after_id = self.after(delay, self._tick)

    def _do_step(self):
        # Robustness: if the episode already finished (for example the
        # user paused exactly on the terminal step and then resumed),
        # roll over into the next episode instead of stepping a closed
        # environment.
        if self.env.done:
            self.episode += 1 if self.step_in_episode else 0
            self._begin_episode()
            return
        state = self.env.state
        eps = self._current_epsilon()
        training = (self.mode_var.get() == "training"
                    and self.agent != "vi")

        # ---- pick the action ------------------------------------------
        if isinstance(self.agent, SarsaLambdaAgent) and training:
            action = self.sarsa_action
        elif training:
            action = self.agent.choose_action(state, eps)
        else:
            action = self._greedy_action(state)

        next_state, reward, done, info = self.env.step(action)

        # ---- learn (only in training mode) ----------------------------
        if training and isinstance(self.agent, QLearningAgent):
            self.agent.visit_counts[state] += 1
            self.agent.update(state, action, reward, next_state, done)
        elif training and isinstance(self.agent, SarsaLambdaAgent):
            self._sarsa_update(state, action, reward, next_state, done, eps)

        # ---- render ----------------------------------------------------
        cell = (next_state[0], next_state[1])
        if self.trail_var.get():
            self.renderer.draw_path_segment(self.last_cell, cell)
        self.last_cell = cell
        self.renderer.update_key_and_door(next_state[2])
        self.renderer.draw_agent(next_state)
        self.renderer.flash_event(cell, info["event"])
        self.energy_bar.update(next_state[3], self.env.initial_energy)
        self._show_event(info["event"])

        # The optimal policy differs before and after the key is taken
        # (k is part of the state), so the arrow overlay must switch to
        # the other layer the moment the key flag changes - otherwise the
        # displayed arrows would silently belong to the wrong layer.
        if next_state[2] != state[2] and self.policy_var.get():
            self._refresh_policy()

        self.episode_reward += reward
        self.step_in_episode += 1
        self._update_info()

        if done:
            success = info["event"] == EV_GOAL
            self.recent_results.append(1 if success else 0)
            text = ("SUCCESS - goal reached" if success
                    else f"FAILED - {info['event'].replace('_', ' ').lower()}")
            self.renderer.show_episode_result(success, text)
            self.episode += 1
            if training:
                self.episodes_trained += 1
            self.pause()
            self.after(700, self._next_episode_if_running)

    def _next_episode_if_running(self):
        """Chain into the next episode automatically."""
        self._begin_episode()
        self.running = True
        self._tick()

    def _sarsa_update(self, s, a, r, s_next, done, eps):
        """One SARSA(lambda) step with eligibility traces (same maths as
        agents/sarsa_lambda.py, unrolled here for the live animation)."""
        agent = self.agent
        agent.visit_counts[s] += 1
        a_next = None if done else agent.choose_action(s_next, eps)
        q_next = 0.0 if done else agent.Q[s_next][a_next]
        delta = r + agent.gamma * q_next - agent.Q[s][a]
        self.sarsa_traces[(s, a)] = 1.0             # replacing traces
        decay = agent.gamma * agent.lam
        dead = []
        for pair, e_val in self.sarsa_traces.items():
            agent.Q[pair[0]][pair[1]] += agent.alpha * delta * e_val
            new_e = e_val * decay
            if new_e < TRACE_THRESHOLD:
                dead.append(pair)
            else:
                self.sarsa_traces[pair] = new_e
        for pair in dead:
            del self.sarsa_traces[pair]
        self.sarsa_action = a_next

    # ------------------------------------------------------------------ #
    # Fast-forward: train without rendering                               #
    # ------------------------------------------------------------------ #
    def fast_forward(self):
        if self.agent is None or self.agent == "vi":
            messagebox.showinfo(
                "not applicable",
                "Fast-forward trains a model-free agent; Value Iteration "
                "has no episodes.")
            return
        if self.mode_var.get() != "training":
            messagebox.showinfo("training mode required",
                                "Switch the mode selector to 'training'.")
            return
        self.pause()
        # continue the decay from the episodes already trained instead of
        # restarting the schedule at epsilon = 1.0 for every burst
        if isinstance(self.agent.eps_schedule, GuiEpsilonSchedule):
            self.agent.eps_schedule.offset = self.episodes_trained
        eps_before = self._current_epsilon()
        self.status_var.set(
            f"training {FAST_FORWARD_EPISODES} episodes without "
            f"animation (epsilon starts at {eps_before:.3f})...")
        self.update_idletasks()
        self.agent.train(FAST_FORWARD_EPISODES,
                         f"gui_{self.algo_var.get().split('(')[0].lower()}",
                         ROOT / "results" / "raw_data",
                         base_seed=9, verbose_every=0)
        self.episode += FAST_FORWARD_EPISODES
        self.episodes_trained += FAST_FORWARD_EPISODES
        self.status_var.set(
            f"trained {self.episodes_trained} episodes in total - "
            f"epsilon now {self._current_epsilon():.3f} - press Start")
        self._begin_episode()

    # ------------------------------------------------------------------ #
    # Status panel                                                        #
    # ------------------------------------------------------------------ #
    def _show_event(self, event):
        colours = {"WALL_HIT": "#e03131", "PENALTY_CELL": "#e8590c",
                   "KEY_PICKUP": "#f08c00", "DOOR_PASS": "#7048e8",
                   "DOOR_LOCKED_ATTEMPT": "#e8590c",
                   "GOAL_REACHED": "#2f9e44",
                   "ENERGY_DEPLETED": "#c92a2a",
                   "STEP_LIMIT": "#868e96", "MOVE": "#495057"}
        self.event_label.config(text=event.replace("_", " "),
                                foreground=colours.get(event, "#495057"))

    def _update_info(self):
        state = self.env.state
        recent = (sum(self.recent_results) / len(self.recent_results)
                  if self.recent_results else 0.0)
        n_states = ("-" if self.agent == "vi" or self.agent is None
                    else str(len(self.agent.Q)))
        values = {
            "episode": str(self.episode),
            "episodes trained": str(self.episodes_trained),
            "step": str(self.step_in_episode),
            "episode reward": f"{self.episode_reward:8.2f}",
            "epsilon": f"{self._current_epsilon():.3f}",
            "key": "YES" if state[2] else "no",
            "energy": f"{state[3]} / {self.env.initial_energy}",
            "recent success (100 ep)": f"{recent:.0%}",
            "states learned": n_states,
        }
        for field, value in values.items():
            self.info_vars[field].set(value)


def main():
    app = MazeApp()
    app.mainloop()


if __name__ == "__main__":
    main()