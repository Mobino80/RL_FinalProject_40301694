"""
gui/renderer.py
---------------
Canvas renderer for the dynamic maze - phase 6.

RL Final Project - Student ID: 40301694

Responsibilities:
  * draw the static map (walls, normal cells, penalty cells, start, key,
    door, goal) with clearly distinct visual codes,
  * draw the agent and keep it in sync with the environment state,
  * make every required EVENT visible at the moment it happens
      - wall / obstacle bump      -> red flash on the agent cell
      - entering a penalty cell   -> orange flash
      - picking up the key        -> the key cell empties, the agent
                                     gets a key badge, the door turns
                                     from "locked" to "open"
      - passing the door          -> door highlight
      - success / failure         -> green / red overlay on the board,
  * draw the LIMITED-ENERGY feature (the chosen extra capability) as a
    live energy bar plus a numeric readout, so the feature is visible in
    the GUI as the spec requires,
  * optionally overlay the greedy-policy arrows for the CURRENT (k, e)
    slice of the state space.

The renderer holds no RL logic: it only translates a state plus an event
into canvas items. All learning happens in gui/app.py and agents/.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.generator import (FREE, WALL, PENALTY, START, KEY, DOOR,
                                    GOAL)
from environments.maze import ACTION_DELTAS

# --------------------------------------------------------------------------
# Visual code of every cell type (distinct colours + distinct glyphs, so the
# board stays readable even for colour-blind readers and in grayscale print)
# --------------------------------------------------------------------------
CELL_STYLE = {
    FREE:    {"fill": "#f2f2ef", "label": ""},
    WALL:    {"fill": "#4a3428", "label": ""},
    PENALTY: {"fill": "#ff9b6a", "label": "!"},
    START:   {"fill": "#9fd8f5", "label": "S"},
    KEY:     {"fill": "#f5cd51", "label": "K"},
    DOOR:    {"fill": "#a06cd5", "label": "D"},
    GOAL:    {"fill": "#5ad18b", "label": "G"},
}

DOOR_OPEN_FILL = "#d9c2f2"      # door colour once the key is held
AGENT_FILL     = "#1f4fd8"
AGENT_KEY_FILL = "#0f8f4f"      # agent carrying the key
GRID_OUTLINE   = "#c8c8c4"

FLASH_COLORS = {
    "WALL_HIT":            "#e03131",
    "DOOR_LOCKED_ATTEMPT": "#e8590c",
    "PENALTY_CELL":        "#ff8c1a",
    "KEY_PICKUP":          "#ffd43b",
    "DOOR_PASS":           "#b197fc",
    "GOAL_REACHED":        "#37b24d",
    "ENERGY_DEPLETED":     "#c92a2a",
    "STEP_LIMIT":          "#868e96",
}


class MazeRenderer:
    """Draws a MazeEnv on a Tkinter canvas."""

    def __init__(self, canvas, map_data, cell_size=34):
        self.canvas = canvas
        self.map_data = map_data
        self.grid = map_data["grid"]
        self.n = map_data["size"]
        self.cs = cell_size
        self.key_pos = tuple(map_data["key"])
        self.door_pos = tuple(map_data["door"])
        self.goal_pos = tuple(map_data["goal"])
        self.initial_energy = map_data["initial_energy"]

        self.cell_items = {}        # (r, c) -> canvas rectangle id
        self.label_items = {}       # (r, c) -> canvas text id
        self.base_fill = {}         # (r, c) -> colour to restore after a
                                    # flash (kept in sync with key/door)
        self.agent_item = None
        self.agent_label = None
        self.policy_items = []
        self.path_items = []
        self.overlay_items = []
        self._flash_after_id = None

    # ------------------------------------------------------------------ #
    # Static board                                                        #
    # ------------------------------------------------------------------ #
    def draw_board(self):
        """(Re)draw the whole static map. Call once per environment."""
        self.canvas.delete("all")
        self.cell_items.clear()
        self.label_items.clear()
        self.base_fill.clear()
        self.agent_item = self.agent_label = None
        self.policy_items, self.path_items, self.overlay_items = [], [], []

        for r in range(self.n):
            for c in range(self.n):
                style = CELL_STYLE[self.grid[r][c]]
                x0, y0 = c * self.cs, r * self.cs
                rect = self.canvas.create_rectangle(
                    x0, y0, x0 + self.cs, y0 + self.cs,
                    fill=style["fill"], outline=GRID_OUTLINE)
                self.cell_items[(r, c)] = rect
                self.base_fill[(r, c)] = style["fill"]
                if style["label"]:
                    self.label_items[(r, c)] = self.canvas.create_text(
                        x0 + self.cs / 2, y0 + self.cs / 2,
                        text=style["label"],
                        font=("Segoe UI", int(self.cs * 0.42), "bold"),
                        fill="#333333")

    def board_pixels(self):
        return self.n * self.cs

    # ------------------------------------------------------------------ #
    # Dynamic elements                                                    #
    # ------------------------------------------------------------------ #
    def draw_agent(self, state):
        """Draw / move the agent. `state` = (row, col, k, energy)."""
        r, c, k, _ = state
        x0 = c * self.cs + self.cs * 0.18
        y0 = r * self.cs + self.cs * 0.18
        x1 = c * self.cs + self.cs * 0.82
        y1 = r * self.cs + self.cs * 0.82
        fill = AGENT_KEY_FILL if k else AGENT_FILL

        if self.agent_item is None:
            self.agent_item = self.canvas.create_oval(
                x0, y0, x1, y1, fill=fill, outline="white", width=2)
            self.agent_label = self.canvas.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2, text="",
                font=("Segoe UI", int(self.cs * 0.34), "bold"),
                fill="white")
        else:
            self.canvas.coords(self.agent_item, x0, y0, x1, y1)
            self.canvas.itemconfig(self.agent_item, fill=fill)
            self.canvas.coords(self.agent_label,
                               (x0 + x1) / 2, (y0 + y1) / 2)
        # a small badge shows that the agent is carrying the key
        self.canvas.itemconfig(self.agent_label, text="K" if k else "")
        self.canvas.tag_raise(self.agent_item)
        self.canvas.tag_raise(self.agent_label)

    def update_key_and_door(self, k):
        """Reflect the key/door state: once the key is taken, the key
        cell becomes an ordinary cell and the door is shown as open."""
        kr, kc = self.key_pos
        if (kr, kc) in self.cell_items:
            fill = (CELL_STYLE[FREE]["fill"] if k
                    else CELL_STYLE[KEY]["fill"])
            self.base_fill[(kr, kc)] = fill
            self.canvas.itemconfig(self.cell_items[(kr, kc)], fill=fill)
            if (kr, kc) in self.label_items:
                self.canvas.itemconfig(self.label_items[(kr, kc)],
                                       text="" if k else "K")
        dr, dc = self.door_pos
        if (dr, dc) in self.cell_items:
            fill = DOOR_OPEN_FILL if k else CELL_STYLE[DOOR]["fill"]
            self.base_fill[(dr, dc)] = fill
            self.canvas.itemconfig(self.cell_items[(dr, dc)], fill=fill)
            if (dr, dc) in self.label_items:
                self.canvas.itemconfig(self.label_items[(dr, dc)],
                                       text="D" if not k else "open")

    def flash_event(self, cell, event, duration_ms=180):
        """Briefly recolour a cell so the event is visible in real time.

        The colour is restored from `base_fill` rather than from whatever
        is currently shown, so two overlapping flashes can never leave a
        cell stuck in a flash colour.
        """
        if event not in FLASH_COLORS or cell not in self.cell_items:
            return
        item = self.cell_items[cell]
        self.canvas.itemconfig(item, fill=FLASH_COLORS[event])
        self.canvas.after(
            duration_ms,
            lambda: self.canvas.itemconfig(
                item, fill=self.base_fill.get(cell,
                                              CELL_STYLE[FREE]["fill"])))

    def show_episode_result(self, success, text):
        """Translucent-looking banner at the end of an episode."""
        self.clear_overlay()
        w = self.board_pixels()
        colour = "#37b24d" if success else "#c92a2a"
        rect = self.canvas.create_rectangle(
            w * 0.08, w * 0.42, w * 0.92, w * 0.58,
            fill=colour, outline="white", width=2)
        label = self.canvas.create_text(
            w / 2, w / 2, text=text, fill="white",
            font=("Segoe UI", max(11, int(self.cs * 0.5)), "bold"))
        self.overlay_items = [rect, label]

    def clear_overlay(self):
        for item in self.overlay_items:
            self.canvas.delete(item)
        self.overlay_items = []

    # ------------------------------------------------------------------ #
    # Policy overlay                                                      #
    # ------------------------------------------------------------------ #
    def draw_policy(self, action_of_cell):
        """Draw greedy arrows. `action_of_cell` maps (r, c) -> action or
        None. The caller decides which (k, energy) slice to show."""
        self.clear_policy()
        for (r, c), action in action_of_cell.items():
            if action is None or self.grid[r][c] == WALL:
                continue
            if (r, c) == self.goal_pos:
                continue
            dr, dc = ACTION_DELTAS[action]
            cx = c * self.cs + self.cs / 2
            cy = r * self.cs + self.cs / 2
            L = self.cs * 0.30
            arrow = self.canvas.create_line(
                cx - dc * L * 0.6, cy - dr * L * 0.6,
                cx + dc * L, cy + dr * L,
                arrow="last", width=2, fill="#495057")
            self.policy_items.append(arrow)
        if self.agent_item is not None:
            self.canvas.tag_raise(self.agent_item)
            self.canvas.tag_raise(self.agent_label)

    def clear_policy(self):
        for item in self.policy_items:
            self.canvas.delete(item)
        self.policy_items = []

    # ------------------------------------------------------------------ #
    # Trajectory trace                                                    #
    # ------------------------------------------------------------------ #
    def draw_path_segment(self, cell_from, cell_to):
        """Leave a faint trail behind the agent (the 'final path' view)."""
        if cell_from == cell_to:
            return
        x0 = cell_from[1] * self.cs + self.cs / 2
        y0 = cell_from[0] * self.cs + self.cs / 2
        x1 = cell_to[1] * self.cs + self.cs / 2
        y1 = cell_to[0] * self.cs + self.cs / 2
        line = self.canvas.create_line(x0, y0, x1, y1, width=3,
                                       fill="#1f4fd8", stipple="gray50")
        self.path_items.append(line)
        # keep the trail above the board but below the agent marker
        if self.agent_item is not None:
            self.canvas.tag_raise(self.agent_item)
            self.canvas.tag_raise(self.agent_label)

    def clear_path(self):
        for item in self.path_items:
            self.canvas.delete(item)
        self.path_items = []


class EnergyBar:
    """Small canvas widget showing the LIMITED-ENERGY extra feature."""

    def __init__(self, canvas, width, height=18):
        self.canvas = canvas
        self.w, self.h = width, height
        self.bg = canvas.create_rectangle(0, 0, width, height,
                                          fill="#e9ecef", outline="#adb5bd")
        self.bar = canvas.create_rectangle(0, 0, 0, height,
                                           fill="#2f9e44", outline="")
        self.text = canvas.create_text(width / 2, height / 2, text="",
                                       font=("Segoe UI", 9, "bold"),
                                       fill="#212529")

    def update(self, energy, initial_energy):
        frac = max(0.0, min(1.0, energy / max(1, initial_energy)))
        self.canvas.coords(self.bar, 0, 0, self.w * frac, self.h)
        # green -> amber -> red as the budget runs out
        colour = ("#2f9e44" if frac > 0.5
                  else "#f59f00" if frac > 0.2 else "#e03131")
        self.canvas.itemconfig(self.bar, fill=colour)
        self.canvas.itemconfig(
            self.text, text=f"energy  {energy} / {initial_energy}")
        self.canvas.tag_raise(self.text)