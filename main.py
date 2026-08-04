"""
main.py
-------
Entry point of the project - launches the graphical interface.

RL Final Project - Student ID: 40301694

Usage (from the project root):

    python main.py                      # open the GUI

Related commands (see README.md):

    python environments/generator.py        # regenerate the source map
    python experiments/run_value_iteration.py
    python experiments/run_q_learning.py
    python experiments/run_sarsa_lambda.py
    python experiments/run_transfer.py
    python experiments/analysis.py          # all visual outputs
    pytest tests/ -v                        # unit tests
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    source_map = ROOT / "environments" / "maps" / "source_map.json"
    if not source_map.exists():
        print("The source map is missing. Generating it first...")
        from environments import generator
        data = generator.generate_valid_map()
        generator.save_map(data)

    try:
        from gui.app import MazeApp
    except ImportError as exc:
        print("Could not start the interface:", exc)
        print("Tkinter is required. On Debian/Ubuntu install it with "
              "'sudo apt-get install python3-tk'.")
        return 1

    MazeApp().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())