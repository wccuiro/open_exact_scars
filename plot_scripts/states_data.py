"""
Reading the output of the `states` (exact) and `states_traj` (trajectories) binaries.

Files: states_<state>_L<L>_g<γ>.csv (exact) and states_traj_<state>_L<L>_g<γ>.csv (trajectories), state minus-x,
neel or neel-x. A job killed at the 24 h limit leaves an incomplete last line; that line is dropped and the run is
marked as cut. A run with γ > 0 whose signal never reached STOP is also cut (the job ended at the line boundary).

One run per (state, L, γ) is selected, in this order: exact and complete, trajectories and complete, exact cut,
trajectories cut. So the exact evolution is used wherever it finished (L ≤ 14), and the trajectories above that and
where the exact L = 14 run was cut.
"""

import io
import re

import numpy as np
import pandas as pd

STOP = 1e-3
NAME = re.compile(r"states_(traj_)?(minus-x|neel-x|neel)_L(\d+)_g([0-9.e+-]+)\.csv$")
SIGNALS = {"minus-x": lambda r: np.hypot(r["mx"], r["my"]), "neel": lambda r: r["sz"].abs(),
           "neel-x": lambda r: np.hypot(r["sx"], r["sy"])}
PREFERENCE = [("exact", False), ("traj", False), ("exact", True), ("traj", True)]


def read(path):
    """The complete rows of one output file and whether the run was cut."""
    text = path.read_text()
    cut = not text.endswith("\n")
    if cut:
        text = text[:text.rfind("\n") + 1]
    return pd.read_csv(io.StringIO(text)), cut


def load(directory, states=("minus-x", "neel", "neel-x")):
    """{(method, state, L, γ): (run, cut)} for every file with at least two complete rows."""
    runs = {}
    for f in sorted(directory.glob("states_*_L*_g*.csv")):
        m = NAME.match(f.name)
        if m is None or m.group(2) not in states:
            continue
        run, cut = read(f)
        if len(run) < 2:
            continue
        state, gamma = m.group(2), float(run["gamma"].iloc[0])
        cut = cut or (gamma > 0 and SIGNALS[state](run).iloc[-1] >= STOP)
        runs["traj" if m.group(1) else "exact", state, int(run["L"].iloc[0]), gamma] = run, cut
    return runs


def select(runs):
    """{(state, L, γ): (method, run, cut)}, one run each, chosen as described in the module docstring."""
    chosen = {}
    for key in sorted({k[1:] for k in runs}):
        for method, cut in PREFERENCE:
            if (method, *key) in runs and runs[method, *key][1] == cut:
                chosen[key] = method, *runs[method, *key]
                break
    return chosen
