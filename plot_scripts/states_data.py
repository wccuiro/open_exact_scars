"""
Reading the output of the `states` (exact) and `states_traj` (trajectories) binaries.

Files: states_<state>_L<L>_g<γ>.csv (exact), states_traj_<state>_L<L>_g<γ>.csv (trajectories, diffusive
unravelling) and states_traj_<state>_L<L>_g<γ>_jump_s<N>.csv (jump unravelling, one file per traj-seed N), state
minus-x, neel or neel-x. A job killed at the 24 h limit (or by a node failure) leaves an incomplete last line; that
line is dropped and the run is marked as cut. A run with γ > 0 whose signal never reached STOP is also cut (the job
ended at the line boundary, or the run reached its --t-max first, as the small-γ runs with t-max = 3/(γ(L − 1)) do).

The jump files of one (state, L, γ) are independent samples: they are pooled into one run (method "jump") on the
time grid common to all of them (seeds that met the stop criterion end at different t), weighted by their
trajectory counts; the standard errors combine the spread within and between the seeds.

One run per (state, L, γ) is selected, in this order: exact, jump, diffusive trajectories, first the complete runs,
then the cut ones. So the exact evolution is used wherever it finished (L ≤ 14), and the trajectories above that and
where the exact L = 14 run was cut; at small γ the exact runs (L ≤ 12) are cut by their t-max, and the jump runs
are preferred over the diffusive cross-check (L = 12, 14, γ = 0.002, 0.005).
"""

import io
import re

import numpy as np
import pandas as pd

STOP = 1e-3
NAME = re.compile(r"states_(traj_)?(minus-x|neel-x|neel)_L(\d+)_g([0-9.e+-]+?)(_jump_s\d+)?\.csv$")
COLUMNS = ["mx", "my", "mz", "sx", "sy", "sz"]
SIGNALS = {"minus-x": lambda r: np.hypot(r["mx"], r["my"]), "neel": lambda r: r["sz"].abs(),
           "neel-x": lambda r: np.hypot(r["sx"], r["sy"])}
PREFERENCE = [(m, cut) for cut in (False, True) for m in ("exact", "jump", "traj")]


def read(path):
    """The complete rows of one output file and whether the run was cut."""
    text = path.read_text()
    cut = not text.endswith("\n")
    if cut:
        text = text[:text.rfind("\n") + 1]
    return pd.read_csv(io.StringIO(text)), cut


def pool(parts):
    """One run from the (run, cut) of several traj-seeds: mean weighted by the trajectory counts, and the standard
    error from the pooled variance (within plus between seeds), on the common time grid."""
    n_t = min(len(run) for run, _ in parts)
    runs = [run.iloc[:n_t].reset_index(drop=True) for run, _ in parts]
    n = np.array([run["trajectories"].iloc[0] for run in runs], dtype=float)
    total = n.sum()
    out = runs[0][["L", "gamma", "t"]].copy()
    out["trajectories"] = int(total)
    for c in COLUMNS:
        mean = sum(w * run[c] for w, run in zip(n, runs)) / total
        within = sum((w - 1) * w * run[c + "_err"] ** 2 for w, run in zip(n, runs))  # (n − 1) · variance
        between = sum(w * (run[c] - mean) ** 2 for w, run in zip(n, runs))
        out[c] = mean
        out[c + "_err"] = np.sqrt((within + between) / (total - 1) / total)
    return out, any(cut for _, cut in parts)


def load(directory, states=("minus-x", "neel", "neel-x")):
    """{(method, state, L, γ): (run, cut)} for every file with at least two complete rows; method exact, traj
    (diffusive) or jump (all traj-seeds pooled)."""
    runs, seeds = {}, {}
    for f in sorted(directory.glob("states_*_L*_g*.csv")):
        m = NAME.match(f.name)
        if m is None or m.group(2) not in states:
            continue
        run, cut = read(f)
        if len(run) < 2:
            continue
        method = "jump" if m.group(5) else "traj" if m.group(1) else "exact"
        key = method, m.group(2), int(run["L"].iloc[0]), float(run["gamma"].iloc[0])
        if method == "jump":
            seeds.setdefault(key, []).append((run, cut))
        else:
            runs[key] = run, cut
    for key, parts in seeds.items():
        runs[key] = pool(parts)
    for key, (run, cut) in list(runs.items()):
        state, gamma = key[1], key[3]
        runs[key] = run, cut or (gamma > 0 and SIGNALS[state](run).iloc[-1] >= STOP)
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
