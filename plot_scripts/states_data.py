"""
Reading the output of the `states` (exact) and `states_traj` (trajectories) binaries.

Files: states_<state>_L<L>_g<γ>.csv (exact), states_traj_<state>_L<L>_g<γ>[_jump][_s<N>].csv (trajectories,
diffusive or jump unravelling; _s<N>: one file per traj-seed N), state minus-x, neel or neel-x (σᶻ dephasing and
realization seed 1 only: files with _ch… or _seed<N> are not read). A job killed at the 24 h limit (or by a node
failure) leaves an incomplete last line; that line is dropped and the run is marked as cut. A run with γ > 0 whose
signal never met the stop criterion is also cut (the job ended at the line boundary, or the run reached its --t-max
first, as the small-γ runs with t-max = 3/(γ(L − 1)) do). The criterion is the binaries' own: signal below STOP
(exact), below max(STOP, 3 × its standard error) (trajectories, radial error √(σ_x² + σ_y²) for m_⊥ and s_⊥).

The traj-seed files of one (state, L, γ, method) are independent samples: they are pooled into one run on the times
common to all of them (matched by value; seeds that met the stop criterion end at different t), weighted by their
trajectory counts; the standard errors (and the covariances mxmy_cov, sxsy_cov where every file has them) combine
the spread within and between the seeds. A file without traj-seed next to an _s1 file of the same run is dropped:
without --traj-seed the noise seed is --seed (1), so the two are the same sample.

One run per (state, L, γ) is selected: complete runs before cut ones; among the complete runs exact, jump, diffusive;
among the cut runs exact, then the trajectory run (jump or diffusive) with the smaller standard error of the signal
at its 1/e crossing. So the exact evolution is used wherever it finished (L ≤ 14), and the trajectories above that
and where the exact L = 14 run was cut; at small γ the exact runs (L ≤ 12) are cut by their t-max, and at L = 14,
γ = 0.002, 0.005 (jump and diffusive both cut) the diffusive cross-check is the more precise run.
"""

import io
import re

import numpy as np
import pandas as pd

STOP = 1e-3
NAME = re.compile(r"states_(traj_)?(minus-x|neel-x|neel)_L(\d+)_g([0-9.e+-]+?)(_jump)?(_s\d+)?\.csv$")
COLUMNS = ["mx", "my", "mz", "sx", "sy", "sz"]
SIGNALS = {"minus-x": lambda r: np.hypot(r["mx"], r["my"]), "neel": lambda r: r["sz"].abs(),
           "neel-x": lambda r: np.hypot(r["sx"], r["sy"])}
# standard error of the signal as the Rust stop criterion takes it (radial bound for m_⊥, s_⊥)
SIGNAL_ERRORS = {"minus-x": lambda r: np.hypot(r["mx_err"], r["my_err"]), "neel": lambda r: r["sz_err"],
                 "neel-x": lambda r: np.hypot(r["sx_err"], r["sy_err"])}
COVARIANCES = {"mxmy_cov": ("mx", "my"), "sxsy_cov": ("sx", "sy")}
METHODS = ("exact", "jump", "traj")


def read(path):
    """The complete rows of one output file and whether the run was cut."""
    text = path.read_text()
    cut = not text.endswith("\n")
    if cut:
        text = text[:text.rfind("\n") + 1]
    return pd.read_csv(io.StringIO(text)), cut


def pool(parts):
    """One run from the (run, cut) of several traj-seeds: mean weighted by the trajectory counts, and the standard
    error from the pooled variance (within plus between seeds), on the times common to all of them."""
    common = set(parts[0][0]["t"])
    for run, _ in parts[1:]:
        common &= set(run["t"])
    runs = [run[run["t"].isin(common)].reset_index(drop=True) for run, _ in parts]
    assert all(np.array_equal(run["t"], runs[0]["t"]) for run in runs)
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
    if all(cov in run for run in runs for cov in COVARIANCES):
        for cov, (a, b) in COVARIANCES.items():
            within = sum((w - 1) * w * run[cov] for w, run in zip(n, runs))
            between = sum(w * (run[a] - out[a]) * (run[b] - out[b]) for w, run in zip(n, runs))
            out[cov] = (within + between) / (total - 1) / total
    return out, any(cut for _, cut in parts)


def load(directory, states=("minus-x", "neel", "neel-x")):
    """{(method, state, L, γ): (run, cut)} for every file with at least two complete rows; method exact, traj
    (diffusive) or jump (traj-seed files pooled)."""
    runs, files = {}, {}
    for f in sorted(directory.glob("states_*_L*_g*.csv")):
        m = NAME.match(f.name)
        if m is None or m.group(2) not in states:
            continue
        run, cut = read(f)
        if len(run) < 2:
            continue
        method = "jump" if m.group(5) else "traj" if m.group(1) else "exact"
        key = method, m.group(2), int(run["L"].iloc[0]), float(run["gamma"].iloc[0])
        files.setdefault(key, {})[m.group(6) or ""] = run, cut
    for key, parts in files.items():
        if "" in parts and "_s1" in parts:
            print(f"states_data: {key}: the file without traj-seed is the same sample as _s1; dropped")
            del parts[""]
        runs[key] = next(iter(parts.values())) if len(parts) == 1 else pool(list(parts.values()))
    for key, (run, cut) in list(runs.items()):
        state, gamma = key[1], key[3]
        threshold = STOP if key[0] == "exact" else max(STOP, 3 * SIGNAL_ERRORS[state](run).iloc[-1])
        runs[key] = run, cut or (gamma > 0 and SIGNALS[state](run).iloc[-1] >= threshold)
    return runs


def error_at_te(state, run):
    """Standard error of the signal at the first time its envelope drops below 1/e (inf if it never does)."""
    signal = SIGNALS[state](run).to_numpy()
    envelope = np.maximum.accumulate(signal[::-1])[::-1]
    below = np.flatnonzero(envelope < np.exp(-1))
    return SIGNAL_ERRORS[state](run).iloc[below[0]] if len(below) else np.inf


def select(runs):
    """{(state, L, γ): (method, run, cut)}, one run each, chosen as described in the module docstring."""
    chosen = {}
    for key in sorted({k[1:] for k in runs}):
        for cut in (False, True):
            candidates = [m for m in METHODS if (m, *key) in runs and runs[m, *key][1] == cut]
            if not candidates:
                continue
            method = candidates[0]
            if cut and method != "exact" and len(candidates) > 1:
                method = min(candidates, key=lambda m: error_at_te(key[0], runs[m, *key][0]))
            chosen[key] = method, *runs[method, *key]
            break
    return chosen
