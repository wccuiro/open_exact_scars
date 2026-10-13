"""
Compare the ITensor TDVP runs and the trajectories with the exact evolution (ED).

Figures (in data/):
  tdvp_convergence.png      error of the TDVP magnetization against time, one curve per bond
                            dimension, one panel per L; error = max over α = x, y, z of |m_α − m_α(ED)|
  trajectories_vs_ed.png    ⟨σˣ⟩ from the trajectories (band = ± one standard error) on top of ED
Printed: the same errors as tables.

Input:  data/evolution_L<L>.csv            (cargo run --release)
        data/tdvp_L<L>_chi<χ>_dt<dt>.csv   (julia --project=. lindblad_tdvp.jl L χ1,χ2,...)
        data/trajectories_L<L>.csv         (cargo run --release --bin trajectories)
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DATA = Path(__file__).resolve().parent.parent / "data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
BLUE, ORANGE = "#2a78d6", "#eb6834"
COMPONENTS = ["mx", "my", "mz"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def exact(L, gamma):
    df = pd.read_csv(DATA / f"evolution_L{L}.csv")
    return df[np.isclose(df["gamma"], gamma)].reset_index(drop=True)


def error_against_exact(run, ed):
    """(t, max over components of |m − m_ED|) at the times of `run` that are on the ED time grid."""
    steps = run["t"].to_numpy() / (ed["t"].iloc[1] - ed["t"].iloc[0])
    on_grid = np.abs(steps - np.rint(steps)) < 1e-6
    index = np.rint(steps[on_grid]).astype(int)
    err = np.max([np.abs(run[c].to_numpy()[on_grid] - ed[c].to_numpy()[index]) for c in COMPONENTS], axis=0)
    return run["t"].to_numpy()[on_grid], err


def tdvp():
    runs = {}  # L -> {(chi, dt): DataFrame}
    for path in DATA.glob("tdvp_L*_chi*_dt*.csv"):
        m = re.fullmatch(r"tdvp_L(\d+)_chi(\d+)_dt([0-9.]+)\.csv", path.name)
        df = pd.read_csv(path)
        if m and len(df) > 1:
            runs.setdefault(int(m.group(1)), {})[(int(m.group(2)), float(m.group(3)))] = df
    if not runs:
        print("no tdvp_L*_chi*_dt*.csv in data/")
        return
    sizes = sorted(L for L in runs if (DATA / f"evolution_L{L}.csv").exists())  # only sizes with an ED reference
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.6 * len(sizes), 4.0), sharey=True, squeeze=False,
                             constrained_layout=True)
    print("TDVP: max over x, y, z of |m - m_ED|")
    for ax, L in zip(axes[0], sizes):
        keys = sorted(runs[L])
        dims = sorted({chi for chi, _ in keys})
        color = {chi: BLUES(i / max(len(dims) - 1, 1)) for i, chi in enumerate(dims)}
        for chi, dt in keys:
            df = runs[L][(chi, dt)]
            t, err = error_against_exact(df, exact(L, df["gamma"].iloc[0]))
            coarse = dt == max(d for c, d in keys if c == chi)
            ax.plot(t[1:], err[1:], color=color[chi], linestyle="-" if coarse else (0, (2, 2)),
                    label=f"χ = {chi}" + ("" if coarse else f", dt = {dt:g}"))
            at = lambda time: f"{err[np.searchsorted(t, time)]:.1e}" if t[-1] >= time else "   -   "
            per_step = (df["seconds"].iloc[-1] - df["seconds"].iloc[len(df) // 2]) / (len(df) - 1 - len(df) // 2)
            print(f"  L = {L}  chi = {chi:4d}  dt = {dt:<5g}  error at t = 5, 10, 20: {at(5)}  {at(10)}  {at(20)}"
                  f"   max {err.max():.1e}   {per_step:.2f} s/step")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_title(f"L = {L}  (exact at χ = {4 ** (L // 2)})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=8)
    axes[0, 0].set_ylabel("max$_α$ |⟨σ$^α$⟩ − ED|")
    fig.savefig(DATA / "tdvp_convergence.png")
    print(f"wrote {DATA / 'tdvp_convergence.png'}")


def trajectories():
    files = sorted(DATA.glob("trajectories_L*.csv"), key=lambda p: int(p.stem.split("_L")[1]))
    if not files:
        print("no trajectories_L*.csv in data/")
        return
    fig, axes = plt.subplots(len(files), 1, figsize=(9, 1.9 * len(files)), sharex=True, squeeze=False,
                             constrained_layout=True)
    print("trajectories: deviation from ED in units of the standard error, and spread of one trajectory")
    for ax, path in zip(axes[:, 0], files):
        df = pd.read_csv(path)
        L, gamma, n = int(df["L"].iloc[0]), df["gamma"].iloc[0], int(df["trajectories"].iloc[0])
        ed = exact(L, gamma)
        ax.plot(ed["t"], ed["mx"], color=INK, linewidth=1.0, label="ED")
        ax.fill_between(df["t"], df["mx"] - df["mx_err"], df["mx"] + df["mx_err"], color=ORANGE, alpha=0.35, linewidth=0)
        ax.plot(df["t"], df["mx"], color=ORANGE, linewidth=1.0, label="trajectories ± standard error")
        ax.set_ylabel("⟨σ$^x$⟩")
        ax.set_title(f"L = {L}, {n} trajectories")
        z = [(df[c] - ed[c]).to_numpy()[1:] / df[c + "_err"].to_numpy()[1:] for c in COMPONENTS]
        print(f"  L = {L}  {n} trajectories  rms deviation / error: " +
              "  ".join(f"{c} {np.sqrt(np.mean(v ** 2)):.2f}" for c, v in zip(COMPONENTS, z)) +
              f"   largest single-trajectory std: mx {df['mx_std'].max():.3f}  my {df['my_std'].max():.3f}  mz {df['mz_std'].max():.3f}")
    axes[0, 0].legend(loc="upper right", ncols=2, fontsize=8)
    axes[-1, 0].set_xlabel("t")
    axes[-1, 0].set_xlim(0, df["t"].max())
    fig.savefig(DATA / "trajectories_vs_ed.png")
    print(f"wrote {DATA / 'trajectories_vs_ed.png'}")


if __name__ == "__main__":
    tdvp()
    trajectories()
