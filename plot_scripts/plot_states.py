"""
Time series of the `states` runs (exact evolution), one panel per γ, one line per L.

  states_scar.png      scar |−⟩^⊗L: m_⊥(t) = √(m_x² + m_y²) against γt (log scale), with e^(−2γt)
  states_neel.png      z-Néel: |s_z(t)|, s_z = (1/L) Σ_j (−1)^j ⟨σᶻ_j⟩ (log scale)
  states_neel-x.png    x-Néel |+−+−…⟩: s_⊥(t) = √(s_x² + s_y²) (log scale), with e^(−2γt)
  states_neel_n.png    z-Néel: ⟨n⟩/L, n = number of 1-bits (down spins) = Σ_j (1 − σᶻ_j)/2,
                       so ⟨n⟩/L = (1 − m_z)/2; 1/2 at t = 0 and in the steady state 𝟙/2^L

The runs stop once the decay signal (m_⊥ or |s_z|) stays below 1e-3, so ⟨n⟩ may not have relaxed yet.

Input:  states_<state>_L<L>_g<γ>.csv in data/states/ (or the directory given as the first argument)
Output: the three figures, in the same directory
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "states"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]  # small L light, large L dark

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.2, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(state):
    runs = {}
    for f in DIR.glob(f"states_{state}_L*_g*.csv"):
        run = pd.read_csv(f)
        if len(run) > 1:
            runs[int(run["L"].iloc[0]), float(run["gamma"].iloc[0])] = run
    return runs


def grid(runs, gammas, signal, x, ylabel, title, file, log=True, ref=None, hline=None):
    if not runs:
        return
    sizes = sorted({l for l, _ in runs})
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    colors = {l: blues(i / max(len(sizes) - 1, 1)) for i, l in enumerate(sizes)}
    ncols = 4
    nrows = -(-len(gammas) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.3 * nrows), squeeze=False, layout="constrained")
    for ax, g in zip(axes.flat, gammas):
        for l in sizes:
            if (l, g) in runs:
                run = runs[l, g]
                ax.plot(x(run, g), signal(run), color=colors[l])
        if ref is not None and g > 0:
            xs = np.linspace(0, ax.get_xlim()[1], 200)
            ax.plot(xs, ref(xs, g), color=MUTED, ls=":", lw=1.1)
        if hline is not None:
            ax.axhline(hline, color=MUTED, ls=":", lw=1.1)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.5)
        ax.set_title(f"γ = {g:g}")
    for ax in axes.flat[len(gammas):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("γt" if x(None, 1.0) is None else "t")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=colors[l], lw=2, label=f"L = {l}") for l in sizes]
    if ref is not None:
        handles.append(Line2D([], [], color=MUTED, ls=":", label="e^(−2γt), H_I = 0"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.savefig(DIR / file)
    print(f"wrote {DIR / file}")


def main():
    scar, neel, neel_x = load("minus-x"), load("neel"), load("neel-x")
    g_scar = sorted({g for _, g in scar if g > 0})
    g_neel = sorted({g for _, g in neel})
    g_neel_x = sorted({g for _, g in neel_x})

    def gt(run, g):
        return None if run is None else run["t"] * g

    def t(run, g):
        return 0 if run is None else run["t"]

    grid(scar, g_scar, lambda r: np.hypot(r["mx"], r["my"]), gt, "m_⊥",
         "Scar |−⟩^⊗L: transverse magnetization against γt", "states_scar.png", ref=lambda x, g: np.exp(-2 * x))
    grid(neel, g_neel, lambda r: r["sz"].abs(), t, "|s_z|", "z-Néel: staggered magnetization", "states_neel.png")
    grid(neel_x, g_neel_x, lambda r: np.hypot(r["sx"], r["sy"]), t, "s_⊥",
         "x-Néel |+−+−…⟩: staggered transverse magnetization", "states_neel-x.png",
         ref=lambda x, g: np.exp(-2 * g * x))
    grid(neel, g_neel, lambda r: (1 - r["mz"]) / 2, t, "⟨n⟩ / L",
         "z-Néel: number of down spins ⟨n⟩/L = (1 − m_z)/2", "states_neel_n.png", log=False, hline=0.5)


if __name__ == "__main__":
    main()
