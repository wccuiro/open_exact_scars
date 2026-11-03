"""
Time series of the `states` runs (exact evolution) and `states_traj` runs (trajectories), one panel per γ, one line
per L. One run per (L, γ), chosen by states_data.select: exact where it finished (L ≤ 14), trajectories otherwise.

  states_scar.png      scar |−⟩^⊗L: m_⊥(t) = √(m_x² + m_y²) against γt (log scale), with e^(−2γt)
  states_neel.png      z-Néel: |s_z(t)|, s_z = (1/L) Σ_j (−1)^j ⟨σᶻ_j⟩ (log scale)
  states_neel-x.png    x-Néel |+−+−…⟩: s_⊥(t) = √(s_x² + s_y²) (log scale), with e^(−2γt)
  states_neel_n.png    z-Néel: ⟨n⟩/L, n = number of 1-bits (down spins) = Σ_j (1 − σᶻ_j)/2,
                       so ⟨n⟩/L = (1 − m_z)/2; 1/2 at t = 0 and in the steady state 𝟙/2^L

The runs stop once the decay signal (m_⊥ or |s_z|) stays below 1e-3, so ⟨n⟩ may not have relaxed yet.

Input:  states_<state>_L<L>_g<γ>.csv and states_traj_<state>_L<L>_g<γ>.csv in data/used_data/ (or the directory given
        as the first argument)
Only γ = 0 and γ ≥ 0.01 (notes_layout.GAMMA_MIN), at most 4 panels per figure: <name>_1.png, <name>_2.png, ...
Output: the figures, in the same directory
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import states_data
from notes_layout import grid_shape, keep_gamma, page_name, pages

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
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
    chosen = states_data.select(states_data.load(DIR, states=(state,)))
    return {(l, g): run for (_, l, g), (_, run, _) in chosen.items()}


def grid(runs, gammas, signal, x, ylabel, title, stem, log=True, ref=None, hline=None):
    """One panel per γ, at most MAX_PANELS per figure: <stem>.png, or <stem>_1.png, <stem>_2.png, ..."""
    if not runs:
        return
    sizes = sorted({l for l, _ in runs})
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    colors = {l: blues(i / max(len(sizes) - 1, 1)) for i, l in enumerate(sizes)}
    chunks = pages(gammas)
    for k, chunk in enumerate(chunks, 1):
        nrows, ncols = grid_shape(len(chunk))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), squeeze=False,
                                 layout="constrained")
        for ax, g in zip(axes.flat, chunk):
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
        for ax in axes[-1]:
            ax.set_xlabel("γt" if x(None, 1.0) is None else "t")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        handles = [Line2D([], [], color=colors[l], lw=2, label=f"L = {l}") for l in sizes]
        if ref is not None:
            handles.append(Line2D([], [], color=MUTED, ls=":", label="e^(−2γt), H_I = 0"))
        fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 5))
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
        file = page_name(stem, k, len(chunks))
        fig.savefig(DIR / file)
        plt.close(fig)
        print(f"wrote {DIR / file}")


def main():
    scar, neel, neel_x = load("minus-x"), load("neel"), load("neel-x")
    g_scar = sorted({g for _, g in scar if g > 0 and keep_gamma(g)})
    g_neel = sorted({g for _, g in neel if keep_gamma(g)})
    g_neel_x = sorted({g for _, g in neel_x if keep_gamma(g)})

    def gt(run, g):
        return None if run is None else run["t"] * g

    def t(run, g):
        return 0 if run is None else run["t"]

    grid(scar, g_scar, lambda r: np.hypot(r["mx"], r["my"]), gt, "m_⊥",
         "Scar |−⟩^⊗L: transverse magnetization against γt", "states_scar", ref=lambda x, g: np.exp(-2 * x))
    grid(neel, g_neel, lambda r: r["sz"].abs(), t, "|s_z|", "z-Néel: staggered magnetization", "states_neel")
    grid(neel_x, g_neel_x, lambda r: np.hypot(r["sx"], r["sy"]), t, "s_⊥",
         "x-Néel |+−+−…⟩: staggered transverse magnetization", "states_neel-x",
         ref=lambda x, g: np.exp(-2 * g * x))
    grid(neel, g_neel, lambda r: (1 - r["mz"]) / 2, t, "⟨n⟩ / L",
         "z-Néel: number of down spins ⟨n⟩/L = (1 − m_z)/2", "states_neel_n", log=False, hline=0.5)


if __name__ == "__main__":
    main()
