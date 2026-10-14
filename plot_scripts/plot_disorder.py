"""
Oscillations of ⟨σˣ⟩, ⟨σʸ⟩ and ⟨σᶻ⟩ with random c coefficients, from product states on a cone around −x with
random azimuths (θ = 0 is the scar state |−⟩^⊗L), exact evolution.

One column per cone angle θ; top row γ = 0, bottom row the dissipative γ. Each panel shows the
site average m_α(t) = (1/L) Σ_j ⟨σ^α_j⟩ for every L, one figure per α. For α = x, y the bottom row
also shows the free-spin envelope ±A(0) e^(−2γt), A(0) = √(m_x(0)² + m_y(0)²) at the largest L
(free spins keep m_z constant, so the z figure has no reference).

Input:  data/disorder_L<L>.csv   (cd rust && cargo run --release --bin disorder -- --lmax 8)
Output: data/disorder_mx.png, data/disorder_my.png, data/disorder_mz.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"mx": "⟨σˣ⟩", "my": "⟨σʸ⟩", "mz": "⟨σᶻ⟩"}
DATA = Path(__file__).resolve().parent.parent / "data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # small L light, large L dark

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def main():
    files = sorted(DATA.glob("disorder_L*.csv"), key=lambda p: int(p.stem.split("_L")[1]))
    if not files:
        raise SystemExit(f"no disorder_L*.csv in {DATA}; see the docstring for the run")
    runs = {int(p.stem.split("_L")[1]): pd.read_csv(p, usecols=["gamma", "theta", "t", "mx", "my", "mz"]) for p in files}
    sizes = sorted(runs)
    colors = dict(zip(sizes, RAMP[-len(sizes):] if len(sizes) <= len(RAMP) else plt.cm.Blues(np.linspace(0.4, 1, len(sizes)))))
    any_run = runs[sizes[-1]]
    thetas = sorted(any_run["theta"].unique())
    gamma = max(any_run["gamma"].unique())
    for component in LABELS:
        plot(runs, sizes, colors, thetas, gamma, component)


def plot(runs, sizes, colors, thetas, gamma, component):
    fig, axes = plt.subplots(2, len(thetas), figsize=(2.6 * len(thetas), 4.6), sharex=True, sharey=True, squeeze=False)
    for col, theta in enumerate(thetas):
        for row, g in enumerate([0.0, gamma]):
            ax = axes[row, col]
            for l in sizes:
                run = runs[l][(runs[l]["gamma"] == g) & np.isclose(runs[l]["theta"], theta)]
                ax.plot(run["t"], run[component], color=colors[l], lw=1.0, label=f"L = {l}")
            if g > 0 and component != "mz":
                t = run["t"].to_numpy()
                a0 = np.hypot(run["mx"].iloc[0], run["my"].iloc[0])
                for sign in (1, -1):
                    ax.plot(t, sign * a0 * np.exp(-2 * g * t), color=MUTED, ls=":", lw=1.2,
                            label="±A(0) e^(−2γt)" if sign > 0 else None)
            ax.set_title(f"θ = {theta / np.pi:g}π, γ = {g:g}")
            if component != "mz":
                ax.set_ylim(-1.05, 1.05)
        axes[1, col].set_xlabel("t")
    for row in range(2):
        axes[row, 0].set_ylabel(LABELS[component])
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    out = DATA / f"disorder_{component}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
