"""
⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩ (site averages) against time, starting from |−⟩^⊗L, one curve per L.

Left column:  γ = 0, the check: perfect oscillations, identical for every L.
Right column: the fixed small γ of the run: does the decay depend on L?

Input:  data/evolution_L<L>.csv   (cd rust && cargo run --release)
Output: data/evolution.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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

files = sorted(DATA.glob("evolution_L*.csv"), key=lambda p: int(p.stem.split("_L")[1]))
if not files:
    raise SystemExit(f"no evolution_L*.csv in {DATA}; run `cargo run --release` in rust/ first")
runs = {int(p.stem.split("_L")[1]): pd.read_csv(p) for p in files}
blues = LinearSegmentedColormap.from_list("blues", RAMP)
colors = {L: blues(i / max(len(runs) - 1, 1)) for i, L in enumerate(runs)}
gamma = max(df["gamma"].max() for df in runs.values())

observables = [("mx", "⟨σ$^x$⟩"), ("my", "⟨σ$^y$⟩"), ("mz", "⟨σ$^z$⟩")]
fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True, sharey=True, constrained_layout=True)
for row, (column, label) in zip(axes, observables):
    for ax, g in zip(row, [0.0, gamma]):
        for L, df in runs.items():
            run = df[df["gamma"] == g]
            ax.plot(run["t"], run[column], color=colors[L])
        ax.set_xlim(run["t"].min(), run["t"].max())
    row[0].set_ylabel(label)
axes[0, 0].set_title("γ = 0  (check: perfect oscillations)")
axes[0, 1].set_title(f"γ = {gamma:g}")
axes[0, 0].set_ylim(-1.1, 1.1)
for ax in axes[-1]:
    ax.set_xlabel("t")
fig.legend(handles=[Line2D([], [], color=c, linewidth=2, label=f"L = {L}") for L, c in colors.items()],
           loc="outside upper center", ncols=len(colors))

out = DATA / "evolution.png"
fig.savefig(out)
print(f"wrote {out}")
