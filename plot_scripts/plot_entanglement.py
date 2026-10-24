"""
Half-chain entanglement entropy of every eigenstate of H (γ = 0) against its energy, one panel per L.
Bulk eigenstates as small blue dots, the L + 1 scars |Φ_n⟩ (tower weight > 1/2) as orange circles, and the Page
value L_A ln 2 − 2^{L_A} / 2^{L − L_A + 1} (random state) as a dotted line.

Input:  entanglement_L<L>.csv (from the `entanglement` binary: L, L_A, E, S, tower) in data/
        used_data/ (or the directory given as the first argument); every size found is plotted
Output: entanglement.png, in the same directory
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.2, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

files = sorted(DIR.glob("entanglement_L*.csv"), key=lambda f: int(re.search(r"_L(\d+)", f.name).group(1)))
if not files:
    sys.exit(f"no entanglement_L*.csv in {DIR}")

ncols = min(len(files), 4)
nrows = -(-len(files) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), squeeze=False, layout="constrained")
for ax, file in zip(axes.flat, files):
    df = pd.read_csv(file)
    l, l_a = int(df["L"].iloc[0]), int(df["L_A"].iloc[0])
    scar = df["tower"] > 0.5
    bulk = df[~scar]
    ax.scatter(bulk["E"], bulk["S"], s=min(6.0, 3000 / len(bulk)), color=BLUE, alpha=0.5, linewidths=0)
    ax.scatter(df.loc[scar, "E"], df.loc[scar, "S"], s=28, color=ORANGE, edgecolors=SURFACE, linewidths=1.0, zorder=3)
    page = l_a * np.log(2) - 0.5 * 2.0**l_a / 2.0 ** (l - l_a)
    ax.axhline(page, color=MUTED, ls=":", lw=1.1)
    ax.set_ylim(-0.1, l_a * np.log(2) * 1.08)
    ax.set_title(f"L = {l}, L_A = {l_a}")
for ax in axes.flat[len(files):]:
    ax.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("E")
for ax in axes[:, 0]:
    ax.set_ylabel("S_A")
handles = [
    Line2D([], [], color=BLUE, marker="o", ls="", ms=4, alpha=0.6, label="bulk eigenstates"),
    Line2D([], [], color=ORANGE, marker="o", ls="", ms=6, label="scars |Φ_n⟩"),
    Line2D([], [], color=MUTED, ls=":", label="Page value"),
]
fig.legend(handles=handles, loc="outside upper center", ncols=3)
fig.savefig(DIR / "entanglement.png")
print(f"wrote {DIR / 'entanglement.png'}")
