"""
Operator entanglement of the right eigenmatrices X of the Lindbladian, one figure per γ, one row per L:
left column S_op(ρ_L) against E_L = Tr(ρ_L H), right column S_op(ρ_R) against E_R = Tr(ρ_R H), with
ρ_L = X X† / Tr(X X†) and ρ_R = X† X / Tr(X† X). Modes with tower weight ‖P X P‖ / ‖X‖ > 0.9 in orange, the
rest in blue; dotted line: the largest possible value 2 L_A ln 2.

𝓛 is real in the Pauli basis, so the eigenmatrix of λ* is X† and ρ_R(λ) = ρ_L(λ*): the two columns show the same
set of points (checked: 1e-12), each mode appearing in the left column at λ and in the right one at λ*.
At γ = 0 every X is one |n⟩⟨m| (ρ_L = |n⟩⟨n|), so each level appears 2^L times at (E_n, 2 S_n).

Input:  lindblad_entanglement_L<L>_g<γ>.csv (from the `lindblad_entanglement` binary: L, L_A, gamma, re, im, E_L, S_L,
        E_R, S_R, tower) in data/used_data/ (or the directory given as the first argument); every size found
Output: lindblad_entanglement_g<γ>.png, in the same directory
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

runs = {}
for f in DIR.glob("lindblad_entanglement_L*_g*.csv"):
    l, g = re.fullmatch(r"lindblad_entanglement_L(\d+)_g(.+)\.csv", f.name).groups()
    runs.setdefault(g, []).append((int(l), f))
if not runs:
    sys.exit(f"no lindblad_entanglement_L*_g*.csv in {DIR}")
handles = [
    Line2D([], [], color=BLUE, marker="o", ls="", ms=4, alpha=0.6, label="Liouvillian modes"),
    Line2D([], [], color=ORANGE, marker="o", ls="", ms=5, label="tower modes (weight > 0.9)"),
    Line2D([], [], color=MUTED, ls=":", label="2 L_A ln 2 (maximum)"),
]

for g, sizes in sorted(runs.items(), key=lambda item: float(item[0])):
    files = [f for _, f in sorted(sizes)][-2:]  # the two largest L: 2 × 2 panels (notes_layout.MAX_PANELS)
    fig, axes = plt.subplots(len(files), 2, figsize=(7.6, 2.9 * len(files)), squeeze=False, layout="constrained")
    for row, file in zip(axes, files):
        df = pd.read_csv(file)
        l, l_a, gamma = int(df["L"].iloc[0]), int(df["L_A"].iloc[0]), df["gamma"].iloc[0]
        tower = df["tower"] > 0.9
        for ax, side in zip(row, ("L", "R")):
            e, s = df[f"E_{side}"], df[f"S_{side}"]
            ax.scatter(e[~tower], s[~tower], s=5, color=BLUE, alpha=0.5, linewidths=0)
            ax.scatter(e[tower], s[tower], s=22, color=ORANGE, edgecolors=SURFACE, linewidths=0.8, zorder=3)
            ax.axhline(2 * l_a * np.log(2), color=MUTED, ls=":", lw=1.1)
            ax.set_ylim(-0.1, 2 * l_a * np.log(2) * 1.06)
            ax.set_title(f"ρ_{side}, L = {l}, L_A = {l_a}, γ = {gamma:g}")
            ax.set_xlabel(f"E_{side} = Tr(ρ_{side} H)")
            ax.set_ylabel(f"S_op(ρ_{side})")
    fig.legend(handles=handles, loc="outside upper center", ncols=3)
    fig.savefig(DIR / f"lindblad_entanglement_g{g}.png")
    plt.close(fig)
    print(f"wrote {DIR / f'lindblad_entanglement_g{g}.png'}")
