"""
Operator entanglement entropy of |n⟩⟨m| for every pair of eigenstates n, m of H (γ = 0), against E_m − E_n,
one panel per L.

The operator Schmidt decomposition of |n⟩⟨m| is the product of the two state decompositions: with
|n⟩ = Σ_i √p_i |a_i⟩|b_i⟩ and |m⟩ = Σ_j √q_j |a′_j⟩|b′_j⟩, |n⟩⟨m| = Σ_ij √(p_i q_j) |a_i⟩⟨a′_j| ⊗ |b_i⟩⟨b′_j|,
and the |a_i⟩⟨a′_j| (and |b_i⟩⟨b′_j|) are Hilbert–Schmidt orthonormal. So the operator entanglement (normalized
by the Hilbert–Schmidt norm, which is 1) is exactly S_op = S_n + S_m, with S the half-chain entropies of the
`entanglement` binary (checked against an explicit SVD of |n⟩⟨m| for random states, L = 8: 2e-15).

Pairs: bulk–bulk as a density (log colour scale, 4^L pairs in all), scar–bulk and scar–scar as points (orange:
both are scars, including n = m). Dotted line: 2 × Page value.

Input:  entanglement_L<L>.csv (L, L_A, E, S, tower) in data/used_data/ (or the directory given as the first
        argument); every size found is plotted
Output: operator_entanglement.png, in the same directory
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baa9b"
RAMP = ["#bcd5f5", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.2, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

L_SHOW = (6, 8, 10, 12)  # at most 4 panels (notes_layout.MAX_PANELS); L = 7, 9, 11 exist but are not drawn
files = sorted((f for f in DIR.glob("entanglement_L*.csv") if int(re.search(r"_L(\d+)", f.name).group(1)) in L_SHOW),
               key=lambda f: int(re.search(r"_L(\d+)", f.name).group(1)))
if not files:
    sys.exit(f"no entanglement_L*.csv in {DIR}")
blues = LinearSegmentedColormap.from_list("blues", RAMP)

nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows), squeeze=False, layout="constrained")
for ax, file in zip(axes.flat, files):
    df = pd.read_csv(file)
    l, l_a = int(df["L"].iloc[0]), int(df["L_A"].iloc[0])
    e, s, scar = df["E"].to_numpy(), df["S"].to_numpy(), (df["tower"] > 0.5).to_numpy()
    s_max = 2 * l_a * np.log(2)

    # bulk–bulk: all pairs, binned (n, m and m, n both counted)
    eb, sb = e[~scar], s[~scar]
    de = (eb[None, :] - eb[:, None]).ravel()  # E_m − E_n, m the column
    so = (sb[None, :] + sb[:, None]).ravel()
    span = e.max() - e.min()
    counts, xe, ye = np.histogram2d(de, so, bins=(160, 100), range=((-span, span), (0, 1.05 * s_max)))
    del de, so
    ax.pcolormesh(xe, ye, np.ma.masked_equal(counts.T, 0), cmap=blues, norm=LogNorm(vmin=1), rasterized=True)

    # scar–bulk (both orders) and scar–scar
    es, ss = e[scar], s[scar]
    de = np.concatenate([(eb[None, :] - es[:, None]).ravel(), (es[None, :] - eb[:, None]).ravel()])
    so = np.concatenate([(sb[None, :] + ss[:, None]).ravel()] * 2)
    ax.scatter(de, so, s=1.5, color=AQUA, alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter((es[None, :] - es[:, None]).ravel(), (ss[None, :] + ss[:, None]).ravel(), s=12, color=ORANGE,
               edgecolors=SURFACE, linewidths=0.6, zorder=3)

    page = l_a * np.log(2) - 0.5 * 2.0**l_a / 2.0 ** (l - l_a)
    ax.axhline(2 * page, color=MUTED, ls=":", lw=1.1)
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.15, 1.05 * s_max)
    ax.set_title(f"L = {l}, L_A = {l_a}")
for ax in axes.flat[len(files):]:
    ax.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("E_m − E_n")
for ax in axes[:, 0]:
    ax.set_ylabel("S_op(|n⟩⟨m|) = S_n + S_m")
handles = [
    Patch(color=BLUE, label="bulk–bulk (density, log scale)"),
    Line2D([], [], color=AQUA, marker="o", ls="", ms=4, label="scar–bulk"),
    Line2D([], [], color=ORANGE, marker="o", ls="", ms=5, label="scar–scar"),
    Line2D([], [], color=MUTED, ls=":", label="2 × Page value"),
]
fig.legend(handles=handles, loc="outside upper center", ncols=4)
fig.savefig(DIR / "operator_entanglement.png")
print(f"wrote {DIR / 'operator_entanglement.png'}")
