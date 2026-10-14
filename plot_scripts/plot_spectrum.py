"""
Liouvillian spectrum from full ED (random c), at the dissipative γ.

Every mode is drawn at its L (with a small horizontal spread) and −Re λ / γ, colored by its tower
weight ‖P R P‖ / ‖R‖. A mode is classed by the largest of its squared weights tower², mixed², bulk²
(they sum to 1). The 3 zero modes (𝟙, |↑…↑⟩⟨↑…↑|, |↓…↓⟩⟨↓…↓|) are left out of the classes; the slowest
remaining bulk mode is highlighted, and the slowest tower modes are listed.

Input:  data/spectrum_L<L>.csv   (cd rust && cargo run --release --bin ed -- --lmax 6)
Output: data/spectrum.png, and a table printed to the terminal
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ZERO = 1e-10

DATA = Path(__file__).resolve().parent.parent / "data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ORANGE = "#eb6834"
TOWER_CMAP = LinearSegmentedColormap.from_list("tower", ["#d6d4cc", "#86b6ef", "#2a78d6", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def main():
    files = sorted(DATA.glob("spectrum_L*.csv"), key=lambda p: int(p.stem.split("_L")[1]))
    if not files:
        raise SystemExit(f"no spectrum_L*.csv in {DATA}; see the docstring for the run")
    spec = pd.concat([pd.read_csv(p) for p in files])
    gamma = spec["gamma"].max()
    spec = spec[spec["gamma"] == gamma].copy()
    spec["rate"] = -spec["re"] / gamma
    weights = spec[["tower", "mixed", "bulk"]].to_numpy() ** 2
    spec["cls"] = np.array(["tower", "mixed", "bulk"])[weights.argmax(axis=1)]
    zero = np.hypot(spec["re"], spec["im"]) < ZERO
    spec.loc[zero, "cls"] = "zero"

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    rng = np.random.default_rng(0)
    order = spec.sort_values("tower")  # tower modes on top
    x = order["L"] + rng.uniform(-0.3, 0.3, len(order))
    sc = ax.scatter(x, order["rate"], c=order["tower"], cmap=TOWER_CMAP, vmin=0, vmax=1, s=4, lw=0)
    fig.colorbar(sc, ax=ax, label="tower weight ‖PRP‖ / ‖R‖")

    print(f"gamma = {gamma}; rates in units of gamma")
    print("  L  zero  tower  mixed   bulk   slowest bulk   slowest tower (5)")
    for l, d in spec.groupby("L"):
        counts = d["cls"].value_counts()
        bulk = d[d["cls"] == "bulk"].sort_values("rate")
        tower = d[d["cls"] == "tower"].sort_values("rate")
        slow = bulk.iloc[0]
        ax.scatter([l], [slow["rate"]], s=60, facecolors="none", edgecolors=ORANGE, lw=1.6,
                   label="slowest bulk mode" if l == spec["L"].min() else None, zorder=3)
        print(f"{l:3d} {counts.get('zero', 0):5d} {counts.get('tower', 0):6d} {counts.get('mixed', 0):6d} "
              f"{counts.get('bulk', 0):6d}   {slow['rate']:8.3f}       {np.round(tower['rate'].to_numpy()[:5], 3)}")

    ax.set_xlabel("L")
    ax.set_ylabel("−Re λ / γ")
    ax.set_xticks(sorted(spec["L"].unique()))
    ax.set_title(f"Liouvillian spectrum, γ = {gamma:g}")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(DATA / "spectrum.png")
    print(f"wrote {DATA / 'spectrum.png'}")


if __name__ == "__main__":
    main()
