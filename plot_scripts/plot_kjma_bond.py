"""Singlet weight <R_{j,j+1}> per bond after one depolarizing kick on |->^L (rust/examples/kjma_bond.rs)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/kjma_bond.csv")
data = pd.read_csv(path)
L, k = int(data.L[0]), int(data.kick_site[0])
# Bond j joins sites j, j+1; distance of its centre j + 1/2 from the kicked site, on the ring.
data["r"] = (data.bond + 0.5 - k + L / 2) % L - L / 2
avg = data.pivot_table(index="t", columns="r", values="singlet", aggfunc="mean")  # depolarizing: mean over x, y, z
total = avg.sum(axis=1)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6), width_ratios=[1.5, 1], layout="constrained")
heat = ax.pcolormesh(avg.columns, avg.index, avg, shading="nearest", cmap="inferno", vmin=0, vmax=avg.values.max())
fig.colorbar(heat, ax=ax, label=r"$\langle R_{j,j+1}\rangle$ (singlet weight of bond $j$)")
ax.set(xlabel="bond centre − kicked site", ylabel="time $t$",
       title=f"One depolarizing kick at $t=0$, $L={L}$")
ax.set_xticks(np.arange(-L // 2 + 0.5, L // 2, 2))

for axis, rows in data.groupby("axis"):
    ax2.plot(*rows.groupby("t").singlet.sum().pipe(lambda s: (s.index, s.values)),
             lw=1.2, ls="--" if axis == "y" else "-",
             label=fr"$\sigma^{axis}$ kick" + (r" (= $\sigma^z$: $\sigma^y|-\rangle \propto \sigma^z|-\rangle$)" if axis == "y" else ""))
ax2.plot(total.index, total.values, "k", lw=2, label="depolarizing (mean)")
ax2.axhline(2 / 3, color="gray", ls=":", lw=1)
ax2.set(xlabel="time $t$", ylabel=r"$\sum_j \langle R_{j,j+1}\rangle$", title="Total singlet weight")
ax2.legend(frameon=False, fontsize=9)
ax2.spines[["right", "top"]].set_visible(False)
out = path.with_suffix(".png")
fig.savefig(out, dpi=150)
print(out, "| t=0 per bond:", avg.iloc[0].round(4).to_dict(), "| total range", total.min(), total.max())
