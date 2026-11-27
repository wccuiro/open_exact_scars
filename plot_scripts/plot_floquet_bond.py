"""Singlet weight <R_{j,j+1}> per bond after every Floquet cycle, depolarizing noise, from |->^L
(rust/examples/floquet_bond.rs). Usage: python plot_scripts/plot_floquet_bond.py [data/floquet_bond.csv]"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/floquet_bond.csv")
data = pd.read_csv(path)
L, gamma, dt = int(data.L[0]), data.gamma[0], data.dt[0]
# γ of D(ρ) = γ Σ_α (σ^α ρ σ^α − ρ) (jumps √γ σ^α, `xyz`); files without a channel column (before 2026-11-26) carry the
# γ of `dep` (√(γ/2) σ^α), twice this one
if "channel" not in data:
    gamma /= 2
heat = data.pivot_table(index="cycle", columns="bond", values="singlet")
total = heat.sum(axis=1)
t = heat.index * dt

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), width_ratios=[1.4, 1], layout="constrained")
mesh = ax.pcolormesh(heat.columns, heat.index, heat, shading="nearest", cmap="inferno", vmin=0)
fig.colorbar(mesh, ax=ax, label=r"$\langle R_{j,j+1}\rangle$ after cycle $n$")
ax.set(xlabel="bond $j$ (sites $j$, $j+1$)", ylabel="Floquet cycle $n$",
       title=fr"Floquet, depolarizing $\gamma={gamma:g}$, $dt={dt:g}$, $L={L}$")
ax.set_xticks(range(L))
sec = ax.secondary_yaxis("right", functions=(lambda n: n * dt, lambda s: s / dt))
sec.set_ylabel("time $t = n\\,dt$")

ax2.plot(t, total, "k", lw=2, label=r"$\sum_j \langle R_{j,j+1}\rangle$")
ax2.plot(t, heat.min(axis=1) * L, color="#b84716", lw=1, ls="--", label=r"$L\times$ min over bonds")
ax2.plot(t, heat.max(axis=1) * L, color="#0c6465", lw=1, ls="--", label=r"$L\times$ max over bonds")
if total.max() > 0.2 * L / 4:  # infinite-temperature value only when it is on the scale of the data
    ax2.axhline(L / 4, color="gray", ls=":", lw=1)
    ax2.text(t.max(), L / 4, r"$L/4$ (infinite temperature)", ha="right", va="bottom", color="gray", fontsize=8)
ax2.set(xlabel="time $t$", ylabel="total singlet weight", title="Sum over bonds")
ax2.legend(frameon=False, fontsize=9)
ax2.spines[["right", "top"]].set_visible(False)
out = path.with_suffix(".png")
fig.savefig(out, dpi=150)
print(out)
for n in [1, 10, 100, 500, 1000, heat.index.max()]:
    if n in heat.index:
        row = heat.loc[n]
        print(f"n={n:5d} t={n*dt:6.1f} sum={row.sum():.5f} min={row.min():.5f} max={row.max():.5f}")
