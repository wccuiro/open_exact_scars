"""Floquet singlet weight per bond, normalized per kick (rust/examples/floquet_bond.rs output).

Depolarizing channel per site and cycle: rho -> (1-p) rho + (p/3) sum_a s^a rho s^a, p = (3/4)(1 - e^{-4 gamma dt}),
gamma of D(rho) = gamma sum_a (s^a rho s^a - rho) (jumps sqrt(gamma) s^a, `xyz`). Files without a channel column (before
2026-11-26) carry the gamma of `dep` (sqrt(gamma/2) s^a), twice this one: it is halved on reading.
Expected kicks after n cycles: N(n) = n L p. Single-kick response at delay tau = (n-1) dt (noise at the end of each
cycle; valid while kicks are isolated):
S(tau) = [sum_j <R_j>(n) - sum_j <R_j>(n-1)] / (L p).
Usage: python plot_scripts/plot_floquet_bond_perkick.py main.csv [other.csv ...]  (heat map from the first file)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(path):
    data = pd.read_csv(path)
    L, gamma, dt = int(data.L[0]), data.gamma[0], data.dt[0]
    if "channel" not in data:
        gamma /= 2
    p = 0.75 * (1 - np.exp(-4 * gamma * dt))
    heat = data.pivot_table(index="cycle", columns="bond", values="singlet")
    return L, gamma, dt, p, heat


paths = [Path(a) for a in sys.argv[1:]] or [Path("data/floquet_bond_g1e-5.csv")]
L, gamma, dt, p, heat = load(paths[0])
n = heat.index.values
per_kick = heat.iloc[1:].div(n[1:] * L * p, axis=0)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8), width_ratios=[1.3, 1], layout="constrained")
mesh = ax.pcolormesh(per_kick.columns, per_kick.index, per_kick, shading="nearest", cmap="inferno", vmin=0)
fig.colorbar(mesh, ax=ax, label=r"$\langle R_{j,j+1}\rangle\,/\,(\text{expected kicks so far})$")
ax.set(xlabel="bond $j$ (sites $j$, $j+1$)", ylabel="Floquet cycle $n$",
       title=fr"Per kick: $\gamma={gamma:g}$, $p={p:.2g}$ per site per cycle, $L={L}$")
ax.set_xticks(range(L))
sec = ax.secondary_yaxis("right", functions=(lambda c: c * dt, lambda s: s / dt))
sec.set_ylabel("time $t = n\\,dt$")

colors = ["k", "#b84716", "#0c6465", "#7a4fa3"]
for path, color in zip(paths, colors):
    L_, g, dt_, p_, h = load(path)
    total = h.sum(axis=1).values
    # The noise sits at the end of each cycle, so the kick of cycle n is seen at delay 0 after cycle n.
    tau = (h.index.values[1:] - 1) * dt_
    response = np.diff(total) / (L_ * p_)
    ax2.plot(tau, response, color=color, lw=1.6, label=fr"$\gamma={g:g}$: one kick, delay $\tau$")
    ax2.plot(tau, total[1:] / (h.index.values[1:] * L_ * p_), color=color, lw=1, ls="--",
             label=fr"$\gamma={g:g}$: running mean")
ax2.axhline(2 / 3, color="gray", ls=":", lw=1)
ax2.text(0, 2 / 3, "2/3", color="gray", fontsize=8, va="bottom")
ax2.set(xlabel=r"delay $\tau$ after the kick (= $n\,dt$)", ylabel=r"total singlet weight per kick, $\sum_j\langle R_{j,j+1}\rangle$",
        title="Single-kick response from the averaged ρ")
ax2.legend(frameon=False, fontsize=8)
ax2.spines[["right", "top"]].set_visible(False)
out = paths[0].with_name(paths[0].stem + "_perkick.png")
fig.savefig(out, dpi=150)
print(out)
total = heat.sum(axis=1).values
resp = np.diff(total) / (L * p)
for c in [1, 2, 5, 10, 20, 40, 100, 300, 1000, 1500]:
    if c <= len(resp):
        print(f"n={c:5d} tau={(c-1)*dt:6.1f} S={resp[c-1]:.4f} spread over bonds={np.ptp(per_kick.loc[c].values)/per_kick.loc[c].mean():.3%}"
              if c in per_kick.index else "")
