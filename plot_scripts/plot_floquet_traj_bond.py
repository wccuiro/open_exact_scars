"""One Floquet jump trajectory: singlet weight <R_{j,j+1}> per bond after every cycle, kicks marked
(rust/examples/floquet_traj_bond.rs: Poisson kicks; floquet_kicks_bond.rs: scheduled kicks, gamma = 0).
Usage: python plot_scripts/plot_floquet_traj_bond.py [csv] [max_cycle] [min_cycle]"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/floquet_traj_bond.csv")
data = pd.read_csv(path)
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else data.cycle.max()
n_min = int(sys.argv[3]) if len(sys.argv) > 3 else 0
data = data[(data.cycle >= n_min) & (data.cycle <= n_max)].reset_index(drop=True)
L, gamma, dt = int(data.L[0]), data.gamma[0], data.dt[0]
# γ of D(ρ) = γ Σ_α (σ^α ρ σ^α − ρ) (jumps √γ σ^α, `xyz`); files without a channel column (before 2026-11-26) carry the
# γ of `dep` (√(γ/2) σ^α), twice this one
if "channel" not in data:
    gamma /= 2
heat = data.pivot_table(index="cycle", columns="bond", values="singlet")
kicks = data[data.kick_axis >= 0]
# Depolarizing jumps (axis 3) are averaged exactly over their Pauli branches: the map is Tr(rho R), not one trajectory.
what = "average over the jumps" if (kicks.kick_axis == 3).any() else "one trajectory"

# Fidelity of floquet_kicks_bond (6th argument): same name with "fidelity" for "bond"; drawn only if it covers the window.
fid_path = path.with_name(path.name.replace("bond", "fidelity", 1))
fid = pd.read_csv(fid_path) if fid_path != path and fid_path.exists() else None
if fid is not None and fid.cycle.max() < data.cycle.max():
    print(fid_path, "stops at cycle", fid.cycle.max(), "<", data.cycle.max(), ": fidelity not drawn")
    fid = None

if fid is None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.2), width_ratios=[1.4, 1], layout="constrained")
else:
    fig, axd = plt.subplot_mosaic([["map", "total"], ["map", "fid"]], figsize=(10.5, 6.4), width_ratios=[1.4, 1],
                                  layout="constrained")
    ax, ax2, ax3 = axd["map"], axd["total"], axd["fid"]
mesh = ax.pcolormesh(heat.columns, heat.index, heat, shading="nearest", cmap="inferno", vmin=0, vmax=0.5)
fig.colorbar(mesh, ax=ax, label=r"$\langle R_{j,j+1}\rangle$ after cycle $n$" + f" ({what})")
# Site j sits between bond j-1 and bond j: x = j - 1/2 (site 0 drawn at both edges of the ring).
for axis, marker, name in [(0, "o", r"$\sigma^x$"), (1, "s", r"$\sigma^y$"), (2, "D", r"$\sigma^z$"),
                           (3, "*", "depolarizing")]:
    k = kicks[kicks.kick_axis == axis]
    if k.empty:
        continue
    x = k.bond - 0.5
    size = 70 if len(kicks) < 10 else 22
    ax.scatter(x, k.cycle, s=size, marker=marker, facecolor="none", edgecolor="cyan", lw=1.2, label=f"{name} kick on site")
    edge = k[k.bond == 0]
    ax.scatter(edge.bond + L - 0.5, edge.cycle, s=size, marker=marker, facecolor="none", edgecolor="cyan", lw=1.2)
if gamma == 0:
    title = fr"Scheduled kicks, no noise, $L={L}$, $dt={dt:g}$"
else:
    per_cycle = L * 0.75 * (1 - np.exp(-4 * gamma * dt))
    title = fr"Jump trajectory, depolarizing $\gamma={gamma:.3g}$ ({per_cycle:.2g} kicks/cycle), $L={L}$"
ax.set(xlabel="bond $j$ (sites $j$, $j+1$); markers on the kicked site", ylabel="Floquet cycle $n$",
       xlim=(-0.6, L - 0.4), ylim=(heat.index.min() - 0.5, heat.index.max() + 0.5), title=title)
ax.set_xticks(range(L))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncols=3, frameon=False, fontsize=8)
sec = ax.secondary_yaxis("right", functions=(lambda c: c * dt, lambda s: s / dt))
sec.set_ylabel("time $t = n\\,dt$")

total = heat.sum(axis=1)
ax2.plot(heat.index * dt, total, "k", lw=1.5)
ax2.axhline(L / 4, color="gray", ls=":", lw=1)
ax2.text(heat.index.max() * dt, L / 4, r"$L/4$ (infinite temperature)", ha="right", va="bottom", color="gray", fontsize=8)
ax2.set(xlabel="time $t$", ylabel=r"$\sum_j\langle R_{j,j+1}\rangle$", title="Total singlet weight\n" + f"({what})")
ax2.spines[["right", "top"]].set_visible(False)
jump_times = sorted(kicks.cycle.unique() * dt)
if fid is not None:
    f = fid[(fid.cycle >= n_min) & (fid.cycle <= n_max)]
    # a jump acts right after its cycle, before the measurement: steps-post puts the change at the jump time
    ax3.plot(f.cycle * dt, f.fidelity, "k", lw=1.5, drawstyle="steps-post")
    for value, name in [(1 / 3, "1/3"), (1 / 9, "1/9")]:
        ax3.axhline(value, color="gray", ls=":", lw=1)
        ax3.text(heat.index.max() * dt, value, name, ha="right", va="bottom", color="gray", fontsize=8)
    ax3.set(xlabel="time $t$", ylabel=r"$F=\langle\psi_0(t)|\rho(t)|\psi_0(t)\rangle$", ylim=(0, 1.05),
            title="Fidelity with the noiseless\n" + r"scar state $\psi_0(t)$")
    ax3.text(0.03, 0.97, "$F=1$ before the first jump", transform=ax3.transAxes, va="top", fontsize=8)
    ax3.spines[["right", "top"]].set_visible(False)
    ax2.sharex(ax3)
    around = {n_min, n_max} | {c + d for c in kicks.cycle.unique() for d in (-1, 0, 1)}
    print("fidelity (t: F)", {round(c * dt, 4): round(v, 8) for c, v in zip(f.cycle, f.fidelity) if c in around},
          "| largest change between consecutive cycles without a jump",
          f.fidelity.diff()[~f.cycle.isin(kicks.cycle.unique())].abs().max())
for a in [ax2] + ([ax3] if fid is not None else []):
    for s in jump_times if len(jump_times) <= 10 else []:  # scheduled jumps only, not every Poisson kick
        a.axvline(s, color="tab:cyan", lw=0.8, ls="--")
suffix = (f"_n{n_min}-{n_max}" if n_min > 0 else f"_n{n_max}") if len(sys.argv) > 2 else ""
out = path.with_name(path.stem + suffix + ".png")
fig.savefig(out, dpi=150)
marks = [0, 10, 20, 50, 100, 200, 299, 300, 301, 310, 320, 350, 400, 600, 800, 1100]
print(out, "kicks:", len(kicks), "| total at", [(int(c), round(total[c], 3)) for c in marks if c in total.index])
