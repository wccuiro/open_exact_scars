"""Fidelity F(t) = <psi0(t)|rho(t)|psi0(t)> with the noiseless scar state after scheduled depolarizing jumps
(rust/examples/floquet_kicks_bond.rs, 6th argument), one curve per run, next to the total singlet weight of the same
run (the bond CSV of the same name with "bond" for "fidelity"). Jumps are read from the bond CSV (kick_axis >= 0).
Usage: python plot_scripts/plot_kicks_fidelity.py out.png fidelity_1.csv [fidelity_2.csv ...]"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

out = Path(sys.argv[1])
files = [Path(p) for p in sys.argv[2:]]
styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "D", "v", "P"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
for i, path in enumerate(files):
    f = pd.read_csv(path)
    bonds = pd.read_csv(path.with_name(path.name.replace("fidelity", "bond", 1)))
    L, dt = int(f.L[0]), f.dt[0]
    kicks = bonds[bonds.kick_axis >= 0]
    label = ", ".join(f"site {int(k.bond)} at $t={k.cycle * dt:g}$" for k in kicks.itertuples())
    style = dict(ls=styles[i % len(styles)], marker=markers[i % len(markers)], markevery=(4 * i, 16), ms=5, lw=1.4,
                 mfc="none")
    # a jump acts right after its cycle, before the measurement: steps-post puts the change at the jump time
    ax.plot(f.cycle * dt, f.fidelity, drawstyle="steps-post", label=label, **style)
    total = bonds.groupby("cycle").singlet.sum()
    ax2.plot(total.index * dt, total, label=label, **style)
    after = f[f.cycle >= kicks.cycle.max()].fidelity
    print(path.name, "| jumps", label, "| F first", f.fidelity.iloc[0], "last", f.fidelity.iloc[-1],
          "| max |F - 1/3| after the last jump", (after - 1 / 3).abs().max(),
          "| branches at the end:", f.branches.iloc[-1])
for value, name in [(1 / 3, "1/3"), (1 / 9, "1/9")]:
    ax.axhline(value, color="gray", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], value, name, ha="right", va="bottom", color="gray", fontsize=8)
ax.set(xlabel="time $t$", ylabel=r"$F=\langle\psi_0(t)|\rho(t)|\psi_0(t)\rangle$", ylim=(0, 1.05),
       title=f"Fidelity with the noiseless scar state, $L={L}$, $dt={dt:g}$")
ax.text(0.03, 0.97, "$F=1$ before the jump", transform=ax.transAxes, va="top", fontsize=8)
ax2.set(xlabel="time $t$", ylabel=r"$\sum_j\langle R_{j,j+1}\rangle$", title="Total singlet weight (same runs)")
for a in (ax, ax2):
    a.spines[["right", "top"]].set_visible(False)
    a.legend(title="depolarizing jump on", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
fig.savefig(out, dpi=150)
print(out)
