"""One depolarizing jump on |->^L under the noiseless Floquet circuit: singlet weight, bond dimension and fidelity.
Inputs (data/kjma_jumps, same stem): floquet_kicks_bond_<s>.csv and floquet_kicks_fidelity_<s>.csv
(rust/examples/floquet_kicks_bond.rs), floquet_kick_bonddim_<s>.csv (rust/examples/floquet_kick_bonddim.rs).
Panels: (a) <R_{j,j+1}> (average over the jump), (b) bond dimension chi (discarded weight <= 1e-6) of the damaged
branch at every cut, (c) chi at the middle cut for both branches, (d) fidelity with the noiseless scar state.
Usage: python plot_scripts/plot_kick_bonddim.py [stem, default L20_dt0.025_site10] [dir, default data/kjma_jumps]"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

stem = sys.argv[1] if len(sys.argv) > 1 else "L20_dt0.025_site10"
folder = Path(sys.argv[2] if len(sys.argv) > 2 else "data/kjma_jumps")
bonds = pd.read_csv(folder / f"floquet_kicks_bond_{stem}.csv")
fid = pd.read_csv(folder / f"floquet_kicks_fidelity_{stem}.csv")
bd = pd.read_csv(folder / f"floquet_kick_bonddim_{stem}.csv")
L, dt, site = int(bd.L[0]), bd.dt[0], int(bd.site[0])
mid = L // 2
dam, scar = bd[bd.branch == "y|z"], bd[bd.branch == "x"]

fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
(ax_r, ax_chi), (ax_mid, ax_f) = axs

heat = bonds.pivot_table(index="t", columns="bond", values="singlet")
mesh = ax_r.pcolormesh(heat.columns, heat.index, heat, shading="nearest", cmap="inferno", vmin=0, vmax=0.5)
fig.colorbar(mesh, ax=ax_r, label=r"$\langle R_{j,j+1}\rangle$ (average over the jump)")
ax_r.set(xlabel="bond $j$ (sites $j$, $j+1$)", ylabel="time $t$", xticks=range(0, L, 2),
         title=f"(a) singlet weight, jump on site {site} at $t=0$")

cmap = dam.pivot_table(index="t", columns="ell", values="chi_1e-6")
mesh = ax_chi.pcolormesh(cmap.columns, cmap.index, cmap, shading="nearest", cmap="viridis",
                         norm=LogNorm(vmin=1, vmax=2 ** mid))
fig.colorbar(mesh, ax=ax_chi, label=r"bond dimension $\chi$ (discarded weight $\leq 10^{-6}$)")
ax_chi.axvline(site, color="w", lw=0.8, ls=":")
ax_chi.set(xlabel=r"cut $\ell$ (between sites $\ell-1$ and $\ell$)", ylabel="time $t$", xticks=range(1, L, 2),
           title="(b) damaged branch ($\\sigma^y$, $\\sigma^z$ image, weight 2/3)")

m = dam[dam.ell == mid]
ax_mid.semilogy(m.t, m["chi_1e-10"], "C0-", lw=1.6, label=r"damaged branch, discarded $\leq 10^{-10}$")
ax_mid.semilogy(m.t, m["chi_1e-6"], "C1-", lw=1.6, label=r"damaged branch, discarded $\leq 10^{-6}$")
s = scar[scar.ell == mid]
ax_mid.semilogy(s.t, s["chi_1e-10"], "ks", ms=6, mfc="none",
                label=r"scar branch ($\sigma^x$ image $=\psi_0(t)$, weight 1/3), any threshold")
ax_mid.axhline(2 ** mid, color="gray", ls=":", lw=1)
ax_mid.text(m.t.max(), 2 ** mid, f"maximum $2^{{{mid}}}$", ha="right", va="bottom", color="gray", fontsize=8)
ax_mid.set(xlabel="time $t$", ylabel=r"bond dimension $\chi$ at the middle cut", ylim=(0.7, 2 ** mid * 2.5),
           title=f"(c) middle cut $\\ell={mid}$")
ax_mid.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.08))
ax_mid.spines[["right", "top"]].set_visible(False)

ax_f.plot(fid.t, fid.fidelity, "k", lw=1.6, drawstyle="steps-post", label="$F$ (average over the jump)")
parts = fid.branches.str.split(";", expand=True)
for col, (label, color) in enumerate([("scar branch", "C2"), ("damaged branch", "C3")]):
    fb = parts[col].str.split(":").str[2].astype(float)
    ax_f.plot(fid.t, fb, color=color, lw=1.2, ls="--", label=f"{label}: $|\\langle\\psi_0(t)|\\psi_b(t)\\rangle|^2$")
ax_f.axhline(1 / 3, color="gray", ls=":", lw=1)
ax_f.set(xlabel="time $t$", ylabel="fidelity with $\\psi_0(t)$", ylim=(-0.03, 1.05),
         title=r"(d) $F=\langle\psi_0(t)|\rho(t)|\psi_0(t)\rangle$")
ax_f.legend(frameon=False, fontsize=8, loc="center right")
ax_f.spines[["right", "top"]].set_visible(False)

fig.suptitle(f"One depolarizing jump, noiseless Floquet circuit, $L={L}$, $dt={dt:g}$")
out = folder / f"floquet_kick_bonddim_{stem}.png"
fig.savefig(out, dpi=150)
print(out)
for tt in (0, 0.5, 1, 1.5, 2, 3, 4):
    r = dam[(dam.ell == mid) & ((dam.t - tt).abs() < 1e-9)]
    if len(r):
        r = r.iloc[0]
        print(f"t={tt}: damaged middle cut chi(1e-10)={r['chi_1e-10']} chi(1e-6)={r['chi_1e-6']} S={r.S:.3f}")
print("scar branch: max chi over all cuts and times", scar["chi_1e-10"].max(), "max S", scar.S.max())
print("fidelity: min", fid.fidelity.min(), "max", fid.fidelity.max())
