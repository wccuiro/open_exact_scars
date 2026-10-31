"""
Operator spreading of the evolving state (lean runs, `states --obs-dt --op-ent-dt`), the state-side counterpart of the
operator-size argument in Mori (arXiv:2311.10304, Eqs. 7–8; his paper has no figure of it).

ρ(t) = 2^−L Σ_P r_P P; weights r_P² / Σ r² (identity included). Plotted against t (log axis):
  mean size / L      mean number of non-identity sites of the strings (product state: 1/2; random pure state: 3/4)
  mean support / L   shortest arc of the ring containing all non-identity sites
  S_op               operator entanglement of ρ across sites 0 … L/2 − 1 (pure state: 2 S_A)
Unitary scrambling moves weight to large strings (size → 3/4); dissipation removes it (size → 0 at the identity).

Top row: scar, x-Néel, z-Néel, L = 12, σᶻ dephasing, γ = 0, 0.01, 0.1, 1 (the Néel runs at γ > 0 stop at 1.5 t_e);
a marker at each run's t_e. Bottom row: scar, γ = 0.05, L = 12, every channel except decay.

Input:  data/used_data/lean/states_obs_*.csv, data/used_data/channels_rates.csv (fit_channels.py), or the first argument
Output: data/used_data/operator_spreading.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_states_inverse_L as fsi

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
L = 12
GAMMAS = (0, 0.01, 0.1, 1)
STYLES = dict(zip(GAMMAS, ("-", "--", "-.", ":")))
CHANNELS = {"z": "", "x": "_chx", "y": "_chy", "dep": "_chdep", "thermal": "_chthermal_n1"}
CH_COLORS = dict(zip(CHANNELS, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc7", "#b8860b"]))
CH_NAMES = {"z": "σᶻ", "x": "σˣ", "y": "σʸ", "dep": "depolarizing", "thermal": "thermal n̄ = 1"}
GAMMA_CH = 0.05


def observables(f):
    d = pd.read_csv(f)
    support = d[[f"supp{k}" for k in range(L + 1)]].to_numpy() @ np.arange(L + 1)
    return d["t"].to_numpy(), d["mean_size"].to_numpy() / L, support / L, d["t"][d["S_op"].notna()].to_numpy(), \
        d["S_op"].dropna().to_numpy()


def t_e(rates, state, ch, g):
    r = rates[(rates["state"] == state) & (rates["channel"] == ch) & (rates["L"] == L) & np.isclose(rates["gamma"], g)]
    return 1 / r["Gamma_e"].iloc[0] if len(r) and np.isfinite(r["Gamma_e"].iloc[0]) else np.nan


def draw(axes, t, size, supp, ts, sop, te, **kw):
    marker = dict(color=kw["color"], ms=5, mfc=fsi.SURFACE, mew=1.1, zorder=3)
    for ax, x, y in zip(axes, (t, t, ts), (size, supp, sop)):
        ax.plot(x[x > 0], y[x > 0], **kw)
        if np.isfinite(te) and te <= x[-1]:
            ax.plot(te, np.interp(te, x, y), "o", **marker)


def main():
    rates = pd.read_csv(DIR / "channels_rates.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for state in fsi.STATES:
        for g in GAMMAS:
            t, size, supp, ts, sop = observables(DIR / "lean" / f"states_obs_{state}_L{L}_g{g:g}.csv")
            draw(axes[0], t, size, supp, ts, sop, t_e(rates, state, "z", g) if g > 0 else np.nan,
                 ls=STYLES[g], color=fsi.SERIES[state], label=f"{fsi.NAMES[state].split(' (')[0]}, γ = {g:g}")
    for ch, sfx in CHANNELS.items():
        t, size, supp, ts, sop = observables(DIR / "lean" / f"states_obs_minus-x_L{L}_g{GAMMA_CH:g}{sfx}.csv")
        draw(axes[1], t, size, supp, ts, sop, t_e(rates, "minus-x", ch, GAMMA_CH), color=CH_COLORS[ch],
             label=CH_NAMES[ch])
    ylabels = ("mean size / L", "mean support / L", "S_op (operator entanglement of ρ)")
    for row, title in enumerate((f"σᶻ dephasing, L = {L}", f"scar, γ = {GAMMA_CH:g}, L = {L}")):
        for ax, yl in zip(axes[row], ylabels):
            ax.set(xscale="log", xlim=(0.02, 40 if row == 0 else 30), xlabel="t", ylabel=yl, title=f"{yl.split(' (')[0]}: {title}")
        for ax, ref in zip(axes[row][:2], (0.75, None)):
            if ref:
                ax.axhline(ref, color=fsi.MUTED, ls=":", lw=0.9)
            ax.axhline(0.5 if ref else 0.73, color=fsi.MUTED, ls="--", lw=0.7)
    axes[0][2].legend(fontsize=7, ncol=3, loc="upper left")
    axes[1][2].legend(fontsize=7.5, loc="upper left")
    fig.text(0.01, -0.05, "Size and support: weights r_P² / Σ r² of the Pauli strings of ρ (identity included); dashed "
             "gray: product state (size 1/2, support 0.73 at L = 12), dotted: random pure state (size 3/4). Open "
             "circles: t_e of the run (1/e of its signal).\nScar at γ = 0 stays a product state (size 1/2, S_op = 0); "
             "the Néel states scramble by t ≈ 0.5, before dissipation acts; the Néel runs at γ > 0 stop at 1.5 t_e "
             "≈ 0.3. Exact `states`, random c (seed 1), lean runs.", fontsize=7.5, color=fsi.INK_SECONDARY)
    fig.savefig(DIR / "operator_spreading.png")
    print(f"wrote {DIR / 'operator_spreading.png'}")


if __name__ == "__main__":
    main()
