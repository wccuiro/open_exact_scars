"""
Decay against L at γ = 0.01 with the current coefficients, from the existing `states` / `states_traj` runs (the
appendix figures that replace the previous-coefficient decay_fit.png, probe_timeseries.png, probe_scaling.png).

  rates_vs_L_scar.png     left: scar m_⊥(t) at γ = 0.01 for every L (log scale);
                          right: Γ_e/γ against L with a power law a L^b and an exponential a e^(cL) fitted on all L,
                          and the cap L − 1
  rates_vs_L_series.png   scar m_⊥, x-Néel s_⊥, z-Néel |s_z| against t, γ = 0.01 (solid) and γ = 0 (dashed),
                          scar L ≤ 12, Néel L = 8, 12 (exact)
  rates_vs_L_scaling.png  Γ_e against L (log-log): scar Γ_e/γ at γ = 0.01 with the power-law fit (left);
                          x-Néel and z-Néel Γ_e at γ = 0 and 0.01 (right)

Γ_e is the 1/e rate of the envelope (states_rates.csv from fit_states.py; CLAUDE.md "Decay rates"). One run per
(state, L, γ) (states_data.select): exact for L ≤ 12 (14), trajectories above.

Input:  data/used_data (or the directory given as the first argument)
Output: the three figures in the same directory
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import states_data
from fit_states import INK_SECONDARY, MUTED

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
GAMMA = 0.01
SERIES_L_MAX = 12
NEEL_SERIES_L = (8, 12)
NEEL_L_MIN = {"neel-x": 8, "neel": 10}   # scaling panel: sizes without finite-size revivals at small γ   # Néel panels: two sizes only (all L and both γ overlap into noise)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"minus-x": "scar |−⟩^⊗L: m_⊥", "neel-x": "x-Néel |+−+−…⟩: s_⊥", "neel": "z-Néel |↑↓↑↓…⟩: |s_z|"}


def ramp(sizes):
    cmap = LinearSegmentedColormap.from_list("blues", RAMP)
    return {l: cmap(i / max(len(sizes) - 1, 1)) for i, l in enumerate(sizes)}


def power_and_exp(l, y):
    """Fits y = a L^b and y = a e^(cL) (straight lines in log y); returns (b, a), (c, a) and the rms relative residuals."""
    pb = np.polyfit(np.log(l), np.log(y), 1)
    pe = np.polyfit(l, np.log(y), 1)
    rms = lambda fit: np.sqrt(np.mean((np.exp(fit) / y - 1) ** 2))
    return pb, pe, rms(np.polyval(pb, np.log(l))), rms(np.polyval(pe, l))


def main():
    chosen = states_data.select(states_data.load(DIR))
    rates = pd.read_csv(DIR / "states_rates.csv")
    rates = rates[rates["selected"]]

    # ---- scar: m_⊥ at γ = 0.01 and Γ_e/γ against L
    scar = rates[(rates["state"] == "minus-x") & np.isclose(rates["gamma"], GAMMA)].sort_values("L")
    sizes = scar["L"].tolist()
    colors = ramp(sizes)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    for l in sizes:
        _, run, _ = chosen["minus-x", l, GAMMA]
        ax.plot(run["t"], np.hypot(run["mx"], run["my"]), color=colors[l], lw=1.1, label=f"L = {l}")
    ts = np.linspace(0, 150, 200)
    ax.plot(ts, np.exp(-2 * GAMMA * ts), color=MUTED, ls=":", lw=1.1, label="e^(−2γt)")
    ax.axhline(np.exp(-1), color=MUTED, lw=0.6, ls="-.")
    ax.set(yscale="log", ylim=(1e-3, 1.3), xlim=(0, 150), xlabel="t", ylabel="m_⊥",
           title=f"Scar m_⊥, γ = {GAMMA:g} (dash-dot: 1/e)")
    ax.legend(fontsize=7, ncols=2)
    lv, y = scar["L"].to_numpy(float), (scar["Gamma_e"] / GAMMA).to_numpy()
    pb, pe, rb, re = power_and_exp(lv, y)
    ls = np.linspace(3.7, 20.5, 200)
    bx.plot(ls, ls - 1, color=MUTED, ls=":", lw=1.1, label="cap L − 1")
    bx.plot(ls, np.exp(np.polyval(pb, np.log(ls))), color=BLUE, lw=1.3,
            label=f"power law ∝ L^{pb[0]:.2f}  (rms residual {100 * rb:.1f} %)")
    bx.plot(ls, np.exp(np.polyval(pe, ls)), color=ORANGE, lw=1.3,
            label=f"exponential ∝ e^({pe[0]:.3f} L)  (rms residual {100 * re:.1f} %)")
    exact = (scar["method"] == "exact").to_numpy()
    bx.plot(lv[exact], y[exact], "o", color="k", ms=5, label="Γ_e/γ, exact")
    bx.plot(lv[~exact], y[~exact], "s", color="k", ms=4.5, mfc="white", label="Γ_e/γ, trajectories")
    bx.set(xlabel="L", ylabel="Γ_e / γ", title=f"Scar 1/e rate against L, γ = {GAMMA:g}")
    bx.legend(fontsize=7)
    fig.savefig(DIR / "rates_vs_L_scar.png")
    print(f"scar Γ_e/γ: power law exponent {pb[0]:.3f} (rms {rb:.3f}), exponential c = {pe[0]:.4f} (rms {re:.3f})")

    # ---- time series of the three states, L ≤ 12
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, state, t_max in zip(axes, ("minus-x", "neel-x", "neel"), (150, 10, 10)):
        signal = states_data.SIGNALS[state]
        ls_state = sorted({k[1] for k in chosen if k[0] == state and k[1] <= SERIES_L_MAX})
        if state != "minus-x":
            ls_state = [x for x in ls_state if x in NEEL_SERIES_L]
        cols = ramp(ls_state)
        for l in ls_state:
            for g, style in ((GAMMA, "-"), (0.0, "--")):
                if (state, l, g) in chosen:
                    run = chosen[state, l, g][1]
                    sel = run["t"] <= t_max
                    ax.plot(run["t"][sel], signal(run)[sel], style, color=cols[l], lw=1.0 if style == "-" else 0.8)
        ax.set(yscale="log", ylim=(1e-3, 1.5), xlim=(0, t_max), xlabel="t", title=NAMES[state])
        ax.legend(handles=[Line2D([], [], color=cols[l], lw=2, label=f"L = {l}") for l in ls_state], fontsize=7,
                  loc="lower left" if state == "minus-x" else "upper right")
    axes[0].set_ylabel("signal")
    fig.text(0.01, -0.04, f"solid: γ = {GAMMA:g}; dashed: γ = 0 (scar: m_⊥ = 1). Exact evolution; scar L ≤ {SERIES_L_MAX}, "
             f"Néel states L = {', '.join(map(str, NEEL_SERIES_L))}. The thermal signals reach a finite-size floor "
             "(≈ 0.01 – 0.1) after t ≈ 1, at both γ.",
             fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "rates_vs_L_series.png")

    # ---- rates against L, log-log
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained", sharey=False)
    ax.plot(lv, y, "o", color=BLUE, ms=6, label=f"scar, γ = {GAMMA:g}")
    ax.plot(ls, np.exp(np.polyval(pb, np.log(ls))), color=BLUE, lw=1.1, label=f"∝ L^{pb[0]:.2f}")
    ax.plot(ls, ls - 1, color=MUTED, ls=":", lw=1.1, label="cap L − 1")
    ax.axhline(2, color=MUTED, ls="--", lw=0.9, label="2γ: free spins")
    ax.set(xscale="log", yscale="log", xlabel="L", ylabel="Γ_e / γ", title="Scar: rate grows with L (finite-size cap)")
    ax.set_xticks([4, 6, 8, 10, 12, 16, 20], ["4", "6", "8", "10", "12", "16", "20"])
    ax.set_yticks([2, 3, 4, 6, 8, 10, 15, 20], ["2", "3", "4", "6", "8", "10", "15", "20"])
    ax.minorticks_off()
    ax.legend(fontsize=7.5)
    for state, color, marker in (("neel-x", AQUA, "D"), ("neel", ORANGE, "s")):
        for g, fill in ((GAMMA, True), (0.0, False)):
            d = rates[(rates["state"] == state) & np.isclose(rates["gamma"], g)
                      & (rates["L"] >= NEEL_L_MIN[state])].sort_values("L")
            bx.plot(d["L"], d["Gamma_e"], marker + ("-" if fill else "--"), color=color, ms=5, lw=0.9,
                    mfc=color if fill else "white", label=f"{NAMES[state].split(':')[0]}, γ = {g:g}")
    bx.set(xscale="log", ylim=(0, 6), xlabel="L", ylabel="Γ_e",
           title="Thermal references: rate set by H, L-independent")
    bx.set_xticks([8, 10, 12, 16, 20], ["8", "10", "12", "16", "20"])
    bx.legend(fontsize=7.5)
    fig.text(0.01, -0.04, "Γ_e: 1/e rate of the envelope. Left: log-log. Right: x-Néel from L = 8, z-Néel from L = 10; the "
             "smaller sizes have finite-size revivals and never reach 1/e at small γ (left out, as in the 1/L fits).",
             fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "rates_vs_L_scaling.png")
    print(f"wrote rates_vs_L_scar.png, rates_vs_L_series.png, rates_vs_L_scaling.png in {DIR}")


if __name__ == "__main__":
    main()
