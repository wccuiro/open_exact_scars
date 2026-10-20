"""
Decay rates against γ for the scar state |−⟩^⊗L and the z- and x-Néel states (exact evolution, `states` binary).

  signal  minus-x: m_⊥(t) = √(m_x² + m_y²)       neel: |s_z(t)|, s_z = (1/L) Σ_j (−1)^j ⟨σᶻ_j⟩
          neel-x: s_⊥(t) = √(s_x² + s_y²)
  envelope E(t) = max_{t' ≥ t} signal(t'), non-increasing (removes the zero crossings of s_z)

Two rates per run:
  Gamma_e     1/t_e, with t_e the first time E < 1/e (log-linear interpolation between steps)
  Gamma_tail  −slope of ln E fitted on [t(E < TAIL_HI), t(E < TAIL_LO)]; σ the standard error of the
              slope. NaN if E never reaches TAIL_LO (e.g. Néel at γ = 0, which stops at a finite-size floor).

Input:  states_<state>_L<L>_g<γ>.csv (exact, `states` binary) and states_traj_<state>_L<L>_g<γ>.csv (trajectories,
        `states_traj` binary) in data/used_data/ (or the directory given as the first argument)
Output: in the same directory
        states_rates.csv   state, method, L, gamma, cut, selected, Gamma_e, Gamma_tail, sigma_tail, t_tail_min,
                           t_tail_max; one row per run, `selected` marks the one run per (state, L, γ) chosen by
                           states_data.select (exact where it finished, else trajectories)
        states_rates.png   Γ against γ (log-log) of the selected runs, one panel per state, one line per L
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from states_data import SIGNALS, load, select

TAIL_HI, TAIL_LO = 1e-2, 1e-3

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#bcd5f5", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]  # small L light, large L dark

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def crossing(t, e, level):
    """First time the non-increasing envelope e drops below level (log-linear interpolation), or NaN."""
    i = np.nonzero(e < level)[0]
    if len(i) == 0:
        return np.nan
    i = i[0]
    if i == 0:
        return t[0]
    a, b = np.log(e[i - 1]), np.log(e[i])
    return t[i - 1] + (t[i] - t[i - 1]) * (a - np.log(level)) / (a - b)


def rates(state, run):
    t = run["t"].to_numpy()
    s = SIGNALS[state](run).to_numpy()
    e = np.maximum.accumulate(s[::-1])[::-1]
    t_e = crossing(t, e, np.exp(-1))
    t0, t1 = crossing(t, e, TAIL_HI), crossing(t, e, TAIL_LO)
    gamma_tail = sigma = np.nan
    if np.isfinite(t1):
        sel = (t >= t0) & (t <= t1)
        if sel.sum() >= 3:
            (slope, _), cov = np.polyfit(t[sel], np.log(e[sel]), 1, cov=True)
            gamma_tail, sigma = -slope, np.sqrt(cov[0, 0])
    return 1 / t_e, gamma_tail, sigma, t0, t1


def main():
    runs = load(DIR)
    chosen = {(state, l, g): method for (state, l, g), (method, _, _) in select(runs).items()}
    rows = [(state, method, l, g, cut, chosen[state, l, g] == method, *rates(state, run))
            for (method, state, l, g), (run, cut) in runs.items()]
    table = pd.DataFrame(rows, columns=["state", "method", "L", "gamma", "cut", "selected", "Gamma_e", "Gamma_tail",
                                        "sigma_tail", "t_tail_min", "t_tail_max"]
                         ).sort_values(["state", "L", "gamma", "method"])
    table.to_csv(DIR / "states_rates.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 1000):
        print(table.to_string(index=False, float_format="%.4g"))
    table = table[table["selected"]]

    states = [s for s in ("minus-x", "neel", "neel-x") if s in set(table["state"])]
    sizes = sorted(table["L"].unique())
    blues = LinearSegmentedColormap.from_list("blues", RAMP[1:])
    colors = {l: blues(i / max(len(sizes) - 1, 1)) for i, l in enumerate(sizes)}
    fig, axes = plt.subplots(1, len(states), figsize=(4.2 * len(states), 3.4), squeeze=False)
    titles = {"minus-x": "Scar |−⟩^⊗L: m_⊥", "neel": "z-Néel: |s_z|", "neel-x": "x-Néel: s_⊥"}
    for ax, state in zip(axes[0], states):
        d = table[(table["state"] == state) & (table["gamma"] > 0)]
        g = np.array(sorted(d["gamma"].unique()))
        ax.plot(g, 2 * g, color=MUTED, lw=0.9, ls=":", label="2γ")
        for l in sizes:
            dl = d[d["L"] == l]
            if dl.empty:
                continue
            ax.plot(dl["gamma"], dl["Gamma_e"], "-o", ms=3, color=colors[l], label=f"L = {l}")
            ax.plot(dl["gamma"], dl["Gamma_tail"].where(dl["method"] == "exact"), "--", lw=1, color=colors[l])
            g0 = table[(table["state"] == state) & (table["L"] == l) & (table["gamma"] == 0)]["Gamma_e"]
            if len(g0) and np.isfinite(g0.iloc[0]) and g0.iloc[0] > 0:
                ax.axhline(g0.iloc[0], color=colors[l], lw=0.7, ls="-.")
        ax.set(xscale="log", yscale="log", xlabel="γ", ylabel="Γ", title=titles[state])
    axes[0][0].legend(fontsize=7.5)
    fig.text(0.01, -0.04, f"solid: 1/e rate Γ_e;  dashed: slope of ln(envelope) between {TAIL_HI:g} and {TAIL_LO:g} "
             "(exact runs only; trajectories reach their statistical floor there);  dash-dot: Γ_e at γ = 0",
             fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "states_rates.png")
    print(f"wrote {DIR / 'states_rates.csv'} and {DIR / 'states_rates.png'}")


if __name__ == "__main__":
    main()
