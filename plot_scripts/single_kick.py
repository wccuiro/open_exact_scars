"""
Single-kick test of the small-γ scar decay: the damage D(τ) of one σᶻ_k kick (`single_flip`), and the prediction

    m_⊥(t) ≈ exp[−γ ∫₀ᵗ D̄(τ) dτ]

(independent kicks: a kick multiplies m_⊥ by 1 − D/L, kicks arrive at rate γL) against the measured m_⊥.

D̄(τ) = mean of D over the 16 phases φ_m and the kicked sites k (all sites for L ≤ 20; sites 0, L/4, L/2, 3L/4 for
L = 22, 24, so those sizes carry a site-sampling error). D(0) = 2. For t beyond the last τ (25 for L ≤ 20), D̄ is
continued with its mean over the last SAT_WINDOW units of τ (D_sat). L = 22, 24 stop at τ = 8 and are used only for
γ where the predicted 1/e time is below 8.

Comparison: Γ_e predicted (γ ∫₀^{t_e} D̄ = 1) against Γ_e measured (fit_small_gamma.rate on the run chosen by
states_data.select), at every (L, γ) of the scar runs with γ ≤ GAMMA_MAX; and the thermodynamic rate Γ_∞ (from
states_inverse_L.csv, γ ≥ 0.01) against the prediction from the largest size whose τ range covers t_e.

Input:  single_flip_L<L>*.csv, the scar run files, states_inverse_L.csv in data/used_data/ (or the first argument)
Output: in the same directory
        single_kick_damage.csv     L, tau, D_mean, D_err (standard error over phases and sites)
        single_kick_rates.csv      L, gamma, Gamma_measured, sigma_measured, Gamma_predicted
        single_kick_damage.png     D̄(τ) for every L, with L − 1
        single_kick_rates.png      predicted against measured Γ_e/γ; Γ_∞ against the large-L prediction
        single_kick_series.png     measured m_⊥ envelope and prediction for a few (L, γ)
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import states_data
from fit_small_gamma import m_perp, rate

GAMMA_MAX = 0.05
SAT_WINDOW = 5.0
SERIES_CASES = [(12, 0.002), (16, 0.0005), (16, 0.005), (20, 0.0005), (20, 0.002), (20, 0.01)]

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]  # small light, large dark
MEASURED, PREDICTED = "#2a78d6", "#eb6834"  # categorical slots 1, 2

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def damage():
    """{L: DataFrame(tau, D_mean, D_err)} from all single_flip files of the directory."""
    out = {}
    for f in DIR.glob("single_flip_L*.csv"):
        l = int(re.match(r"single_flip_L(\d+)", f.name).group(1))
        d = pd.read_csv(f, usecols=["m", "k", "tau", "D"])
        g = d.groupby("tau")["D"]
        out[l] = pd.DataFrame({"tau": g.mean().index, "D_mean": g.mean().to_numpy(),
                               "D_err": (g.std() / np.sqrt(g.count())).to_numpy()})
    return dict(sorted(out.items()))


def predicted_log(dmg, t):
    """γ-free exponent ∫₀ᵗ D̄ dτ at the times t (D̄ continued with D_sat past its last τ); NaN past τ_max for sizes
    whose D̄ has not saturated (τ_max < 25)."""
    tau, d = dmg["tau"].to_numpy(), dmg["D_mean"].to_numpy()
    cum = np.concatenate([[0], np.cumsum(np.diff(tau) * (d[1:] + d[:-1]) / 2)])
    out = np.interp(t, tau, cum)
    past = t > tau[-1]
    if tau[-1] >= 25:
        d_sat = d[tau >= tau[-1] - SAT_WINDOW].mean()
        out[past] = cum[-1] + d_sat * (t[past] - tau[-1])
    else:
        out[past] = np.nan
    return out


def predicted_rate(dmg, gamma):
    t = np.linspace(0, 4 / gamma, 200001)
    x = gamma * predicted_log(dmg, t)
    i = np.nonzero(x >= 1)[0]
    return np.nan if len(i) == 0 or not np.isfinite(x[i[0]]) else 1 / t[i[0]]


def main():
    dmgs = damage()
    pd.concat([d.assign(L=l) for l, d in dmgs.items()])[["L", "tau", "D_mean", "D_err"]].to_csv(
        DIR / "single_kick_damage.csv", index=False)
    for l, d in dmgs.items():
        tail = d[d["tau"] >= d["tau"].max() - SAT_WINDOW]["D_mean"].mean()
        print(f"L = {l}: D̄(1) = {np.interp(1, d['tau'], d['D_mean']):.2f}, D̄(4) = {np.interp(4, d['tau'], d['D_mean']):.2f},"
              f" D̄(8) = {np.interp(8, d['tau'], d['D_mean']):.2f}, mean over the last {SAT_WINDOW:g} of τ ≤ "
              f"{d['tau'].max():g}: {tail:.2f} (L − 1 = {l - 1})")

    runs = states_data.select(states_data.load(DIR, states=("minus-x",)))
    rows = []
    for (_, l, g), (method, run, _) in runs.items():
        if 0 < g <= GAMMA_MAX and l in dmgs:
            gm, sm = rate(run)
            rows.append((l, g, method, gm, sm, predicted_rate(dmgs[l], g)))
    table = pd.DataFrame(rows, columns=["L", "gamma", "method", "Gamma_measured", "sigma_measured",
                                        "Gamma_predicted"]).sort_values(["L", "gamma"])
    table.to_csv(DIR / "single_kick_rates.csv", index=False)
    t = table.assign(meas=table["Gamma_measured"] / table["gamma"], pred=table["Gamma_predicted"] / table["gamma"],
                     ratio=table["Gamma_measured"] / table["Gamma_predicted"])
    with pd.option_context("display.width", 200):
        print(t.pivot_table(index="gamma", columns="L", values="ratio").round(3).to_string())

    inf = pd.read_csv(DIR / "states_inverse_L.csv")
    inf = inf[inf["state"] == "minus-x"].sort_values("gamma")
    big = max(dmgs)
    inf["Gamma_pred_big"] = [predicted_rate(dmgs[big], g) for g in inf["gamma"]]
    print(f"Γ_∞ against the prediction from D̄ at L = {big}:")
    print(inf[["gamma", "Gamma_inf", "sigma_Gamma_inf", "Gamma_pred_big"]].to_string(index=False, float_format="%.4g"))

    sizes = list(dmgs)
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    colors = {l: blues(i / max(len(sizes) - 1, 1)) for i, l in enumerate(sizes)}

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for l, d in dmgs.items():
        ax.plot(d["tau"], d["D_mean"], color=colors[l], label=f"L = {l}")
        ax.plot([d["tau"].max() + 0.6, d["tau"].max() + 1.6], [l - 1, l - 1], color=colors[l], lw=1.0) \
            if d["tau"].max() >= 25 else None
    ax.set(xlabel="τ", ylabel="D̄(τ)", title="Damage of one σᶻ kick on the scar state", ylim=(0, None))
    ax.legend(fontsize=7.5, loc="lower right", ncols=2)
    fig.text(0.01, -0.08, "D̄ = Σ_j [1 − Re c_j], averaged over 16 phases and the kicked sites (all sites for L ≤ 20;\n"
             "4 sites for L = 22, 24, τ ≤ 8). D̄(0) = 2. Short bars right of τ = 25: L − 1.", fontsize=7.5,
             color=INK_SECONDARY)
    fig.savefig(DIR / "single_kick_damage.png")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
    lo, hi = 2, max(sizes)
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls=":", lw=1.1, label="measured = predicted")
    for l in sizes:
        d = t[t["L"] == l]
        if len(d):
            ax.errorbar(d["pred"], d["meas"], yerr=d["sigma_measured"] / d["gamma"], fmt="o", ms=3.5, capsize=1.5,
                        color=colors[l], label=f"L = {l}")
    ax.set(xlabel="predicted Γ_e/γ (single kick)", ylabel="measured Γ_e/γ",
           title=f"Finite L, γ ≤ {GAMMA_MAX:g}")
    ax.legend(fontsize=7.5)
    bx.plot(inf["gamma"], 2 * inf["gamma"], color=MUTED, ls=":", lw=1.0, label="2γ")
    bx.errorbar(inf["gamma"], inf["Gamma_inf"], yerr=inf["sigma_Gamma_inf"], fmt="-o", ms=3.5, capsize=1.5,
                color=MEASURED, label="Γ_∞ (A + B/L)")
    bx.plot(inf["gamma"], inf["Gamma_pred_big"], "--s", ms=3.5, color=PREDICTED,
            label=f"single kick, D̄ of L = {big} (τ ≤ {dmgs[big]['tau'].max():g})")
    bx.set(xscale="log", yscale="log", xlabel="γ", ylabel="Γ", title="Thermodynamic rate against the single-kick "
           "prediction")
    bx.legend(fontsize=7.5)
    fig.text(0.01, -0.06, "Prediction: γ ∫₀^{t_e} D̄(τ) dτ = 1. Left: one point per (L, γ), error bars: statistical "
             "error of the trajectory Γ_e. Right: missing predictions have t_e beyond the τ range of the L = "
             f"{big} damage.", fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "single_kick_rates.png")

    cases = [c for c in SERIES_CASES if (c[0], c[1]) in {(l, g) for (_, l, g) in runs} and c[0] in dmgs]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.2), layout="constrained", squeeze=False)
    for ax, (l, g) in zip(axes.flat, cases):
        method, run, _ = runs["minus-x", l, g]
        tt = run["t"].to_numpy()
        m, err = m_perp(run)
        e = np.maximum.accumulate(m[::-1])[::-1]
        ax.plot(g * tt, e, color=MEASURED, lw=1.2)
        if err.any():
            ax.fill_between(g * tt, e - err, e + err, color=MEASURED, alpha=0.2, lw=0)
        ax.plot(g * tt, np.exp(-g * predicted_log(dmgs[l], tt)), color=PREDICTED, ls="--", lw=1.2)
        ax.plot(g * tt, np.exp(-2 * g * tt), color=MUTED, ls=":", lw=1.0)
        ax.set(yscale="log", ylim=(0.02, 1.1), title=f"L = {l}, γ = {g:g} ({'exact' if method == 'exact' else 'trajectories'})")
    for ax in axes.flat[len(cases):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("γt")
    for ax in axes[:, 0]:
        ax.set_ylabel("envelope of m_⊥")
    fig.legend(handles=[Line2D([], [], color=MEASURED, label="measured (band: ± 1 standard error)"),
                        Line2D([], [], color=PREDICTED, ls="--", label="single kick, exp(−γ ∫ D̄)"),
                        Line2D([], [], color=MUTED, ls=":", label="free spins, e^(−2γt)")],
               loc="outside lower center", ncols=3)
    fig.suptitle("Scar decay against the single-kick prediction", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.savefig(DIR / "single_kick_series.png")
    print(f"wrote single_kick_damage.csv, single_kick_rates.csv and three figures in {DIR}")


if __name__ == "__main__":
    main()
