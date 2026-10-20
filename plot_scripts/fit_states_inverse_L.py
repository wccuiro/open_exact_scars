"""
Thermodynamic decay rates of the `states` runs: Γ_e against 1/L for every γ, extrapolated to 1/L = 0.

Γ_e is the 1/e rate of the envelope from fit_states.py (states_rates.csv, run that first):
m_⊥ for the scar |−⟩^⊗L (minus-x), |s_z| for the z-Néel state (neel), s_⊥ for the x-Néel state (neel-x).
Only the selected run per (state, L, γ) is used: exact evolution where it finished (L ≤ 14), trajectories otherwise
(L ≥ 16, and L = 14 at γ ≤ 0.02 where the exact run was cut at 24 h).
Only γ ≥ GAMMA_FIT_MIN is extrapolated on L ≥ 8: the small-γ scar runs (γ = 0.0005 … 0.007, L ≤ 20) sit at the
finite-size cap Γ_e/γ ≈ L − 1 for the small L. The scar fit on the large sizes only (L ≥ 14, fit_small_gamma.py,
states_inverse_L_large.csv) is shown in the Γ_∞ figure as an estimate next to the lower bounds (not converged at
γ < 0.01).
Two fits per (state, γ):
  linear     Γ_e = A + B/L             on L ≥ LIN_MIN
  quadratic  Γ_e = A + B/L + C/L²      on L ≥ QUAD_MIN
Γ_∞ = A of the linear fit; its error is the larger of the fit error and the difference to the quadratic intercept.

Néel at γ ≤ NEEL_REVIVAL_GAMMA: L = 6 and 8 are left out. Their closed-system |s_z| has finite-size revivals above
1/e (Γ_e at γ = 0 is 0.02 and 0.54, against ≈ 5 for L = 10, 12), so at weak dephasing Γ_e measures how fast the
revivals are damped, not the thermal decay. The linear fit then goes through L = 10, 12 exactly and there is no
quadratic fit; the error is taken as the size of the correction, |A − Γ_e(L_max)|.
x-Néel at γ ≤ NEEL_X_REVIVAL_GAMMA: L = 6 is left out of the quadratic fit for the same reason (its closed-system
s_⊥ does not reach 1/e: Γ_e = 0.02 at γ = 0, 0.3 – 0.9 at γ = 0.01 – 0.05, against ≈ 4.8 for L ≥ 8).
γ = 0 is not fitted: its Γ_e is shown as measured, the mean over the sizes the fits use at small γ (L ≥ 8; z-Néel
L = 10, 12; error = spread over L). Scar at γ = 0: Γ_e = 0 (no decay at any L).
Néel at every γ: no quadratic fit either. Only even L exist, and L = 6 lies ≈ 0.3 below L = 8 … 12 (which agree
to a few 0.01), a small-size jump rather than a smooth 1/L trend; the error is again |A − Γ_e(L_max)|.

Input:  states_rates.csv in data/used_data/ (or the directory given as the first argument)
Output: in the same directory
        states_inverse_L.csv            state, gamma, sizes, A_lin, sigma_A_lin, B_lin, A_quad, Gamma_inf, sigma_Gamma_inf
        states_inverse_L.png            Γ_e against 1/L with the linear fits, one panel per state
                                        (circles: exact, squares: trajectories; γ = 0: diamonds, not fitted)
        states_thermodynamic_rate.png   Γ_∞ against γ, all states, with the γ = 0 rate (symlog axes); for the scar
                                        at γ < GAMMA_FIT_MIN, where Γ_∞ cannot be extrapolated, Γ_e of the largest
                                        size as a lower bound on Γ_∞ (Γ_e grows with L there), and the power law
                                        Γ_∞(γ) = Γ_∞(COLLAPSE_ANCHOR) (γ/COLLAPSE_ANCHOR)^(1 − ν) implied by the
                                        collapse of fit_small_gamma.py (small_gamma_collapse.csv: ν and its range),
                                        and the scar fit on L ≥ 14 (states_inverse_L_large.csv); run
                                        fit_small_gamma.py first
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

LIN_MIN, QUAD_MIN = 8, 6
GAMMA_FIT_MIN = 0.01
NEEL_REVIVAL_GAMMA, NEEL_REVIVAL_SIZES = 0.02, (6, 8)
NEEL_X_REVIVAL_GAMMA, NEEL_X_REVIVAL_SIZES = 0.05, (4, 6)
G_LIN, RATE_LIN = 0.01, 0.02  # symlog axes of the 1/L figure (scar panel): linear below, so that Γ_e = 0 fits
G_LIN_INF, RATE_LIN_INF = 2e-4, 2e-3  # the same for the Γ_∞ figure, low enough for the small-γ lower bounds
LARGE_L_MIN = 14  # as in fit_small_gamma.py (label only)
COLLAPSE_ANCHOR = 0.05  # γ at which the collapse power law is pinned to the extrapolated Γ_∞

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]  # small γ light, large γ dark
SERIES = {"minus-x": "#2a78d6", "neel": "#eb6834", "neel-x": "#1baf7a"}  # categorical slots 1, 2, 3
NAMES = {"minus-x": "Scar |−⟩^⊗L (m_⊥)", "neel": "z-Néel (|s_z|)", "neel-x": "x-Néel (s_⊥)"}
STATES = ("minus-x", "neel", "neel-x")
GAMMA0 = INK_SECONDARY  # γ = 0 points in the 1/L figure

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fit(x, y, degree):
    """Polynomial fit in x = 1/L; returns coefficients (highest power first) and σ of the intercept
    (NaN unless there are more than degree + 2 points)."""
    if len(x) > degree + 2:
        p, cov = np.polyfit(x, y, degree, cov=True)
        return p, np.sqrt(cov[-1, -1])
    return np.polyfit(x, y, degree), np.nan


def sizes_used(state, gamma, sizes, minimum):
    if state == "neel" and gamma <= NEEL_REVIVAL_GAMMA:
        sizes = [l for l in sizes if l not in NEEL_REVIVAL_SIZES]
    if state == "neel-x" and gamma <= NEEL_X_REVIVAL_GAMMA:
        sizes = [l for l in sizes if l not in NEEL_X_REVIVAL_SIZES]
    return [l for l in sizes if l >= minimum]


def extrapolate(rates):
    rows = []
    for (state, g), dg in rates[rates["gamma"] > 0].groupby(["state", "gamma"]):
        dg = dg.set_index("L")["Gamma_e"]
        lin = sizes_used(state, g, dg.index, LIN_MIN)
        quad = sizes_used(state, g, dg.index, QUAD_MIN)
        p1, s1 = fit(1 / np.array(lin), dg[lin].to_numpy(), 1)
        a_quad = fit(1 / np.array(quad), dg[quad].to_numpy(), 2)[0][-1] if state != "neel" else np.nan
        spread = abs(p1[-1] - a_quad) if np.isfinite(a_quad) else abs(p1[-1] - dg[max(lin)])
        rows.append((state, g, " ".join(map(str, lin)), p1[-1], s1, p1[0], a_quad, p1[-1],
                     np.nanmax([s1, spread])))
    return pd.DataFrame(rows, columns=["state", "gamma", "sizes", "A_lin", "sigma_A_lin", "B_lin", "A_quad",
                                       "Gamma_inf", "sigma_Gamma_inf"])


def closed_system(rates):
    """γ = 0 rate per state, not fitted: mean and spread of Γ_e over the sizes the γ > 0 fits use at small γ.
    Scar: 0 (m_⊥ = 1 at every L, Γ_e is NaN in states_rates.csv because the envelope never reaches 1/e)."""
    rows = []
    for state, d in rates[rates["gamma"] == 0].groupby("state"):
        if state == "minus-x":
            rows.append((state, "", 0.0, 0.0))
            continue
        d = d.set_index("L")["Gamma_e"]
        v = d[sizes_used(state, 0.0, d.index, LIN_MIN)].dropna()
        rows.append((state, " ".join(map(str, v.index)), v.mean(), v.std()))
    return pd.DataFrame(rows, columns=["state", "sizes", "Gamma_0", "sigma_Gamma_0"])


def inverse_l_figure(rates, table):
    states = [s for s in STATES if s in set(table["state"])]
    fig, axes = plt.subplots(1, len(states), figsize=(5.4 * len(states), 4.0), squeeze=False, layout="constrained")
    gammas = sorted(table["gamma"].unique())
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    colors = {g: blues(i / max(len(gammas) - 1, 1)) for i, g in enumerate(gammas)}
    xs = np.linspace(0, 1 / 4, 100)
    for ax, state in zip(axes[0], states):
        for g in gammas:
            d = rates[(rates["state"] == state) & (rates["gamma"] == g)].sort_values("L")
            row = table[(table["state"] == state) & (table["gamma"] == g)].iloc[0]
            used = d["L"].isin([int(l) for l in row["sizes"].split()])
            x, y = 1 / d["L"].to_numpy(), d["Gamma_e"].to_numpy()
            traj = (d["method"] != "exact").to_numpy()
            for sel, marker in ((~traj, "o"), (traj, "s")):
                ax.plot(x[used & sel], y[used & sel], marker, ms=4 if marker == "o" else 3.5, color=colors[g],
                        label=f"γ = {g:g}" if marker == "o" else None)
                ax.plot(x[~used & sel], y[~used & sel], marker, ms=4, mfc="none", mew=0.9, color=colors[g])
            xf = xs[xs <= x[used].max()]
            ax.plot(xf, row["A_lin"] + row["B_lin"] * xf, "--", lw=1, color=colors[g])
        # γ = 0, not fitted (scar: Γ_e = 0 at every L; NaN in states_rates.csv since m_⊥ never reaches 1/e)
        d = rates[(rates["state"] == state) & (rates["gamma"] == 0)].sort_values("L")
        if len(d):
            used = d["L"].isin(sizes_used(state, 0.0, d["L"].tolist(), LIN_MIN)).to_numpy()
            x, y = 1 / d["L"].to_numpy(), d["Gamma_e"].to_numpy()
            if state == "minus-x":
                y = np.zeros_like(x)
            ax.plot(x[used], y[used], "D", ms=4.5, mfc=SURFACE, mew=1.1, color=GAMMA0, label="γ = 0 (not fitted)",
                    zorder=3)
            ax.plot(x[~used], y[~used], "D", ms=4.5, mfc="none", mew=0.7, color=MUTED, zorder=3)
        top = ax.secondary_xaxis("top")
        ls = []
        for l in sorted(rates[rates["state"] == state]["L"].unique()):
            if not ls or 1 / ls[-1] - 1 / l > 0.009:  # skip labels that would overlap
                ls.append(l)
        top.set_xticks([1 / l for l in ls], [str(l) for l in ls])
        top.set_xlabel("L")
        top.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=7.5)
        if state == "minus-x":  # symlog, so that Γ_e = 0 at γ = 0 fits
            ax.set_yscale("symlog", linthresh=RATE_LIN, linscale=0.5)
            ax.set(ylim=(-0.1 * RATE_LIN, None), ylabel="Γ_e")
        else:
            ax.set(yscale="log", ylabel="Γ_e")
        ax.set(xlim=(0, 0.26), xlabel="1/L", title=NAMES[state])
    handles, labels = axes[0][-1].get_legend_handles_labels()
    order = [labels.index("γ = 0 (not fitted)")] + [i for i, l in enumerate(labels) if l != "γ = 0 (not fitted)"]
    axes[0][-1].legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7.5, loc="center left",
                       bbox_to_anchor=(1.01, 0.5))
    fig.text(0.01, -0.05, f"dashed: A + B/L on L ≥ {LIN_MIN}, extrapolated to 1/L = 0; circles: exact, squares: "
             f"trajectories; open markers: not in the linear fit (L < {LIN_MIN}, and z-Néel L = 6, 8 at "
             f"γ ≤ {NEEL_REVIVAL_GAMMA:g}: finite-size revivals)\ndiamonds: γ = 0 (closed system, exact or one trajectory), not fitted; light gray: sizes left out at small γ; scar at "
             f"γ = 0: Γ_e = 0 at every L (perfect revivals). Scar axis linear below Γ_e = {RATE_LIN:g}, logarithmic above", fontsize=7.5,
             color=INK_SECONDARY)
    fig.savefig(DIR / "states_inverse_L.png")


def lower_bounds(rates):
    """Scar at 0 < γ < GAMMA_FIT_MIN: Γ_e of the largest size, a lower bound on Γ_∞ (Γ_e grows with L there)."""
    d = rates[(rates["state"] == "minus-x") & (rates["gamma"] > 0) & (rates["gamma"] < GAMMA_FIT_MIN)]
    d = d.dropna(subset=["Gamma_e"])
    return d.loc[d.groupby("gamma")["L"].idxmax()].sort_values("gamma")


def thermodynamic_figure(table, gamma0, bounds, large):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    g = np.array(sorted(set(table["gamma"]) | set(bounds["gamma"])))
    gs = np.concatenate([np.linspace(0, G_LIN_INF, 20), np.geomspace(G_LIN_INF, g.max(), 200)])  # from γ = 0
    ax.plot(gs, 2 * gs, color=MUTED, ls=":", lw=1, label="free spins, 2γ")
    collapse = DIR / "small_gamma_collapse.csv"
    for state in [s for s in STATES if s in set(table["state"])]:
        d = table[table["state"] == state].sort_values("gamma")
        ax.errorbar(d["gamma"], d["Gamma_inf"], yerr=d["sigma_Gamma_inf"], fmt="-o", ms=4, capsize=2,
                    color=SERIES[state], label=NAMES[state])
        first_g, first_r = d["gamma"].iloc[0], d["Gamma_inf"].iloc[0]
        if state == "minus-x" and len(bounds):
            ax.plot(bounds["gamma"], bounds["Gamma_e"], "^", ms=5, mfc=SURFACE, mew=1.1, color=SERIES[state],
                    label="scar Γ_e at the largest L (L = " + ", ".join(map(str, sorted(set(bounds["L"])))) +
                    "): lower bound")
            if collapse.exists():
                c = pd.read_csv(collapse).iloc[0]
                anchor = d[np.isclose(d["gamma"], COLLAPSE_ANCHOR)]["Gamma_inf"].iloc[0]
                gs = np.concatenate([np.linspace(0, G_LIN_INF, 20)[:-1],  # down to γ = 0, where the law gives 0
                                     np.geomspace(G_LIN_INF, COLLAPSE_ANCHOR, 80)])
                law = {nu: anchor * (gs / COLLAPSE_ANCHOR) ** (1 - nu) for nu in (c["nu"], c["nu_min"], c["nu_max"])}
                ax.fill_between(gs, law[c["nu_min"]], law[c["nu_max"]], color=SERIES[state], alpha=0.12, lw=0)
                ax.plot(gs, law[c["nu"]], "--", lw=1.0, color=SERIES[state],
                        label=f"collapse: Γ_∞ ∝ γ^(1 − ν), ν = {c['nu']:.2f} (band {c['nu_min']:.2f} – {c['nu_max']:.2f})")
            first_g, first_r = bounds["gamma"].iloc[0], bounds["Gamma_e"].iloc[0]
        if state == "minus-x" and len(large):
            ax.errorbar(large["gamma"], large["A"], yerr=large["sigma_A"], fmt="s", ms=4.5, capsize=2, mew=1.1,
                        mfc=SURFACE, color=SERIES[state], label=f"scar, A + B/L on L ≥ {LARGE_L_MIN} only", zorder=3)
        z = gamma0[gamma0["state"] == state]
        if len(z):  # γ = 0, joined to the smallest γ > 0
            z = z.iloc[0]
            if state != "minus-x":  # the scar's γ = 0 point is reached by the collapse power law instead
                ax.plot([0, first_g], [z["Gamma_0"], first_r], ":", lw=1.4, color=SERIES[state])
            ax.errorbar([0], [z["Gamma_0"]], yerr=[z["sigma_Gamma_0"]], fmt="D", ms=6, mew=1.4,
                        mfc=SURFACE, capsize=2, color=SERIES[state], zorder=4)
    ax.set_xscale("symlog", linthresh=G_LIN_INF, linscale=0.4)
    ax.set_yscale("symlog", linthresh=RATE_LIN_INF, linscale=0.4)
    ax.set_xticks([0, 1e-3, 1e-2, 1e-1, 1, 10], ["0", "10⁻³", "10⁻²", "10⁻¹", "1", "10"])
    ax.set_yticks([0, 1e-2, 1e-1, 1, 10], ["0", "10⁻²", "10⁻¹", "1", "10"])
    ax.set(xlim=(-0.3 * G_LIN_INF, 1.5 * g.max()), ylim=(-0.4 * RATE_LIN_INF, 60), xlabel="γ", ylabel="Γ",
           title="Thermodynamic decay rate Γ_∞ (1/L → 0)")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.text(0.01, -0.12, f"Γ_∞: 1/e rate of the envelope, A + B/L extrapolated to 1/L = 0 (γ ≥ {GAMMA_FIT_MIN:g}).\n"
             f"Triangles: γ < {GAMMA_FIT_MIN:g}, where every L ≤ 20 sits at the finite-size cap Γ_e ≈ γ(L − 1) and "
             "Γ_∞ cannot be extrapolated;\nΓ_e grows with L there, so Γ_e of the largest size bounds Γ_∞ from below. "
             f"Dashed and band: power law through Γ_∞(γ = {COLLAPSE_ANCHOR:g}), continued to γ = 0. Open squares: scar fit on L ≥ {LARGE_L_MIN} "
             "(not converged at γ < 0.01).\n"
             "Open diamonds: γ = 0 (closed system), mean Γ_e over L ≥ 8 (z-Néel L = 10, 12), not fitted; "
             "scar Γ = 0 (perfect revivals).\n"
             f"Axes linear below γ = {G_LIN_INF:g} and Γ = {RATE_LIN_INF:g}, logarithmic above (symlog).",
             fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "states_thermodynamic_rate.png")


def main():
    rates = pd.read_csv(DIR / "states_rates.csv")
    rates = rates[rates["selected"]]
    bounds = lower_bounds(rates)
    large_file = DIR / "states_inverse_L_large.csv"  # from fit_small_gamma.py
    large = pd.read_csv(large_file) if large_file.exists() else pd.DataFrame(columns=["gamma", "A", "sigma_A"])
    rates = rates[(rates["gamma"] == 0) | (rates["gamma"] >= GAMMA_FIT_MIN)]
    table = extrapolate(rates)
    table.to_csv(DIR / "states_inverse_L.csv", index=False)
    t = table.assign(**{"Gamma_inf/gamma": table["Gamma_inf"] / table["gamma"]})
    with pd.option_context("display.width", 200):
        print(t.to_string(index=False, float_format="%.3g"))
    for state, d in table.groupby("state"):
        g, r = d["gamma"].to_numpy(), d["Gamma_inf"].to_numpy()
        print(f"{state}: local exponent d ln Γ_∞ / d ln γ:",
              ", ".join(f"{g[i]:g}–{g[i + 1]:g}: {np.log(r[i + 1] / r[i]) / np.log(g[i + 1] / g[i]):.2f}"
                        for i in range(len(g) - 1)))
    gamma0 = closed_system(rates)
    print("γ = 0 (not fitted):")
    print(gamma0.to_string(index=False, float_format="%.3g"))
    inverse_l_figure(rates, table)
    print("scar, γ < GAMMA_FIT_MIN: Γ_e of the largest size (lower bound on Γ_∞):")
    print(bounds[["gamma", "L", "Gamma_e"]].to_string(index=False, float_format="%.4g"))
    thermodynamic_figure(table, gamma0, bounds, large)
    print(f"wrote states_inverse_L.csv, states_inverse_L.png, states_thermodynamic_rate.png in {DIR}")


if __name__ == "__main__":
    main()
