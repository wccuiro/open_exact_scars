"""
Thermodynamic decay rates of the `states` runs: Γ_e against 1/L for every γ, extrapolated to 1/L = 0.

Γ_e is the 1/e rate of the envelope from fit_states.py (states_rates.csv, run that first):
m_⊥ for the scar |−⟩^⊗L (minus-x), |s_z| for the z-Néel state (neel), s_⊥ for the x-Néel state (neel-x).
Two fits per (state, γ):
  linear     Γ_e = A + B/L             on L ≥ LIN_MIN
  quadratic  Γ_e = A + B/L + C/L²      on L ≥ QUAD_MIN
Γ_∞ = A of the linear fit; its error is the larger of the fit error and the difference to the quadratic intercept.

Néel at γ ≤ NEEL_REVIVAL_GAMMA: L = 6 and 8 are left out. Their closed-system |s_z| has finite-size revivals above
1/e (Γ_e at γ = 0 is 0.02 and 0.54, against ≈ 5 for L = 10, 12), so at weak dephasing Γ_e measures how fast the
revivals are damped, not the thermal decay. The linear fit then goes through L = 10, 12 exactly and there is no
quadratic fit; the error is taken as the size of the correction, |A − Γ_e(L_max)|.
Néel at every γ: no quadratic fit either. Only even L exist, and L = 6 lies ≈ 0.3 below L = 8 … 12 (which agree
to a few 0.01), a small-size jump rather than a smooth 1/L trend; the error is again |A − Γ_e(L_max)|.

Input:  states_rates.csv in data/states/ (or the directory given as the first argument)
Output: in the same directory
        states_inverse_L.csv            state, gamma, sizes, A_lin, sigma_A_lin, B_lin, A_quad, Gamma_inf, sigma_Gamma_inf
        states_inverse_L.png            Γ_e against 1/L with the linear fits, one panel per state
        states_thermodynamic_rate.png   Γ_∞ against γ, all states
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

LIN_MIN, QUAD_MIN = 8, 6
NEEL_REVIVAL_GAMMA, NEEL_REVIVAL_SIZES = 0.02, (6, 8)

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "states"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]  # small γ light, large γ dark
SERIES = {"minus-x": "#2a78d6", "neel": "#eb6834", "neel-x": "#1baf7a"}  # categorical slots 1, 2, 3
NAMES = {"minus-x": "Scar |−⟩^⊗L (m_⊥)", "neel": "z-Néel (|s_z|)", "neel-x": "x-Néel (s_⊥)"}
STATES = ("minus-x", "neel", "neel-x")

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


def inverse_l_figure(rates, table):
    states = [s for s in STATES if s in set(table["state"])]
    fig, axes = plt.subplots(1, len(states), figsize=(5.4 * len(states), 4.0), squeeze=False, layout="constrained")
    gammas = sorted(table["gamma"].unique())
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    colors = {g: blues(i / max(len(gammas) - 1, 1)) for i, g in enumerate(gammas)}
    xs = np.linspace(0, 1 / 4, 100)
    for ax, state in zip(axes[0], states):
        # scar: Γ/γ on a linear scale; Néel states: Γ on a log scale (not ∝ γ)
        scale = (lambda g: g) if state == "minus-x" else (lambda g: 1.0)
        for g in gammas:
            d = rates[(rates["state"] == state) & (rates["gamma"] == g)].sort_values("L")
            row = table[(table["state"] == state) & (table["gamma"] == g)].iloc[0]
            used = d["L"].isin([int(l) for l in row["sizes"].split()])
            x, y = 1 / d["L"].to_numpy(), d["Gamma_e"].to_numpy() / scale(g)
            ax.plot(x[used], y[used], "o", ms=4, color=colors[g], label=f"γ = {g:g}")
            ax.plot(x[~used], y[~used], "o", ms=4, mfc="none", mew=0.9, color=colors[g])
            xf = xs[xs <= x[used].max()]
            ax.plot(xf, (row["A_lin"] + row["B_lin"] * xf) / scale(g), "--", lw=1, color=colors[g])
        top = ax.secondary_xaxis("top")
        ls = sorted(rates[rates["state"] == state]["L"].unique())
        top.set_xticks([1 / l for l in ls], [str(l) for l in ls])
        top.set_xlabel("L")
        top.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=7.5)
        if state == "minus-x":
            ax.axhline(2, color=MUTED, ls=":", lw=1)
            ax.set(ylabel="Γ_e / γ")
        else:
            ax.set(yscale="log", ylabel="Γ_e")
        ax.set(xlim=(0, 0.26), xlabel="1/L", title=NAMES[state])
    axes[0][-1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.01, -0.05, f"dashed: A + B/L on L ≥ {LIN_MIN}, extrapolated to 1/L = 0; open markers: not used in the fit "
             f"(Néel L = 6, 8 at γ ≤ {NEEL_REVIVAL_GAMMA:g}: finite-size revivals)", fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "states_inverse_L.png")


def thermodynamic_figure(rates, table):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    g = np.array(sorted(table["gamma"].unique()))
    ax.plot(g, 2 * g, color=MUTED, ls=":", lw=1, label="free spins, 2γ")
    neel0 = rates[(rates["state"] == "neel") & (rates["gamma"] == 0)]
    if len(neel0):
        r0 = neel0.sort_values("L").iloc[-1]
        ax.axhline(r0["Gamma_e"], color=SERIES["neel"], ls="-.", lw=0.9,
                   label=f"z-Néel at γ = 0 (L = {int(r0['L'])})")
    for state in [s for s in STATES if s in set(table["state"])]:
        d = table[table["state"] == state]
        ax.errorbar(d["gamma"], d["Gamma_inf"], yerr=d["sigma_Gamma_inf"], fmt="-o", ms=4, capsize=2,
                    color=SERIES[state], label=NAMES[state])
    ax.set(xscale="log", yscale="log", xlabel="γ", ylabel="Γ", title="Thermodynamic decay rate Γ_∞ (1/L → 0)")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.text(0.01, -0.06, "Γ_∞: 1/e rate of the envelope, A + B/L extrapolated to 1/L = 0", fontsize=7.5,
             color=INK_SECONDARY)
    fig.savefig(DIR / "states_thermodynamic_rate.png")


def main():
    rates = pd.read_csv(DIR / "states_rates.csv")
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
    inverse_l_figure(rates, table)
    thermodynamic_figure(rates, table)
    print(f"wrote states_inverse_L.csv, states_inverse_L.png, states_thermodynamic_rate.png in {DIR}")


if __name__ == "__main__":
    main()
