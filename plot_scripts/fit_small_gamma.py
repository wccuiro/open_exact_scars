"""
Small-γ scar runs: the 1/e rate Γ_e of m_⊥ against L at γ = 0.0005 … 0.05, the finite-size cap Γ_e/γ ≈ L − 1, a
scaling collapse, and the jump-unravelling cross-checks.

Γ_e is the 1/e rate of the envelope (fit_states.rates); one run per (L, γ) from states_data.select: exact for
L ≤ 12 (the small-γ exact runs end at t-max = 3/(γ(L − 1)), past the 1/e time), jump trajectories (traj-seeds pooled)
for L = 12 … 20 at γ < 0.01, diffusive trajectories at γ ≥ 0.01, L ≥ 16. Statistical error of a trajectory Γ_e:
σ_Γ = Γ² σ_m / |dE/dt| at the crossing, with σ_m the standard error of m_⊥ there. Trajectory errors are correlated
between the γ of one L: every γ uses the same uniform random numbers (traj-seed), so the kick sequences are the
same up to the time scale 1/γ.

Collapse: Γ_e/γ = ξ f(L/ξ) with ξ = γ^(−ν) (f(x) ≈ x − 1/ξ-corrections at small x: the cap L − 1; f → const = Γ_∞/γ
· γ^ν at large x). ν is chosen to minimize the spread of y = (Γ_e/γ) γ^ν at common x = L γ^ν (linear interpolation in
log x between the points of each γ), over γ ≤ COLLAPSE_GAMMA_MAX and L ≥ COLLAPSE_L_MIN.

Large-size extrapolation: A + B/L fitted on L ≥ LARGE_L_MIN only (14, 16, 18, 20; at least 3 sizes, so not
γ = 0.0007, 0.0014, which stop at L = 16), unweighted; error = the larger of the fit error and the change of A when
L = LARGE_L_CHECK is added (A_check). Not converged at γ < 0.01 (the sizes are near the cap): an estimate of Γ_∞.

Cross-checks (the plan's references for the jump runs at small γ): jump against exact (L = 12, γ ≥ 0.002) and against
diffusive trajectories (L = 12, 14, γ = 0.002, 0.005): rms of (m_⊥,jump − m_⊥,ref)/σ while m_⊥,ref > 0.05, and the
relative difference of Γ_e.

Input:  the scar run files in data/used_data/ (or the directory given as the first argument)
Output: in the same directory
        small_gamma_rates.csv          L, gamma, method, Gamma_e, sigma_Gamma_e, Gamma_e/gamma
        small_gamma_checks.csv         L, gamma, reference, rms_z, Gamma_jump, Gamma_ref, rel_diff
        small_gamma_collapse.csv       nu, nu_min, nu_max (cost within a factor 2 of the minimum); read by
                                       fit_states_inverse_L.py for the small-γ part of the Γ_∞ figure
        states_inverse_L_large.csv     gamma, sizes, A, sigma_A, B, A_check (the large-size fit); read by
                                       fit_states_inverse_L.py (Γ_∞ figure) and single_kick.py
        small_gamma_rates.png          Γ_e/γ against L, one line per γ, with L − 1, the large-size fits and their L → ∞
                                       value; and the collapse
        states_inverse_L_large.png     Γ_e/γ against 1/L with the large-size fits
        small_gamma_crosscheck.png     m_⊥(t) of jump, exact and diffusive runs where they overlap
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import states_data
from fit_states import crossing

GAMMA_MAX = 0.05
COLLAPSE_GAMMA_MAX, COLLAPSE_L_MIN = 0.05, 8
LARGE_L_MIN, LARGE_L_CHECK = 14, 12
X_INF = 23.5  # position of L = ∞ on the L axis of small_gamma_rates.png

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]  # small γ light, large γ dark
SERIES = {"exact": "#0b0b0b", "jump": "#2a78d6", "traj": "#eb6834"}  # reference ink, categorical slots 1, 2
NAMES = {"exact": "exact ρ", "jump": "jump trajectories", "traj": "diffusive trajectories"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def m_perp(run):
    m = np.hypot(run["mx"], run["my"]).to_numpy()
    if "mx_err" not in run:
        return m, np.zeros_like(m)
    err = np.hypot(run["mx"] * run["mx_err"], run["my"] * run["my_err"]).to_numpy() / np.maximum(m, 1e-12)
    return m, err


def rate(run):
    """Γ_e and its statistical error (0 for the exact evolution)."""
    t = run["t"].to_numpy()
    m, err = m_perp(run)
    e = np.maximum.accumulate(m[::-1])[::-1]
    t_e = crossing(t, e, np.exp(-1))
    if not np.isfinite(t_e):
        return np.nan, np.nan
    i = max(np.searchsorted(t, t_e), 1)
    lo, hi = max(i - 4, 0), min(i + 4, len(t) - 1)
    slope = abs((e[hi] - e[lo]) / (t[hi] - t[lo]))
    return 1 / t_e, np.interp(t_e, t, err) / slope / t_e ** 2 if slope > 0 else np.nan


def rates_table(runs):
    rows = []
    for (state, l, g), (method, run, _) in states_data.select(runs).items():
        if 0 < g <= GAMMA_MAX:
            gamma_e, sigma = rate(run)
            rows.append((l, g, method, gamma_e, sigma, gamma_e / g))
    return pd.DataFrame(rows, columns=["L", "gamma", "method", "Gamma_e", "sigma_Gamma_e", "Gamma_e/gamma"]
                        ).sort_values(["gamma", "L"])


def collapse_cost(table, nu):
    """Mean squared log-spread of y = (Γ_e/γ)γ^ν between the curves of different γ at common x = Lγ^ν."""
    curves = []
    for g, d in table.groupby("gamma"):
        d = d.sort_values("L")
        curves.append((np.log(d["L"] * g ** nu), np.log(d["Gamma_e/gamma"] * g ** nu)))
    cost, n = 0.0, 0
    for i, (xi, yi) in enumerate(curves):
        for xj, yj in curves[i + 1:]:
            sel = (xj >= xi.min()) & (xj <= xi.max())
            if sel.any():
                cost += np.sum((np.interp(xj[sel], xi, yi) - yj[sel]) ** 2)
                n += sel.sum()
    return cost / n if n else np.inf


def linear_fit(x, y):
    """A + B x: coefficients (B, A) and σ of A (NaN with fewer than 4 points)."""
    if len(x) > 3:
        p, cov = np.polyfit(x, y, 1, cov=True)
        return p, np.sqrt(cov[1, 1])
    return np.polyfit(x, y, 1), np.nan


def large_l_fits(table):
    rows = []
    for g, d in table.dropna(subset=["Gamma_e"]).groupby("gamma"):
        d = d.set_index("L")["Gamma_e"].sort_index()
        big = [l for l in d.index if l >= LARGE_L_MIN]
        check = [l for l in d.index if l >= LARGE_L_CHECK]
        if len(big) < 3:
            continue
        p, s = linear_fit(1 / np.array(big), d[big].to_numpy())
        a_check = linear_fit(1 / np.array(check), d[check].to_numpy())[0][1]
        rows.append((g, " ".join(map(str, big)), p[1], np.nanmax([s, abs(p[1] - a_check)]), p[0], a_check))
    return pd.DataFrame(rows, columns=["gamma", "sizes", "A", "sigma_A", "B", "A_check"])


def rates_figure(table, nu, large):
    gammas = sorted(table["gamma"].unique())
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    colors = {g: blues(i / max(len(gammas) - 1, 1)) for i, g in enumerate(gammas)}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained", width_ratios=[1.15, 1])
    ls = np.arange(4, 21)
    ax.plot(ls, ls - 1, color=MUTED, ls=":", lw=1.1, label="L − 1")
    for g in gammas:
        d = table[table["gamma"] == g].sort_values("L")
        ax.errorbar(d["L"], d["Gamma_e/gamma"], yerr=d["sigma_Gamma_e"] / g, fmt="-o", ms=3.5, capsize=1.5,
                    color=colors[g], label=f"γ = {g:g}")
        f = large[large["gamma"] == g]
        if len(f):
            f = f.iloc[0]
            lf = np.linspace(LARGE_L_MIN, 20, 20)
            ax.plot(lf, (f["A"] + f["B"] / lf) / g, "--", lw=1.0, color=colors[g])
            ax.plot([20, X_INF], [(f["A"] + f["B"] / 20) / g, f["A"] / g], ":", lw=1.0, color=colors[g])
            ax.errorbar([X_INF], [f["A"] / g], yerr=[f["sigma_A"] / g], fmt="D", ms=4.5, capsize=2, color=colors[g])
        c = table[(table["gamma"] == g) & (table["L"] >= COLLAPSE_L_MIN)].sort_values("L")
        if g <= COLLAPSE_GAMMA_MAX and len(c):
            bx.plot(c["L"] * g ** nu, c["Gamma_e/gamma"] * g ** nu, "-o", ms=3.5, color=colors[g])
    ax.axvspan(21.2, 22.3, color=SURFACE, zorder=2)  # gap between the finite sizes and L = ∞
    ax.set_xticks([4, 6, 8, 10, 12, 14, 16, 18, 20, X_INF], ["4", "6", "8", "10", "12", "14", "16", "18", "20", "∞"])
    ax.set(xlabel="L", ylabel="Γ_e / γ", title="Scar m_⊥: 1/e rate against L, and its L → ∞ limit",
           xlim=(3.5, X_INF + 1), ylim=(0, None))
    ax.legend(fontsize=7, ncols=2, loc="upper left")
    xs = np.geomspace(*bx.get_xlim(), 50)
    bx.plot(xs, xs, color=MUTED, ls=":", lw=1.1, label="y = x (cap L − 1 at large L)")
    bx.set(xscale="log", yscale="log", xlabel=f"L γ^{nu:.2f}", ylabel=f"(Γ_e/γ) γ^{nu:.2f}",
           title=f"Collapse with ξ = γ^(−ν), ν = {nu:.2f} (L ≥ {COLLAPSE_L_MIN})")
    bx.legend(fontsize=7.5, loc="lower right")
    fig.text(0.01, -0.08, "Exact ρ for L ≤ 12, jump trajectories (traj-seeds pooled) for L = 12 … 20 at γ < 0.01, "
             "diffusive trajectories at γ ≥ 0.01 and L ≥ 16; error bars: statistical error of the trajectory Γ_e.\n"
             f"Dashed: A + B/L fitted on L ≥ {LARGE_L_MIN}; diamonds at L = ∞: its intercept A/γ (error: max of the fit "
             f"error and the change when L = {LARGE_L_CHECK} is added; not converged at γ < 0.01).\n"
             "Trajectory errors are correlated between the γ of one L (same random numbers, kick times rescaled "
             "by 1/γ).", fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "small_gamma_rates.png")


def large_l_figure(table, large):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    all_g = sorted(table["gamma"].unique())
    colors = {g: blues(i / max(len(all_g) - 1, 1)) for i, g in enumerate(all_g)}  # same colours as above
    x = np.linspace(1 / 22, 1 / 7.5, 100)
    ax.plot(x, 1 / x - 1, color=MUTED, ls=":", lw=1.1, label="cap L − 1")
    for g in sorted(large["gamma"]):
        d = table[(table["gamma"] == g) & (table["L"] >= 8)].sort_values("L")
        row = large[large["gamma"] == g].iloc[0]
        used = d["L"] >= LARGE_L_MIN
        exact = d["method"] == "exact"
        for sel, marker in ((exact, "o"), (~exact, "s")):
            ax.plot(1 / d["L"][used & sel], d["Gamma_e/gamma"][used & sel], marker, ms=4, color=colors[g])
            ax.plot(1 / d["L"][~used & sel], d["Gamma_e/gamma"][~used & sel], marker, ms=4, mfc="none", mew=0.9,
                    color=colors[g])
        xf = np.linspace(0, 1 / LARGE_L_MIN, 20)
        ax.plot(xf, (row["A"] + row["B"] * xf) / g, "--", lw=1, color=colors[g], label=f"γ = {g:g}")
        ax.errorbar([0], [row["A"] / g], yerr=[row["sigma_A"] / g], fmt="D", ms=4, capsize=2, color=colors[g])
    top = ax.secondary_xaxis("top")
    top.set_xticks([1 / l for l in (8, 10, 12, 14, 16, 20)], ["8", "10", "12", "14", "16", "20"])
    top.set_xlabel("L")
    top.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=7.5)
    ax.set(xlim=(-0.004, 0.13), ylim=(0, None), xlabel="1/L", ylabel="Γ_e / γ",
           title=f"Scar: A + B/L on the large sizes, L ≥ {LARGE_L_MIN}")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(0.01, -0.07, f"filled: sizes in the fit (L ≥ {LARGE_L_MIN}); open: L = 8 … 12, not fitted; circles exact, "
             "squares trajectories; diamonds at 1/L = 0: A/γ,\n"
             f"error = max(fit error, change of A when L = {LARGE_L_CHECK} is added). γ = 0.0007, 0.0014 have only "
             "L ≤ 16 and are left out.", fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "states_inverse_L_large.png")


def checks(runs):
    rows, panels = [], []
    for (method, state, l, g), (run, _) in sorted(runs.items()):
        if method != "jump":
            continue
        for ref in ("exact", "traj"):
            if (ref, state, l, g) not in runs:
                continue
            r = runs[ref, state, l, g][0]
            t = run["t"].to_numpy()
            m, err = m_perp(run)
            mr = np.interp(t, r["t"], m_perp(r)[0])
            sel = (mr > 0.05) & (err > 0)
            z = (m - mr)[sel] / err[sel]
            gj, gr = rate(run)[0], rate(r)[0]
            rows.append((l, g, ref, np.sqrt(np.mean(z ** 2)), gj, gr, gj / gr - 1))
            panels.append((l, g, ref, run, r))
    return pd.DataFrame(rows, columns=["L", "gamma", "reference", "rms_z", "Gamma_jump", "Gamma_ref", "rel_diff"]), panels


def crosscheck_figure(panels):
    keys = sorted({(l, g) for l, g, *_ in panels})
    ncols = 4
    nrows = -(-len(keys) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows), squeeze=False, layout="constrained")
    for ax, (l, g) in zip(axes.flat, keys):
        drawn = False
        for pl, pg, ref, run, r in panels:
            if (pl, pg) != (l, g):
                continue
            if not drawn:
                m, err = m_perp(run)
                ax.fill_between(run["t"], m - err, m + err, color=SERIES["jump"], alpha=0.25, lw=0)
                ax.plot(run["t"], m, color=SERIES["jump"], lw=1.0)
                drawn = True
            ax.plot(r["t"], m_perp(r)[0], color=SERIES[ref], lw=1.0, ls="--" if ref == "traj" else "-")
        ax.set(yscale="log", ylim=(0.03, 1.2), title=f"L = {l}, γ = {g:g}")
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("m_⊥")
    handles = [mpl.lines.Line2D([], [], color=SERIES["jump"], label="jump (band: ± 1 standard error)"),
               mpl.lines.Line2D([], [], color=SERIES["exact"], label=NAMES["exact"]),
               mpl.lines.Line2D([], [], color=SERIES["traj"], ls="--", label=NAMES["traj"])]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    fig.suptitle("Jump trajectories against the exact evolution and the diffusive unravelling", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.savefig(DIR / "small_gamma_crosscheck.png")


def main():
    runs = states_data.load(DIR, states=("minus-x",))
    table = rates_table(runs)
    table.to_csv(DIR / "small_gamma_rates.csv", index=False)
    print(table.pivot_table(index="gamma", columns="L", values="Gamma_e/gamma").round(2).to_string())
    print("relative statistical error of the trajectory Γ_e:")
    rel = table.assign(rel=table["sigma_Gamma_e"] / table["Gamma_e"])
    print(rel[rel["method"] != "exact"].pivot_table(index="gamma", columns="L", values="rel").round(3).to_string())

    fit = table[(table["gamma"] <= COLLAPSE_GAMMA_MAX) & (table["L"] >= COLLAPSE_L_MIN)]
    nus = np.linspace(0.2, 1.0, 81)
    costs = np.array([collapse_cost(fit, nu) for nu in nus])
    nu = nus[np.argmin(costs)]
    near = nus[costs <= 2 * costs.min()]
    print(f"collapse: ν = {nu:.2f} (cost within a factor 2 of the minimum for ν = {near.min():.2f} … {near.max():.2f});"
          f" cost at ν = 0.5: {collapse_cost(fit, 0.5):.4f}, at the minimum {costs.min():.4f}")
    pd.DataFrame({"nu": [nu], "nu_min": [near.min()], "nu_max": [near.max()]}).to_csv(
        DIR / "small_gamma_collapse.csv", index=False)
    large = large_l_fits(table)
    large.to_csv(DIR / "states_inverse_L_large.csv", index=False)
    print(f"A + B/L on L ≥ {LARGE_L_MIN} (A_check: L ≥ {LARGE_L_CHECK}):")
    print(large.assign(**{"A/gamma": large["A"] / large["gamma"]}).to_string(index=False, float_format="%.4g"))
    rates_figure(table, nu, large)
    large_l_figure(table, large)

    check, panels = checks(runs)
    check.to_csv(DIR / "small_gamma_checks.csv", index=False)
    print(check.to_string(index=False, float_format="%.3g"))
    crosscheck_figure(panels)
    print(f"wrote small_gamma_rates.csv, small_gamma_checks.csv, small_gamma_collapse.csv, states_inverse_L_large.csv, "
          f"states_inverse_L_large.png, small_gamma_rates.png, small_gamma_crosscheck.png "
          f"in {DIR}")


if __name__ == "__main__":
    main()
