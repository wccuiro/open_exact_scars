"""
Thermodynamic decay rates under depolarizing noise from the Floquet circuit: the depolarizing panel of
channels_thermodynamic_rate.png (fit_channels.py, notes Fig. `fig:chinf`) with F^N in place of e^{𝓛t}.

    python plot_scripts/floquet_dep_rates.py [data_dir]          default: data/ of the repo

Runs (<data>/floquet_dep/; Leonardo jobs from data/bashes/floquet/make_floquet_dep_jobs.sh): channel dep (jumps
√(γ/2) σ^α, every Pauli factor decays at 2γ, as the exact dep runs), γ = 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10,
20, scar (m_⊥), x-Néel (s_⊥), z-Néel (|s_z|), random c seed 1 (the realizations of the exact runs). Sizes divisible
by 4 only (the same circuit, d = 4, at every L):
  floquet       ρ, L = 8, 12 (every γ), 16 (γ ≤ 2)        floquet_<state>_L<L>_g<γ>_dt<dt>_chdep.csv
  floquet_traj  L = 16, 20, 24 (γ ≤ 0.1)                  floquet_traj_<state>_L<L>_g<γ>_dt<dt>_chdep.csv
--dt 0.1 (scar), 0.02 (Néel states); Trotter step dt · min(1, 0.5/γ). Only the runs at these dt are used (the dt/2
runs are for a Trotter extrapolation), and ρ before trajectories at the same L. Runs stop at 2 t₁ (t₁ = start of
the latest stretch below 1/e); Néel ρ at L ≤ 12, γ ≤ 0.2 to t = 2.5.

Γ_e = 1/t_e from the envelope at 1/e (fit_states.py), on the run's own output grid (the Trotter step); trajectory
error σ_Γ = Γ² σ_S(t_e)/|dE/dt| (slope over ± 4 output steps, as fit_small_gamma.rate), σ_S with the covariance of
the means (m_⊥, s_⊥). Γ_∞ = A of Γ_e = A + B/L on L ≥ 8, unweighted, z-Néel without L = 8 at γ ≤ 0.02
(fit_states_inverse_L.sizes_used: finite-size revivals); error = max(fit error, |A − A without the smallest size|), as
fit_channels.py for the channels other than σᶻ.

Exact reference: the dep runs of fit_channels.py (<data>/used_data/lean/: exact L = 8, 10, 12, diffusive trajectories
L = 16, 20 for γ ≤ 2) and its Γ_∞ (channels_inverse_L.csv, L = 8 – 20 including L = 10). Their output grid is coarse
(0.1 up to γ ≈ 0.5, while t_e ≈ 0.2 for the Néel states), which biases the linear interpolation of the 1/e crossing.
For the comparison (panels c, d) the Floquet signal is therefore also evaluated at the output times of the exact run
of the same (state, L, γ) (linear interpolation of the fine Floquet grid; columns *_grid), and the exact Γ_e are
refitted on the same sizes (Gamma_inf_exact_same): the differences are then the Trotter error alone.

Output in <data>/floquet_dep/: floquet_dep_rates.csv (state, L, gamma, method, Gamma_e, sigma_Gamma_e, Gamma_e_grid,
Gamma_e_exact, t_end), floquet_dep_inverse_L.csv (state, gamma, sizes, A_lin, sigma_A_lin, B_lin, Gamma_inf,
sigma_Gamma_inf, Gamma_inf_grid, Gamma_inf_exact_same, Gamma_inf_exact, sigma_Gamma_inf_exact), floquet_dep_rates.png.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import fit_states_inverse_L as fsi  # also sets the plot style of the notes
from fit_states import crossing
from states_data import SIGNALS, read

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
RUNS, USED = DATA / "floquet_dep", DATA / "used_data"
NAME = re.compile(r"floquet_(traj_)?(minus-x|neel-x|neel)_L(\d+)_g([0-9.]+?)_dt([0-9.]+)_chdep\.csv$")
LIN_MIN, MIN_SIZES = 8, 3
SHORT = {"minus-x": "scar, $m_\\perp$", "neel-x": "x-Néel, $s_\\perp$", "neel": "z-Néel, $|s_z|$"}
L_MARKERS = {8: "o", 12: "s", 16: "^", 20: "D", 24: "v"}
DT = {"minus-x": 0.1, "neel-x": 0.02, "neel": 0.02}  # main Trotter step (--dt) per state


def signal_error(state, run):
    if state == "neel":
        return run["sz_err"].to_numpy()
    a, b, cov = ("mx", "my", "mxmy_cov") if state == "minus-x" else ("sx", "sy", "sxsy_cov")
    x, y = run[a].to_numpy(), run[b].to_numpy()
    var = x ** 2 * run[a + "_err"] ** 2 + y ** 2 * run[b + "_err"] ** 2 + 2 * x * y * run[cov]
    return np.sqrt(np.maximum(var.to_numpy(), 0)) / np.maximum(np.hypot(x, y), 1e-300)


def t_e_of(t, s):
    return crossing(t, np.maximum.accumulate(s[::-1])[::-1], np.exp(-1))


def rate(state, run, traj):
    """Γ_e and its statistical error (0 for ρ), on the run's own grid."""
    t = run["t"].to_numpy()
    s = SIGNALS[state](run).to_numpy()
    t_e = t_e_of(t, s)
    if not np.isfinite(t_e):
        return np.nan, np.nan
    if not traj:
        return 1 / t_e, 0.0
    e = np.maximum.accumulate(s[::-1])[::-1]
    i = max(np.searchsorted(t, t_e), 1)
    lo, hi = max(i - 4, 0), min(i + 4, len(t) - 1)
    slope = abs((e[hi] - e[lo]) / (t[hi] - t[lo]))
    err = np.interp(t_e, t, signal_error(state, run))
    return 1 / t_e, err / slope / t_e ** 2 if slope > 0 else np.nan


def rate_on_grid(state, run, grid):
    """Γ_e of the run sampled at the times `grid` (linear interpolation, times inside the run only)."""
    t = run["t"].to_numpy()
    grid = grid[grid <= t[-1] + 1e-9]
    return 1 / t_e_of(grid, np.interp(grid, t, SIGNALS[state](run).to_numpy()))


def exact_run(state, l, g):
    """The exact dep run fit_channels.py uses for (state, L, γ): exact ρ if it exists, else trajectories."""
    for prefix in ("states", "states_traj"):
        f = USED / "lean" / f"{prefix}_{state}_L{l}_g{g}_chdep.csv"
        if f.exists():
            run, _ = read(f)
            if len(run) >= 2:
                return run
    return None


def collect():
    rows = []
    for f in sorted(RUNS.glob("floquet_*_chdep.csv")):
        m = NAME.match(f.name)
        if m is None:
            continue
        run, _ = read(f)
        if len(run) < 2:
            continue
        state, l, g, traj = m.group(2), int(m.group(3)), float(m.group(4)), bool(m.group(1))
        gamma_e, sigma = rate(state, run, traj)
        ref = exact_run(state, l, m.group(4))
        if ref is None:
            grid_rate = exact = np.nan
        else:
            grid_rate = rate_on_grid(state, run, ref["t"].to_numpy())
            exact = 1 / t_e_of(ref["t"].to_numpy(), SIGNALS[state](ref).to_numpy())
        rows.append((state, l, g, float(m.group(5)), "traj" if traj else "rho", gamma_e, sigma, grid_rate, exact,
                     run["t"].iloc[-1]))
    d = pd.DataFrame(rows, columns=["state", "L", "gamma", "dt", "method", "Gamma_e", "sigma_Gamma_e", "Gamma_e_grid",
                                    "Gamma_e_exact", "t_end"])
    # the main Trotter step only (the dt/2 runs are for a Trotter extrapolation), ρ before trajectories at one L
    d = d[d["dt"] == d["state"].map(DT)]
    d = d.sort_values("method").drop_duplicates(["state", "L", "gamma"])  # "rho" < "traj"
    return d.drop(columns="dt").sort_values(["state", "gamma", "L"])


def linear(state, g, d):
    """(sizes, A, σ_A, B, error) of A + B/L on the sizes of fit_states_inverse_L.sizes_used (None if too few)."""
    d = d.dropna()
    lin = fsi.sizes_used(state, g, sorted(d.index), LIN_MIN)
    if len(lin) < MIN_SIZES:
        return None
    p1, s1 = fsi.fit(1 / np.array(lin), d[lin].to_numpy(), 1)
    upper = lin[1:]
    spread = abs(p1[-1] - np.polyfit(1 / np.array(upper), d[upper].to_numpy(), 1)[-1])
    return lin, p1[-1], s1, p1[0], np.nanmax([s1, spread])


def extrapolate(rates):
    exact = pd.read_csv(USED / "channels_inverse_L.csv")
    exact = exact[exact["channel"] == "dep"].set_index(["state", "gamma"])
    rows = []
    for (state, g), d in rates.groupby(["state", "gamma"]):
        d = d.set_index("L")
        own = linear(state, g, d["Gamma_e"])
        if own is None:
            continue
        lin, a, s_a, b, err = own
        # the comparison: Floquet on the exact grid and exact, both on the sizes where both exist
        both = d[["Gamma_e_grid", "Gamma_e_exact"]].dropna()
        grid, same = linear(state, g, both["Gamma_e_grid"]), linear(state, g, both["Gamma_e_exact"])
        ex = exact.loc[(state, g)] if (state, g) in exact.index else None
        rows.append((state, g, " ".join(map(str, lin)), a, s_a, b, a, err,
                     np.nan if grid is None else grid[1], np.nan if same is None else same[1],
                     np.nan if ex is None else ex["Gamma_inf"], np.nan if ex is None else ex["sigma_Gamma_inf"]))
    return pd.DataFrame(rows, columns=["state", "gamma", "sizes", "A_lin", "sigma_A_lin", "B_lin", "Gamma_inf",
                                       "sigma_Gamma_inf", "Gamma_inf_grid", "Gamma_inf_exact_same", "Gamma_inf_exact",
                                       "sigma_Gamma_inf_exact"])


def figure(rates, table):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    gs = np.geomspace(table["gamma"].min(), table["gamma"].max(), 100)

    # (a) Γ_∞ against γ: Floquet (filled) and exact (open, dashed; Fig. chinf, depolarizing panel)
    ax_a.plot(gs, 2 * gs, color=fsi.MUTED, ls=":", lw=1, label="$2\\gamma$")
    for state in fsi.STATES:
        d = table[table["state"] == state].sort_values("gamma")
        ax_a.errorbar(d["gamma"], d["Gamma_inf_exact"], yerr=d["sigma_Gamma_inf_exact"], fmt="--o", ms=6.5, lw=0.9,
                      capsize=2, mfc="none", mew=1.0, color=fsi.SERIES[state], alpha=0.7)
        ax_a.errorbar(d["gamma"], d["Gamma_inf"], yerr=d["sigma_Gamma_inf"], fmt="-o", ms=3.2, lw=1.3, capsize=2,
                      color=fsi.SERIES[state], label=SHORT[state])
    ax_a.plot([], [], "-o", ms=3.2, color=fsi.INK_SECONDARY, label="Floquet, $L$ = 8 – 20")
    ax_a.plot([], [], "--o", ms=6.5, mfc="none", color=fsi.INK_SECONDARY, label="exact (Fig. 19)")
    ax_a.set(xscale="log", yscale="log", xlabel="$\\gamma$", ylabel="$\\Gamma_\\infty$",
             title="(a) Depolarizing noise: $\\Gamma_\\infty$ ($A + B/L$)")
    ax_a.legend(fontsize=7.5, loc="upper left")

    # (b) scar Γ_e against 1/L with the fits; exact per size (crosses) and its Γ_∞ (cross at 1/L = 0)
    gammas = sorted(table["gamma"].unique())
    blues = LinearSegmentedColormap.from_list("blues", fsi.RAMP)
    colors = {g: blues(i / max(len(gammas) - 1, 1)) for i, g in enumerate(gammas)}
    xs = np.linspace(0, 1 / 8, 50)
    for g in gammas:
        d = rates[(rates["state"] == "minus-x") & (rates["gamma"] == g)].sort_values("L")
        t = table[(table["state"] == "minus-x") & (table["gamma"] == g)]
        if d.empty or t.empty:
            continue
        t = t.iloc[0]
        ax_b.errorbar(1 / d["L"], d["Gamma_e"], yerr=d["sigma_Gamma_e"], fmt="o", ms=3.5, capsize=1.5,
                      color=colors[g], label=f"{g:g}")
        ax_b.plot(1 / d["L"], d["Gamma_e_exact"], "x", ms=5, mew=0.9, color=colors[g])
        ax_b.plot(xs, t["A_lin"] + t["B_lin"] * xs, "--", lw=0.9, color=colors[g])
        ax_b.plot(0, t["Gamma_inf_exact"], "x", ms=6, mew=1.2, color=colors[g], clip_on=False)
    ax_b.set(yscale="log", xlim=(-0.004, 0.13), xlabel="$1/L$", ylabel="$\\Gamma_e$",
             title="(b) Scar: $\\Gamma_e$ against $1/L$ (×: exact)")
    ax_b.set_xticks([0, 1 / 20, 1 / 16, 1 / 12, 1 / 8], ["0", "1/20", "1/16", "1/12", "1/8"])
    ax_b.legend(title="$\\gamma$", fontsize=6.5, title_fontsize=7, ncol=2, loc="lower left")

    # (c) the same size on the same output grid: Γ_e(Floquet)/Γ_e(exact) − 1, the Trotter error
    ax_c.axhline(0, color=fsi.MUTED, lw=0.8)
    for state in fsi.STATES:
        for l, marker in L_MARKERS.items():
            d = rates[(rates["state"] == state) & (rates["L"] == l)].dropna(subset=["Gamma_e_exact"])
            d = d.sort_values("gamma")
            if d.empty:
                continue
            ax_c.errorbar(d["gamma"], 100 * (d["Gamma_e_grid"] / d["Gamma_e_exact"] - 1),
                          yerr=100 * d["sigma_Gamma_e"] / d["Gamma_e_exact"], fmt=marker, ls="-", lw=0.6, ms=3.5,
                          capsize=1.5, color=fsi.SERIES[state], mfc=fsi.SERIES[state] if l <= 12 else "none")
    handles = [plt.Line2D([], [], color=fsi.SERIES[s], lw=1.5, label=SHORT[s]) for s in fsi.STATES]
    handles += [plt.Line2D([], [], color=fsi.INK_SECONDARY, marker=m, ls="none", ms=4,
                           mfc=fsi.INK_SECONDARY if l <= 12 else "none",
                           label=f"$L$ = {l}" + (" (ρ)" if l <= 12 else " (traj.)")) for l, m in L_MARKERS.items()]
    ax_c.legend(handles=handles, fontsize=6.8, ncol=2)
    ax_c.set(xscale="log", xlabel="$\\gamma$", ylabel="$\\Gamma_e^{\\rm Floquet}/\\Gamma_e^{\\rm exact} - 1$ (%)",
             title="(c) Each size, same output grid as exact")

    # (d) extrapolated rates: same grid and same sizes (filled) / as plotted in (a) (open)
    ax_d.axhline(0, color=fsi.MUTED, lw=0.8)
    for k, state in enumerate(fsi.STATES):
        d = table[table["state"] == state].sort_values("gamma")
        shift = 1.07 ** (k - 1)  # small horizontal offsets so the error bars do not overlap
        ax_d.errorbar(d["gamma"] * shift, 100 * (d["Gamma_inf_grid"] / d["Gamma_inf_exact_same"] - 1),
                      fmt="-o", ms=3.5, color=fsi.SERIES[state], label=SHORT[state])
        err = np.hypot(d["sigma_Gamma_inf"], d["sigma_Gamma_inf_exact"]) / d["Gamma_inf_exact"]
        ax_d.errorbar(d["gamma"] * shift, 100 * (d["Gamma_inf"] / d["Gamma_inf_exact"] - 1), yerr=100 * err,
                      fmt="o", ms=5, mfc="none", mew=0.9, lw=0.7, capsize=1.5, color=fsi.SERIES[state], alpha=0.7)
    ax_d.plot([], [], "-o", ms=3.5, color=fsi.INK_SECONDARY, label="same grid, same sizes")
    ax_d.plot([], [], "o", ms=5, mfc="none", color=fsi.INK_SECONDARY, label="as in (a), with errors")
    ax_d.set(xscale="log", xlabel="$\\gamma$",
             ylabel="$\\Gamma_\\infty^{\\rm Floquet}/\\Gamma_\\infty^{\\rm exact} - 1$ (%)",
             title="(d) Extrapolated rates, Floquet against exact")
    ax_d.legend(fontsize=7)
    fig.tight_layout()
    out = RUNS / "floquet_dep_rates.png"
    fig.savefig(out)
    print(f"-> {out}")


def main():
    rates = collect()
    table = extrapolate(rates)
    rates.to_csv(RUNS / "floquet_dep_rates.csv", index=False, float_format="%.6g")
    table.to_csv(RUNS / "floquet_dep_inverse_L.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(rates.assign(err_grid=rates["Gamma_e_grid"] / rates["Gamma_e_exact"] - 1).to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
        print(table.assign(rel=table["Gamma_inf"] / table["Gamma_inf_exact"] - 1,
                           rel_same=table["Gamma_inf_grid"] / table["Gamma_inf_exact_same"] - 1).to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
    if not table.empty:
        figure(rates, table)


if __name__ == "__main__":
    main()
