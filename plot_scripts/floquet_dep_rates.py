"""
Thermodynamic decay rates under depolarizing noise from the Floquet circuit: the depolarizing panel of
channels_thermodynamic_rate.png (fit_channels.py, notes Fig. `fig:chinf`) with F^N in place of e^{𝓛t}.

    python plot_scripts/floquet_dep_rates.py [data_dir [used_dir]]

data_dir (default: data/ of the repo) holds floquet_dep/; used_dir (default: <data_dir>/used_data, or, where that
does not exist, the used_data/ of the WSL repo, WSL_USED) holds the exact references. The Windows checkout has no
used_data/: the exact runs stay in the WSL repo.

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
fit_channels.py for the channels other than σᶻ. γ_dep ≥ 5 has only L = 8, 12 (no ρ at L = 16 there; the two agree to
≤ 0.15 %): Γ_∞ = Γ_e(L = 12), error |Γ_e(12) − Γ_e(8)|.

γ convention: the runs use --channel dep (jumps √(γ_dep/2) σ^α, every Pauli at 2γ_dep), and so do the exact runs and
channels_inverse_L.csv. The outputs quote γ = γ_dep/2 (jumps √γ σ^α, dissipator γ Σ_j Σ_α (σ^α ρ σ^α − ρ), every
Pauli at 4γ; CLAUDE.md "Depolarizing γ"): column gamma, with gamma_dep alongside; the figure's axis is γ, the free-spin
line 4γ. The rates Γ are unchanged (time units).

Exact reference: the dep runs of fit_channels.py (<data>/used_data/lean/: exact L = 8, 10, 12, diffusive trajectories
L = 16, 20 for γ ≤ 2) and its Γ_∞ (channels_inverse_L.csv, L = 8 – 20 including L = 10). Their output grid is coarse
(0.1 up to γ ≈ 0.5, while t_e ≈ 0.2 for the Néel states), which biases the linear interpolation of the 1/e crossing.
For the comparison (panels c, d) the Floquet signal is therefore also evaluated at the output times of the exact run
of the same (state, L, γ) (linear interpolation of the fine Floquet grid; columns *_grid), and the exact Γ_e are
refitted on the same sizes (Gamma_inf_exact_same): the differences are then the Trotter error alone.

Output in <data>/floquet_dep/: floquet_dep_rates.csv (state, L, gamma, gamma_dep, method, Gamma_e, sigma_Gamma_e,
Gamma_e_grid, Gamma_e_exact, t_end), floquet_dep_inverse_L.csv (state, gamma, gamma_dep, sizes, A_lin, sigma_A_lin,
B_lin, Gamma_inf, sigma_Gamma_inf, Gamma_inf_grid, Gamma_inf_exact_same, Gamma_inf_exact, sigma_Gamma_inf_exact),
floquet_dep_thermodynamic_rate.png (one panel: Γ_∞ against γ, Floquet filled, exact open, as the depolarizing panel of
Fig. 19). The same-grid comparison columns are printed and written, not plotted.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_states_inverse_L as fsi  # also sets the plot style of the notes
from fit_states import crossing
from states_data import SIGNALS, read

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
WSL_USED = Path(r"\\wsl.localhost\Ubuntu\home\wccuirom\repos\open_exact_scars\data\used_data")
RUNS = DATA / "floquet_dep"
if len(sys.argv) > 2:
    USED = Path(sys.argv[2])
else:
    USED = DATA / "used_data" if (DATA / "used_data").is_dir() or not WSL_USED.is_dir() else WSL_USED
NAME = re.compile(r"floquet_(traj_)?(minus-x|neel-x|neel)_L(\d+)_g([0-9.]+?)_dt([0-9.]+)_chdep\.csv$")
LIN_MIN, MIN_SIZES = 8, 3
DT ={"minus-x": 0.1, "neel-x": 0.02, "neel": 0.02}  # main Trotter step (--dt) per state


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
            # two sizes only (γ_dep ≥ 5: L = 8, 12, which agree to ≤ 0.15 %): Γ_e of the larger, error the difference
            two = sorted(d["Gamma_e"].dropna().index)
            if len(two) != 2:
                continue
            ge = d.loc[two, "Gamma_e"].to_numpy()
            own = two, ge[1], np.nan, np.nan, abs(ge[1] - ge[0])
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


def figure(table):
    """The depolarizing panel of Fig. chinf with the Floquet circuit: Γ_∞ against γ (convention √γ σ^α), one panel."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    gs = np.geomspace(table["gamma"].min(), table["gamma"].max(), 100)
    ax.plot(gs, 4 * gs, color=fsi.MUTED, ls=":", lw=1, label="$4\\gamma$ (free spin)")
    for state in fsi.STATES:
        d = table[table["state"] == state].sort_values("gamma")
        ax.errorbar(d["gamma"], d["Gamma_inf_exact"], yerr=d["sigma_Gamma_inf_exact"], fmt="--o", ms=6.5, lw=0.9,
                    capsize=2, mfc="none", mew=1.0, color=fsi.SERIES[state], alpha=0.6)
        ax.errorbar(d["gamma"], d["Gamma_inf"], yerr=d["sigma_Gamma_inf"], fmt="-o", ms=3.5, lw=1.3, capsize=2,
                    color=fsi.SERIES[state], label=fsi.NAMES[state])
    ax.plot([], [], "-o", ms=3.5, color=fsi.INK_SECONDARY, label="Floquet circuit")
    ax.plot([], [], "--o", ms=6.5, mfc="none", color=fsi.INK_SECONDARY, alpha=0.6, label="exact Lindblad (Fig. 19)")
    ax.set(xscale="log", yscale="log", xlabel="$\\gamma$   (jumps $\\sqrt{\\gamma}\\,\\sigma^\\alpha_j$)",
           ylabel="$\\Gamma_\\infty$")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Depolarizing noise, Floquet circuit: thermodynamic decay rate $\\Gamma_\\infty$ ($A + B/L$, "
                 "$1/L \\to 0$)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    out = RUNS / "floquet_dep_thermodynamic_rate.png"
    fig.savefig(out)
    print(f"-> {out}")


def to_convention(d):
    """γ of the files and of channels_inverse_L.csv (--channel dep, jumps √(γ_dep/2) σ^α) → γ = γ_dep/2 (jumps √γ σ^α,
    CLAUDE.md "Depolarizing γ"); both columns are written."""
    d = d.rename(columns={"gamma": "gamma_dep"})
    d.insert(d.columns.get_loc("gamma_dep"), "gamma", d["gamma_dep"] / 2)
    return d


def main():
    rates = collect()
    table = extrapolate(rates)
    rates, table = to_convention(rates), to_convention(table)
    rates.to_csv(RUNS / "floquet_dep_rates.csv", index=False, float_format="%.6g")
    table.to_csv(RUNS / "floquet_dep_inverse_L.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(rates.assign(err_grid=rates["Gamma_e_grid"] / rates["Gamma_e_exact"] - 1).to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
        print(table.assign(rel=table["Gamma_inf"] / table["Gamma_inf_exact"] - 1,
                           rel_same=table["Gamma_inf_grid"] / table["Gamma_inf_exact_same"] - 1).to_string(
            index=False, float_format=lambda x: f"{x:.4g}"))
    if not table.empty:
        figure(table)


if __name__ == "__main__":
    main()
