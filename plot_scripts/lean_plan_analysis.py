"""
Step A of lean-data-plan.md: analysis of the existing data only (no new runs).

  A1  F(t) = −(1/γ) ln m⊥(t) of the scar at L = 16, 18, 20, γ ≤ 0.02 (the single-kick formula predicts
      F(t) = ∫₀ᵗ D̄). Its slope dF/dt, averaged over one precession period 2π/h, is the effective damage, compared
      with D̄(τ) of single_flip. The slope s = dD̄/dτ of the linear growth is fitted on D̄ of L = 24 (τ = 4 … 8), and
      on the period-averaged slope of F (period_slope) up to t = L/s, before the damage of a kick wraps around the
      ring (it reaches the far side at t ≈ L/s, s = 2 × the spreading velocity); the same is done on ∫D̄ of the same L.
  A2  single_flip L = 20 (16 phases × 20 sites) and L = 24 (16 phases × 4 sites): spread of D(τ) over phases and over
      sites, and the actual error of D̄ when only n_p equally spaced phases × n_s equally spaced sites are used
      (every offset of the subsample, worst case over τ ≤ 25 or 8).
  A3  Diffusive and jump runs at L = 16, 18, 20: the spread of one trajectory, σ_1 = σ_m(t_e) √n, at the 1/e time,
      and the number of trajectories for a 2 % Γ_e (σ_Γ/Γ_e = e σ_m).
  A4  Predicted t_e of the planned runs (B1: L = 22, γ = 0.005, 0.01, 0.02, 0.05; B1′: L = 24, γ = 0.01, 0.02):
      from Γ_e(L = 20) (an upper bound on t_e, as Γ_e grows with L) and from the single-kick formula with D̄ of
      L = 22, 24 (known to τ = 8; continued either flat at D̄(8), an upper bound on t_e, or linearly with slope s).

Input:  data/used_data (or the directory given as the first argument): run CSVs, single_flip CSVs, states_rates.csv
Output: lean_A1_damage.png, lean_A2_subsampling.csv, lean_A3_spread.csv, lean_A4_t_e.csv in the same directory;
        all numbers printed
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import states_data
from fit_small_gamma import m_perp, rate
from fit_states import INK_SECONDARY, MUTED

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
A1_SIZES, A1_GAMMA_MAX = (16, 18, 20), 0.02
M_MIN = 0.05            # A1: use m⊥ above this (and above 3 σ_m)
T_FIT_MIN = 4.0         # A1: start of the quadratic fit of F(t) and of the linear fit of D̄
PERIOD = 2 * np.pi      # precession period 2π/h, h = 1
TARGET = 0.02           # A2, A3: relative error aimed at
B1 = [(22, 0.005), (22, 0.01), (22, 0.02), (22, 0.05), (24, 0.01), (24, 0.02)]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"]
ORANGE = "#eb6834"


def flip_table(l):
    """D(m, k, τ) of single_flip at size l as a 3-d array [τ, m, k], with the τ grid, phases and sites."""
    files = sorted(DIR.glob(f"single_flip_L{l}*.csv"))
    d = pd.concat([pd.read_csv(f, usecols=["m", "k", "tau", "D"]) for f in files])
    taus, ms, ks = (np.sort(d[c].unique()) for c in ("tau", "m", "k"))
    cube = d.pivot_table(index="tau", columns=["m", "k"], values="D").to_numpy().reshape(len(taus), len(ms), len(ks))
    return taus, ms, ks, cube


def mean_damage(l):
    taus, _, _, cube = flip_table(l)
    return taus, cube.mean(axis=(1, 2))


def integral(taus, d):
    return np.concatenate([[0], np.cumsum(np.diff(taus) * (d[1:] + d[:-1]) / 2)])


def period_slope(t, f, t_wrap):
    """Slope s of the damage from F(t) = ∫D: D_eff(t) = [F(t + P/2) − F(t − P/2)] / P averages D over one precession
    period P (removing its oscillation); s is the slope of a straight-line fit of D_eff on [T_FIT_MIN + P/2,
    min(t_wrap, end of data) − P/2]. NaN if that window is shorter than 2 time units."""
    lo, hi = T_FIT_MIN + PERIOD / 2, min(t_wrap, t[-1]) - PERIOD / 2
    if hi - lo < 2:
        return np.nan, np.nan
    tc = np.linspace(lo, hi, 50)
    d_eff = (np.interp(tc + PERIOD / 2, t, f) - np.interp(tc - PERIOD / 2, t, f)) / PERIOD
    return np.polyfit(tc, d_eff, 1)[0], hi + PERIOD / 2


def linear_slope(taus, d, lo, hi):
    sel = (taus >= lo) & (taus <= hi)
    return np.polyfit(taus[sel], d[sel], 1)[0]


def a1(chosen):
    """F(t) = −ln m⊥/γ against ∫D̄, and the slope s."""
    t24, d24 = mean_damage(24)
    s_flip = linear_slope(t24, d24, T_FIT_MIN, t24[-1])
    print(f"A1: D̄ of L = 24, slope on τ = {T_FIT_MIN:g} … {t24[-1]:g}: s = {s_flip:.3f}  "
          f"(wrap-around time L/s = {16 / s_flip:.1f}, {18 / s_flip:.1f}, {20 / s_flip:.1f} for L = 16, 18, 20)")
    gammas = sorted({g for (st, l, g) in chosen if st == "minus-x" and l in A1_SIZES and 0 < g <= A1_GAMMA_MAX})
    cmap = LinearSegmentedColormap.from_list("blues", RAMP)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    rows = []
    for ax, l in zip(axes, A1_SIZES):
        taus, dl = mean_damage(l)
        t_wrap = l / s_flip
        for i, g in enumerate(gammas):
            if ("minus-x", l, g) not in chosen:
                continue
            method, run, _ = chosen["minus-x", l, g]
            t = run["t"].to_numpy()
            m, err = m_perp(run)
            ok = (m > M_MIN) & (m > 3 * err)
            ok &= np.cumprod(ok).astype(bool)  # up to the first time it fails
            f = -np.log(m[ok]) / g
            tt = t[ok]
            col = cmap(i / max(len(gammas) - 1, 1))
            ax.plot(tt, f, color=col, lw=1.0, label=f"γ = {g:g} ({method})")
            s_fit, t_hi = period_slope(tt, f, t_wrap)
            # F at fixed times, against the single-kick ∫D̄ of the same L
            for t0 in (4.0, 8.0, 12.0):
                if tt[-1] >= t0:
                    rows.append((l, g, method, t0, np.interp(t0, tt, f), np.interp(t0, taus, integral(taus, dl)),
                                 s_fit, period_slope(taus, integral(taus, dl), t_wrap)[0], t_hi))
        tg = np.linspace(0, min(taus[-1], 30), 300)
        ax.plot(tg, np.interp(tg, taus, integral(taus, dl)), color="k", lw=1.6, ls="--", label=f"∫D̄, single flip L = {l}")
        ax.plot(t24, integral(t24, d24), color=ORANGE, lw=1.6, ls=":", label="∫D̄, single flip L = 24 (τ ≤ 8)")
        ax.axvline(t_wrap, color=MUTED, lw=0.8, ls="-.")
        ax.set(xlim=(0, 30), ylim=(0, 350), xlabel="t", title=f"L = {l}: F(t) = −(1/γ) ln m⊥   (dash-dot: L/s)")
    axes[0].set_ylabel("F(t)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.text(0.01, -0.04, f"F(t) shown while m⊥ > {M_MIN:g} and > 3σ. Single-kick formula: F(t) = ∫₀ᵗ D̄(τ) dτ for every γ. "
             f"L/s with s = {s_flip:.2f} (slope of D̄ at L = 24, τ = {T_FIT_MIN:g} … 8): the damage reaches around the ring.",
             fontsize=7.5, color=INK_SECONDARY)
    fig.savefig(DIR / "lean_A1_damage.png")
    table = pd.DataFrame(rows, columns=["L", "gamma", "method", "t", "F", "int_D_flip", "s_lindblad", "s_flip_same_L",
                                        "t_fit_max"])
    table["ratio"] = table["F"] / table["int_D_flip"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format="%.3g"))
    return s_flip, table


def a2():
    """Error of D̄ from n_p equally spaced phases × n_s equally spaced sites."""
    rows = []
    for l in (20, 24):
        taus, ms, ks, cube = flip_table(l)
        full = cube.mean(axis=(1, 2))
        sd_phase = cube.mean(axis=2).std(axis=1, ddof=1)   # spread of the site-averaged D over phases
        sd_site = cube.mean(axis=1).std(axis=1, ddof=1)    # spread of the phase-averaged D over sites
        sd_all = cube.reshape(len(taus), -1).std(axis=1, ddof=1)
        print(f"A2: L = {l} ({len(ms)} phases × {len(ks)} sites, τ ≤ {taus[-1]:g}): max over τ of the relative spread "
              f"over phases {np.max(sd_phase / full):.3f}, over sites {np.max(sd_site / full):.3f}, "
              f"of single runs {np.max(sd_all / full):.3f}")
        for n_p in (1, 2, 4, 8, 16):
            for n_s in sorted({1, 2, 4, len(ks)} | ({5, 10} if len(ks) == 20 else set())):
                if n_p > len(ms) or n_s > len(ks) or len(ms) % n_p or len(ks) % n_s:
                    continue
                errs = []
                for op in range(len(ms) // n_p):
                    for os in range(len(ks) // n_s):
                        sub = cube[:, op::len(ms) // n_p, :][:, :, os::len(ks) // n_s]
                        errs.append(np.max(np.abs(sub.mean(axis=(1, 2)) / full - 1)))
                rows.append((l, n_p, n_s, n_p * n_s, np.median(errs), np.max(errs)))
    table = pd.DataFrame(rows, columns=["L", "phases", "sites", "runs", "median_max_rel_err", "worst_max_rel_err"])
    table.to_csv(DIR / "lean_A2_subsampling.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format="%.4f"))
    return table


def a3(chosen, rates):
    rows = []
    for (state, l, g), (method, run, _) in chosen.items():
        if state != "minus-x" or l not in A1_SIZES or g == 0 or method == "exact":
            continue
        gamma_e, sigma_g = rate(run)
        if not np.isfinite(gamma_e):
            continue
        t = run["t"].to_numpy()
        m, err = m_perp(run)
        n = run["trajectories"].iloc[0]
        s_m = np.interp(1 / gamma_e, t, err)
        s1 = s_m * np.sqrt(n)
        rows.append((l, g, method, n, gamma_e, sigma_g / gamma_e, s_m, s1, int(np.ceil((np.e * s1 / TARGET) ** 2))))
    table = pd.DataFrame(rows, columns=["L", "gamma", "method", "trajectories", "Gamma_e", "rel_err_Gamma",
                                        "sigma_m_at_te", "sigma_one_traj", "n_for_2pct"]).sort_values(["method", "L", "gamma"])
    table.to_csv(DIR / "lean_A3_spread.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("A3:")
        print(table.to_string(index=False, float_format="%.4g"))
    return table


def a4(rates, s_flip):
    rows = []
    r20 = rates[(rates["state"] == "minus-x") & (rates["L"] == 20)].set_index("gamma")["Gamma_e"]
    for l, g in B1:
        taus, dl = mean_damage(l)
        t = np.linspace(0, 60, 60001)
        out = {}
        for name, slope in (("flat", 0.0), ("linear", s_flip)):
            d = np.interp(t, taus, dl)
            d[t > taus[-1]] = dl[-1] + slope * (t[t > taus[-1]] - taus[-1])
            x = g * integral(t, d)
            out[name] = t[np.argmax(x >= 1)]
        g20 = r20.get(g, np.nan)
        t_up = 1 / g20 if np.isfinite(g20) else np.nan
        rows.append((l, g, t_up, out["flat"], out["linear"], 1.5 * np.nanmax([t_up, out["flat"]])))
    table = pd.DataFrame(rows, columns=["L", "gamma", "t_e_from_L20", "t_e_kick_flat", "t_e_kick_linear", "t_max_1.5"])
    table.to_csv(DIR / "lean_A4_t_e.csv", index=False)
    print("A4: predicted t_e (L = 20 measured: upper bound; single kick with D̄ of the same L continued flat or linearly)")
    print(table.to_string(index=False, float_format="%.4g"))
    return table


def main():
    chosen = states_data.select(states_data.load(DIR, states=("minus-x",)))
    rates = pd.read_csv(DIR / "states_rates.csv")
    rates = rates[rates["selected"]]
    s_flip, _ = a1(chosen)
    a2()
    a3(chosen, rates)
    a4(rates, s_flip)
    print(f"wrote lean_A1_damage.png, lean_A2_subsampling.csv, lean_A3_spread.csv, lean_A4_t_e.csv in {DIR}")


if __name__ == "__main__":
    main()
