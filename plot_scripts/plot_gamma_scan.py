"""
Decay rate against γ at fixed L, for the scar and the non-scar probes.

  P1 scar:    m_⊥, fitted with e^(−Γt) on the window γt ∈ GT_WINDOW
  P2 energy:  C_H̃, same window
  P3 thermal: ΔΓ = Γ(γ) − Γ₀, slope of −ln(|C⁺_γ| / |C⁺_0|) on t ∈ [0, P3_T]

The P1/P2 window is fixed in γt, not in t: a window fixed in t (the plan's [5, until 1e-3]) covers a
different part of the multi-exponential decay at each γ (at γ = 0.1 the fast part is over by t = 5).
GT_WINDOW = [0.05, 0.6] is the window [5, 60] at γ = 0.01. The rates on the plan's window are also
written to the CSV (probe P1_plan, P2_plan). P3 decays thermally by t ≈ 1, so its window is fixed in t.

Each set of rates is fitted with Γ = a γ^p (p = 1: rate linear in γ) for γ ≤ FIT_GAMMA_MAX, the
weak-noise range; larger γ is shown but not fitted.

Input:  data/probe_{P1,P2,P3}_<tag>.csv for every tag in TAGS that exists; each tag's rates use its own
        γ = 0 run. Strong γ needs a fine time step for the window γt ∈ GT_WINDOW, hence two runs:
        (cd rust && cargo run --release --bin probes -- --lmin 6 --lmax 6 --dt 0.05
                    --gammas 0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0 --tag gscan
                 && cargo run --release --bin probes -- --lmin 6 --lmax 6 --dt 0.002 --gammas 3,6 --tag gscan_strong)
Output: data/gamma_scan.png (all probes), data/gamma_scan_scar.png (scar only), data/rates_gamma_scan.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_probes import AQUA, BLUE, DATA, MUTED, ORANGE, SURFACE, line_fit, window_p12

TAGS = ("gscan", "gscan_strong")
GT_WINDOW = (0.05, 0.6)
P3_T = 0.5
FIT_GAMMA_MAX = 0.05


def main():
    rows = []
    for tag in TAGS:
        if (DATA / f"probe_P1_{tag}.csv").exists():
            rows += tag_rates(tag)
    rates = pd.DataFrame(rows, columns=["probe", "L", "gamma", "Gamma", "sigma_Gamma", "t_fit_min", "t_fit_max"])
    rates = rates.sort_values(["probe", "gamma"]).reset_index(drop=True)
    rates.to_csv(DATA / "rates_gamma_scan.csv", index=False)
    (l,) = rates["L"].unique()
    gammas = sorted(rates["gamma"].unique())

    window = f"γt ∈ [{GT_WINDOW[0]:g}, {GT_WINDOW[1]:g}]"
    series = [("P1", f"scar m⊥ (P1), {window}", BLUE, "o"),
              ("P2", f"energy H̃ (P2), {window}", ORANGE, "s"),
              ("P3_delta", f"thermal k = π (P3), ΔΓ on t ∈ [0, {P3_T:g}]", AQUA, "D")]
    print(f"L = {l}: Gamma/gamma per gamma {gammas}, and the fit Gamma = a gamma^p")
    figure(rates, series, l, gammas, "gamma_scan.png")
    figure(rates, series[:1], l, gammas, "gamma_scan_scar.png")
    for probe in ("P1_plan", "P2_plan"):
        d = rates[rates["probe"] == probe]
        print(f"  {probe:9s} {np.round(d['Gamma'].to_numpy() / d['gamma'].to_numpy(), 2)}   (plan window, t from 5 until 1e-3)")


def tag_rates(tag):
    """Rates of every probe and γ > 0 of one run."""
    p1, p2, p3 = (pd.read_csv(DATA / f"probe_{p}_{tag}.csv") for p in ("P1", "P2", "P3"))
    (l,) = p1["L"].unique()
    gammas = sorted(g for g in p1["gamma"].unique() if g > 0)
    rows = []
    for probe, df, col in (("P1", p1, "m_perp"), ("P2", p2, "C")):
        for g in gammas:
            run = df[df["gamma"] == g]
            t, s = run["t"].to_numpy(), run[col].to_numpy()
            t0, t1 = GT_WINDOW[0] / g, GT_WINDOW[1] / g
            sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
            slope, err, _ = line_fit(t[sel], np.log(s[sel]))
            rows.append((probe, l, g, -slope, err, t0, t1))
            t0, t1 = window_p12(t, s)
            sel = (t >= t0) & (t <= t1)
            if sel.sum() > 2:  # at strong γ the signal is below 1e-3 before t = 5
                slope, err, _ = line_fit(t[sel], np.log(s[sel]))
                rows.append((f"{probe}_plan", l, g, -slope, err, t0, t1))
    r0 = p3[(p3["gamma"] == 0.0) & (p3["t"] <= P3_T + 1e-9)]
    for g in gammas:
        rg = p3[(p3["gamma"] == g) & (p3["t"] <= P3_T + 1e-9)]  # runs differ in length, same time grid
        assert np.allclose(r0["t"].to_numpy(), rg["t"].to_numpy())
        slope, err, _ = line_fit(r0["t"].to_numpy(), np.log(rg["C_abs"].to_numpy() / r0["C_abs"].to_numpy()))
        rows.append(("P3_delta", l, g, -slope, err, 0.0, P3_T))
    return rows


def figure(rates, series, l, gammas, name):
    """Γ against γ (log-log, with the fit Γ = a γ^p) and Γ/γ against γ, for the given probes."""
    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(10, 4.0))
    fine = np.geomspace(min(gammas) / 1.3, max(gammas) * 1.3, 100)
    fit_range = np.geomspace(min(gammas) / 1.3, FIT_GAMMA_MAX * 1.3, 100)
    ax.plot(fine, 2 * fine, color=MUTED, ls=":", lw=1.0, label="2γ: free spin")
    ax_r.axhline(2, color=MUTED, ls=":", lw=1.0, label="2: free spin")
    for probe, label, color, marker in series:
        d = rates[rates["probe"] == probe]
        g, rate = d["gamma"].to_numpy(), d["Gamma"].to_numpy()
        weak = g <= FIT_GAMMA_MAX + 1e-12
        (p, ln_a), cov = np.polyfit(np.log(g[weak]), np.log(rate[weak]), 1, cov=True)
        ax.plot(fit_range, np.exp(ln_a) * fit_range**p, color=color, lw=1.1)
        ax.plot(g, rate, ls="none", marker=marker, ms=7, color=color,
                label=f"{label}:  Γ ∝ γ^({p:.3f} ± {np.sqrt(cov[0, 0]):.3f}) for γ ≤ {FIT_GAMMA_MAX:g}")
        ax_r.plot(g, rate / g, color=color, marker=marker, ms=7, lw=1.2, label=label)
        print(f"  {name}: {probe:9s} {np.round(rate / g, 2)}   p = {p:.3f} ± {np.sqrt(cov[0, 0]):.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("γ")
    ax.set_ylabel("decay rate Γ")
    ax.set_title(f"Decay rate against γ, L = {l}")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1)
    ax_r.set_xscale("log")
    ax_r.set_ylim(0, None)
    ax_r.set_xlabel("γ")
    ax_r.set_ylabel("Γ / γ")
    ax_r.set_title("Same rates divided by γ (flat = linear in γ)")
    ax_r.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DATA / name)
    print(f"wrote {DATA / name}")


if __name__ == "__main__":
    main()
