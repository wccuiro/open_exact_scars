"""
Single-kick prediction of the scar-class rates under depolarizing noise (step 5 of the rate histograms).

    ~/anaconda3/bin/python plot_scripts/state_hist_kick.py [data/state_hist]

Reads single_flip_L16_cos<c>_kickxyz_sites0-4-8-12.csv (`scripts/run_state_hist.py kick`): D_αα(τ) of one σ^α kick on
the coherent state at θ, averaged over the phases and kicked sites, D̄(τ) = D̄_xx + D̄_yy + D̄_zz. Prediction:
a(t)/a(0) ≈ exp[−(γ/3) ∫₀ᵗ D̄ dτ], so Γ_e^pred = 1/t_e with (γ/3) ∫₀^{t_e} D̄ = 1 (D̄ continued past τ = 16 by its mean
over the last 2 units). Compared with the measured scar Γ_e (summary, larger of the two sizes) at every γ.
Writes state_hist_kick.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_hist_data as shd  # noqa: E402

NAME = re.compile(r"single_flip_L(\d+)_cos(-?[0-9.]+)_kickxyz_sites[0-9-]+\.csv$")


def damage(path):
    df = pd.read_csv(path, usecols=["m", "k", "kick", "tau", "D"])
    per = df.groupby(["kick", "tau"]).D.mean().unstack(0)
    return per.index.to_numpy(), per


def predicted_rate(tau, dsum, gamma):
    dt = tau[1] - tau[0]
    tail = dsum[tau >= tau[-1] - 2].mean()
    integral = np.concatenate([[0], np.cumsum((dsum[1:] + dsum[:-1]) / 2 * dt)])
    target = 3.0 / gamma
    if integral[-1] >= target:
        k = np.searchsorted(integral, target)
        te = tau[k - 1] + (target - integral[k - 1]) / (integral[k] - integral[k - 1]) * dt
    else:
        te = tau[-1] + (target - integral[-1]) / tail
    return 1.0 / te


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else shd.DATA
    summary = shd.summarize(data)
    rows = []
    for path in sorted(data.glob("single_flip_L*_kickxyz_*.csv")):
        m = NAME.match(path.name)
        if not m:
            continue
        c = float(m.group(2))
        tau, per = damage(path)
        dsum = per[["x", "y", "z"]].sum(axis=1).to_numpy()
        print(f"cos θ = {c:+.1f}: D̄(0) = {dsum[0]:.3f} (xx {per['x'].iloc[0]:.3f}, yy {per['y'].iloc[0]:.3f}, "
              f"zz {per['z'].iloc[0]:.3f}), D̄(τ = 4, 8, 16) = "
              + ", ".join(f"{np.interp(t, tau, dsum):.2f}" for t in (4, 8, 16)))
        for g in sorted(summary.gamma.unique()):
            if g == 0:
                continue
            meas = summary[(summary.family == "coherent") & (summary.gamma == g) & (summary.cos_theta == c)]
            if not len(meas):
                continue
            r = meas.sort_values("L").iloc[-1]
            pred = predicted_rate(tau, dsum, g)
            rows.append(dict(cos_theta=c, gamma=g, L_measured=int(r.L), Gamma_measured=r.Gamma_e,
                             sigma_measured=r.sigma_Gamma, Gamma_predicted_L16=pred, ratio=r.Gamma_e / pred))
    out = pd.DataFrame(rows)
    out.to_csv(data / "state_hist_kick.csv", index=False)
    if len(out):
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
