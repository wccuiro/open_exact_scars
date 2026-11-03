"""
Spreading of the damage of one kick, and what it implies for the scar rate as γ → 0 with L → ∞ first.

(a) Damage profile d_r(τ) = 1 − Re c_j at distance r = |j − k| (ring) from the kicked site k, mean over phases and
    kicked sites (single_flip L = 24, lean run: 4 phases × sites 0, 6, 12, 18, τ ≤ 16). The body of the profile
    (d_r = BODY) moves ballistically; it reaches the antipode r = 12 at τ ≈ 4, after which D̄ = Σ_r d_r only fills up.
(b) D̄(τ) for L = 12 … 24 (L = 24 from the lean run, which equals the 16-phase τ ≤ 8 run to 0.02), and the
    thermodynamic estimate: D̄_24 up to T0, then linear with the slope SLOPES (2 = slope of D̄_24 on τ = 1.5 … 3,
    before wrapping, ≈ 2 v_body h; 2.7 if the interior height grows from 0.7 to its finite-size saturation 0.85).
(c) Γ_∞ predicted from that D̄_∞ (γ ∫₀^{t_e} D̄ = 1, independent kicks) against the extrapolated Γ_∞: σᶻ, A + B/L on
    L ≥ 8 (channels_inverse_L.csv) and on L ≥ 14 (states_inverse_L_large.csv). Small γ: Γ_∞ ≈ √(cγ/2).
Three panels (notes_layout: at most 4), γ ≥ 0.01 only.

Input:  data/used_data/ (or the first argument): single_flip_L*.csv, lean/single_flip_L24_*.csv,
        channels_rates.csv, channels_inverse_L.csv (fit_channels.py), states_inverse_L_large.csv
Output: spreading.png, spreading_prediction.csv (gamma, c, Gamma_pred) in the same directory
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import fit_states_inverse_L as fsi

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
L24 = DIR / "lean" / "single_flip_L24_sites0-6-12-18.csv"
T0, SLOPES, BODY = 3.0, (2.0, 2.7), 0.35
PROFILE_TAUS = (0.5, 1, 2, 3, 4, 6, 8, 16)
GAMMAS = np.geomspace(0.01, 0.3, 40)  # notes: γ ≥ 0.01 only (notes_layout.GAMMA_MIN)


def profile(path, l):
    d = pd.read_csv(path)
    dmg = 1 - d[[f"re_c{j}" for j in range(l)]].to_numpy()
    r = np.minimum((np.arange(l)[None, :] - d["k"].to_numpy()[:, None]) % l,
                   (d["k"].to_numpy()[:, None] - np.arange(l)[None, :]) % l)
    tau = d["tau"].round(6).to_numpy()
    return {t: np.array([dmg[(tau == t)][r[tau == t] == rr].mean() for rr in range(l // 2 + 1)])
            for t in np.unique(tau)}


def mean_damage():
    out = {}
    for f in sorted(DIR.glob("single_flip_L*.csv")):
        l = int(f.name.split("_L")[1].split("_")[0].split(".")[0])
        if l < 24:
            d = pd.read_csv(f, usecols=["tau", "D"])
            out[l] = d.groupby(d["tau"].round(6))["D"].mean()
    d = pd.read_csv(L24, usecols=["tau", "D"])
    out[24] = d.groupby(d["tau"].round(6))["D"].mean()
    return out


def predicted(d24, gamma, c):
    tau, dm = d24.index.to_numpy(), d24.to_numpy()
    t = np.linspace(0, 4 / np.sqrt(gamma) + 50, 200001)
    dt = np.where(t <= T0, np.interp(t, tau, dm), np.interp(T0, tau, dm) + c * (t - T0))
    cum = np.concatenate([[0], np.cumsum(np.diff(t) * (dt[1:] + dt[:-1]) / 2)])
    return 1 / t[np.argmax(gamma * cum >= 1)]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained", squeeze=False)
    axes = axes.reshape(1, 3)
    blues = LinearSegmentedColormap.from_list("blues", fsi.RAMP)

    ax = axes[0, 0]
    prof = profile(L24, 24)
    for i, t in enumerate(PROFILE_TAUS):
        ax.plot(range(13), prof[t], "-o", ms=3, color=blues(i / (len(PROFILE_TAUS) - 1)), label=f"τ = {t:g}")
    ax.axhline(BODY, color=fsi.MUTED, ls=":", lw=0.9)
    ax.set(xlabel="distance r from the kicked site", ylabel="damage d_r = 1 − Re c", title="(a) Damage profile of one kick, L = 24")
    ax.legend(fontsize=7.5, ncol=2)
    ts = [t for t in prof if 0.5 <= t <= 3.5]
    fr = []
    for t in ts:
        p = prof[t]
        i = np.nonzero(p < BODY)[0][0]
        fr.append(i - 1 + (p[i - 1] - BODY) / (p[i - 1] - p[i]))
    v_body = np.polyfit(ts, fr, 1)[0]
    ax.set_title(f"(a) Damage profile, L = 24 (front d = {BODY:g}: v = {v_body:.2f})")

    ax = axes[0, 1]
    dmg = mean_damage()
    sizes = sorted(dmg)
    for i, l in enumerate(sizes):
        ax.plot(dmg[l].index, dmg[l].to_numpy(), color=blues(i / (len(sizes) - 1)), label=f"L = {l}")
    d24 = dmg[24]
    tt = np.linspace(T0, 25, 50)
    for c, ls in zip(SLOPES, ("--", ":")):
        ax.plot(tt, np.interp(T0, d24.index, d24) + c * (tt - T0), ls, color=fsi.SERIES["neel"],
                label=f"L → ∞ estimate, slope {c:g}")
    ax.set(xlabel="τ", ylabel="D̄(τ)", xlim=(0, 25), ylim=(0, 32), title="(b) Mean damage D̄(τ)")
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[0, 2]
    rows = [(g, c, predicted(d24, g, c)) for c in SLOPES for g in GAMMAS]
    pred = pd.DataFrame(rows, columns=["gamma", "c", "Gamma_pred"])
    pred.to_csv(DIR / "spreading_prediction.csv", index=False)
    for c, ls in zip(SLOPES, ("--", ":")):
        p = pred[pred["c"] == c]
        ax.plot(p["gamma"], p["Gamma_pred"], ls, color=fsi.SERIES["neel"], label=f"single kick, ballistic D̄_∞ (slope {c:g})")
    ax.plot(GAMMAS, 2 * GAMMAS, ":", color=fsi.MUTED, lw=0.9, label="2γ")
    ch = pd.read_csv(DIR / "channels_inverse_L.csv")
    z = ch[(ch["state"] == "minus-x") & (ch["channel"] == "z") & (ch["gamma"] <= 0.3)]
    ax.errorbar(z["gamma"], z["Gamma_inf"], yerr=z["sigma_Gamma_inf"], fmt="o", ms=4, capsize=2,
                color=fsi.SERIES["minus-x"], label="Γ_∞, σᶻ, A + B/L on L ≥ 8")
    lg = pd.read_csv(DIR / "states_inverse_L_large.csv")
    lg = lg[lg["gamma"] >= GAMMAS[0] * (1 - 1e-9)]
    ax.errorbar(lg["gamma"], lg["A"], yerr=lg["sigma_A"], fmt="s", ms=4, capsize=2, mfc=fsi.SURFACE,
                color=fsi.SERIES["minus-x"], label="Γ_∞, σᶻ, A + B/L on L ≥ 14")
    ax.set(xscale="log", yscale="log", xlabel="γ", ylabel="Γ", title="(c) Scar Γ_∞ and the single-kick prediction")
    ax.legend(fontsize=7.5)

    fig.savefig(DIR / "spreading.png")
    print(f"v_body = {v_body:.2f}; slope of D̄_24 on τ = 1.5 … 3: {(d24[3.0] - d24[1.5]) / 1.5:.2f}")
    print(pred[np.isin(pred["c"], SLOPES)].pivot(index="gamma", columns="c", values="Gamma_pred").iloc[::6].to_string(
        float_format="%.4f"))
    print(f"wrote spreading.png, spreading_prediction.csv in {DIR}")


if __name__ == "__main__":
    main()
