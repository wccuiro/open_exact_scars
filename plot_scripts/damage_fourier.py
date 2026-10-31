"""
Fourier transform of the damage of one kick (single_flip files, no new runs).

d_r(τ) = 1 − Re c_{k+r} at signed distance r = 0 … L − 1 (ring) from the kicked site k, mean over phases and kicked
sites; d(k, τ) = Σ_r e^{ikr} d_r(τ), k = 2πq/L. d(0, τ) = D̄(τ), and d(k, 0) = 2 for every k (only the kicked site is
damaged at τ = 0). If the damage were a conserved density, d(0, τ) would stay at 2 and only k ≠ 0 would relax.

Front: rms width σ(τ) of the profile (minimal-image distance, weights d_r) and the half width in k, k_½(τ), where
d(k, τ)/d(0, τ) = 1/2 (linear interpolation between the q of the ring; NaN once it drops below 2π/L). For a ballistic
front both ∝ τ, for a diffusive one ∝ √τ; the speed is the slope of σ on τ ∈ FIT (before the front wraps the ring).

Sizes: L = 12 … 20 (16 phases × all sites, τ ≤ 25), L = 22 (16 phases × 4 sites, τ ≤ 8), L = 24 from the lean run
(4 phases × 4 sites, τ ≤ 16, in place of the τ ≤ 8 file). Realization seed 1 only.

Input:  data/used_data/ (or the first argument): single_flip_L<L>*.csv, lean/single_flip_L24_*.csv
Output: in the same directory
        damage_fourier.csv        L, tau, q, k, d_re, d_im, d_err (standard error of d_re over phases and sites)
        damage_fourier_front.csv  L, tau, D, sigma, k_half
        damage_fourier.png        (a), (b) d(k, τ) against τ at L = 24, 20; (c) d(k, τ)/d(0, τ) against kτ, L = 24;
                                  (d) σ(τ) and 1/k_½(τ) for every L
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import fit_states_inverse_L as fsi

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
FILES = {12: "single_flip_L12.csv", 14: "single_flip_L14.csv", 16: "single_flip_L16.csv", 18: "single_flip_L18.csv",
         20: "single_flip_L20.csv", 22: "single_flip_L22_sites0-5-11-16.csv",
         24: "lean/single_flip_L24_sites0-6-12-18.csv"}
COLLAPSE_TAUS = (0.5, 1, 1.5, 2, 2.5, 3)
FIT = (0.5, 1.5)


def transform(path, l):
    """Per τ: mean profile d_r, and d(k) with its standard error over the runs (phase, kicked site)."""
    d = pd.read_csv(path)
    dmg = 1 - d[[f"re_c{j}" for j in range(l)]].to_numpy()
    k = d["k"].to_numpy()
    dr = np.take_along_axis(dmg, (np.arange(l)[None, :] + k[:, None]) % l, axis=1)
    dk = np.fft.ifft(dr, axis=1) * l  # Σ_r e^{+2πiqr/L} d_r
    tau = d["tau"].round(6).to_numpy()
    taus = np.unique(tau)
    prof = np.array([dr[tau == t].mean(0) for t in taus])
    mean = np.array([dk[tau == t].mean(0) for t in taus])
    err = np.array([dk[tau == t].real.std(0, ddof=1) / np.sqrt((tau == t).sum()) for t in taus])
    return taus, prof, mean, err


def k_half(ratio, l):
    """k where d(k)/d(0) first drops to 1/2, interpolated over q = 0 … L/2."""
    q = np.nonzero(ratio[: l // 2 + 1] < 0.5)[0]
    if len(q) == 0 or q[0] < 2:
        return np.nan
    i = q[0]
    return 2 * np.pi / l * (i - 1 + (ratio[i - 1] - 0.5) / (ratio[i - 1] - ratio[i]))


def main():
    rows, front, data = [], [], {}
    for l, name in FILES.items():
        taus, prof, dk, err = transform(DIR / name, l)
        data[l] = (taus, prof, dk, err)
        r = np.minimum(np.arange(l), l - np.arange(l))
        for t, p, d, e in zip(taus, prof, dk, err):
            for q in range(l // 2 + 1):
                rows.append((l, t, q, 2 * np.pi * q / l, d[q].real, d[q].imag, e[q]))
            front.append((l, t, d[0].real, np.sqrt((r**2 * p).sum() / p.sum()), k_half(d.real / d[0].real, l)))
    pd.DataFrame(rows, columns=["L", "tau", "q", "k", "d_re", "d_im", "d_err"]).to_csv(DIR / "damage_fourier.csv",
                                                                                      index=False)
    front = pd.DataFrame(front, columns=["L", "tau", "D", "sigma", "k_half"])
    front.to_csv(DIR / "damage_fourier_front.csv", index=False)

    blues = LinearSegmentedColormap.from_list("blues", fsi.RAMP)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, l, tag in ((axes[0, 0], 24, "a"), (axes[0, 1], 20, "b")):
        taus, _, dk, err = data[l]
        for q in range(1, l // 2 + 1):
            ax.plot(taus, dk[:, q].real, color=blues(q / (l // 2)), lw=1.0, label=f"k = {q}·2π/L" if q in (1, 2, 3, l // 2) else None)
        ax.plot(taus, dk[:, 0].real, color=fsi.SERIES["neel"], lw=2, label="k = 0 (= D̄)")
        ax.axhline(2, color=fsi.MUTED, ls=":", lw=0.9, label="conserved density: d(0, τ) = 2")
        ax.set(xlabel="τ", ylabel="Re d(k, τ)", title=f"({tag}) Damage in k space, L = {l}")
        ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    taus, _, dk, _ = data[24]
    kk = 2 * np.pi * np.arange(13) / 24
    for i, t in enumerate(COLLAPSE_TAUS):
        j = np.argmin(np.abs(taus - t))
        ax.plot(kk * t, dk[j, :13].real / dk[j, 0].real, "-o", ms=3, color=blues(i / (len(COLLAPSE_TAUS) - 1)),
                label=f"τ = {t:g}")
    ax.axhline(0, color=fsi.AXIS, lw=0.8)
    ax.set(xlabel="k τ", ylabel="d(k, τ) / d(0, τ)", xlim=(0, 3),
           title="(c) Profile shape, L = 24: ballistic collapse in kτ")
    ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    sizes = sorted(FILES)
    v = {}
    for i, l in enumerate(sizes):
        f = front[front["L"] == l]
        c = blues(i / (len(sizes) - 1))
        ax.plot(f["tau"], f["sigma"], color=c, label=f"L = {l}")
        ax.plot(f["tau"], 1 / f["k_half"], "--", color=c, lw=1.0)
        w = f[(f["tau"] >= FIT[0]) & (f["tau"] <= FIT[1])]
        v[l] = np.polyfit(w["tau"], w["sigma"], 1)[0]
    ax.set(xlabel="τ", ylabel="σ (solid), 1/k_½ (dashed)", xlim=(0, 8),
           title=f"(d) Front width; slope of σ on τ = {FIT[0]:g} … {FIT[1]:g}: {v[24]:.2f} (L = 24)")
    ax.legend(fontsize=7.5, ncol=2)
    fig.savefig(DIR / "damage_fourier.png")

    print("Re d(k, τ), L = 24 (columns q = 0 … 12):")
    taus, _, dk, err = data[24]
    for t in (0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16):
        j = np.argmin(np.abs(taus - t))
        print(f"  τ = {t:4g}: " + " ".join(f"{x:6.2f}" for x in dk[j, :13].real) + f"   (max err {err[j, :13].max():.2f})")
    print("max |Im d| / D̄ per L: " + ", ".join(f"{l}: {np.abs(data[l][2].imag).max() / data[l][2][1:, 0].real.min():.1e}"
                                            for l in sizes))
    print("slope dσ/dτ on τ = %g … %g: " % FIT + ", ".join(f"L = {l}: {v[l]:.2f}" for l in sizes))
    f = front[front["L"] == 24]
    for lo, hi in ((0.5, 1.5), (1, 2)):  # k_½ is below the ring resolution 2π/24 from τ ≈ 2.3 on
        w = f[(f["tau"] >= lo) & (f["tau"] <= hi)]
        print(f"L = 24, τ = {lo:g} … {hi:g}: d ln σ/d ln τ = {np.polyfit(np.log(w['tau']), np.log(w['sigma']), 1)[0]:.2f}, "
              f"d ln(1/k_½)/d ln τ = {np.polyfit(np.log(w['tau']), -np.log(w['k_half']), 1)[0]:.2f}")
    print(f"wrote damage_fourier.csv, damage_fourier_front.csv, damage_fourier.png in {DIR}")


if __name__ == "__main__":
    main()
