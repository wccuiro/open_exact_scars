"""
Half-chain entanglement of the scars |Φ_n⟩ (Dicke states) against the analytic results of the notes (Theory,
"Scar entanglement"):

  exact      S = −Σ_ℓ p_ℓ ln p_ℓ, p_ℓ = C(L_A, ℓ) C(L − L_A, n − ℓ) / C(L, n) (hypergeometric Schmidt weights)
  Gaussian   S(Φ_{L/2}) ≈ ½ ln(π e L / 8) (the entropy of a discretized Gaussian with variance ≈ L/16)

Panels: (a) S against n for every L with data, the scars found by diagonalizing H (tower weight > 1/2, n = L/2 − E/h)
and the exact formula; (b) the middle scar S(Φ_{⌊L/2⌋}) against L: data, exact formula up to L = 200, Gaussian
estimate, Page value of a random state (volume law) for contrast.

Input:  entanglement_L<L>.csv (from the `entanglement` binary) in data/used_data (or the directory given as the first
        argument)
Output: scar_entropy.png in the same directory
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gammaln

from plot_states import MUTED, RAMP  # also sets the shared rcParams

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
H = 1.0
ORANGE = "#eb6834"


def log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def dicke_entropy(l, n, l_a=None):
    l_a = l // 2 if l_a is None else l_a
    ell = np.arange(max(0, n - (l - l_a)), min(l_a, n) + 1)
    p = np.exp(log_binom(l_a, ell) + log_binom(l - l_a, n - ell) - log_binom(l, n))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    files = sorted(DIR.glob("entanglement_L*.csv"), key=lambda f: int(re.search(r"_L(\d+)", f.name).group(1)))
    if not files:
        sys.exit(f"no entanglement_L*.csv in {DIR}")
    blues = LinearSegmentedColormap.from_list("blues", RAMP)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    mid = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        l = int(df["L"].iloc[0])
        scars = df[df["tower"] > 0.5].copy()
        scars["n"] = np.rint(l / 2 - scars["E"] / H).astype(int)
        scars = scars.sort_values("n")
        exact = np.array([dicke_entropy(l, n) for n in scars["n"]])
        err = np.abs(scars["S"].to_numpy() - exact).max()
        print(f"L = {l}: {len(scars)} scars, max |S − exact| = {err:.1e}")
        c = blues(i / max(len(files) - 1, 1))
        ns = np.arange(l + 1)
        ax.plot(ns, [dicke_entropy(l, n) for n in ns], color=c, lw=1.0)
        ax.plot(scars["n"], scars["S"], "o", ms=4, color=c, label=f"L = {l}")
        m = scars[scars["n"] == l // 2]
        mid.append((l, float(m["S"].iloc[0])))
    ax.set(xlabel="n (down spins)", ylabel="S_A of |Φ_n⟩", title="(a) scars: diagonalization (dots) and exact formula")
    ax.legend(ncols=2, fontsize=7.5)

    ls = np.arange(4, 201)
    bx.plot(ls, [dicke_entropy(l, l // 2) for l in ls], color=ORANGE, lw=1.4, label="exact, hypergeometric weights")
    bx.plot(ls, 0.5 * np.log(np.pi * np.e * ls / 8), color=MUTED, ls="--", lw=1.1, label="½ ln(π e L / 8)")
    l_mid, s_mid = np.array(mid).T
    bx.plot(l_mid, s_mid, "o", ms=5, color="#1c5cab", label="diagonalization of H")
    l_page = np.arange(4, 25)
    bx.plot(l_page, (l_page // 2) * np.log(2) - 0.5 * 2.0 ** (2 * (l_page // 2) - l_page), color=MUTED, ls=":",
            label="Page value (random state)")
    bx.set_xscale("log")
    bx.set(xlabel="L", ylabel="S_A of |Φ_⌊L/2⌋⟩", ylim=(0, 4), title="(b) middle scar: logarithmic in L")
    bx.legend(fontsize=7.5)
    out = DIR / "scar_entropy.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
