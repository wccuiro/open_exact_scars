"""Compare square-root fits to the Floquet chapter's thermodynamic scar rates."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    folder = Path(__file__).resolve().parents[1] / "data" / "floquet_dep"
    table = pd.read_csv(folder / "floquet_dep_inverse_L.csv")
    scar = table.query("state == 'minus-x' and 0 < gamma <= 1").sort_values("gamma")
    gamma = scar.gamma.to_numpy()
    rate = scar.Gamma_inf.to_numpy()
    error = scar.sigma_Gamma_inf.to_numpy()
    ratio = rate / np.sqrt(gamma)
    weak = gamma <= 0.05  # All four extrapolations include L = 8 through 24.
    # Least squares in log(rate), with exponent fixed at 1/2 and equal weight per point.
    weak_fit = np.exp(np.log(ratio[weak]).mean())
    full_fit = np.exp(np.log(ratio).mean())
    exponent = np.polyfit(np.log(gamma[weak]), np.log(rate[weak]), 1)[0]
    print(f"Weak A={weak_fit:.6f}, full A={full_fit:.6f}, weak free exponent={exponent:.6f}")

    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=(7.4, 5.7), layout="constrained",
                                 gridspec_kw={"height_ratios": [1.55, 1]})
    x = np.geomspace(gamma.min(), gamma.max(), 300)
    blue, orange, ink = "#06716e", "#b35318", "#172a43"
    for panel in (ax, bx):
        panel.axvspan(gamma.min(), 0.05, color="#eaf4f3", zorder=0)
        panel.grid(axis="y", which="major", alpha=.18)
        panel.spines[["top", "right"]].set_visible(False)
    ax.loglog(x, weak_fit * np.sqrt(x), color=blue, lw=2, label=fr"$A={weak_fit:.2f}$  ·  weak fit")
    ax.loglog(x, full_fit * np.sqrt(x), color=orange, ls="--", lw=1.7,
              label=fr"$A={full_fit:.2f}$  ·  full fit")
    ax.errorbar(gamma, rate, yerr=error, fmt="o", color=ink, capsize=3, ms=5,
                label="Floquet scar rate", zorder=4)
    ax.set(ylabel=r"Decay rate  $\Gamma_\infty$",
           title=r"Scar decay under depolarization: fits to $\Gamma_\infty=A\sqrt{\gamma}$")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    bx.axhline(weak_fit, color=blue, lw=2)
    bx.axhline(full_fit, color=orange, ls="--", lw=1.7)
    bx.errorbar(gamma, ratio, yerr=error / np.sqrt(gamma), color=ink, fmt="o", capsize=3, ms=5)
    bx.set(xlabel=r"Depolarization strength  $\gamma$", ylabel=r"$\Gamma_\infty / \sqrt{\gamma}$",
           ylim=(1.55, 4.55))
    bx.set_xticks(gamma, [f"{g:g}" for g in gamma], minor=False)
    bx.minorticks_off()
    fig.savefig(folder / "floquet_scar_sqrt_gamma.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
