"""Plot the L=12 pulse sequence and its matching continuously dissipative reference."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    folder = Path(__file__).resolve().parents[1] / "data"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "floquet_on_off_L12_g0.025.csv"
    run = pd.read_csv(path)
    ref = pd.read_csv(folder / "floquet_dep" / "floquet_minus-x_L12_g0.05_dt0.1_chdep.csv")
    threshold = 1 / np.e
    reference_amplitude = np.hypot(ref.mx, ref.my)
    t_decay = ref.loc[reference_amplitude <= threshold, "t"].iloc[0]
    assert (run.m_perp.iloc[:-1] > threshold).all() and run.m_perp.iloc[-1] <= threshold
    end = run.t.iloc[-1]
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 5.4), layout="constrained")
    top.set_title(f"L = 12     ·     γ on = 0.025     ·     window = {t_decay / 3:.1f}", loc="left", pad=20)
    rates = run.gamma_previous.iloc[1:].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(rates)) + 1, len(rates)]
    for i, j in zip(boundaries[:-1], boundaries[1:]):
        start, stop = run.t.iloc[i], run.t.iloc[j]
        on = rates[i] > 0
        for ax in (top, bottom):
            ax.axvspan(start, stop, color="#f8dfae" if on else "#cde8f0", alpha=0.62, lw=0)
        top.text((start + stop) / 2, 1.015, "ON" if on else "OFF", ha="center", va="bottom",
                 transform=top.get_xaxis_transform(), fontsize=9, color="#704817" if on else "#24596b")
    mask = ref.t <= end + 1e-8
    top.plot(ref.t[mask], reference_amplitude[mask], "--", color="#747371", lw=1.4, label="Continuously on")
    top.plot(run.t, run.m_perp, color="#183452", lw=2.4, label="Pulsed")
    top.axhline(threshold, color="#9e3b25", ls=":", lw=1.5, label="1/e threshold")
    top.plot(end, run.m_perp.iloc[-1], "o", color="#9e3b25", clip_on=False, ms=5)
    top.set(ylabel="Transverse amplitude  $m_\\perp$", ylim=(0, 1.05))
    top.legend(frameon=False, loc="lower left", ncols=3, fontsize=9)
    bottom.plot(run.t, run.mx, color="#234d73", lw=2, label="$m_x$")
    bottom.plot(run.t, run.my, color="#923e65", lw=1.8, ls="--", label="$m_y$")
    bottom.axhline(0, lw=.6, color="#69737a")
    bottom.set(xlabel="Time", ylabel="Magnetization per site", ylim=(-1.06, 1.06), xlim=(0, end))
    bottom.legend(loc="upper left", ncols=2, frameon=False)
    for ax in (top, bottom):
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
