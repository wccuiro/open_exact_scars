"""
Fit the decay of the oscillations at the fixed γ of the run, and its dependence on L.

Step 1, for each L: the amplitude A(t) = √(⟨σˣ⟩² + ⟨σʸ⟩²) is fitted with A = A₀ e^(−Γt)
        (straight line in ln A against t, on the window where A > A_MIN).
Step 2: Γ(L) is fitted with a power law, Γ = a L^b, and with an exponential, Γ = a e^(bL)
        (straight lines in ln Γ against ln L and against L).

Input:  data/evolution_L<L>.csv   (cd rust && cargo run --release)
Output: data/decay_fit.png, data/decay_rates.csv (L, gamma, rate, rate_over_gamma)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

A_MIN = 0.05  # the fit uses the times where the amplitude is above this

DATA = Path(__file__).resolve().parent.parent / "data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # small L light, large L dark
ORANGE = "#eb6834"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

files = sorted(DATA.glob("evolution_L*.csv"), key=lambda p: int(p.stem.split("_L")[1]))
if not files:
    raise SystemExit(f"no evolution_L*.csv in {DATA}; run `cargo run --release` in rust/ first")
runs = {int(p.stem.split("_L")[1]): pd.read_csv(p) for p in files}
blues = LinearSegmentedColormap.from_list("blues", RAMP)
colors = {L: blues(i / max(len(runs) - 1, 1)) for i, L in enumerate(runs)}
gamma = max(df["gamma"].max() for df in runs.values())

fig, (ax_t, ax_L) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

# step 1: exponential fit of the amplitude, one L at a time
rates = {}
for L, df in runs.items():
    run = df[df["gamma"] == gamma]
    t, amp = run["t"].to_numpy(), np.hypot(run["mx"], run["my"]).to_numpy()
    below = np.nonzero(amp < A_MIN)[0]
    end = below[0] if len(below) else len(t)
    slope, intercept = np.polyfit(t[:end], np.log(amp[:end]), 1)
    rates[L] = -slope
    ax_t.plot(t, amp, color=colors[L], label=f"L = {L}:  Γ = {-slope:.4f} = {-slope / gamma:.2f} γ")
    ax_t.plot(t[:end], np.exp(intercept + slope * t[:end]), color=INK, linewidth=0.8, linestyle=(0, (4, 3)))
ax_t.axhline(A_MIN, color=MUTED, linewidth=0.8, label=f"fit window: amplitude above {A_MIN:g}")
ax_t.plot([], [], color=INK, linewidth=0.8, linestyle=(0, (4, 3)), label="fit A₀ e$^{-Γt}$")
ax_t.set_yscale("log")
ax_t.set_ylim(1e-3, 1.5)
ax_t.set_xlim(0, t[-1])
ax_t.set_xlabel("t")
ax_t.set_ylabel("amplitude √(⟨σ$^x$⟩² + ⟨σ$^y$⟩²)")
ax_t.set_title(f"Decay of the oscillations, γ = {gamma:g}")
ax_t.legend(loc="lower left", fontsize=8)

# step 2: Γ against L
sizes = np.array(list(rates))
rate = np.array(list(rates.values()))
b_pow, ln_a_pow = np.polyfit(np.log(sizes), np.log(rate), 1)
b_exp, ln_a_exp = np.polyfit(sizes, np.log(rate), 1)
power = lambda L: np.exp(ln_a_pow) * L**b_pow
expo = lambda L: np.exp(ln_a_exp + b_exp * L)
rms = lambda model: np.sqrt(np.mean((np.log(rate) - np.log(model(sizes))) ** 2))

fine = np.linspace(sizes[0] - 0.3, sizes[-1] + 0.3, 200)
ax_L.plot(fine, power(fine) / gamma, color=RAMP[2],
          label=f"power law  Γ ∝ L$^{{{b_pow:.2f}}}$   (rms residual {100 * rms(power):.1f} %)")
ax_L.plot(fine, expo(fine) / gamma, color=ORANGE,
          label=f"exponential  Γ ∝ e$^{{{b_exp:.3f} L}}$   (rms residual {100 * rms(expo):.1f} %)")
ax_L.plot(sizes, rate / gamma, linestyle="none", marker="o", markersize=7, color=INK, markeredgecolor=SURFACE,
          markeredgewidth=1.2, label="fitted Γ", zorder=3)
ax_L.set_xticks(sizes)
ax_L.set_xlabel("L")
ax_L.set_ylabel("Γ / γ")
ax_L.set_title("Decay rate against L")
ax_L.legend(loc="upper left", fontsize=8)

pd.DataFrame({"L": sizes, "gamma": gamma, "rate": rate, "rate_over_gamma": rate / gamma}).to_csv(
    DATA / "decay_rates.csv", index=False)
fig.savefig(DATA / "decay_fit.png")

print(f"gamma = {gamma:g}, fit window: amplitude > {A_MIN:g}")
for L in sizes:
    print(f"  L = {L}:  Gamma = {rates[L]:.5f} = {rates[L] / gamma:.3f} gamma")
print(f"power law:    Gamma = {np.exp(ln_a_pow) / gamma:.3f} gamma * L^{b_pow:.3f}        rms residual {100 * rms(power):.2f} %")
print(f"exponential:  Gamma = {np.exp(ln_a_exp) / gamma:.3f} gamma * exp({b_exp:.4f} L)   rms residual {100 * rms(expo):.2f} %")
print(f"wrote {DATA / 'decay_fit.png'} and {DATA / 'decay_rates.csv'}")
