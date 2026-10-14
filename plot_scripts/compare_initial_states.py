"""
Decay of the oscillations from the scar state |−⟩^⊗L against a random product state, exact evolution.

Same fit as fit_decay.py, applied to both initial states at the fixed γ of the runs:
Step 1, for each L: the amplitude A(t) = √(⟨σˣ⟩² + ⟨σʸ⟩²) is fitted with A = A₀ e^(−Γt)
        (straight line in ln A against t, from t = 0 up to the first time A drops below A_MIN).
Step 2: Γ(L) is fitted with a power law, Γ = a L^b, and with an exponential, Γ = a e^(bL).

The γ = 0 curves of the random state are drawn too: they show how much of its decay happens
without any noise.

Input:  data/evolution_L<L>.csv          (cd rust && cargo run --release -- --lmax 10)
        data/evolution_random_L<L>.csv   (cd rust && cargo run --release -- --lmax 10 --init random)
Output: data/initial_state_comparison.png, data/decay_rates_initial_states.csv
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
BLUE, ORANGE = "#2a78d6", "#eb6834"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(prefix):
    files = sorted(DATA.glob(f"{prefix}_L*.csv"), key=lambda p: int(p.stem.split("_L")[1]))
    if not files:
        raise SystemExit(f"no {prefix}_L*.csv in {DATA}; see the docstring for the runs")
    return {int(p.stem.split("_L")[1]): pd.read_csv(p) for p in files}


def amplitude(df, g):
    run = df[df["gamma"] == g]
    return run["t"].to_numpy(), np.hypot(run["mx"], run["my"]).to_numpy()


def fit(t, amp):
    """Γ, ln A₀ and the end of the fit window."""
    below = np.nonzero(amp < A_MIN)[0]
    end = below[0] if len(below) else len(t)
    slope, intercept = np.polyfit(t[:end], np.log(amp[:end]), 1)
    return -slope, intercept, end


cases = {"scar state |−⟩$^{⊗L}$": load("evolution"), "random product state": load("evolution_random")}
sizes = sorted(set.intersection(*(set(runs) for runs in cases.values())))
gamma = max(df["gamma"].max() for runs in cases.values() for df in runs.values())
blues = LinearSegmentedColormap.from_list("blues", RAMP)
colors = {L: blues(i / max(len(sizes) - 1, 1)) for i, L in enumerate(sizes)}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
rates, table = {}, []
for ax, (name, runs) in zip(axes, cases.items()):
    rates[name] = {}
    for L in sizes:
        t, amp = amplitude(runs[L], gamma)
        rate, intercept, end = fit(t, amp)
        rates[name][L] = rate
        table.append({"initial_state": name.split()[0], "L": L, "gamma": gamma, "rate": rate,
                      "rate_over_gamma": rate / gamma, "A0": np.exp(intercept), "t_window": t[end - 1]})
        ax.plot(t, amp, color=colors[L], label=f"L = {L}:  Γ = {rate / gamma:.2f} γ")
        ax.plot(t[:end], np.exp(intercept - rate * t[:end]), color=INK, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axhline(A_MIN, color=MUTED, linewidth=0.8, label=f"fit window: amplitude above {A_MIN:g}")
    ax.plot([], [], color=INK, linewidth=0.8, linestyle=(0, (4, 3)), label="fit A₀ e$^{-Γt}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude √(⟨σ$^x$⟩² + ⟨σ$^y$⟩²)")
    ax.set_title(f"{name}, γ = {gamma:g}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2, fontsize=7.5)

# the random state without noise, for reference
ax_random = axes[1]
for L in sizes:
    t0, amp0 = amplitude(cases["random product state"][L], 0.0)
    ax_random.plot(t0, amp0, color=colors[L], linewidth=0.7, alpha=0.45)
ax_random.text(0.98, 0.97, "faint: same states at γ = 0", transform=ax_random.transAxes, ha="right", va="top",
               color=INK_SECONDARY, fontsize=8)

# Γ against L, both initial states
ax_L = axes[2]
fine = np.linspace(sizes[0] - 0.3, sizes[-1] + 0.3, 200)
summary = []
for (name, rate_of), color, marker in zip(rates.items(), [BLUE, ORANGE], ["o", "s"]):
    L_arr = np.array(sizes)
    rate = np.array([rate_of[L] for L in sizes])
    b_pow, ln_a_pow = np.polyfit(np.log(L_arr), np.log(rate), 1)
    b_exp, ln_a_exp = np.polyfit(L_arr, np.log(rate), 1)
    rms_pow = np.sqrt(np.mean((np.log(rate) - (ln_a_pow + b_pow * np.log(L_arr))) ** 2))
    rms_exp = np.sqrt(np.mean((np.log(rate) - (ln_a_exp + b_exp * L_arr)) ** 2))
    summary.append((name, ln_a_pow, b_pow, rms_pow, ln_a_exp, b_exp, rms_exp))
    ax_L.plot(fine, np.exp(ln_a_pow) * fine**b_pow / gamma, color=color, linewidth=1.2,
              label=f"{name}:  Γ ∝ L$^{{{b_pow:.2f}}}$  (rms {100 * rms_pow:.1f} %)")
    ax_L.plot(L_arr, rate / gamma, linestyle="none", marker=marker, markersize=7, color=color,
              markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
ax_L.set_yscale("log")
ax_L.set_xticks(sizes)
ax_L.set_xlabel("L")
ax_L.set_ylabel("Γ / γ")
ax_L.set_title("Fitted decay rate against L (power-law fits)")
ax_L.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)

pd.DataFrame(table).to_csv(DATA / "decay_rates_initial_states.csv", index=False)
fig.savefig(DATA / "initial_state_comparison.png")

print(f"gamma = {gamma:g}, fit window: amplitude > {A_MIN:g}")
print(pd.DataFrame(table).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
for name, ln_a_pow, b_pow, rms_pow, ln_a_exp, b_exp, rms_exp in summary:
    print(f"{name}:")
    print(f"  power law:    Gamma = {np.exp(ln_a_pow) / gamma:.3f} gamma * L^{b_pow:.3f}        rms residual {100 * rms_pow:.2f} %")
    print(f"  exponential:  Gamma = {np.exp(ln_a_exp) / gamma:.3f} gamma * exp({b_exp:.4f} L)   rms residual {100 * rms_exp:.2f} %")
print(f"wrote {DATA / 'initial_state_comparison.png'} and {DATA / 'decay_rates_initial_states.csv'}")
