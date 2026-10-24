"""
The scar at the largest L with data (L = 20): m_x(t) and m_⊥(t) for γ = 0 and a range of γ, with the envelope and
the 1/e rate Γ_e that summarizes it (fit_states.py, CLAUDE.md "Decay rates").

  m_x       the precessing signal, −cos(ht) at γ = 0 (reference 1)
  m_⊥       √(m_x² + m_y²) of the trajectory-averaged (or exact) m_x, m_y
  E(t)      max_{t' ≥ t} m_⊥(t'), the envelope
  ±e^(−Γ_e t)  the exponential that starts at m_⊥(0) = 1 and passes through E at 1/e (t_e = 1/Γ_e, marked);
            it wraps the oscillation of m_x
  e^(−2γt)  free spins (H_I = 0), for comparison

One run per γ (states_data.select: exact, jump or diffusive). Linear axes, so the oscillation stays visible.

Input:  data/used_data (or the directory given as the first argument)
Output: scar_decay_L<L>.png in the same directory
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import states_data
from fit_states import INK, INK_SECONDARY, MUTED, rates

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
GAMMAS = [0, 0.0005, 0.002, 0.005, 0.01, 0.05, 0.2, 1]
T_SPAN = 2.5          # each panel shows t ≤ T_SPAN · t_e (γ > 0), or the whole run (γ = 0)
BLUE, LIGHT, ORANGE = "#2a78d6", "#a9c8f0", "#eb6834"
METHOD = {"exact": "exact ρ", "jump": "jump trajectories", "traj": "diffusive trajectories"}


def main():
    chosen = states_data.select(states_data.load(DIR, states=("minus-x",)))
    l = max(key[1] for key in chosen if key[2] > 0)
    runs = {g: chosen[k] for k in chosen if k[1] == l for g in GAMMAS if np.isclose(k[2], g)}
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), layout="constrained")
    for ax, g in zip(axes.flat, GAMMAS):
        if g not in runs:
            ax.set_visible(False)
            continue
        method, run, _ = runs[g]
        t, mx = run["t"].to_numpy(), run["mx"].to_numpy()
        mp = np.hypot(run["mx"], run["my"]).to_numpy()
        env = np.maximum.accumulate(mp[::-1])[::-1]
        gamma_e = rates("minus-x", run)[0] if g > 0 else 0.0
        t_end = T_SPAN / gamma_e if gamma_e > 0 else t[-1]
        sel = t <= t_end
        ax.plot(t[sel], mx[sel], color=LIGHT, lw=0.6, label="m_x")
        ax.plot(t[sel], mp[sel], color=BLUE, lw=1.3, label="m_⊥")
        ax.plot(t[sel], env[sel], color=INK, lw=0.9, ls="--", label="envelope E")
        ts = np.linspace(0, t[sel][-1], 400)
        wrap = np.exp(-gamma_e * ts)
        ax.plot(ts, wrap, color=ORANGE, lw=1.4, label="±e^(−Γ_e t)")
        ax.plot(ts, -wrap, color=ORANGE, lw=1.4)
        if g > 0:
            ax.plot(ts, np.exp(-2 * g * ts), color=MUTED, ls=":", lw=1.1, label="e^(−2γt), free spins")
            ax.plot([1 / gamma_e], [np.exp(-1)], "o", ms=5, color=ORANGE, mec=INK, mew=0.6, zorder=4)
            ax.axhline(np.exp(-1), color=MUTED, lw=0.6, ls="-.")
            title = f"γ = {g:g} ({METHOD[method]})\nΓ_e = {gamma_e:.3g} = {gamma_e / g:.2f} γ"
        else:
            title = f"γ = 0 ({'one trajectory' if method != 'exact' else 'exact'})\nno decay, Γ_e = 0"
        ax.set(ylim=(-1.08, 1.08), xlim=(0, t[sel][-1]), title=title)
        ax.title.set_fontsize(8.5)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("magnetization")
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), fontsize=8)
    fig.suptitle(f"Scar |−⟩^⊗L at L = {l}: decay and the 1/e rate", x=0.01, ha="left", fontweight="bold",
                 fontsize=10, color=INK)
    fig.text(0.01, -0.03, "Γ_e = 1/t_e, t_e the first time the envelope E (dashed) drops below 1/e (dash-dot line; dot). "
             f"±e^(−Γ_e t) starts at 1 and meets E at t_e; panels end at {T_SPAN:g} t_e.",
             fontsize=7.5, color=INK_SECONDARY)
    out = DIR / f"scar_decay_L{l}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
