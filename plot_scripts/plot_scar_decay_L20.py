"""
How the envelope and the 1/e rate Γ_e (fit_states.py, CLAUDE.md "Decay rates") summarize a decay, on the largest L
with data. Two figures, 4 panels each (notes_layout):

  scar_decay_L<L>.png   scar |−⟩^⊗L, γ = 0, 0.01, 0.1, 1: m_x (the precessing signal, −cos(ht) at γ = 0), m_⊥
  neel_envelope.png     x-Néel (s_x, s_⊥) and z-Néel (s_z, |s_z|), γ = 0.01 and 1: the staggered signals cross
                        zero, so the envelope E is what makes Γ_e well defined

In every panel: E(t) = max_{t' ≥ t} S(t') (dashed), ±e^(−Γ_e t) (the exponential that starts at S(0) = 1 and meets E
at 1/e, t_e = 1/Γ_e marked), and the free-spin e^(−2γt) where it applies (scar, x-Néel). One run per (state, L, γ)
(states_data.select: exact, jump or diffusive). Linear axes, so the oscillation stays visible.

Input:  data/used_data (or the directory given as the first argument)
Output: the two figures, in the same directory
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import states_data
from fit_states import INK, INK_SECONDARY, MUTED, rates

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
SCAR_GAMMAS = [0, 0.01, 0.1, 1]
NEEL_GAMMAS = [0.01, 1]
T_SPAN = 2.5          # scar panels show t ≤ T_SPAN · t_e (γ > 0), or the whole run (γ = 0)
T_NEEL = 3.0          # Néel panels show t ≤ T_NEEL (≈ 15 t_e), so the zero crossings of s_x, s_z are visible
BLUE, LIGHT, ORANGE = "#2a78d6", "#a9c8f0", "#eb6834"
METHOD = {"exact": "exact ρ", "jump": "jump trajectories", "traj": "diffusive trajectories"}
# state: (raw signal, its label, decay signal, its label, free-spin reference e^(−2γt) applies)
SIGNALS = {
    "minus-x": (lambda r: r["mx"], "m_x", lambda r: np.hypot(r["mx"], r["my"]), "m_⊥", True),
    "neel-x": (lambda r: r["sx"], "s_x", lambda r: np.hypot(r["sx"], r["sy"]), "s_⊥", True),
    "neel": (lambda r: r["sz"], "s_z", lambda r: r["sz"].abs(), "|s_z|", False),
}
NAMES = {"minus-x": "scar", "neel-x": "x-Néel", "neel": "z-Néel"}


def largest(chosen, state, gammas):
    """Largest L that has every γ of the list, and its runs."""
    for l in sorted({k[1] for k in chosen if k[0] == state}, reverse=True):
        runs = {g: chosen[k] for k in chosen if k[0] == state and k[1] == l for g in gammas if np.isclose(k[2], g)}
        if len(runs) == len(gammas):
            return l, runs
    raise SystemExit(f"no L with every γ of {gammas} for {state}")


def panel(ax, state, l, g, method, run):
    raw_f, raw_label, sig_f, sig_label, free = SIGNALS[state]
    t, raw, s = run["t"].to_numpy(), raw_f(run).to_numpy(), sig_f(run).to_numpy()
    env = np.maximum.accumulate(s[::-1])[::-1]
    gamma_e = rates(state, run)[0] if g > 0 else 0.0
    t_end = T_NEEL if state != "minus-x" else T_SPAN / gamma_e if gamma_e > 0 else t[-1]
    sel = t <= t_end
    ax.plot(t[sel], raw[sel], color=LIGHT, lw=0.9, label="raw signal (m_x, s_x, s_z)")
    ax.plot(t[sel], s[sel], color=BLUE, lw=1.3, label="decay signal S (m_⊥, s_⊥, |s_z|)")
    ax.plot(t[sel], env[sel], color=INK, lw=0.9, ls="--", label="envelope E")
    ts = np.linspace(0, t[sel][-1], 400)
    wrap = np.exp(-gamma_e * ts)
    ax.plot(ts, wrap, color=ORANGE, lw=1.4, label="±e^(−Γ_e t)")
    ax.plot(ts, -wrap, color=ORANGE, lw=1.4)
    if g > 0:
        if free:
            ax.plot(ts, np.exp(-2 * g * ts), color=MUTED, ls=":", lw=1.1, label="e^(−2γt), free spins")
        ax.plot([1 / gamma_e], [np.exp(-1)], "o", ms=5, color=ORANGE, mec=INK, mew=0.6, zorder=4)
        ax.axhline(np.exp(-1), color=MUTED, lw=0.6, ls="-.")
        title = (f"{NAMES[state]}, L = {l}, γ = {g:g} ({METHOD[method]})\n"
                 f"{raw_label}, {sig_label}: Γ_e = {gamma_e:.3g}")
    else:
        title = f"{NAMES[state]}, L = {l}, γ = 0 ({'one trajectory' if method != 'exact' else 'exact'})\nno decay, Γ_e = 0"
    ax.set(ylim=(-1.08, 1.08), xlim=(0, t[sel][-1]), title=title)
    ax.title.set_fontsize(8.5)


def figure(panels, title, out):
    """2 × 2 panels; the caption in the notes explains the lines."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    for ax, args in zip(axes.flat, panels):
        panel(ax, *args)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("signal")
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncols=3, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10, color=INK)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    chosen = states_data.select(states_data.load(DIR))
    l, runs = largest(chosen, "minus-x", SCAR_GAMMAS)
    figure([("minus-x", l, g, runs[g][0], runs[g][1]) for g in SCAR_GAMMAS],
           f"Scar |−⟩^⊗L at L = {l}: decay and the 1/e rate", DIR / f"scar_decay_L{l}.png")
    neel = []
    for state in ("neel-x", "neel"):
        l, runs = largest(chosen, state, NEEL_GAMMAS)
        neel += [(state, l, g, runs[g][0], runs[g][1]) for g in NEEL_GAMMAS]
    figure(neel, "Néel states: the envelope of a signal that crosses zero", DIR / "neel_envelope.png")


if __name__ == "__main__":
    main()
