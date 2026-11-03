"""
Operator size of ρ(t) at the 1/e time t_e, extrapolated to the thermodynamic limit, against γ, per noise channel.

  size      mean number of non-identity sites of the Pauli strings of ρ = 2^−L Σ_P r_P P, with weights
            r_P² / Σ_Q r_Q² (identity included): column mean_size of the `states --obs-dt` output
  t_e       first time the envelope of the run's own signal (scar m_⊥, x-Néel s_⊥, z-Néel |s_z|) drops below 1/e
            (fit_states.rates)
  n(t_e)    size/L at t_e, linear interpolation on the observable grid
  n_∞       A of n(t_e) = A + B/L on L = 8, 10, 12 (exact runs; the only sizes with the operator observables);
            error: the change of A when L = 8 is dropped (the same rule as the other channels in fit_channels.py)

Reference levels: 1/2 for a product state with all spins in the plane (|−⟩^⊗L, |+−+−…⟩: strings of 𝟙 and σˣ only),
3/4 for a random pure state (every non-identity factor equally likely). Dissipation removes weight from long strings,
so a mixed ρ has a smaller size; scrambling under H_I pushes a pure ρ towards 3/4.

Channels: σᶻ, σˣ, σʸ, depolarizing, thermal (n̄ = 1), decay; decay is left out for the scar (its m_⊥ rises again after
the first 1/e crossing and the runs stopped too early, CLAUDE.md "Lean-plan runs"). Runs that never reach 1/e
(Zeno-protected) are skipped. Only γ ≥ 0.01 (notes_layout.GAMMA_MIN).

Input:  data/used_data/lean/states_<state>_L<L>_g<γ><sfx>.csv and states_obs_<…>.csv (or <dir>/lean, dir = first argument)
Output: operator_size.csv (state, channel, gamma, L, n_te, t_e; and the fits with L = inf) and operator_size.png
        (one panel per state) in data/used_data
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_states import rates
from fit_states_inverse_L import MUTED, NAMES, STATES, SURFACE
from notes_layout import keep_gamma

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
LEAN = DIR / "lean"
SIZES = (8, 10, 12)
SUFFIX = {"z": "", "x": "_chx", "y": "_chy", "dep": "_chdep", "thermal": "_chthermal_n1", "decay": "_chdecay"}
CH_NAMES = {"z": "σᶻ dephasing", "x": "σˣ dephasing", "y": "σʸ dephasing", "dep": "depolarizing",
            "thermal": "thermal, n̄ = 1", "decay": "decay"}
COLORS = dict(zip(SUFFIX, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc7", "#b8860b", "#898781"]))
STYLES = dict(zip(SUFFIX, ["-o", "--s", "-.^", ":D", "-v", "--x"]))
NAME = re.compile(r"states_obs_(minus-x|neel-x|neel)_L(\d+)_g([0-9.]+?)(_ch\w+?)?\.csv$")


def collect():
    rows = []
    for f in sorted(LEAN.glob("states_obs_*_L*_g*.csv")):
        m = NAME.match(f.name)
        if m is None:
            continue
        state, l, g, sfx = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4) or ""
        ch = {v: k for k, v in SUFFIX.items()}.get(sfx)
        if ch is None or l not in SIZES or g == 0 or not keep_gamma(g) or (state == "minus-x" and ch == "decay"):
            continue
        run_file = LEAN / f.name.replace("states_obs_", "states_")
        if not run_file.exists():
            continue
        run = pd.read_csv(run_file)
        gamma_e = rates(state, run)[0]
        if not np.isfinite(gamma_e) or gamma_e <= 0:
            continue  # never reaches 1/e within the run (Zeno-protected)
        obs = pd.read_csv(f)
        t_e = 1 / gamma_e
        if t_e > obs["t"].max():
            continue
        rows.append((state, ch, g, l, np.interp(t_e, obs["t"], obs["mean_size"]) / l, t_e))
    return pd.DataFrame(rows, columns=["state", "channel", "gamma", "L", "n_te", "t_e"])


def extrapolate(d):
    fits = []
    for (state, ch, g), dg in d.groupby(["state", "channel", "gamma"]):
        dg = dg.sort_values("L")
        if list(dg["L"]) != list(SIZES):
            continue
        x, y = 1 / dg["L"].to_numpy(), dg["n_te"].to_numpy()
        a = np.polyfit(x, y, 1)[1]
        a2 = np.polyfit(x[1:], y[1:], 1)[1]
        fits.append((state, ch, g, a, abs(a - a2), y[-1]))
    return pd.DataFrame(fits, columns=["state", "channel", "gamma", "n_inf", "sigma_n_inf", "n_L12"])


def main():
    d = collect()
    fits = extrapolate(d)
    out = pd.concat([d.assign(L=d["L"].astype(str)),
                     fits.rename(columns={"n_inf": "n_te"}).assign(L="inf")[["state", "channel", "gamma", "L", "n_te"]]
                     .merge(fits[["state", "channel", "gamma", "sigma_n_inf"]], on=["state", "channel", "gamma"])])
    out.to_csv(DIR / "operator_size.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(fits.pivot_table(index=["state", "gamma"], columns="channel", values="n_inf").round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, state in zip(axes, STATES):
        for ch in SUFFIX:
            f = fits[(fits["state"] == state) & (fits["channel"] == ch)].sort_values("gamma")
            if f.empty:
                continue
            ax.errorbar(f["gamma"], f["n_inf"], yerr=f["sigma_n_inf"], fmt=STYLES[ch], ms=3.5, capsize=2,
                        color=COLORS[ch], mfc=SURFACE, label=CH_NAMES[ch])
        for level, text in ((0.5, "product state in the plane"), (0.75, "random pure state")):
            ax.axhline(level, color=MUTED, ls=":", lw=1.0)
            ax.text(3.0, level + 0.012, text, fontsize=7, color=MUTED, ha="right")
        ax.set(xscale="log", ylim=(0, 0.85), xlabel="γ", ylabel="n_inf = size/L at t_e (extrapolated)",
               title=NAMES[state])
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labels = [CH_NAMES[ch] for ch in SUFFIX if CH_NAMES[ch] in handles]
    fig.legend([handles[k] for k in labels], labels, loc="outside lower center", ncols=len(labels), fontsize=8)
    fig.suptitle("Operator size of ρ(t_e), extrapolated A + B/L over L = 8, 10, 12", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.savefig(DIR / "operator_size.png")
    print(f"wrote operator_size.csv, operator_size.png in {DIR}")


if __name__ == "__main__":
    main()
