"""
Thermodynamic decay rates for every noise channel: Γ_e against 1/L, extrapolated to 1/L = 0, as fit_states_inverse_L.py
does for σᶻ dephasing.

Channels (rust/src/noise.rs): z, x, y (σ^α dephasing), dep (depolarizing), thermal (n̄ = 1). The decay channel is left
out: the scar m_⊥ under decay rises again after its first 1/e crossing (e.g. L = 8, γ = 0.1: 0.30 → 0.37), and the
runs stop at 1.5 × that first crossing, so their Γ_e depends on where the run ended.

Γ_e = 1/t_e, t_e the first time the envelope max_{t' ≥ t} S(t') drops below 1/e (`rates` in fit_states.py); signal S:
m_⊥ (scar), s_⊥ (x-Néel), |s_z| (z-Néel).

Runs (one per (state, channel, L, γ), exact preferred over trajectories):
  lean/   lean-data-plan.md step B (2026-10-26/30): exact L = 8, 10 (obelix), 12 (Leonardo), every channel, γ = 0.005 … 20;
          diffusive trajectories L = 16 (obelix), 20 (Leonardo), z, x, y, dep, γ = 0.01 … 2; scar σᶻ L = 22 (B1).
  σᶻ only: also every run of data/used_data (states_data.load/select: exact L ≤ 14, trajectories L = 12 … 20), which
          take precedence over a lean run of the same (state, L, γ) (they ran to 1e-3, the lean runs stop at 1.5 t_e).
Only realization seed 1 (the lean _seed<N> files are other realizations).

Fit as in fit_states_inverse_L.py: γ ≥ GAMMA_FIT_MIN, linear Γ_e = A + B/L on L ≥ LIN_MIN, at least MIN_SIZES sizes;
z-Néel without L = 6, 8 at γ ≤ 0.02 and x-Néel without L = 4, 6 at γ ≤ 0.05 (finite-size revivals). Γ_∞ = A.
Error: σᶻ as before (max of the fit error and |A − quadratic intercept on L ≥ 6|; z-Néel |A − Γ_e(L_max)|); other
channels (no L = 6) max of the fit error and |A − A on L ≥ LIN_MIN + 2|.

Input:  data/used_data/ (or the first argument) and its lean/ subdirectory
Output: in data/used_data/
        channels_rates.csv               state, channel, method, L, gamma, Gamma_e (the runs used)
        channels_inverse_L.csv           state, channel, gamma, sizes, A_lin, sigma_A_lin, B_lin, Gamma_inf, sigma_Gamma_inf
        channels_inverse_L.png           Γ_e against 1/L with the fits, rows = channels, columns = states
        channels_thermodynamic_rate.png  Γ_∞ against γ, one panel per channel, three states
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import fit_states_inverse_L as fsi
from fit_states import rates
from states_data import load, read, select

CHANNELS = ("z", "x", "y", "dep", "thermal")
CH_NAMES = {"z": "σᶻ dephasing", "x": "σˣ dephasing", "y": "σʸ dephasing", "dep": "depolarizing",
            "thermal": "thermal, n̄ = 1"}
GAMMA_FIT_MIN, LIN_MIN, MIN_SIZES = 0.01, 8, 3
NAME = re.compile(r"states_(traj_)?(minus-x|neel-x|neel)_L(\d+)_g([0-9.]+?)(?:_ch(x|y|dep|decay|thermal))?(?:_n1)?"
                  r"(_seed\d+)?\.csv$")

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
LEAN = DIR / "lean"


def collect():
    rows = []
    for f in sorted(LEAN.glob("states_*.csv")):
        m = NAME.match(f.name)
        if m is None or m.group(6) or (m.group(5) or "z") not in CHANNELS:
            continue
        run, _ = read(f)
        if len(run) < 2:
            continue
        rows.append((m.group(2), m.group(5) or "z", "traj" if m.group(1) else "exact", int(m.group(3)),
                     float(m.group(4)), rates(m.group(2), run)[0], "lean"))
    for (state, l, g), (method, run, _) in select(load(DIR)).items():
        rows.append((state, "z", method, l, g, rates(state, run)[0], "used_data"))
    d = pd.DataFrame(rows, columns=["state", "channel", "method", "L", "gamma", "Gamma_e", "source"])
    # one run per (state, channel, L, γ): used_data first (σᶻ), then exact, then trajectories
    d["rank"] = (d["source"] == "lean") * 2 + (d["method"] != "exact")
    both = d[d["channel"] == "z"].pivot_table(index=["state", "L", "gamma", "method"], columns="source",
                                              values="Gamma_e").dropna()
    d = d.sort_values("rank").drop_duplicates(["state", "channel", "L", "gamma"]).drop(columns="rank")
    return d.sort_values(["state", "channel", "L", "gamma"]), both


def extrapolate(rates_):
    rows = []
    for (state, ch, g), dg in rates_[rates_["gamma"] >= GAMMA_FIT_MIN].groupby(["state", "channel", "gamma"]):
        dg = dg.dropna(subset=["Gamma_e"]).set_index("L")["Gamma_e"]
        lin = fsi.sizes_used(state, g, sorted(dg.index), LIN_MIN)
        if len(lin) < MIN_SIZES:
            continue
        x, y = 1 / np.array(lin), dg[lin].to_numpy()
        p1, s1 = fsi.fit(x, y, 1)
        quad = fsi.sizes_used(state, g, sorted(dg.index), fsi.QUAD_MIN)
        if ch == "z" and state != "neel" and min(quad) < LIN_MIN and len(quad) >= 4:
            spread = abs(p1[-1] - fsi.fit(1 / np.array(quad), dg[quad].to_numpy(), 2)[0][-1])
        elif ch == "z" and state == "neel":
            spread = abs(p1[-1] - dg[max(lin)])
        else:
            upper = [l for l in lin if l >= LIN_MIN + 2]
            spread = (abs(p1[-1] - np.polyfit(1 / np.array(upper), dg[upper].to_numpy(), 1)[-1]) if len(upper) >= 2
                      else abs(p1[-1] - dg[max(lin)]))
        rows.append((state, ch, g, " ".join(map(str, lin)), p1[-1], s1, p1[0], p1[-1], np.nanmax([s1, spread])))
    return pd.DataFrame(rows, columns=["state", "channel", "gamma", "sizes", "A_lin", "sigma_A_lin", "B_lin",
                                       "Gamma_inf", "sigma_Gamma_inf"])


OTHER = [ch for ch in CHANNELS if ch != "z"]  # σᶻ has its own figures (fit_states_inverse_L.py)


def inverse_l_figure(rates_, table):
    """One figure per state, 2 × 2 panels: the four channels other than σᶻ (notes_layout: at most 4 panels)."""
    gammas = sorted(table["gamma"].unique())
    blues = LinearSegmentedColormap.from_list("blues", fsi.RAMP)
    colors = {g: blues(i / max(len(gammas) - 1, 1)) for i, g in enumerate(gammas)}
    xs = np.linspace(0, 1 / 4, 100)
    for state in fsi.STATES:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), squeeze=False, layout="constrained")
        for ax, ch in zip(axes.flat, OTHER):
            for g in gammas:
                t = table[(table["state"] == state) & (table["channel"] == ch) & (table["gamma"] == g)]
                if t.empty:
                    continue
                t = t.iloc[0]
                d = rates_[(rates_["state"] == state) & (rates_["channel"] == ch) & (rates_["gamma"] == g)
                           ].sort_values("L")
                used = d["L"].isin([int(l) for l in t["sizes"].split()]).to_numpy()
                x, y = 1 / d["L"].to_numpy(), d["Gamma_e"].to_numpy()
                traj = (d["method"] != "exact").to_numpy()
                for sel, marker in ((~traj, "o"), (traj, "s")):
                    ax.plot(x[used & sel], y[used & sel], marker, ms=3.5, color=colors[g],
                            label=f"γ = {g:g}" if marker == "o" else None)
                    ax.plot(x[~used & sel], y[~used & sel], marker, ms=3.5, mfc="none", mew=0.8, color=colors[g])
                xf = xs[xs <= x[used].max()]
                ax.plot(xf, t["A_lin"] + t["B_lin"] * xf, "--", lw=0.9, color=colors[g])
            ax.set(yscale="log", xlim=(0, 0.26), xlabel="1/L", ylabel="Γ_e", title=CH_NAMES[ch])
        handles, labels = {}, {}
        for ax in axes.flat:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        order = sorted(handles, key=lambda s: float(s.split("= ")[1]))
        fig.legend([handles[k] for k in order], order, fontsize=7.5, loc="outside right center")
        fig.suptitle(f"{fsi.NAMES[state]}: Γ_e against 1/L, channels other than σᶻ", x=0.01, ha="left",
                     fontsize=10, fontweight="bold")
        out = DIR / f"channels_inverse_L_{state}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


def thermodynamic_figure(table):
    """channels_thermodynamic_rate.png: Γ_∞ per channel other than σᶻ (2 × 2); channels_scar.png: the scar under
    every channel, with the σᶻ fit on L ≥ 14 and the single-kick prediction (γ ≥ 0.01 only)."""
    g = np.geomspace(table["gamma"].min(), table["gamma"].max(), 100)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), layout="constrained")
    for ax, ch in zip(axes.flat, OTHER):
        ax.plot(g, 2 * g, color=fsi.MUTED, ls=":", lw=1, label="2γ")
        for state in fsi.STATES:
            d = table[(table["state"] == state) & (table["channel"] == ch)].sort_values("gamma")
            ax.errorbar(d["gamma"], d["Gamma_inf"], yerr=d["sigma_Gamma_inf"], fmt="-o", ms=3.5, capsize=2,
                        color=fsi.SERIES[state], label=fsi.NAMES[state])
        ax.set(xscale="log", yscale="log", xlabel="γ", ylabel="Γ", title=CH_NAMES[ch])
    axes.flat[0].legend(fontsize=7.5)
    fig.suptitle("Thermodynamic decay rate Γ_∞ (A + B/L, 1/L → 0) per noise channel", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.savefig(DIR / "channels_thermodynamic_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    styles = dict(zip(CHANNELS, ["-o", "--s", "-.^", ":D", "-v"]))
    colors = dict(zip(CHANNELS, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc7", "#b8860b"]))
    ax.plot(g, 2 * g, color=fsi.MUTED, ls=":", lw=1, label="2γ")
    for ch in CHANNELS:
        d = table[(table["state"] == "minus-x") & (table["channel"] == ch)].sort_values("gamma")
        ax.errorbar(d["gamma"], d["Gamma_inf"], yerr=d["sigma_Gamma_inf"], fmt=styles[ch], ms=3.5, capsize=2,
                    color=colors[ch], mfc=fsi.SURFACE, label=CH_NAMES[ch])
    large, pred = DIR / "states_inverse_L_large.csv", DIR / "spreading_prediction.csv"
    if large.exists():
        lg = pd.read_csv(large)
        lg = lg[lg["gamma"] >= GAMMA_FIT_MIN * (1 - 1e-9)]
        ax.errorbar(lg["gamma"], lg["A"], yerr=lg["sigma_A"], fmt="s", ms=5, capsize=2, mfc="none",
                    color=colors["z"], label="σᶻ, A + B/L on L ≥ 14", zorder=3)
    if pred.exists():
        p = pd.read_csv(pred)
        p = p[p["gamma"] >= GAMMA_FIT_MIN * (1 - 1e-9)]
        lo, hi = (p[p["c"] == c].sort_values("gamma") for c in sorted(p["c"].unique())[:2])
        ax.fill_between(lo["gamma"], lo["Gamma_pred"], hi["Gamma_pred"], color=fsi.SERIES["neel"], alpha=0.35,
                        lw=0, label="σᶻ, single-kick prediction (ballistic D̄_∞)")
    ax.set(xscale="log", yscale="log", xlabel="γ", ylabel="Γ_∞", title="Scar m_⊥ under every channel")
    ax.legend(fontsize=7.5)
    fig.savefig(DIR / "channels_scar.png")
    plt.close(fig)


def main():
    rates_, both = collect()
    rates_.to_csv(DIR / "channels_rates.csv", index=False)
    rel = (both["lean"] / both["used_data"] - 1).abs()
    print(f"σᶻ runs in both lean and used_data: {len(rel)}, max relative difference of Γ_e {rel.max():.1e}")
    table = extrapolate(rates_)
    table.to_csv(DIR / "channels_inverse_L.csv", index=False)
    wide = table.assign(G=table["Gamma_inf"].map("{:.3g}".format) + " ± " + table["sigma_Gamma_inf"].map(
        "{:.2g}".format)).pivot_table(index=["state", "gamma"], columns="channel", values="G", aggfunc="first")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(wide[[c for c in CHANNELS if c in wide.columns]].to_string())
    inverse_l_figure(rates_, table)
    thermodynamic_figure(table)
    print(f"wrote channels_rates.csv, channels_inverse_L.csv, channels_inverse_L_<state>.png, channels_thermodynamic_rate.png, channels_scar.png "
          f"in {DIR}")


if __name__ == "__main__":
    main()
