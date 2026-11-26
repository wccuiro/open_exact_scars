"""
Checks of the Floquet circuit (floquet-circuit.md, `floquet` binary) against the exact evolution (`states`).

    python plot_scripts/floquet_checks.py [data_dir]          default: data/ of the repo

Runs compared (σᶻ noise, γ = 0, 0.01, 0.1, the three states):
  Floquet, current circuit   <data>/floquet/floquet_<state>_L<L>_g<γ>_dt<dt>.csv, L = 8 (dt = 0.1, 0.05, 0.025)
                             and L = 10 (dt = 0.1, 0.05)
  Floquet, old protocol      <data>/non_used_data/floquet_old_protocol/floquet_<state>_L8_g<γ>_dt<dt>.csv (dt = 0.1,
                             0.05; field and noise first, the protocol before 2026-11-25)
  exact                      scar: <data>/used_data/states_minus-x_L<L>_g<γ>.csv (output step 0.1); Néel states:
                             <data>/floquet/states_<state>_L<L>_g<γ>.csv (output step 0.01)

Signal S(t): m_⊥ (scar), s_⊥ (x-Néel), |s_z| (z-Néel); Γ_e = 1/t_e at envelope 1/e (as fit_states.py), each run on
its own output grid (= the Trotter step for Floquet). Deviation: max |S_Floquet − S_exact| over the common output
times t ≤ 3 t_e (exact t_e; the whole run if S never reaches 1/e: the γ = 0 scar), and over the whole run
(max_deviation_all: includes the late finite-size fluctuations of the Néel states at small L, which the Trotter error
dephases). Output in <data>/floquet/: floquet_checks.csv (state, L, gamma, dt, protocol, Gamma_e, Gamma_e_exact,
rel_error, max_deviation, max_deviation_all, t_ref_end = end of the exact run, revival = a finite-size revival of the
exact signal decides t_e, see revival_at_threshold; those rows are left out of panel d) and floquet_checks.png
(4 panels).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_states import crossing  # also sets the plot style of the notes
from states_data import SIGNALS, read

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
FLOQ, OLD, USED = DATA / "floquet", DATA / "non_used_data" / "floquet_old_protocol", DATA / "used_data"
STATES = {"minus-x": "scar, $m_\\perp$", "neel-x": "x-Néel, $s_\\perp$", "neel": "z-Néel, $|s_z|$"}
COLOR = {"minus-x": "#2a78d6", "neel-x": "#d6732a", "neel": "#3c9a5f"}
MARKER = {0.0: "s", 0.01: "o", 0.1: "^"}
GAMMAS = [0.0, 0.01, 0.1]
DTS = {8: [0.1, 0.05, 0.025], 10: [0.1, 0.05]}


def gname(g):
    return "0" if g == 0 else f"{g:g}"


def load(path):
    if not path.exists():
        return None
    run, _ = read(path)
    return run


def exact_run(state, l, g):
    base = USED if state == "minus-x" else FLOQ
    return load(base / f"states_{state}_L{l}_g{gname(g)}.csv")


def gamma_e(state, run):
    t = run["t"].to_numpy()
    s = SIGNALS[state](run).to_numpy()
    e = np.maximum.accumulate(s[::-1])[::-1]
    return 1 / crossing(t, e, np.exp(-1))


def revival_at_threshold(state, ref, g_ref):
    """True if a finite-size revival of the exact signal decides t_e: t_e > 1.5 × the first raw 1/e crossing, or a
    later revival (t > 2 t_e) above 0.9/e. Then Γ_e jumps between the early decay and the revival under small
    changes of the signal, and is no test of the Trotter error (z-Néel at L = 8, γ ≤ 0.01)."""
    if not np.isfinite(g_ref):
        return False
    t = ref["t"].to_numpy()
    s = SIGNALS[state](ref).to_numpy()
    below = np.nonzero(s < np.exp(-1))[0]
    t_first = t[below[0]] if len(below) else np.inf
    late = s[t > 2 / g_ref]
    return bool(1 / g_ref > 1.5 * t_first or (len(late) and late.max() > 0.9 * np.exp(-1)))


def deviation(state, run, ref, t_max=np.inf):
    """max |S_Floquet − S_exact| over the common output times t ≤ t_max."""
    a = pd.Series(SIGNALS[state](run).to_numpy(), index=run["t"].round(6))
    b = pd.Series(SIGNALS[state](ref).to_numpy(), index=ref["t"].round(6))
    common = a.index.intersection(b.index)
    common = common[common <= t_max]
    return float(np.max(np.abs(a[common] - b[common]))) if len(common) else np.nan


def table():
    rows = []
    for state in STATES:
        for l, dts in DTS.items():
            for g in GAMMAS:
                ref = exact_run(state, l, g)
                if ref is None:
                    print(f"missing exact run: {state} L = {l} γ = {g}")
                    continue
                g_ref = gamma_e(state, ref)
                window = 3 / g_ref if np.isfinite(g_ref) else np.inf
                revival = revival_at_threshold(state, ref, g_ref)
                for protocol, base in (("current", FLOQ), ("old", OLD)):
                    for dt in dts:
                        run = load(base / f"floquet_{state}_L{l}_g{gname(g)}_dt{dt:g}.csv")
                        if run is None:
                            continue
                        g_f = gamma_e(state, run)
                        rows.append(dict(state=state, L=l, gamma=g, dt=dt, protocol=protocol, Gamma_e=g_f,
                                         Gamma_e_exact=g_ref, rel_error=g_f / g_ref - 1,
                                         max_deviation=deviation(state, run, ref, window),
                                         max_deviation_all=deviation(state, run, ref),
                                         t_ref_end=float(ref["t"].iloc[-1]), revival=revival))
    return pd.DataFrame(rows)


def main():
    df = table()
    df.to_csv(FLOQ / "floquet_checks.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.3))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) scar time series, L = 8
    for g, ls in ((0.0, ":"), (0.01, "-"), (0.1, "--")):
        ref = exact_run("minus-x", 8, g)
        t_max = 60 if g == 0 else 3 / gamma_e("minus-x", ref)
        sel = ref["t"] <= t_max
        ax_a.plot(ref["t"][sel], SIGNALS["minus-x"](ref)[sel], color="#0b0b0b", ls=ls, lw=1.2,
                  label=f"exact, $\\gamma$ = {gname(g)}")
        for dt, col in ((0.1, "#86b6ef"), (0.025, "#1c5cab")):
            run = load(FLOQ / f"floquet_minus-x_L8_g{gname(g)}_dt{dt:g}.csv")
            if run is not None:
                sel = run["t"] <= t_max
                ax_a.plot(run["t"][sel], SIGNALS["minus-x"](run)[sel], color=col, ls=ls, lw=2.2, alpha=0.75,
                          label=f"Floquet $\\Delta t$ = {dt:g}" if g == 0.01 else None, zorder=1)
    ax_a.set(xlabel="$t$", ylabel="$m_\\perp$", title="(a) Scar, $L = 8$", ylim=(0, 1.05))
    ax_a.legend(fontsize=7.5, loc="center right")

    # (b) Néel signals at γ = 0.01, L = 8, t ≤ 2
    for state, ls in (("neel-x", "-"), ("neel", "--")):
        ref = exact_run(state, 8, 0.01)
        if ref is None:
            continue
        sel = ref["t"] <= 2
        ax_b.plot(ref["t"][sel], SIGNALS[state](ref)[sel], color="#0b0b0b", ls=ls, lw=1.2,
                  label=f"exact, {STATES[state]}")
        for dt, alpha in ((0.1, 0.45), (0.025, 0.9)):
            run = load(FLOQ / f"floquet_{state}_L8_g0.01_dt{dt:g}.csv")
            if run is not None:
                sel = run["t"] <= 2
                ax_b.plot(run["t"][sel], SIGNALS[state](run)[sel], color=COLOR[state], ls=ls, lw=2.2, alpha=alpha,
                          marker="o" if dt == 0.1 else None, ms=3, label=f"Floquet $\\Delta t$ = {dt:g}", zorder=1)
    ax_b.set(xlabel="$t$", ylabel="signal", title="(b) Néel states, $L = 8$, $\\gamma = 0.01$")
    ax_b.legend(fontsize=7, loc="upper right")

    # (c) largest deviation against dt; (d) error of Γ_e against dt (without the runs whose t_e a revival decides)
    cur = df[df.protocol == "current"]
    for state in STATES:
        for g in GAMMAS:
            for l, fill in ((8, True), (10, False)):
                sub = cur[(cur.state == state) & (cur.gamma == g) & (cur.L == l)].sort_values("dt")
                if sub.empty or (state == "minus-x" and g == 0):
                    continue
                kw = dict(color=COLOR[state], marker=MARKER[g], ms=5, lw=1.0,
                          mfc=COLOR[state] if fill else "none", ls="-" if fill else ":")
                ax_c.plot(sub.dt, sub.max_deviation, **kw)
                if not sub.revival.any():
                    ax_d.plot(sub.dt, sub.rel_error.abs(), **kw)
    old = df[df.protocol == "old"]
    for state in STATES:
        for g in GAMMAS:
            sub = old[(old.state == state) & (old.gamma == g)].sort_values("dt")
            if not sub.empty and not (state == "minus-x" and g == 0) and not sub.revival.any():
                ax_d.plot(sub.dt, sub.rel_error.abs(), color=COLOR[state], marker="x", ms=7, lw=0.8, ls="--",
                          alpha=0.7)
    dts = np.array([0.025, 0.1])
    for ax, y0 in ((ax_c, 0.05), (ax_d, 0.05)):
        ax.plot(dts, y0 * dts / 0.1, color="#898781", lw=0.8, ls="-.")
        ax.text(0.03, y0 * 0.3 * 1.15, "$\\propto \\Delta t$", color="#898781", fontsize=8)
        ax.set(xscale="log", yscale="log", xlabel="$\\Delta t$")
        ax.set_xticks([0.025, 0.05, 0.1], ["0.025", "0.05", "0.1"])
        ax.minorticks_off()
    ax_c.set(ylabel="max$_{t \\leq 3t_e}$ |signal − exact|", title="(c) Largest deviation of the signal")
    ax_d.set(ylabel="$|\\Gamma_e/\\Gamma_e^{\\rm exact} - 1|$", title="(d) Error of the decay rate $\\Gamma_e$")
    handles = [plt.Line2D([], [], color=COLOR[s], lw=1.5, label=STATES[s]) for s in STATES]
    handles += [plt.Line2D([], [], color="#52514e", marker=MARKER[g], ls="none", label=f"$\\gamma$ = {gname(g)}")
                for g in GAMMAS]
    handles += [plt.Line2D([], [], color="#52514e", marker="o", ls="-", label="$L$ = 8"),
                plt.Line2D([], [], color="#52514e", marker="o", mfc="none", ls=":", label="$L$ = 10"),
                plt.Line2D([], [], color="#52514e", marker="x", ms=7, ls="--", label="old protocol, $L$ = 8")]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=9, frameon=False, columnspacing=1.0,
               handlelength=1.8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FLOQ / "floquet_checks.png")
    print(f"-> {FLOQ / 'floquet_checks.png'}")


if __name__ == "__main__":
    main()
