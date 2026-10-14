"""
Decay rates of the three probes of scar-vs-thermal-decay-plan.md, exact evolution with random c.

  P1  scar:    m_⊥(t) from ρ₀ = |−⟩⟨−|^⊗L
  P2  energy:  C(t) = Tr[H̃ e^{𝓛†t}(H̃)] / Tr H̃²,  H̃ = H − h Sᶻ P_tower
  P3  thermal: |C⁺(t)| for O⁺ = Σ_j (−1)^j σ⁺_j (k = π, even L)

Fit: straight line in ln(signal) against t, slope −Γ, σ_Γ the standard error of the slope.
  P1, P2: window from t = 5 until the signal first drops below 1e-3 (or the end of the run).
  P3:     the thermal decay is over by t ≈ 1, so the window is [0, t_end], with t_end the first time
          |C⁺| at γ = 0 drops below twice its noise floor (mean of |C⁺| for t > 20 at γ = 0).
          The same window is used at both γ, and ΔΓ = Γ(γ) − Γ₀ is the slope of
          −ln(|C⁺_γ| / |C⁺_0|) on it. That window grows with L (the floor drops), so ΔΓ is also
          printed on common windows [0, T], T in P3_COMMON.
Step 2 (printed): Γ(L) fitted with a power law Γ = a L^b (b with its standard error) and with an
          exponential Γ = a e^(cL), with the rms relative residuals, as in fit_decay.py.

Input:  data/probe_P1.csv, probe_P2.csv, probe_P3.csv, probe_free.csv
        (cd rust && cargo run --release --bin probes)
Output: data/rates.csv               probe, L, gamma, k, Gamma, sigma_Gamma, t_fit_min, t_fit_max
        data/probe_rates.png         Γ/γ against L for P1, P2 and ΔΓ/γ for P3
        data/probe_timeseries.png    the three signals with their fit windows
        data/probe_free.png          sanity check: H_I = 0 against e^(−2γt)
        data/probe_scaling.png       Γ/γ against L (log-log) with the power-law fits, scar and non-scar
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T_START = 5.0     # P1, P2: start of the fit window (after the initial transient)
SIGNAL_MIN = 1e-3  # P1, P2: end of the fit window
FLOOR_T = 20.0     # P3: the noise floor is the mean of |C⁺| at γ = 0 after this time
P3_COMMON = (0.5, 1.0)  # P3: common fit windows [0, T] for ΔΓ

DATA = Path(__file__).resolve().parent.parent / "data"
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # small L light, large L dark
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baa9b"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def line_fit(t, y):
    """Slope of y(t), its standard error, and the intercept."""
    (slope, intercept), cov = np.polyfit(t, y, 1, cov=True)
    return slope, np.sqrt(cov[0, 0]), intercept


def window_p12(t, s):
    below = np.nonzero((s < SIGNAL_MIN) & (t >= T_START))[0]
    end = t[below[0] - 1] if len(below) else t[-1]
    return T_START, end


def main():
    p1, p2, p3 = (pd.read_csv(DATA / f"probe_{p}.csv") for p in ("P1", "P2", "P3"))
    free = pd.read_csv(DATA / "probe_free.csv")
    gamma = max(p1["gamma"].unique())
    sizes = sorted(p1["L"].unique())
    colors = dict(zip(sizes, RAMP[-len(sizes):]))
    rows, fits = [], {}

    for probe, df, col in (("P1", p1, "m_perp"), ("P2", p2, "C")):
        for l in sizes:
            for g in (0.0, gamma):
                run = df[(df["L"] == l) & (df["gamma"] == g)]
                t, s = run["t"].to_numpy(), run[col].to_numpy()
                t0, t1 = window_p12(t, s)
                sel = (t >= t0) & (t <= t1)
                slope, err, intercept = line_fit(t[sel], np.log(s[sel]))
                rows.append((probe, l, g, 0.0, -slope, err, t0, t1))
                fits[probe, l, g] = (t0, t1, slope, intercept)

    for l in sorted(p3["L"].unique()):
        r0 = p3[(p3["L"] == l) & (p3["gamma"] == 0.0)]
        rg = p3[(p3["L"] == l) & (p3["gamma"] == gamma)]
        t, c0, cg = r0["t"].to_numpy(), r0["C_abs"].to_numpy(), rg["C_abs"].to_numpy()
        floor = c0[t > FLOOR_T].mean()
        end = np.nonzero(c0 < 2 * floor)[0][0]
        sel = slice(0, end)  # points before the first one under the floor
        t0, t1 = t[0], t[end - 1]
        k = r0["k"].iloc[0]
        for g, c in ((0.0, c0), (gamma, cg)):
            slope, err, intercept = line_fit(t[sel], np.log(c[sel]))
            rows.append(("P3", l, g, k, -slope, err, t0, t1))
            fits["P3", l, g] = (t0, t1, slope, intercept)
        slope, err, _ = line_fit(t[sel], np.log(cg[sel] / c0[sel]))
        rows.append(("P3_delta", l, gamma, k, -slope, err, t0, t1))
        print(f"P3 L = {l}: floor of |C+| at gamma = 0 is {floor:.3f}, fit window [{t0:g}, {t1:g}]")

    rates = pd.DataFrame(rows, columns=["probe", "L", "gamma", "k", "Gamma", "sigma_Gamma", "t_fit_min", "t_fit_max"])
    rates.to_csv(DATA / "rates.csv", index=False)
    show = rates[rates["gamma"] == gamma].assign(**{"Gamma/gamma": lambda d: d["Gamma"] / gamma,
                                                    "sigma/gamma": lambda d: d["sigma_Gamma"] / gamma})
    print(show[["probe", "L", "Gamma/gamma", "sigma/gamma", "t_fit_min", "t_fit_max"]].to_string(index=False, float_format="%.3f"))
    print("gamma = 0:", rates[rates["gamma"] == 0][["probe", "L", "Gamma"]].to_string(index=False, float_format="%.2e"))

    # P3 on common windows, and the L-scaling of every rate
    scaling = {}
    for name, probe in (("P1 scar", "P1"), ("P2 energy", "P2"), ("P3 thermal ΔΓ, auto window", "P3_delta")):
        d = rates[(rates["probe"] == probe) & (rates["gamma"] == gamma)]
        scaling[name] = (d["L"].to_numpy(), d["Gamma"].to_numpy())
    early = [line_fit(*(lambda d: (d["t"], np.log(d["m_perp"])))(
        p1[(p1["L"] == l) & (p1["gamma"] == gamma) & (p1["t"] >= 5) & (p1["t"] <= 60)]))[0] for l in sizes]
    scaling["P1 scar, [5, 60]"] = (np.array(sizes), -np.array(early))
    for T in P3_COMMON:
        ls, dg = [], []
        for l in sorted(p3["L"].unique()):
            r0 = p3[(p3["L"] == l) & (p3["gamma"] == 0.0)]
            rg = p3[(p3["L"] == l) & (p3["gamma"] == gamma)]
            sel = r0["t"].to_numpy() <= T + 1e-9
            slope, _, _ = line_fit(r0["t"].to_numpy()[sel], np.log(rg["C_abs"].to_numpy()[sel] / r0["C_abs"].to_numpy()[sel]))
            ls.append(l)
            dg.append(-slope)
        scaling[f"P3 thermal ΔΓ, [0, {T:g}]"] = (np.array(ls), np.array(dg))
    print("L-scaling of the rates (Γ/γ):")
    power = {}
    for name, (ls, g) in scaling.items():
        (b, ln_a), cov = np.polyfit(np.log(ls), np.log(g / gamma), 1, cov=True)
        c, ln_c = np.polyfit(ls, np.log(g / gamma), 1)
        res_pow = np.sqrt(np.mean((np.exp(ln_a) * ls**b / (g / gamma) - 1) ** 2))
        res_exp = np.sqrt(np.mean((np.exp(ln_c + c * ls) / (g / gamma) - 1) ** 2))
        print(f"  {name:28s} {np.round(g / gamma, 2)}  power law {np.exp(ln_a):.2f} L^({b:.2f} ± {np.sqrt(cov[0, 0]):.2f}), "
              f"rms {100 * res_pow:.1f} %;  exponential rms {100 * res_exp:.1f} %")
        power[name] = (ls, g / gamma, np.exp(ln_a), b, np.sqrt(cov[0, 0]))
    scaling_figure(power, gamma)

    rate_figure(rates, gamma)
    timeseries_figure(p1, p2, p3, fits, gamma, colors)
    free_figure(free, gamma, colors)


def rate_figure(rates, gamma):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    series = (("P1", "P1 scar, Γ/γ", BLUE, "o"), ("P2", "P2 energy H̃, Γ/γ", ORANGE, "s"),
              ("P3_delta", "P3 thermal k = π, ΔΓ/γ", AQUA, "D"))
    for probe, label, color, marker in series:
        d = rates[(rates["probe"] == probe) & (rates["gamma"] == gamma)]
        ax.errorbar(d["L"], d["Gamma"] / gamma, yerr=d["sigma_Gamma"] / gamma, color=color, marker=marker,
                    ms=6, lw=1.4, capsize=2, label=label)
    ls = np.arange(rates["L"].min(), rates["L"].max() + 1)
    ax.plot(ls, ls - 1, color=MUTED, ls=":", lw=1.2, label="L − 1 (initial tower-weight loss)")
    ax.axhline(2, color=MUTED, ls="--", lw=1.0, label="2 (single spin)")
    ax.set_xticks(ls)
    ax.set_xlabel("L")
    ax.set_ylabel("rate / γ")
    ax.set_title(f"Decay rates, γ = {gamma:g}")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(DATA / "probe_rates.png")
    print(f"wrote {DATA / 'probe_rates.png'}")


def scaling_figure(power, gamma):
    """Log-log Γ/γ against L with the power-law fits; scar left, non-scar right."""
    panels = (
        ("Scar: m⊥ from |−⟩^⊗L", (("P1 scar, [5, 60]", "scar, window [5, 60]", BLUE, "o", "-", True),
                                  ("P1 scar", "scar, window [5, until 1e-3]", BLUE, "o", "--", False))),
        ("Non-scar", (("P2 energy", "energy H̃ (P2)", ORANGE, "s", "-", True),
                      ("P3 thermal ΔΓ, [0, 0.5]", "thermal k = π (P3), ΔΓ on [0, 0.5]", AQUA, "D", "-", True),
                      ("P3 thermal ΔΓ, [0, 1]", "thermal k = π (P3), ΔΓ on [0, 1]", AQUA, "D", "--", False))),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    all_l = np.concatenate([v[0] for v in power.values()])
    fine = np.linspace(all_l.min() - 0.2, all_l.max() + 0.2, 100)
    for ax, (title, series) in zip(axes, panels):
        for name, label, color, marker, ls, filled in series:
            l, g, a, b, db = power[name]
            ax.plot(fine, a * fine**b, color=color, ls=ls, lw=1.2)
            ax.plot(l, g, ls="none", marker=marker, ms=7, color=color, mfc=color if filled else SURFACE, mew=1.6,
                    label=f"{label}:  Γ/γ = {a:.2f} L^({b:.2f} ± {db:.2f})")
        ax.axhline(2, color=MUTED, ls=":", lw=1.0, label="2γ: free spin, e^(−2γt)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(range(int(all_l.min()), int(all_l.max()) + 1))
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(l) for l in range(int(all_l.min()), int(all_l.max()) + 1)])
        ax.set_yticks([1, 2, 3, 4, 6, 8])
        ax.set_yticks([], minor=True)
        ax.set_yticklabels(["1", "2", "3", "4", "6", "8"])
        ax.set_ylim(1, 9)
        ax.set_xlabel("L")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel(f"decay rate Γ / γ   (γ = {gamma:g})")
    fig.tight_layout()
    fig.savefig(DATA / "probe_scaling.png")
    print(f"wrote {DATA / 'probe_scaling.png'}")


def timeseries_figure(p1, p2, p3, fits, gamma, colors):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panels = (("P1", p1, "m_perp", "P1 scar: m⊥"), ("P2", p2, "C", "P2 energy: C(t) of H̃"), ("P3", p3, "C_abs", "P3 thermal: |C⁺|, k = π"))
    for ax, (probe, df, col, title) in zip(axes, panels):
        for l in sorted(df["L"].unique()):
            for g, ls, lw in ((gamma, "-", 1.3), (0.0, "--", 0.8)):
                run = df[(df["L"] == l) & (df["gamma"] == g)]
                ax.plot(run["t"], run[col], color=colors[l], ls=ls, lw=lw, label=f"L = {l}" if g else None)
            t0, t1, slope, intercept = fits[probe, l, gamma]
            tt = np.linspace(t0, t1, 50)
            ax.plot(tt, np.exp(intercept + slope * tt), color=INK, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel("t")
        ax.set_title(title)
        if probe == "P3":
            ax.set_xlim(0, 10)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.text(0.5, -0.02, f"solid: γ = {gamma:g}, dashed: γ = 0, dotted: fit", ha="center", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(DATA / "probe_timeseries.png")
    print(f"wrote {DATA / 'probe_timeseries.png'}")


def free_figure(free, gamma, colors):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for l in sorted(free["L"].unique()):
        run = free[free["L"] == l]
        ax.plot(run["t"], run["m_perp"], color=colors[l], label=f"L = {l}")
    ax.plot(run["t"], run["exact"], color=INK, ls=":", lw=1.2, label="e^(−2γt)")
    dev = (free["m_perp"] - free["exact"]).abs().max()
    ax.set_title(f"H_I = 0, γ = {gamma:g}: max deviation {dev:.1e}")
    ax.set_xlabel("t")
    ax.set_ylabel("m⊥")
    ax.legend(fontsize=8)
    fig.savefig(DATA / "probe_free.png")
    print(f"wrote {DATA / 'probe_free.png'}")


if __name__ == "__main__":
    main()
