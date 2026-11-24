"""
Thermodynamic decay rates per probe state and their histograms (depolarizing noise; runs of scripts/run_state_hist.py,
`leonardo-thermo` grid: every state at L = 6, 8, 10 with exact ρ and L = 12, 14, 16 with trajectories).

    ~/anaconda3/bin/python plot_scripts/plot_state_rate_thermo.py [data/state_hist]

Γ_e(L) of every run from state_hist_data.summarize. Thermodynamic rates (the rule of CLAUDE.md, "Decay rates"):
  - scar class: each coherent state cos θ is the same state at every L. Γ_∞ = A of A + B/L, unweighted, on L ≥ 8;
    error = max(fit error, |A − A_quad|), A_quad the intercept of A + B/L + C/L² on L ≥ 6 (only with ≥ 4 sizes).
    Finite-size revivals: a(t) oscillates with the tower period T = 2π/h, and where its first plateau or revival is
    close to 1/e, t_e moves by a large part of T from one size to the next. A step |t_e(L) − t_e(L + 2)| > JUMP · T
    (smooth size drifts here are ≤ 0.36 T, the revival jumps ≥ 0.45 T) marks the sizes up to L as another revival
    branch: they are left out (column dropped), as the small Néel sizes in CLAUDE.md. Γ_∞ needs ≥ 3 remaining sizes
    with L ≥ 8 and no bound (no 1/e crossing) among them. Without L = 6 there is no quadratic check (too short a
    lever arm); the error is then max(fit error, |A − Γ_e(L_max)|), the z-Néel rule of CLAUDE.md.
  - bulk class: different random product states at each L, labelled by their energy density ε = E/L (random product
    states at the same ε are equivalent as L → ∞; their spread at fixed ε is finite-size). Joint unweighted fit of all
    states on L ≥ 8: Γ_e = a(ε) + b(ε)/L with a quadratic and b linear in ε; Γ_∞(ε) = a(ε); error = max(statistical
    error of a(ε) from the fit covariance, |a − a_quad|), a_quad from the same fit plus c/L² on L ≥ 6. A size with a
    bound among its runs (revivals: L = 6 at γ = 0) is left out of both fits. Check: states
    binned in ε (width 0.1), bin means against L, A + B/L per bin with the scar rule. Each of the 100 states of L = 16
    counts in the histogram at Γ_∞(its ε). γ = 0 (pure runs) the same way: thermalization rate Γ_th,∞(ε).
Outputs (in the data directory):
  state_rate_thermo_scar.csv       γ, cos θ, E/L, Γ_e and σ at each L, A, B, fit error, A_quad, Γ_∞, σ_∞
  state_rate_thermo_bulk.csv       γ, a0, a1, a2, b0, b1 (and the check a0q, a1q, a2q, b0q, b1q, c0q), runs, residual sd
  state_rate_thermo_bins.csv       γ, ε bin, mean ε, bin mean Γ_e, standard error and states per L, A, error, a(ε̄)
  state_rate_histogram_thermo.png  one panel per γ: histograms of Γ_∞ of both classes (bars, same bins; counts on a
                                   log axis), their Γ_e at L = 16 as outlines
  state_rate_inverse_L_scar.png    scar Γ_e against 1/L per γ with the A + B/L fits
  state_rate_inverse_L_bulk.png    bulk bin means against 1/L (γ = 0 and every γ) with the joint fit at the bin ε̄
  state_rate_thermo_vs_energy.png  Γ_∞ against E/L, both classes
  state_rate_thermo_vs_gamma.png   Γ_∞ against γ, both classes, log-log, with 4γ/3, √γ and γ guides
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_hist_data as shd  # noqa: E402

GAMMAS = [0.1, 0.05, 0.03]
SIZES = [6, 8, 10, 12, 14, 16]
L_FIT = 8
L_HIST = 16
JUMP = 0.4  # revival-branch step of t_e between neighbouring sizes, in units of the period 2π/h
EDGES = np.array([-0.25, -0.15, -0.05, 0.05, 0.15, 0.25])
COLORS = {0.1: "#2a78d6", 0.05: "#eb6834", 0.03: "#1baf7a", 0.0: "#52514e"}  # dataviz reference slots 1-3, ink
BULK_COLOR, SCAR_COLOR, GREY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def line_fit(l, g):
    """(A, B, fit error of A) of A + B/L, unweighted (error from the residual scatter)."""
    x = 1.0 / np.asarray(l, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, g, rcond=None)[0]
    dof = len(x) - 2
    s2 = float(np.sum((g - X @ beta) ** 2) / dof) if dof > 0 else float("nan")
    return beta[0], beta[1], math.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])


def extrapolate(l, g):
    """(A, B, fit error, A_quad, error) with the rule of the module docs."""
    l, g = np.asarray(l), np.asarray(g, float)
    m = l >= L_FIT
    a, b, e = line_fit(l[m], g[m])
    if l.min() <= 6 and len(l) >= 4:
        aq = np.polyfit(1.0 / l, g, 2)[-1]  # quadratic on all given sizes, L ≥ 6
        return a, b, e, aq, max(e, abs(a - aq))
    return a, b, e, float("nan"), max(e, abs(a - g[np.argmax(l)]))


def scar_thermo(df):
    rows = []
    sc = df[(df.family == "coherent") & (df.method != "pure")]
    for g in GAMMAS:
        for c in sorted(sc[sc.gamma == g].cos_theta.unique()):
            r = sc[(sc.gamma == g) & (sc.cos_theta == c)].set_index("L").sort_index()
            row = dict(gamma=g, cos_theta=c, E_per_L=0.5 * shd.H * c)
            for l in SIZES:
                row[f"G_L{l}"] = r.Gamma_e.get(l, np.nan)
                row[f"s_L{l}"] = r.sigma_Gamma.get(l, np.nan)
            sizes = [l for l in SIZES if l in r.index]
            cut = 0
            for l1, l2 in zip(sizes[:-1], sizes[1:]):
                if abs(r.t_e[l1] - r.t_e[l2]) > JUMP * shd.PERIOD:
                    cut = l1
            keep = [l for l in sizes if l > cut]
            ok = sum(l >= L_FIT for l in keep) >= 3 and not r.bound[keep].any()
            a, b, e, aq, err = extrapolate(np.array(keep), r.Gamma_e[keep].to_numpy()) if ok else (np.nan,) * 5
            row.update(A=a, B=b, fit_err=e, A_quad=aq, Gamma_inf=a, sigma_inf=err, complete=ok,
                       dropped=" ".join(str(l) for l in sizes if l <= cut))
            rows.append(row)
    return pd.DataFrame(rows)


def bulk_runs(df, g):
    """Bulk runs of γ (pure runs for γ = 0), without the sizes that have a bound among their runs."""
    d = df[(df.family == "bulk") & ((df.method == "pure") if g == 0 else ((df.gamma == g) & (df.method != "pure")))]
    bad = set(d[d.bound].L)
    return d[~d.L.isin(bad)], sorted(bad)


def design(eps, l, quad=False):
    cols = [np.ones_like(eps), eps, eps ** 2, 1 / l, eps / l] + ([1 / l ** 2] if quad else [])
    return np.column_stack(cols)


def bulk_fit(d):
    eps, l, y = d.E_per_L.to_numpy(), d.L.to_numpy(float), d.Gamma_e.to_numpy()
    m = l >= L_FIT
    X = design(eps[m], l[m])
    beta = np.linalg.lstsq(X, y[m], rcond=None)[0]
    res = y[m] - X @ beta
    s2 = float(res @ res / (m.sum() - X.shape[1]))
    cov = s2 * np.linalg.inv(X.T @ X)
    betaq = np.linalg.lstsq(design(eps, l, quad=True), y, rcond=None)[0]
    return dict(beta=beta, cov=cov, betaq=betaq, n=int(m.sum()), resid_sd=math.sqrt(s2))


def a_of(fit, e):
    """(a(ε), error) of the joint fit."""
    gv = np.array([1.0, e, e * e, 0.0, 0.0])
    a = float(gv @ fit["beta"])
    aq = float(np.append(gv, 0.0) @ fit["betaq"])
    return a, max(math.sqrt(float(gv @ fit["cov"] @ gv)), abs(a - aq))


def fit_at(fit, e, l):
    return float(design(np.array([e]), np.array([float(l)]))[0] @ fit["beta"])


def bins_check(d, fit, g):
    rows = []
    for k in range(len(EDGES) - 1):
        lo, hi = EDGES[k], EDGES[k + 1]
        sel = d[(d.E_per_L >= lo) & ((d.E_per_L < hi) | ((k == len(EDGES) - 2) & (d.E_per_L <= hi)))]
        per = sel.groupby("L").Gamma_e.agg(["mean", "std", "size"])
        per = per[per["size"] >= 2]
        row = dict(gamma=g, eps_lo=lo, eps_hi=hi, eps_mean=sel[sel.L >= L_FIT].E_per_L.mean())
        for l in SIZES:
            row[f"mean_L{l}"] = per["mean"].get(l, np.nan)
            row[f"se_L{l}"] = per["std"].get(l, np.nan) / math.sqrt(per["size"].get(l, np.nan))
            row[f"n_L{l}"] = int(per["size"].get(l, 0))
        if (per.index >= L_FIT).sum() >= 3:
            a, b, e, aq, err = extrapolate(per.index.to_numpy(), per["mean"].to_numpy())
            row.update(A=a, B=b, sigma=err, a_joint=a_of(fit, row["eps_mean"])[0])
        rows.append(row)
    return rows


def bulk_thermo(df):
    fits, bins, states = {}, [], {}
    for g in [0.0] + GAMMAS:
        d, bad = bulk_runs(df, g)
        fits[g] = bulk_fit(d)
        fits[g]["dropped"] = bad
        bins += bins_check(d, fits[g], g)
        last = d[d.L == L_HIST].sort_values("init_seed")
        states[g] = pd.DataFrame(dict(init_seed=last.init_seed, E_per_L=last.E_per_L, Gamma_L=last.Gamma_e,
                                      Gamma_inf=[a_of(fits[g], e)[0] for e in last.E_per_L],
                                      sigma_inf=[a_of(fits[g], e)[1] for e in last.E_per_L]))
    return fits, pd.DataFrame(bins), states


def histogram(scar, states, out):
    """Both classes as histograms of Γ_∞ on the same log bins (bars), with their Γ_e at L = 16 as outlines; counts on a
    log axis (9 scar states against 100 bulk states)."""
    fig, axes = plt.subplots(len(GAMMAS), 1, figsize=(6.5, 2.4 * len(GAMMAS)), sharex=True)
    bins = np.logspace(np.log10(0.05), np.log10(20), 61)
    for ax, g in zip(axes, GAMMAS):
        st = states[g]
        sc = scar[(scar.gamma == g) & scar.complete]
        ax.hist(sc.Gamma_inf / shd.H, bins=bins, color=SCAR_COLOR, edgecolor="white", linewidth=0.5,
                label=f"scar Γ_∞ ({len(sc)} states)")
        ax.hist(st.Gamma_inf / shd.H, bins=bins, color=BULK_COLOR, edgecolor="white", linewidth=0.5,
                label=f"bulk Γ_∞ ({len(st)} states)")
        ax.hist(sc[f"G_L{L_HIST}"] / shd.H, bins=bins, histtype="step", color=SCAR_COLOR, linewidth=0.9,
                label=f"scar Γ_e at L = {L_HIST}")
        ax.hist(st.Gamma_L / shd.H, bins=bins, histtype="step", color=INK2, linewidth=0.9,
                label=f"bulk Γ_e at L = {L_HIST}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 200)
        ax.set_ylabel("initial states")
        ax.set_title(f"γ = {g}: {len(sc)} scar states, {len(st)} bulk states", fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    axes[-1].set_xlabel("Γ_∞  (h = 1)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def inverse_l_scar(scar, out):
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(10, 3.4), sharey=False)
    cmap = plt.get_cmap("coolwarm")
    x = np.linspace(0, 1 / 6, 50)
    for ax, g in zip(axes, GAMMAS):
        for p in scar[scar.gamma == g].itertuples():
            col = cmap(0.5 + p.cos_theta / 1.8)
            l = np.array(SIZES)
            gl = np.array([getattr(p, f"G_L{s}") for s in SIZES])
            sl = np.array([getattr(p, f"s_L{s}") for s in SIZES])
            ax.errorbar(1 / l, gl, yerr=sl, fmt="o", ms=3.5, color=col, elinewidth=0.8, capsize=0)
            drop = [int(v) for v in str(p.dropped).split()] if isinstance(p.dropped, str) and p.dropped else []
            if drop:
                ax.plot([1 / v for v in drop], [getattr(p, f"G_L{v}") for v in drop], "o", ms=5, mfc="white",
                        color=GREY, zorder=3)
            if p.complete:
                ax.plot(x[x <= 1 / L_FIT + 1e-9], p.A + p.B * x[x <= 1 / L_FIT + 1e-9], color=col, lw=0.9)
                ax.errorbar(0, p.Gamma_inf, yerr=p.sigma_inf, fmt="D", ms=4, color=col, elinewidth=0.8, capsize=0)
        ax.set_xlim(-0.006, 1 / 6 + 0.008)
        ax.set_xticks([0] + [1 / s for s in SIZES])
        ax.set_xticklabels(["∞"] + [str(s) for s in SIZES], fontsize=7)
        ax.set_xlabel("L (axis linear in 1/L)")
        ax.set_title(f"γ = {g}", fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("scar Γ_e")
    axes[0].plot([], [], "o", ms=5, mfc="white", color=GREY, label="other revival branch (left out)")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.9, 0.9))
    fig.colorbar(sm, ax=axes, label="cos θ", fraction=0.02, pad=0.01)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def inverse_l_bulk(fits, bins, out):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), sharex=True)
    cmap = plt.get_cmap("coolwarm")
    x = np.linspace(1 / 16, 1 / L_FIT, 30)
    for ax, g in zip(axes.flat, [0.0] + GAMMAS):
        for r in bins[bins.gamma == g].itertuples():
            col = cmap(0.5 + r.eps_mean / 0.6)
            l = np.array(SIZES)
            m = np.array([getattr(r, f"mean_L{s}") for s in SIZES])
            se = np.array([getattr(r, f"se_L{s}") for s in SIZES])
            ax.errorbar(1 / l, m, yerr=se, fmt="o", ms=3.5, color=col, elinewidth=0.8, capsize=0,
                        label=f"ε ∈ [{r.eps_lo:+.2f}, {r.eps_hi:+.2f}]")
            xs = np.concatenate([[0], x])
            ax.plot(xs, [fit_at(fits[g], r.eps_mean, 1 / v) if v > 0 else a_of(fits[g], r.eps_mean)[0] for v in xs],
                    color=col, lw=0.9)
            if np.isfinite(r.A):
                ax.errorbar(-0.004, r.A, yerr=r.sigma, fmt="s", ms=3.5, mfc="white", color=col, elinewidth=0.8)
        ax.set_xlim(-0.01, 1 / 6 + 0.008)
        ax.set_xticks([0] + [1 / s for s in SIZES])
        ax.set_xticklabels(["∞"] + [str(s) for s in SIZES], fontsize=7)
        ax.set_title("γ = 0 (thermalization)" if g == 0 else f"γ = {g}", fontsize=9, loc="left", color=INK)
        ax.set_ylabel("bulk Γ_e, mean per ε bin")
    for ax in axes[1]:
        ax.set_xlabel("L (axis linear in 1/L)")
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def vs_energy(scar, fits, bins, out):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    e = np.linspace(-0.25, 0.25, 51)
    for k, g in enumerate([0.0] + GAMMAS):
        col = COLORS[g]
        a = np.array([a_of(fits[g], v) for v in e])
        lab = "γ = 0 (thermalization)" if g == 0 else f"γ = {g}"
        ax.plot(e, a[:, 0], color=col, lw=1.2, label=f"bulk Γ_∞(ε), {lab}")
        ax.fill_between(e, a[:, 0] - a[:, 1], a[:, 0] + a[:, 1], color=col, alpha=0.15, lw=0)
        b = bins[(bins.gamma == g) & np.isfinite(bins.A)]
        ax.errorbar(b.eps_mean + 0.006 * (k - 1.5), b.A, yerr=b.sigma, fmt="s", ms=4, mfc="white", color=col,
                    elinewidth=0.8, capsize=2)
        if g > 0:
            s = scar[(scar.gamma == g) & scar.complete]
            ax.errorbar(s.E_per_L, s.Gamma_inf, yerr=s.sigma_inf, fmt="D", ms=5, color=col, mec=INK, mew=0.5,
                        elinewidth=0.8, label=f"scar Γ_∞, γ = {g}")
    ax.plot([], [], "s", mfc="white", color=INK2, label="bulk, A + B/L per ε bin")
    ax.set_yscale("log")
    ax.set_xlabel("E / L")
    ax.set_ylabel("Γ_∞", rotation=0, labelpad=14)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def vs_gamma(scar, states, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, g in enumerate(GAMMAS):
        s = scar[(scar.gamma == g) & scar.complete]
        ax.errorbar([g] * len(s), s.Gamma_inf, yerr=s.sigma_inf, fmt="D", ms=4.5, color=SCAR_COLOR, mec="white",
                    mew=0.5, elinewidth=0.8, label="scar Γ_∞ (coherent states)" if i == 0 else None)
        v = states[g].Gamma_inf
        ax.errorbar(g, v.median(), yerr=[[v.median() - v.min()], [v.max() - v.median()]], fmt="o", ms=6,
                    color=BULK_COLOR, capsize=3, label="bulk Γ_∞: median, range over states" if i == 0 else None)
    v0 = states[0.0].Gamma_inf
    ax.axhspan(v0.min(), v0.max(), color=COLORS[0.0], alpha=0.12, lw=0, label="bulk, γ = 0: Γ_th,∞ range")
    x = np.logspace(np.log10(0.02), np.log10(0.15), 50)
    ax.plot(x, 4 * x / 3, color=INK2, lw=1, label="free spin 4γ/3")
    ref = scar[(scar.gamma == 0.05) & scar.complete].Gamma_inf.median()
    ax.plot(x, ref * np.sqrt(x / 0.05), color=INK2, lw=1, ls="--", label="∝ √γ")
    ax.plot(x, ref * x / 0.05, color=INK2, lw=1, ls="-.", label="∝ γ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("γ")
    ax.set_xticks([0.02, 0.03, 0.05, 0.1])
    ax.set_xticklabels(["0.02", "0.03", "0.05", "0.1"])
    ax.minorticks_off()
    ax.set_ylabel("Γ_∞", rotation=0, labelpad=14)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else shd.DATA
    df, csv = shd.write_summary(data)
    scar = scar_thermo(df)
    fits, bins, states = bulk_thermo(df)
    scar.to_csv(data / "state_rate_thermo_scar.csv", index=False)
    bins.to_csv(data / "state_rate_thermo_bins.csv", index=False)
    names = ["a0", "a1", "a2", "b0", "b1"]
    pd.DataFrame([dict(gamma=g, n_runs=f["n"], resid_sd=f["resid_sd"],
                       **dict(zip(names, f["beta"])), **{f"{k}_err": math.sqrt(f["cov"][i, i]) for i, k in enumerate(names)},
                       **dict(zip([k + "q" for k in names] + ["c0q"], f["betaq"])))
                  for g, f in fits.items()]).to_csv(data / "state_rate_thermo_bulk.csv", index=False)
    histogram(scar, states, data / "state_rate_histogram_thermo.png")
    inverse_l_scar(scar, data / "state_rate_inverse_L_scar.png")
    inverse_l_bulk(fits, bins, data / "state_rate_inverse_L_bulk.png")
    vs_energy(scar, fits, bins, data / "state_rate_thermo_vs_energy.png")
    vs_gamma(scar, states, data / "state_rate_thermo_vs_gamma.png")
    print(f"{len(df)} runs, summary {csv}")
    for g in GAMMAS:
        s = scar[scar.gamma == g]
        print(f"\nγ = {g}, scar Γ_∞ (A + B/L, L ≥ {L_FIT}; error = max(fit, |A − A_quad|)):")
        for p in s.itertuples():
            gl = " ".join(f"{getattr(p, f'G_L{l}'):.4f}" for l in SIZES)
            print(f"  cos θ = {p.cos_theta:+.1f}: Γ_e(L = 6 … 16) {gl} -> {p.Gamma_inf:.4f} ± {p.sigma_inf:.4f} "
                  f"(fit {p.fit_err:.4f}, quad {p.A_quad:.4f})" + (f", left out L = {p.dropped}" if p.dropped else "")
                  + ("" if p.complete else " NO Γ_∞"))
    for g in [0.0] + GAMMAS:
        f, st = fits[g], states[g]
        b = bins[bins.gamma == g]
        print(f"\nγ = {g}, bulk: a(ε) = {f['beta'][0]:.3f} {f['beta'][1]:+.3f} ε {f['beta'][2]:+.3f} ε², "
              f"b(ε) = {f['beta'][3]:+.2f} {f['beta'][4]:+.2f} ε; residual sd {f['resid_sd']:.3f} ({f['n']} runs)"
              + (f"; left out L = {f['dropped']} (bounds)" if f["dropped"] else ""))
        print(f"  Γ_∞ over the {len(st)} states of L = {L_HIST}: median {st.Gamma_inf.median():.3f}, range "
              f"{st.Gamma_inf.min():.3f} – {st.Gamma_inf.max():.3f}; error {st.sigma_inf.min():.3f} – "
              f"{st.sigma_inf.max():.3f}; Γ_e at L = {L_HIST}: {st.Gamma_L.min():.3f} – {st.Gamma_L.max():.3f}")
        for r in b.itertuples():
            means = " ".join(f"{getattr(r, f'mean_L{l}'):.3f}({getattr(r, f'n_L{l}')})" for l in SIZES)
            print(f"  bin [{r.eps_lo:+.2f}, {r.eps_hi:+.2f}] ε̄ {r.eps_mean:+.3f}: {means} -> A {r.A:.3f} ± {r.sigma:.3f}, "
                  f"joint a(ε̄) {r.a_joint:.3f}")


if __name__ == "__main__":
    main()
