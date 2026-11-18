"""
Decay-rate histograms over initial states under depolarizing noise (runs of scripts/run_state_hist.py).

    ~/anaconda3/bin/python plot_scripts/plot_state_rate_histogram.py [data/state_hist]

Reads every finished run (state_hist_data.summarize, also written to summary_noisedepol.csv) and plan.json (the
saturation sizes of the size scan). Classes:
  - scar: coherent states, cos θ = 0, ±0.2, …; at each γ the two sizes chosen from the scan. A point counts if the
    two Γ_e agree (state_hist_data.agree); it is drawn at the larger size, finite-size points in grey and left out
    of the counts; Γ_e bounds (no 1/e crossing before t-max) as arrows.
  - bulk: bulk product states; the histogram is the trajectories at L = 12; the class is converged in L if the mean
    Γ_e at L = 10 (exact) and L = 12 agree within 2 standard errors of the means (state-to-state spread / √N).
Figures (in the data directory):
  state_rate_histogram.png   one panel per γ: bulk histogram (bars), scar points (markers, labelled cos θ)
  state_rate_vs_gamma.png    Γ_e against γ, both classes, log-log, with 4γ/3, √γ and γ guides
  state_rate_vs_energy.png   Γ_e against E/L, both classes
  state_rate_size_scan.png   size scan: Γ_e of the coherent state cos θ = 0 against L, per γ
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_hist_data as shd  # noqa: E402

GAMMAS = [0.1, 0.05, 0.03]
COLORS = {0.1: "#2a78d6", 0.05: "#eb6834", 0.03: "#1baf7a"}  # categorical slots 1-3 (dataviz reference palette)
BULK_COLOR, SCAR_COLOR, GREY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def classes(df, plan):
    """Scar points and bulk rows per γ."""
    sizes = {float(g): (tuple(s) if s else None) for g, s in plan.get("scan_sizes", {}).items()}
    scar, bulk, notes = {}, {}, []
    for g in GAMMAS:
        pts = []
        s = sizes.get(g)
        sub = df[(df.family == "coherent") & (df.gamma == g)]
        if s is None:
            if g in sizes:
                notes.append(f"γ = {g}: scar class not converged in L (size scan), no scar runs")
        else:
            for c in sorted(sub.cos_theta.unique()):
                rows = {int(r.L): r for r in sub[sub.cos_theta == c].itertuples()}
                if not all(l in rows for l in s):
                    notes.append(f"γ = {g}, cos θ = {c:g}: missing size(s) {[l for l in s if l not in rows]}")
                    continue
                r1, r2 = rows[s[0]], rows[s[1]]
                bound = bool(r1.bound or r2.bound)
                ok = (not bound) and shd.agree(r1.Gamma_e, r1.sigma_Gamma, r2.Gamma_e, r2.sigma_Gamma)
                if not ok:
                    why = "bound (no 1/e crossing)" if bound else \
                        f"L = {s[0]}: {r1.Gamma_e:.4f} ± {r1.sigma_Gamma:.4f}, L = {s[1]}: {r2.Gamma_e:.4f} ± {r2.sigma_Gamma:.4f}"
                    notes.append(f"γ = {g}, cos θ = {c:g}: finite-size, {why}")
                pts.append(dict(cos=c, E=c / 2, G=r2.Gamma_e, sG=r2.sigma_Gamma, ok=ok, bound=bound, L=s))
        scar[g] = pts
        b10 = df[(df.family == "bulk") & (df.gamma == g) & (df.L == 10) & (df.method == "exact")]
        b12 = df[(df.family == "bulk") & (df.gamma == g) & (df.L == 12) & (df.method == "traj")]
        conv = None
        if len(b10) > 1 and len(b12) > 1:
            m10, m12 = b10.Gamma_e.mean(), b12.Gamma_e.mean()
            e10, e12 = b10.Gamma_e.std() / math.sqrt(len(b10)), b12.Gamma_e.std() / math.sqrt(len(b12))
            conv = abs(m10 - m12) <= 2 * math.hypot(e10, e12)
            notes.append(f"γ = {g}, bulk: mean Γ_e L = 10 {m10:.3f} ± {e10:.3f} ({len(b10)} states), L = 12 {m12:.3f} "
                         f"± {e12:.3f} ({len(b12)}): {'agree' if conv else 'DO NOT agree'}")
        for r in b12[b12.nbar_flag].itertuples():
            notes.append(f"γ = {g}, bulk seed {r.init_seed}: |n̄⊥| = {r.nbar_norm:.3f} > 0.2 a(0) = {0.2 * r.a0:.3f}")
        for r in b12[b12.bound].itertuples():
            notes.append(f"γ = {g}, bulk seed {r.init_seed}: Γ_e < {r.Gamma_e:.3f} (bound)")
        bulk[g] = (b12, b10, conv)
    return scar, bulk, notes


def histogram(scar, bulk, out):
    fig, axes = plt.subplots(len(GAMMAS), 1, figsize=(6.5, 2.4 * len(GAMMAS)), sharex=True)
    bins = np.logspace(np.log10(0.05), np.log10(20), 61)
    for ax, g in zip(axes, GAMMAS):
        b12, _, conv = bulk[g]
        vals = b12.Gamma_e[~b12.bound].to_numpy()
        n_bulk = len(vals)
        if n_bulk:
            ax.hist(vals / shd.H, bins=bins, color=BULK_COLOR, edgecolor="white", linewidth=0.5,
                    label=f"bulk, L = 12 ({n_bulk})")
        top = max(ax.get_ylim()[1], 3)
        pts = scar[g]
        n_ok = sum(p["ok"] for p in pts)
        for k, p in enumerate(sorted(pts, key=lambda p: p["G"])):
            y = top * (0.15 + 0.7 * (k % 9) / 9)
            col = SCAR_COLOR if p["ok"] else GREY
            ax.errorbar(p["G"] / shd.H, y, xerr=p["sG"] / shd.H if p["sG"] > 0 else None, fmt="D", ms=5, color=col,
                        mec="white", mew=0.8, elinewidth=1, zorder=3)
            if p["bound"]:
                ax.annotate("", xy=(p["G"] / shd.H / 1.6, y), xytext=(p["G"] / shd.H, y),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1))
            ax.annotate(f"{p['cos']:+.1f}" if p["cos"] else "0", (p["G"] / shd.H, y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK2)
        if pts:
            ax.plot([], [], "D", color=SCAR_COLOR, ms=5, label=f"scar, L = {pts[0]['L'][1]} (cos θ)")
        if any(not p["ok"] for p in pts):
            ax.plot([], [], "D", color=GREY, ms=5, label="scar, finite-size")
        ax.set_xscale("log")
        ax.set_ylabel("initial states")
        conv_txt = "" if conv is None else (", bulk L = 10, 12 agree" if conv else ", bulk L = 10, 12 differ")
        ax.set_title(f"γ = {g}: {n_ok} scar states, {n_bulk} bulk states{conv_txt}", fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Γ_e / h")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def vs_gamma(df, scar, bulk, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, g in enumerate(GAMMAS):
        ok = [p for p in scar[g] if p["ok"]]
        if ok:
            ax.plot([g] * len(ok), [p["G"] for p in ok], "D", ms=5, color=SCAR_COLOR, mec="white", mew=0.6,
                    label="scar (coherent states)" if i == 0 else None)
        b12 = bulk[g][0]
        if len(b12):
            v = b12.Gamma_e[~b12.bound]
            ax.errorbar(g, v.mean(), yerr=v.std(), fmt="o", ms=6, color=BULK_COLOR, capsize=3,
                        label="bulk, L = 12: mean ± spread" if i == 0 else None)
    pure = df[(df.family == "bulk") & (df.method == "pure")]
    for l, ls in [(16, "--"), (12, ":")]:
        p = pure[pure.L == l]
        if len(p):
            ax.axhline(p.Gamma_e.mean(), color=BULK_COLOR, ls=ls, lw=1, label=f"bulk, γ = 0, L = {l}: mean Γ_th")
    x = np.logspace(np.log10(0.02), np.log10(0.15), 50)
    ax.plot(x, 4 * x / 3, color=INK2, lw=1, label="free spin 4γ/3")
    sc = [p["G"] for p in scar.get(0.05, []) if p["ok"]]
    ref = np.median(sc) if sc else 0.25
    ax.plot(x, ref * np.sqrt(x / 0.05), color=INK2, lw=1, ls="--", label="∝ √γ")
    ax.plot(x, ref * x / 0.05, color=INK2, lw=1, ls="-.", label="∝ γ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("γ")
    ax.set_ylabel("Γ_e")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def vs_energy(scar, bulk, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for g in GAMMAS:
        b12 = bulk[g][0]
        if len(b12):
            ax.plot(b12.E_per_L, b12.Gamma_e, "o", ms=3, color=COLORS[g], alpha=0.7, label=f"bulk, γ = {g}")
        ok = [p for p in scar[g] if p["ok"]]
        if ok:
            ax.plot([p["E"] for p in ok], [p["G"] for p in ok], "D", ms=5, color=COLORS[g], mec=INK, mew=0.6,
                    label=f"scar, γ = {g}")
    ax.set_yscale("log")
    ax.set_xlabel("E / L")
    ax.set_ylabel("Γ_e")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def size_scan(df, plan, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sc = df[(df.family == "coherent") & (df.cos_theta == 0)]
    sizes = plan.get("scan_sizes", {})
    for g in GAMMAS:
        r = sc[sc.gamma == g].sort_values("L")
        if not len(r):
            continue
        s = sizes.get(str(g))
        ax.errorbar(r.L, r.Gamma_e, yerr=r.sigma_Gamma, fmt="o-", ms=5, lw=1, color=COLORS[g], capsize=3,
                    label=f"γ = {g}: " + (f"saturates at L = {s[0]}" if s else "not converged"))
    ax.set_xlabel("L")
    ax.set_ylabel("Γ_e (coherent state, cos θ = 0)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else shd.DATA
    df, csv = shd.write_summary(data)
    plan_file = data / "plan.json"
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    scar, bulk, notes = classes(df, plan)
    histogram(scar, bulk, data / "state_rate_histogram.png")
    vs_gamma(df, scar, bulk, data / "state_rate_vs_gamma.png")
    vs_energy(scar, bulk, data / "state_rate_vs_energy.png")
    size_scan(df, plan, data / "state_rate_size_scan.png")
    print(f"{len(df)} runs, summary {csv}")
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
