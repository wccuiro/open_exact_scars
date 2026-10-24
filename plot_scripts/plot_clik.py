"""
CLIK eigenvalues of the Lindbladian (julia/clik.jl) against the exact spectrum, for the tower (scar) modes.

Reference: the Rust ED with tower weights, tower = ‖P X P‖ / ‖X‖ for the right eigenmatrix X (P the tower projector):
periodic ring lindblad_entanglement_L<L>_g<γ>.csv, open chain spectrum_open_L<L>.csv (`ed --open 1`, rows of this γ).
They are the same 𝓛 as clik.jl's (its dense backend checks this to 5e-12).
Targets: the N_TARGETS slowest distinct eigenvalues with tower weight > TOWER_MIN and Im λ ≥ 0 (the spectrum is
symmetric under λ → λ*, and a complex-time (α > 0) space is only trusted for Im λ ≥ 0). Error of a target at time T:
distance to the closest Krylov eigenvalue (Im λ ≥ −1e-6) of the block |z| ≤ T.

Exact tower modes, for every L and either boundary: λ = 0 (kernel: 𝟙, |Φ_0⟩⟨Φ_0|, |Φ_L⟩⟨Φ_L|) and
λ = −2nγ ± i h n (|Φ_0⟩⟨Φ_n|, |Φ_{L−n}⟩⟨Φ_L|, two-fold), marked in the spectrum panels.

Input:  clik_<backend>_L<L>_g<γ>[_open][_scar][_chi<χ>]_eigs.csv (T, n_states, rank, re, im) and the reference files,
        in data/used_data/ (or the directory given as the first argument; references also looked up in its parent)
Output: clik_errors.csv (run, L, gamma, open, init, chi, T, n_states, rank, target, target_re, target_im, tower, error),
        clik_spectrum.png (slow part of the spectrum, ED and Krylov at the largest T, one panel per run),
        clik_convergence.png (error against T, one panel per run), in the same directory
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "used_data"
N_TARGETS = 8
TOWER_MIN = 0.9
H = 1.0  # tower spacing (rust/src/parameters.rs)
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = SERIES[0], SERIES[1]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 1.2, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def reference(l, g, open_chain):
    """(eigenvalues, tower weights) of the Rust ED, or None."""
    name = f"spectrum_open_L{l}.csv" if open_chain else f"lindblad_entanglement_L{l}_g{g}.csv"
    for f in (DIR / name, DIR.parent / name, DIR.parent / "used_data" / name):
        if f.exists():
            d = pd.read_csv(f)
            d = d[np.isclose(d["gamma"], float(g))]
            return d["re"].to_numpy() + 1j * d["im"].to_numpy(), d["tower"].to_numpy()
    return None


def targets(lam, tower):
    """Slowest distinct tower modes with Im λ ≥ 0, sorted by Re λ descending: list of (λ, tower weight)."""
    keep = (lam.imag > -1e-9) & (tower > TOWER_MIN)
    lam, tower = lam[keep], tower[keep]
    order = np.lexsort((-lam.imag, -lam.real))
    distinct = []
    for x, w in zip(lam[order], tower[order]):
        if all(abs(x - y) > 1e-7 for y, _ in distinct):
            distinct.append((x, w))
        if len(distinct) == N_TARGETS:
            break
    return distinct


runs = []
for f in sorted(DIR.glob("clik_*_eigs.csv")):
    m = re.fullmatch(r"clik_(dense|mps)_L(\d+)_g([^_]+)(_open)?(_scar)?(?:_chi(\d+))?_eigs\.csv", f.name)
    if m is None:
        continue
    backend, l, g, op, scar, chi = m.groups()
    ref = reference(int(l), g, bool(op))
    if ref is None:
        print(f"{f.name}: no reference spectrum, skipped")
        continue
    runs.append(dict(name=f.name[: -len("_eigs.csv")], backend=backend, L=int(l), g=g, open=bool(op),
                     init="scar" if scar else "random", chi=int(chi) if chi else 0, df=pd.read_csv(f), ref=ref))
if not runs:
    sys.exit(f"no clik_*_eigs.csv with a reference spectrum in {DIR}")

rows = []
for run in runs:
    tg = targets(*run["ref"])
    for T, block in run["df"].groupby("T"):
        lam = block["re"].to_numpy() + 1j * block["im"].to_numpy()
        lam = lam[lam.imag > -1e-6]
        for k, (x, w) in enumerate(tg):
            rows.append(dict(run=run["name"], L=run["L"], gamma=float(run["g"]), open=int(run["open"]),
                             init=run["init"], chi=run["chi"], T=T, n_states=block["n_states"].iloc[0],
                             rank=block["rank"].iloc[0], target=k, target_re=x.real, target_im=x.imag, tower=w,
                             error=np.min(np.abs(lam - x))))
errors = pd.DataFrame(rows)
errors.to_csv(DIR / "clik_errors.csv", index=False)

# summary at the largest T
last = errors[errors["T"] == errors.groupby("run")["T"].transform("max")]
for name, block in last.groupby("run", sort=False):
    b = block.iloc[0]
    print(f"{name}: T = {b['T']:g}, {b['n_states']} states, rank {b['rank']}")
    for _, r in block.iterrows():
        print(f"    λ = {r['target_re']:+.6f} {r['target_im']:+.6f}i  (tower {r['tower']:.3f})   error {r['error']:.2e}")


def title(run):
    method = "dense" if run["backend"] == "dense" else f"MPS χ = {run['chi']}"
    return f"L = {run['L']} {'open' if run['open'] else 'ring'}, {run['init']}, {method}"


# ---- spectrum panels: ED and Krylov at the largest T, one per run --------------------------------------
NCOL = min(4, len(runs))
NROW = -(-len(runs) // NCOL)
fig, axes = plt.subplots(NROW, NCOL, figsize=(2.8 * NCOL, 3.0 * NROW), squeeze=False, layout="constrained",
                         sharey=True)
for ax in axes.flat[len(runs):]:
    ax.set_visible(False)
for ax, run in zip(axes.flat, runs):
    gamma = float(run["g"])
    lam, tower = run["ref"]
    ax.scatter(lam.real, lam.imag, s=9, color=AXIS, lw=0, label="ED", zorder=1)
    t = tower > TOWER_MIN
    ax.scatter(lam[t].real, lam[t].imag, s=30, marker="s", facecolor="none", edgecolor=INK_SECONDARY, lw=0.8,
               label=f"ED, tower weight > {TOWER_MIN}", zorder=2)
    df = run["df"][run["df"]["T"] == run["df"]["T"].max()]
    ax.scatter(df["re"], df["im"], s=14, marker="x", lw=1.0, color=ORANGE, label="CLIK, largest T", zorder=3)
    n = np.arange(0, run["L"] + 1)
    ax.scatter(-2 * gamma * n, H * n, s=60, marker="+", color=BLUE, lw=1.0, label="exact −2nγ + ihn", zorder=4)
    ax.set_xlim(-7 * gamma, 0.5 * gamma)
    ax.set_ylim(-0.3, 3.3)
    ax.set_title(title(run), fontsize=8)
    ax.set_xlabel("Re λ")
for row in axes:
    row[0].set_ylabel("Im λ")
axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.savefig(DIR / "clik_spectrum.png")

# ---- convergence: error against T, one panel per run ---------------------------------------------
fig, axes = plt.subplots(NROW, NCOL, figsize=(2.8 * NCOL, 3.9 * NROW), squeeze=False, layout="constrained",
                         sharey=True)
for ax in axes.flat[len(runs):]:
    ax.set_visible(False)
for ax, run in zip(axes.flat, runs):
    block = errors[errors["run"] == run["name"]]
    for k, tb in block.groupby("target"):
        x = tb.iloc[0]
        ax.semilogy(tb["T"], np.maximum(tb["error"], 1e-16), color=SERIES[k], marker="o", ms=2.5,
                    label=f"{x['target_re']:+.4f} {x['target_im']:+.3f}i")
    ax.set_title(title(run), fontsize=8)
    ax.set_xlabel("T")
    ax.legend(title="ED tower mode λ", fontsize=6.5, title_fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2)
for row in axes:
    row[0].set_ylabel("|λ_CLIK − λ_ED|")
fig.savefig(DIR / "clik_convergence.png")
print("->", DIR / "clik_errors.csv", DIR / "clik_spectrum.png", DIR / "clik_convergence.png")
