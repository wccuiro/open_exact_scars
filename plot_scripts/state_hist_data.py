"""
Reading the `state_hist` runs (decay-rate histograms over initial states, depolarizing noise) and their decay rates.

Files (in data/state_hist/, written by rust/src/bin/state_hist.rs and state_hist_ed.rs):
  states_L<L>_g<γ>_noisedepol_<init>.csv        exact ρ
  states_traj_L<L>_g<γ>_noisedepol_<init>.csv   diffusive trajectories
  states_pure_L<L>_g0_<init>.csv                one state vector at γ = 0
  nbar_L<L>_initbulk-product_s<N>.csv           n̄⊥_j: from ED (γ = 0 diagonal ensemble, L ≤ 12) or, with
                                                `source=late` in its first line (L ≥ 14), the average over t ≥ LATE of
                                                the γ = 0 run (`state_hist --write-nbar 5`)
with <init> = initcoherent_cos<cos θ> (scar class) or initbulk-product_s<N> (bulk class). First line `# key=value …`,
last line `# end t=… reason=… wall_s=… setup_s=…`.

Observable a(t) = (1/L) Σ_j |n⊥_j(t) − n̄⊥_j| (column a; n̄ = 0 for the tower states). For the γ = 0 runs a is
recomputed here from the per-site columns: n̄ from ED for L ≤ 12, else the late-time average of the run itself
(t ≥ LATE).

Rates, for one run with a(0) = a0:
  - envelope E(t) = max of a over [t, t + 2π/h] (the next period; truncated at the end of the run);
  - Γ_e = 1/t_e, t_e = first time E < a0/e, interpolated linearly in ln E between the bracketing output times; if E
    never falls below a0/e, Γ_e = 1/t_end is an upper bound (`bound`);
  - σ_Γ (trajectories) = Γ_e² σ_a(t_e) / |dE/dt|, the slope over ± 4 output steps (as for the other runs of the
    project); 0 for exact and pure runs;
  - fit: least-squares slope of ln a(t) on the points with a/a0 in [e^−2.5, e^−0.3], a > 5 × floor and (trajectories)
    a > 3 σ_a; floor = rms of a over t ≥ LATE in the γ = 0 run of the same state and L (0 for coherent states, where
    a is constant at γ = 0);
  - nbar_flag: (1/L) Σ_j |n̄⊥_j| > 0.2 a0.
Size comparisons (`agree`): |Γ₁ − Γ₂| ≤ AGREE_SIGMAS · √(σ₁² + σ₂²), each σ at least SYS_REL · Γ (the exact runs
have no statistical error; 0.5 % stands for the precision at which a rate is called converged).
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "state_hist"
H = 1.0
PERIOD = 2 * math.pi / H
LATE = 5.0
SYS_REL = 0.005
AGREE_SIGMAS = 2.0
NAME = re.compile(r"states_(?:(traj|pure)_)?L(\d+)_g([0-9.e+-]+?)(_noisedepol)?_init"
                  r"(?:coherent_cos(-?[0-9.e+-]+)|bulk-product_s(\d+))\.csv$")


def parse_name(path):
    m = NAME.match(Path(path).name)
    if not m:
        return None
    method = m.group(1) or "exact"
    cos = None if m.group(5) is None else float(m.group(5))
    seed = None if m.group(6) is None else int(m.group(6))
    return dict(method=method, L=int(m.group(2)), gamma=float(m.group(3)),
                family="coherent" if cos is not None else "bulk", cos_theta=cos, init_seed=seed)


def read_run(path):
    """(meta dict, DataFrame, end dict or None) of one finished run."""
    lines = Path(path).read_text().splitlines()
    meta = dict(kv.split("=", 1) for kv in lines[0][2:].split())
    end = None
    if lines[-1].startswith("# end"):
        end = dict(kv.split("=", 1) for kv in lines[-1][len("# end "):].split())
    df = pd.read_csv(path, comment="#")
    return meta, df, end


def sites(df, l):
    return (df[[f"x{j}" for j in range(l)]].to_numpy(), df[[f"y{j}" for j in range(l)]].to_numpy())


def a_with(df, l, nbar):
    x, y = sites(df, l)
    return np.hypot(x - nbar[:, 0], y - nbar[:, 1]).mean(axis=1)


def read_nbar(path):
    df = pd.read_csv(path, comment="#")
    return df[["nx", "ny"]].to_numpy()


def nbar_source(path):
    """'late' if the n̄ file was written from a late-time average (first line `source=late`), else 'ed'."""
    with open(path) as f:
        return "late" if "source=late" in f.readline() else "ed"


def nbar_file(l, seed, data=DATA):
    return Path(data) / f"nbar_L{l}_initbulk-product_s{seed}.csv"


def envelope(t, a, period=PERIOD):
    """Running maximum of a over [t, t + period] (the next period), on the output grid."""
    env = np.empty_like(a)
    j = 0
    for i in range(len(t)):
        j = max(j, i)
        while j + 1 < len(t) and t[j + 1] <= t[i] + period + 1e-9:
            j += 1
        env[i] = a[i:j + 1].max()
    return env


def gamma_e(t, a, a_err, a0):
    """(Γ_e, σ_Γ, t_e, bound) as in the module docs."""
    env = envelope(t, a)
    level = a0 / math.e
    below = np.nonzero(env < level)[0]
    if len(below) == 0:
        return 1.0 / t[-1], 0.0, float("nan"), True
    k = below[0]
    if k == 0:
        return float("inf"), 0.0, 0.0, False
    e1, e2 = env[k - 1], env[k]
    if e2 > 0:
        f = (math.log(level) - math.log(e1)) / (math.log(e2) - math.log(e1))
    else:
        f = (level - e1) / (e2 - e1)
    te = t[k - 1] + f * (t[k] - t[k - 1])
    g = 1.0 / te
    sigma = 0.0
    if a_err is not None and np.any(a_err > 0):
        lo, hi = max(k - 4, 0), min(k + 4, len(t) - 1)
        slope = (env[hi] - env[lo]) / (t[hi] - t[lo])
        s_a = np.interp(te, t, a_err)
        sigma = g * g * s_a / abs(slope) if slope != 0 else float("inf")
    return g, sigma, te, False


def fit_rate(t, a, a_err, a0, floor):
    """(rate, t_min, t_max, points) of the ln a fit, NaN if fewer than 3 points qualify."""
    r = a / a0
    mask = (r <= math.exp(-0.3)) & (r >= math.exp(-2.5)) & (a > 5 * floor)
    if a_err is not None:
        mask &= a > 3 * a_err
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    slope = np.polyfit(t[mask], np.log(a[mask]), 1)[0]
    return -slope, float(t[mask].min()), float(t[mask].max()), n


def agree(g1, s1, g2, s2):
    s1 = max(s1, SYS_REL * abs(g1))
    s2 = max(s2, SYS_REL * abs(g2))
    return abs(g1 - g2) <= AGREE_SIGMAS * math.hypot(s1, s2)


def pure_reference(l, seed, data=DATA):
    """(n̄ (L × 2), source, floor) for a bulk state from its γ = 0 run (None if missing): n̄ from its n̄ file (ED, or
    the late-time average written by `--write-nbar`), else the late-time average of the γ = 0 run; floor = rms of
    a(t ≥ LATE) in that run with that n̄."""
    pure = Path(data) / f"states_pure_L{l}_g0_initbulk-product_s{seed}.csv"
    ed = nbar_file(l, seed, data)
    if not pure.exists():
        return (read_nbar(ed) if ed.exists() else None), (nbar_source(ed) if ed.exists() else None), float("nan")
    _, df, _ = read_run(pure)
    late = df["t"].to_numpy() >= LATE - 1e-9
    if ed.exists():
        nbar, source = read_nbar(ed), nbar_source(ed)
    else:
        x, y = sites(df[late], l)
        nbar, source = np.stack([x.mean(axis=0), y.mean(axis=0)], axis=1), "late"
    a = a_with(df, l, nbar)
    return nbar, source, float(np.sqrt(np.mean(a[late] ** 2)))


def summarize(data=DATA):
    """One row per finished run (the summary CSV)."""
    rows = []
    refs = {}
    for path in sorted(Path(data).glob("states_*.csv")):
        info = parse_name(path)
        if info is None:
            continue
        meta, df, end = read_run(path)
        if end is None:
            continue
        l = info["L"]
        t = df["t"].to_numpy()
        a = df["a"].to_numpy()
        a_err = df["a_err"].to_numpy() if info["method"] == "traj" and int(meta["trajectories"]) > 1 else None
        nbar_norm = float(meta["nbar_norm"])
        floor = 0.0
        source = meta["nbar_source"]
        if info["family"] == "bulk":
            key = (l, info["init_seed"])
            if key not in refs:
                refs[key] = pure_reference(l, info["init_seed"], data)
            nbar, src, floor = refs[key]
            if info["method"] == "pure" and nbar is not None:
                a = a_with(df, l, nbar)
                nbar_norm = float(np.hypot(nbar[:, 0], nbar[:, 1]).mean())
                source = src
        a0 = a[0]
        g, sg, te, bound = gamma_e(t, a, a_err, a0)
        fit, f_lo, f_hi, f_n = fit_rate(t, a, a_err, a0, floor if np.isfinite(floor) else 0.0)
        w = df["W"].to_numpy()
        rows.append(dict(
            method=info["method"], L=l, gamma=info["gamma"], family=info["family"], cos_theta=info["cos_theta"],
            init_seed=info["init_seed"], trajectories=int(meta["trajectories"]), E=float(meta["E"]),
            E_per_L=float(meta["E_per_L"]), a0=a0, nbar_norm=nbar_norm, nbar_source=source,
            nbar_flag=nbar_norm > 0.2 * a0, floor=floor, t_e=te, Gamma_e=g, sigma_Gamma=sg, bound=bound,
            Gamma_fit=fit, fit_t_min=f_lo, fit_t_max=f_hi, fit_points=f_n,
            W_te=float(np.interp(te, t, w)) if np.isfinite(te) else float("nan"),
            t_end=float(end["t"]), reason=end["reason"], wall_s=float(end["wall_s"]), file=path.name))
    return pd.DataFrame(rows)


def write_summary(data=DATA):
    df = summarize(data)
    out = Path(data) / "summary_noisedepol.csv"
    df.to_csv(out, index=False)
    return df, out


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
    df, out = write_summary(d)
    print(f"{len(df)} runs -> {out}")
