"""
Time and RAM estimates for the three methods, on this machine and on a 112-core / 0.5 TB node.

All numbers are for one run at γ = 0.01 up to t = 100 with dt = 0.1 (1000 steps). They are
extrapolated from the timings measured on this machine (i7-12650H, 16 threads, 15 GB), listed in
MEASURED below, with the scaling laws written out in each function. Everything that is an
assumption rather than a measurement is in ASSUMED.

Output: printed, and data/resource_tables.md
"""

from math import ceil, gcd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
STEPS = 1000

MEASURED = dict(
    # exact evolution (cargo run --release), 16 threads: seconds per stored matrix element per substep
    ed_seconds_per_element=3.03e-9,          # from L = 10: 0.286 s per step, 1.57e7 elements, 6 substeps
    ed_seconds_per_step={8: 0.026, 9: 0.065, 10: 0.286},
    ed_elements_per_row=(149.7, 15.4),       # at L = 10, and the increase per added site (L = 8, 9, 10)
    ed_norm=(55.1, 6.45),                    # norm bound at L = 10 and its increase per site; substeps = ceil(norm · dt)
    # trajectories, one trajectory on one thread: seconds per 1000 steps
    traj_core_seconds={8: 4.16, 10: 19.6, 12: 82.1, 14: 434.0, 16: 1886.0},
    traj_parallel_speedup=9.6,               # 16 threads on this machine do the work of 9.6 single threads (L = 10)
    traj_std_times_L=0.97,                   # spread of ⟨σˣ⟩ of one trajectory ≈ 0.97 / L at its largest (L = 4 … 8)
    # TDVP (julia, 8 BLAS threads), MPO bond dimension 74: seconds per time step, {(L, χ): s}
    tdvp_seconds_per_step={(8, 16): 0.49, (8, 32): 1.17, (8, 64): 4.62, (8, 128): 6.53,
                           (12, 16): 1.04, (12, 32): 2.97, (16, 16): 1.47, (16, 32): 5.86},
    tdvp_seconds_per_unit=4.6e-4,            # fitted to the above, see tdvp(); reproduces them within 20 %
    tdvp_base_ram_gb=1.3,                    # julia + ITensors with small tensors (measured 0.9 – 1.6 GB)
)

ASSUMED = dict(
    traj_target_error=0.0025,                # wanted standard error of ⟨σˣ⟩ (5 % of an amplitude of 0.05)
    cluster_cores=112,
    cluster_ram_gb=500,
    local_ram_gb=15,
    cluster_ed_speedup=5.0,                  # sparse matrix-vector product is limited by memory bandwidth, not cores
    cluster_core_vs_local_thread=1.0,        # one cluster core ≈ one thread of this machine running alone
    tdvp_cluster_speedup=1.0,                # dense linear algebra at these sizes does not use 112 cores; run 14 jobs instead
)


def duration(seconds):
    if seconds > 2 * 365 * 86400:
        return "> 2 years"
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.0f} s"


def memory(gb):
    if gb < 0.001:
        return "< 1 MB"
    return f"{gb * 1000:.0f} MB" if gb < 1 else f"{gb:.1f} GB"


def table(title, header, rows, notes=()):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    lines += [""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


# ---- exact evolution in Liouville space ---------------------------------------------------------

def ed(L):
    """(seconds on this machine, GB): time ∝ stored elements × substeps, RAM = matrix + orbit table + vectors."""
    dim = sum(4 ** gcd(d, L) for d in range(L)) // L
    per_row, slope = MEASURED["ed_elements_per_row"]
    elements = dim * (per_row + slope * (L - 10))
    norm, norm_slope = MEASURED["ed_norm"]
    substeps = ceil((norm + norm_slope * (L - 10)) * 0.1)
    seconds = STEPS * MEASURED["ed_seconds_per_step"].get(L, MEASURED["ed_seconds_per_element"] * elements * substeps)
    gb = (12 * elements + 4 * 4 ** L + 60 * dim) / 1e9
    return seconds, gb


def ed_table(cluster):
    speed = ASSUMED["cluster_ed_speedup"] if cluster else 1.0
    limit = ASSUMED["cluster_ram_gb"] if cluster else ASSUMED["local_ram_gb"]
    rows = []
    for L in range(8, 17):
        seconds, gb = ed(L)
        fits = "yes" if gb < 0.7 * limit else ("tight" if gb < limit else "no")
        rows.append((L, duration(seconds / speed), memory(gb), fits))
    notes = ["Exact (round-off only). Time measured at L = 8, 9, 10; RAM measured at L = 9, 10 (0.05, 0.20 GB, as predicted).",
             "The code currently stops at L = 12 (orbit table); larger sizes need that limit lifted."]
    if cluster:
        notes.append(f"Assumes the node is {speed:g}× faster than this machine for this memory-bound kernel (not measured).")
    return table("Exact evolution (Rust, Liouville space)", ("L", "time", "RAM", "fits in RAM"), rows, notes)


# ---- trajectories ---------------------------------------------------------------------------------

def trajectory(L):
    """(core-seconds per trajectory, GB per trajectory, trajectories needed)."""
    known = MEASURED["traj_core_seconds"]
    if L in known:
        core = known[L]
    else:
        # cost ∝ (number of Pauli terms ∝ L) × 2^L × substeps; substeps = ceil(1.5 ‖H‖ dt/2), ‖H‖ ≈ 0.98 L
        substeps = lambda n: ceil(1.5 * 0.98 * n * 0.05)
        core = known[16] * (L / 16) * 2 ** (L - 16) * substeps(L) / substeps(16)
    gb = 48 * 2 ** L / 1e9  # state + two work vectors, 16 bytes per amplitude
    needed = ceil((MEASURED["traj_std_times_L"] / L / ASSUMED["traj_target_error"]) ** 2)
    return core, gb, needed


def trajectory_table(cluster):
    if cluster:
        workers = ASSUMED["cluster_cores"] * ASSUMED["cluster_core_vs_local_thread"]
        slots, limit = ASSUMED["cluster_cores"], ASSUMED["cluster_ram_gb"]
    else:
        workers, slots, limit = MEASURED["traj_parallel_speedup"], 16, ASSUMED["local_ram_gb"]
    rows = []
    for L in range(8, 30, 2):
        core, gb, needed = trajectory(L)
        concurrent = max(1, min(slots, int(0.9 * limit / gb)))
        effective = min(concurrent, workers)
        rows.append((L, duration(core), needed, duration(needed * core / effective),
                     memory(gb * concurrent), concurrent if concurrent < slots else "all"))
    notes = [f"Trajectories needed for a standard error of {ASSUMED['traj_target_error']} on ⟨σˣ⟩, using the measured "
             "single-trajectory spread ≈ 0.97 / L (extrapolated beyond L = 8). The error falls as 1/√N.",
             "Core time measured up to L = 16, extrapolated above (a 2-step check at L = 20 was about 25 % faster).",
             "Statistical error only; the time-step error is O(dt²) and below 1e-3 at dt = 0.1 (checked at L = 4, 5)."]
    if cluster:
        notes.append("Assumes one cluster core is as fast as one thread of this machine running alone (not measured).")
    return table("Trajectories (Rust, state vector)",
                 ("L", "core time per trajectory", "trajectories", "total time", "RAM", "run at once"), rows, notes)


# ---- TDVP -----------------------------------------------------------------------------------------

TDVP_ERROR = {16: "6e-2", 32: "5e-2", 64: "2e-2", 128: "5e-4 up to t = 3 (not run further)", 256: "exact for L = 8",
              1024: "exact for L = 10"}


def tdvp(L, chi):
    """(seconds, GB). Bond k has dimension χ_k = min(χ, 4^k, 4^(L−k)); the two-site update at bond k acts
    on a tensor of χ_{k−1} · 16 · χ_{k+1} entries and costs ∝ χ_{k−1} χ_{k+1} (1 + √(χ_{k−1} χ_{k+1}) / 150)."""
    dims = [min(chi, 4 ** min(k, L - k)) for k in range(L + 1)]
    pairs = [dims[k - 1] * dims[k + 1] for k in range(1, L)]
    per_step = MEASURED["tdvp_seconds_per_step"].get(
        (L, chi), MEASURED["tdvp_seconds_per_unit"] * sum(p * (1 + p ** 0.5 / 150) for p in pairs))
    # environments (χ_k² · 74 each), plus Krylov vectors (30) and contraction temporaries (2 · 74) of the largest update
    gb = MEASURED["tdvp_base_ram_gb"] + 8 * (74 * sum(d * d for d in dims) + max(pairs) * 16 * (30 + 2 * 74)) / 1e9
    return STEPS * per_step, gb


def tdvp_table(cluster):
    limit = ASSUMED["cluster_ram_gb"] if cluster else ASSUMED["local_ram_gb"]
    rows = []
    for chi in (16, 32, 64, 128, 256, 1024):
        cells = []
        for L in (8, 16, 32):
            if 4 ** (L // 2) < chi:
                cells.append("-")
                continue
            seconds, gb = tdvp(L, chi)
            cells.append(f"{duration(seconds / ASSUMED['tdvp_cluster_speedup'] if cluster else seconds)}, {memory(gb)}"
                         + ("" if gb < limit else " (too big)"))
        rows.append((chi, TDVP_ERROR[chi], *cells))
    notes = ["Error = largest deviation of ⟨σ^α⟩ from the exact result at L = 8 up to t = 10 (measured).",
             "Time measured for χ = 16 – 128 at L = 8 and χ = 16, 32 at L = 16; the rest is the scaling law fitted to them.",
             "Below the exact bond dimension 4^(L/2) the error does not converge: at L = 6 it is 1e-2 at χ = 32 "
             "and 3e-5 at χ = 64 (exact)."]
    if cluster:
        notes.append("One run does not get faster with 112 cores (assumed); the node can run about 14 such runs at once.")
    return table("ITensor TDVP (Julia, density matrix as an MPS)",
                 ("bond dimension χ", "error", "L = 8: time, RAM", "L = 16", "L = 32"), rows, notes)


if __name__ == "__main__":
    text = "\n".join([
        "# Time and RAM, one run: γ = 0.01, t = 100, dt = 0.1 (1000 steps)", "",
        "## This machine: 16 threads, 15 GB", "",
        ed_table(False), trajectory_table(False), tdvp_table(False),
        f"## Cluster node: {ASSUMED['cluster_cores']} cores, {ASSUMED['cluster_ram_gb']} GB", "",
        ed_table(True), trajectory_table(True), tdvp_table(True),
    ])
    print(text)
    (DATA / "resource_tables.md").write_text(text)
    print(f"wrote {DATA / 'resource_tables.md'}")
