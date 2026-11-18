#!/usr/bin/env python3
"""
Local driver of the decay-rate histograms under depolarizing noise (rust binaries `state_hist`, `state_hist_ed`):
benchmark, budget, then the runs, as single-threaded jobs in parallel on (physical cores − 1). Local only.

    ~/anaconda3/bin/python scripts/run_state_hist.py bench    # wall time per unit t, budget table -> plan.json
    ~/anaconda3/bin/python scripts/run_state_hist.py run      # size scan, then the scar and bulk classes
    ~/anaconda3/bin/python scripts/run_state_hist.py status   # finished and pending runs
    ~/anaconda3/bin/python scripts/run_state_hist.py kick     # step 5 (optional): single_flip kicks at L = 16
    ~/anaconda3/bin/python scripts/run_state_hist.py leonardo # the runs not finished here, as Leonardo jobs
    ~/anaconda3/bin/python scripts/run_state_hist.py leonardo-thermo   # sizes L = 6 … 16 for every state (below)
    options: --workers N (default 9: 10 physical cores, 6P + 4E, of which WSL reports 8), --deadline-hours 24

Restart: every run writes <name>.part and renames it to <name>.csv when it ends; a run whose .csv exists is skipped,
so the driver can be stopped (Ctrl-C or kill: the running jobs are killed and their .part files are redone later)
and started again without losing finished runs. `run` reads plan.json (written by `bench`): which items the budget
dropped, and the deadline (first `run` + --deadline-hours): no job is started that is expected to end after it, and
jobs still running at the deadline are killed.

Grid (γ = 0.1, 0.05, 0.03; depolarizing noise; realization seed 1):
  1. size scan: coherent cos θ = 0; exact ρ at L = 8, 10, trajectories (64) at L = 12, 14; t-max 60; L = 16 only for a
     γ where Γ_e(12) and Γ_e(14) do not agree (state_hist_data.agree).
  2. scar class: cos θ = 0, ±0.2, ±0.4, ±0.6, ±0.8 at the two smallest sizes from the saturation size of the scan
     (L ≤ 10 exact, L ≥ 12 trajectories, 64), t-max = 3/Γ_e of the scan at that (L, γ); no runs for a γ that does not
     saturate.
  3. bulk class: init seeds 1 … 100; n̄ by ED at L = 10, 12; γ = 0 state vectors at L = 10 (floor of the exact runs),
     12, 16; exact ρ at L = 10 and trajectories (32) at L = 12 for every γ; t-max 10, output step 0.02 up to t = 1.5,
     then 0.1.
Budget order of the drops (if the estimate exceeds 24 h): L = 16 runs; cos θ = ±0.2, ±0.6; bulk seeds 31 … 100;
γ = 0.03.
Per-state thermodynamic rates (`leonardo-thermo`, Leonardo jobs): every probe state at L = 6, 8, 10 (exact ρ) and
12, 14, 16 (trajectories) for every γ, for Γ_∞ = A of A + B/L. Scar class: the 9 coherent states (fixed θ, so the
same state and the same E/L at every L), 64 trajectories, t-max 60 with the adaptive stop. Bulk class: bulk product
states 1 … 100 at every L (the generator is seeded with L: different states at different L, same energy window; the
thermodynamic rate is taken per energy density, Γ_∞(ε)), 32 trajectories, t-max 10, γ = 0 runs at every L. n̄: ED
for L ≤ 12 (L = 6, 8 computed on the laptop, in the tarball), late-time average t ≥ 5 of the γ = 0 run for L = 14
(`state_hist --write-nbar 5`, first phase of the job that needs it) and L = 16 (written on the laptop from the
existing γ = 0 runs, in the tarball).
Step 5 (`kick`, only if it ends before the deadline): single_flip at L = 16, cos θ = 0, ±0.4, kicks x, y, z,
4 phases × sites 0, 4, 8, 12, τ ≤ 16; one job per θ with 3 threads (prediction: plot_scripts/state_hist_kick.py).
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "plot_scripts"))
import state_hist_data as shd  # noqa: E402

BIN = REPO / "rust" / "target" / "release"
OUT = REPO / "data" / "state_hist"
LOGS = OUT / "logs"
BENCH = OUT / "bench"
PLAN = OUT / "plan.json"

GAMMAS = [0.1, 0.05, 0.03]
COS_ALL = [0.0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8]
COS_DROPPABLE = {0.2, -0.2, 0.6, -0.6}
N_BULK, N_BULK_MIN = 100, 30
TRAJ_SCAN, TRAJ_SCAR, TRAJ_BULK = 64, 64, 32
SCAN_SIZES = [8, 10, 12, 14]
T_MAX_SCAN, T_MAX_BULK = 60.0, 10.0
BULK_GRID = ["--dt", "0.1", "--dt-fine", "0.02", "--t-fine", "1.5"]
HOLD = 2.0 + 2 * math.pi  # stop-hold + one period
# scar Γ_e guesses for the budget only (σᶻ results at γ' = 2γ/3, L ≈ 12 … 16)
GAMMA_GUESS = {0.1: 0.37, 0.05: 0.26, 0.03: 0.19}
SIZES_GUESS = {0.1: (10, 12), 0.05: (12, 14), 0.03: (14, 16)}
BUDGET_H = 24.0

_running = set()
_lock = threading.Lock()
_stop = threading.Event()


def log(msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "driver.log", "a") as f:
        f.write(line + "\n")


def fmt(x):
    return f"{x:g}"


class Job:
    def __init__(self, name, args, out, est, threads=1, binary="state_hist"):
        self.name, self.args, self.out, self.est, self.threads, self.binary = name, args, out, est, threads, binary

    def done(self):
        return self.out.exists()

    def cmdline(self):
        return " ".join([str(BIN / self.binary)] + self.args)

    def active(self):
        """Run by a process that this driver did not start (e.g. one kept running across a driver restart): a process
        with exactly this command line exists."""
        return self.cmdline() in running_cmdlines()


_ps_cache = [0.0, set()]


def running_cmdlines():
    """Command lines of every running process (refreshed at most every 5 s)."""
    if time.time() - _ps_cache[0] > 5:
        out = subprocess.run(["ps", "-eo", "args", "--no-headers"], capture_output=True, text=True).stdout
        _ps_cache[:] = [time.time(), {line.strip() for line in out.splitlines()}]
    return _ps_cache[1]


def state_job(method, l, gamma, init_args, tag, est, extra=(), data=OUT):
    if method == "pure":
        out = data / f"states_pure_L{l}_g0_{tag}.csv"
        args = ["--method", "pure", "--L", str(l)]
    else:
        prefix = "states" if method == "exact" else "states_traj"
        out = data / f"{prefix}_L{l}_g{fmt(gamma)}_noisedepol_{tag}.csv"
        args = ["--method", method, "--L", str(l), "--gamma", fmt(gamma), "--noise", "depol"]
    return Job(out.stem, args + list(init_args) + list(extra), out, est)


def coherent(c):
    tag = f"initcoherent_cos{fmt(c)}" if c != 0 else "initcoherent_cos0"
    return ["--init", "coherent", "--cos-theta", fmt(c)], tag


def bulk(seed, l=None, data=OUT):
    args = ["--init", "bulk-product", "--init-seed", str(seed)]
    if l is not None and l <= 12:
        args += ["--nbar", str(shd.nbar_file(l, seed, data))]
    return args, f"initbulk-product_s{seed}"


def execute(job, data=OUT):
    env = dict(os.environ, SCARS_DATA_DIR=str(data), RAYON_NUM_THREADS=str(job.threads),
               MKL_NUM_THREADS=str(job.threads))
    LOGS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log(f"START {job.name} (expected {job.est / 60:.1f} min)")
    with open(LOGS / f"{job.name}.log", "w") as f:
        p = subprocess.Popen([str(BIN / job.binary)] + job.args, stdout=f, stderr=subprocess.STDOUT, env=env)
        with _lock:
            _running.add(p)
        p.wait()
        with _lock:
            _running.discard(p)
    status = "DONE" if p.returncode == 0 else f"FAILED ({p.returncode})"
    log(f"{status} {job.name} in {(time.time() - start) / 60:.1f} min")


class Pool:
    """`workers` threads that always start the cheapest queued job; jobs can be added while it runs (`submit`), and
    `hold`/`release` keep the workers waiting for jobs that are still to be added."""

    def __init__(self, workers, deadline=None, hold=False):
        self.heap, self.count, self.holds = [], 0, int(hold)
        self.external = []  # jobs run by processes this driver did not start: they take a worker slot while alive
        self.own = 0
        self.cond = threading.Condition()
        self.deadline = deadline
        self.workers = workers
        self.threads = []

    def start(self):
        """Starts the workers (after the first jobs are queued: a worker that finds nothing queued and no hold ends)."""
        self.threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(self.workers)]
        for t in self.threads:
            t.start()

    def submit(self, jobs):
        import heapq
        jobs = [j for j in jobs if not j.done()]
        for j in jobs:
            if j.active():
                log(f"RUNNING ELSEWHERE {j.name}: not started here, it takes a worker slot while it runs")
        with self.cond:
            self.external += [j for j in jobs if j.active()]
        jobs = [j for j in jobs if not j.active()]
        with self.cond:
            for j in jobs:
                heapq.heappush(self.heap, (j.est, self.count, j))
                self.count += 1
            self.cond.notify_all()
        log(f"queued {len(jobs)} jobs, {sum(j.est for j in jobs) / 3600:.2f} core-h")

    def hold(self):
        with self.cond:
            self.holds += 1

    def release(self):
        with self.cond:
            self.holds -= 1
            self.cond.notify_all()

    def worker(self):
        import heapq
        while True:
            with self.cond:
                while not _stop.is_set() and ((not self.heap and self.holds > 0) or (self.heap and self.full())):
                    self.cond.wait(timeout=30)
                if not self.heap or _stop.is_set():
                    return
                _, _, job = heapq.heappop(self.heap)
                self.own += 1
            try:
                if job.done():
                    continue
                if self.deadline is not None and time.time() + job.est > self.deadline:
                    log(f"SKIP {job.name}: expected end after the deadline")
                    continue
                execute(job)
            finally:
                with self.cond:
                    self.own -= 1
                    self.cond.notify_all()

    def full(self):
        """All worker slots taken (own jobs plus external ones still running)."""
        self.external = [j for j in self.external if j.active()]
        return self.own + len(self.external) >= self.workers

    def join(self):
        while any(t.is_alive() for t in self.threads):
            if self.deadline is not None and time.time() > self.deadline and not _stop.is_set():
                log("deadline reached: killing the running jobs")
                _stop.set()
                kill_all()
            time.sleep(5)


def run_jobs(jobs, workers, deadline=None):
    """Runs the jobs whose output is missing, cheapest first, `workers` at a time; returns when all have ended."""
    todo = [j for j in jobs if not j.done()]
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} finished before, {len(todo)} to run, expected "
        f"{sum(j.est for j in todo) / 3600:.2f} core-h ({sum(j.est for j in todo) / 3600 / workers:.2f} h on {workers} workers)")
    pool = Pool(workers, deadline)
    pool.submit(todo)
    pool.start()
    pool.join()


def kill_all():
    with _lock:
        for p in list(_running):
            p.kill()


def handler(signum, frame):
    log(f"signal {signum}: stopping, killing the running jobs")
    _stop.set()
    kill_all()
    sys.exit(1)


# ---------------------------------------------------------------------------------------------------------------------
# bench and budget

def bench(workers):
    BENCH.mkdir(parents=True, exist_ok=True)
    for f in BENCH.glob("states_*"):
        f.unlink()  # scratch output of an earlier bench
    init_c, tag_c = coherent(0.0)
    init_b, tag_b = bulk(1)
    cases = [  # (key, method, L, n_traj, t_max, extra)
        ("exact_8", "exact", 8, 1, 2.0, ["--dt", "0.1"]),
        ("exact_10", "exact", 10, 1, 1.0, ["--dt", "0.1"]),
        ("exact_10_fine", "exact", 10, 1, 0.4, ["--dt", "0.02", "--dt-fine", "0.02"]),
        ("traj_12", "traj", 12, 4, 1.0, ["--dt", "0.1"]),
        ("traj_12_fine", "traj", 12, 4, 0.4, ["--dt", "0.02", "--dt-fine", "0.02"]),
        ("traj_14", "traj", 14, 4, 1.0, ["--dt", "0.1"]),
        ("traj_16", "traj", 16, 2, 1.0, ["--dt", "0.1"]),
        ("pure_12", "pure", 12, 1, 1.0, ["--dt", "0.1"]),
        ("pure_12_fine", "pure", 12, 1, 0.4, ["--dt", "0.02", "--dt-fine", "0.02"]),
        ("pure_16", "pure", 16, 1, 1.0, ["--dt", "0.1"]),
        ("pure_16_fine", "pure", 16, 1, 0.4, ["--dt", "0.02", "--dt-fine", "0.02"]),
    ]
    results = {}

    def run_case(key, method, l, n, t_max, extra, copy=None):
        d = BENCH / (key if copy is None else f"{key}_c{copy}")
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("states_*"):
            f.unlink()
        init, tag = (init_b, tag_b) if method == "pure" else (init_c, tag_c)
        args = ["--method", method, "--L", str(l)] + ([] if method == "pure" else ["--gamma", "0.1", "--noise", "depol"])
        args += init + ["--t-max", fmt(t_max), "--trajectories", str(n)] + extra
        if method != "traj":
            args = [a for i, a in enumerate(args) if not (a == "--trajectories" or (i > 0 and args[i - 1] == "--trajectories"))]
        env = dict(os.environ, SCARS_DATA_DIR=str(d), RAYON_NUM_THREADS="1")
        subprocess.run([str(BIN / "state_hist")] + args, env=env, check=True, capture_output=True)
        f = next(d.glob("states_*.csv"))
        _, df, end = shd.read_run(f)
        return float(end["wall_s"]) / (float(end["t"]) * n)

    for key, method, l, n, t_max, extra in cases:
        results[key] = run_case(key, method, l, n, t_max, extra)
        log(f"bench {key}: {results[key]:.3f} s per unit t" + (" per trajectory" if method == "traj" else ""))
    # concurrency: `workers` copies at once against one alone
    for key, method, l, n, t_max, extra in [("conc_exact_10", "exact", 10, 1, 0.5, ["--dt", "0.1"]),
                                            ("conc_traj_14", "traj", 14, 2, 1.0, ["--dt", "0.1"])]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            costs = list(pool.map(lambda c: run_case(key, method, l, n, t_max, extra, copy=c), range(workers)))
        single = results["exact_10" if method == "exact" else "traj_14"]
        results[key] = sum(costs) / len(costs) / single
        log(f"bench {key}: {workers} copies at once are {results[key]:.2f}x slower than one alone")
    (BENCH / "bench.json").write_text(json.dumps(results, indent=1))
    return results


def cost_per_t(b, method, l, fine_frac=0.0):
    """Single-job wall seconds per unit t (per trajectory for traj), with a fraction fine_frac at output step 0.02,
    under full load (concurrency factor)."""
    def at(key):
        if key in b:
            return b[key]
        raise KeyError(key)
    if method == "exact":
        base = {8: at("exact_8"), 10: at("exact_10")}[l]
        fine = at("exact_10_fine") * base / at("exact_10")
        return ((1 - fine_frac) * base + fine_frac * fine) * b["conc_exact_10"]
    if method == "traj":
        base = {12: at("traj_12"), 14: at("traj_14"), 16: at("traj_16")}[l]
        fine = at("traj_12_fine") * base / at("traj_12")
        return ((1 - fine_frac) * base + fine_frac * fine) * b["conc_traj_14"]
    base = {10: at("pure_12") / 4.5, 12: at("pure_12"), 16: at("pure_16")}[l]
    fine = {10: at("pure_12_fine") / 4.5, 12: at("pure_12_fine"), 16: at("pure_16_fine")}[l]
    return ((1 - fine_frac) * base + fine_frac * fine) * b["conc_traj_14"]


def scar_t(gamma, t_max):
    g = GAMMA_GUESS[gamma]
    return min(t_max, 1.5 / g + HOLD)


def items(b, drop):
    """Budget items: (name, runs, hours per run) for the grid without the dropped parts."""
    gammas = [g for g in GAMMAS if not ("gamma0.03" in drop and g == 0.03)]
    cos = [c for c in COS_ALL if not ("cos" in drop and c in COS_DROPPABLE)]
    n_bulk = N_BULK_MIN if "bulk" in drop else N_BULK
    l16 = "L16" not in drop
    method = lambda l: "exact" if l <= 10 else "traj"
    ntraj = lambda l, n: 1 if l <= 10 else n
    out = []
    for l in SCAN_SIZES + ([16] if l16 else []):
        gs = gammas if l != 16 else [g for g in gammas if g != 0.1]  # assumed: L = 16 needed at γ = 0.05, 0.03
        for g in gs:
            h = scar_t(g, T_MAX_SCAN) * cost_per_t(b, method(l), l) * ntraj(l, TRAJ_SCAN) / 3600
            out.append((f"scan L={l} γ={g}", 1, h))
    for g in gammas:
        sizes = SIZES_GUESS[g]
        if not l16 and 16 in sizes:
            sizes = (12, 14)
        for l in sizes:
            h = scar_t(g, 3 / GAMMA_GUESS[g]) * cost_per_t(b, method(l), l) * ntraj(l, TRAJ_SCAR) / 3600
            out.append((f"scar L={l} γ={g} ({len(cos) - 1} cos θ)", len(cos) - 1, h))
    for l in [10, 12] + ([16] if l16 else []):
        out.append((f"bulk γ=0 pure L={l}", n_bulk, T_MAX_BULK * cost_per_t(b, "pure", l, 0.15) / 3600))
    for g in gammas:
        out.append((f"bulk exact L=10 γ={g}", n_bulk, T_MAX_BULK * cost_per_t(b, "exact", 10, 0.15) / 3600))
        out.append((f"bulk traj L=12 γ={g}", n_bulk, T_MAX_BULK * cost_per_t(b, "traj", 12, 0.15) * TRAJ_BULK / 3600))
    return out


def budget(b, workers):
    drop = []
    order = ["L16", "cos", "bulk", "gamma0.03"]
    while True:
        rows = items(b, drop)
        total = sum(n * h for _, n, h in rows) / workers
        if total <= BUDGET_H or len(drop) == len(order):
            break
        drop.append(order[len(drop)])
    print(f"\n{'item':40s} {'runs':>5s} {'h per run':>10s} {'core-h':>8s}")
    for name, n, h in rows:
        print(f"{name:40s} {n:5d} {h:10.3f} {n * h:8.2f}")
    print(f"total {sum(n * h for _, n, h in rows):.1f} core-h = {total:.1f} h on {workers} workers (budget {BUDGET_H} h); "
          f"dropped: {', '.join(drop) if drop else 'nothing'}")
    return drop, total


# ---------------------------------------------------------------------------------------------------------------------
# the runs

def scan_rates(gammas):
    df = shd.summarize(OUT)
    sc = df[(df.family == "coherent") & (df.cos_theta == 0) & (df.method != "pure")]
    rates = {}
    for g in gammas:
        rows = sc[sc.gamma == g].sort_values("L")
        rates[g] = {int(r.L): (r.Gamma_e, r.sigma_Gamma, bool(r.bound)) for r in rows.itertuples()}
    return rates


def saturation(r):
    """Smallest scanned L whose Γ_e agrees with every larger scanned size, and the next size; None if none."""
    sizes = sorted(r)
    for i, l in enumerate(sizes[:-1]):
        if all(shd.agree(r[l][0], r[l][1], r[m][0], r[m][1]) for m in sizes[i + 1:]):
            return l, sizes[i + 1]
    return None


def run(workers, deadline_hours):
    plan = json.loads(PLAN.read_text())
    if "deadline" not in plan:
        plan["deadline"] = time.time() + deadline_hours * 3600
        PLAN.write_text(json.dumps(plan, indent=1))
    deadline = plan["deadline"]
    b = plan["bench"]
    drop = plan["drop"]
    gammas = [g for g in GAMMAS if not ("gamma0.03" in drop and g == 0.03)]
    cos_list = [c for c in COS_ALL if not ("cos" in drop and c in COS_DROPPABLE)]
    n_bulk = N_BULK_MIN if "bulk" in drop else N_BULK
    log(f"run: workers {workers}, dropped {drop or 'nothing'}, deadline {time.ctime(deadline)}")
    method = lambda l: "exact" if l <= 10 else "traj"
    ntraj = lambda l, n: ["--trajectories", str(n)] if l > 10 else []
    est = lambda l, t, n: t * cost_per_t(b, method(l), l) * (n if l > 10 else 1)

    # 1. size scan
    init, tag = coherent(0.0)
    jobs = [state_job(method(l), l, g, init, tag, est(l, scar_t(g, T_MAX_SCAN), TRAJ_SCAN),
                      ["--t-max", fmt(T_MAX_SCAN), "--dt", "0.1"] + ntraj(l, TRAJ_SCAN))
            for g in gammas for l in SCAN_SIZES]
    run_jobs(jobs, workers, deadline)
    rates = scan_rates(gammas)
    extra16 = []
    if "L16" not in drop:
        for g in gammas:
            r = rates[g]
            if 12 in r and 14 in r and not shd.agree(r[12][0], r[12][1], r[14][0], r[14][1]):
                extra16.append((g, state_job("traj", 16, g, init, tag, est(16, scar_t(g, T_MAX_SCAN), TRAJ_SCAN),
                                             ["--t-max", fmt(T_MAX_SCAN), "--dt", "0.1"] + ntraj(16, TRAJ_SCAN))))
        if extra16:
            log(f"L = 16 for γ = {', '.join(str(g) for g, _ in extra16)}")
    pending = {g for g, _ in extra16}

    # 2. n̄ by ED for the bulk states (before the bulk runs that read it)
    for l in [10, 12]:
        missing = [s for s in range(1, n_bulk + 1) if not shd.nbar_file(l, s).exists()]
        if missing:
            job = Job(f"nbar_L{l}", ["--L", str(l), "--init-seeds", f"1-{n_bulk}"], shd.nbar_file(l, n_bulk), 60,
                      threads=max(1, workers - 2), binary="state_hist_ed")
            execute(job)

    def scar_jobs(g, sizes_g):
        jobs = []
        for l in sizes_g:
            g_scan = rates[g][l][0]
            t_max = math.ceil(30 / g_scan) / 10  # 3/Γ_e, rounded up to 0.1
            for c in cos_list:
                init_c, tag_c = coherent(c)
                jobs.append(state_job(method(l), l, g, init_c, tag_c, est(l, min(t_max, 1.5 / g_scan + HOLD), TRAJ_SCAR),
                                      ["--t-max", fmt(t_max), "--dt", "0.1"] + ntraj(l, TRAJ_SCAR)))
        return jobs

    def decide(g):
        s = saturation(rates[g])
        log(f"scan γ = {g}: " + ", ".join(f"L={l}: {v[0]:.4f} ± {v[1]:.4f}" for l, v in sorted(rates[g].items()))
            + f" -> sizes {s if s else 'not converged'}")
        plan.setdefault("scan_sizes", {})[str(g)] = s
        PLAN.write_text(json.dumps(plan, indent=1))
        return s

    # 3. one pool, cheapest first: the L = 16 scan runs, the bulk class, the scar class of every γ that is decided;
    #    the scar runs of the other γ are added when their L = 16 run has finished
    jobs = [j for _, j in extra16]
    for g in gammas:
        if g not in pending:
            s = decide(g)
            if s:
                jobs += scar_jobs(g, s)
    for s in range(1, n_bulk + 1):
        for l in [10, 12] + ([] if "L16" in drop else [16]):
            init, tag = bulk(s, l)
            jobs.append(state_job("pure", l, 0.0, init, tag, T_MAX_BULK * cost_per_t(b, "pure", l, 0.15),
                                  ["--t-max", fmt(T_MAX_BULK)] + BULK_GRID))
        for g in gammas:
            init, tag = bulk(s, 10)
            jobs.append(state_job("exact", 10, g, init, tag, T_MAX_BULK * cost_per_t(b, "exact", 10, 0.15),
                                  ["--t-max", fmt(T_MAX_BULK)] + BULK_GRID))
            init, tag = bulk(s, 12)
            jobs.append(state_job("traj", 12, g, init, tag, T_MAX_BULK * cost_per_t(b, "traj", 12, 0.15) * TRAJ_BULK,
                                  ["--t-max", fmt(T_MAX_BULK), "--trajectories", str(TRAJ_BULK)] + BULK_GRID))
    todo = [j for j in jobs if not j.done()]
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} finished before, {len(todo)} to run now"
        + (f", scar runs of γ = {sorted(pending)} after their L = 16 run" if pending else ""))
    pool = Pool(workers, deadline, hold=True)
    pool.submit(todo)
    pool.start()

    def follow_up():
        nonlocal rates
        submitted = {j.name for j in todo if not j.active()}
        while pending and not _stop.is_set():
            for g, j in extra16:
                if g in pending and j.done():
                    rates = scan_rates(gammas)
                    s = decide(g)
                    if s:
                        pool.submit(scar_jobs(g, s))
                    pending.discard(g)
                elif g in pending and not j.active() and j.name not in submitted:
                    pool.submit([j])  # its process ended without output: run it here
                    submitted.add(j.name)
            time.sleep(60)
        pool.release()

    threading.Thread(target=follow_up, daemon=True).start()
    pool.join()
    df, out = shd.write_summary(OUT)
    log(f"summary: {len(df)} runs -> {out}")


KICK_COS = [0.0, 0.4, -0.4]


def kick_jobs(b, threads, data=OUT):
    core_s = 48 * 16 * b["pure_16"] * b["conc_traj_14"]
    jobs = []
    for c in KICK_COS:
        out = Path(data) / f"single_flip_L16_cos{fmt(c) if c else '0'}_kickxyz_sites0-4-8-12.csv"
        args = ["--L", "16", "--phases", "4", "--sites", "0,4,8,12", "--t-max", "16", "--dt", "0.1",
                "--cos-theta", fmt(c), "--kick", "x,y,z"]
        jobs.append(Job(out.stem, args, out, core_s / threads, threads=threads, binary="single_flip"))
    return jobs


def kick(workers):
    plan = json.loads(PLAN.read_text())
    deadline = plan.get("deadline", time.time() + BUDGET_H * 3600)
    threads = max(1, workers // len(KICK_COS))
    jobs = kick_jobs(plan["bench"], threads)
    log(f"kick: {len(jobs)} single_flip jobs, {threads} threads each, expected {jobs[0].est / 3600:.2f} h each")
    run_jobs(jobs, len(KICK_COS), deadline)


# ---------------------------------------------------------------------------------------------------------------------
# Leonardo: the runs not finished here, as Slurm jobs (generated on the laptop, copied by hand like data/bashes)

LEO_REPO = PurePosixPath("/leonardo/home/userexternal/wccuirom/repos/open_exact_scars")
LEO_OUT = LEO_REPO / "data" / "state_hist"
LEO_JOBS = REPO / "data" / "bashes" / "state_hist"  # $REPO/data/bashes/state_hist on Leonardo
PARTIAL = OUT / "partial"  # .part files of runs stopped by hand (kept, never rerun into)
TARBALL = REPO / "data" / "state_hist_to_leonardo.tar.gz"
NODE_CORES = 112
MULTI_THREADS = 16  # trajectory runs at L ≥ 14: 16 threads each, 7 at a time

JOB_SCRIPT = r"""#!/bin/bash
# {desc}
# Generated by scripts/run_state_hist.py leonardo ({date}). Run list {name}.txt: one run per line,
# name|output|threads|binary arguments; a run whose output exists in $OUT is skipped (resubmitting never redoes data).
# {parallel} runs at a time; per-run log (with /usr/bin/time -v) in $OUT/logs_leonardo/<name>.log.

#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --job-name={name}
#SBATCH --error={name}.err
#SBATCH --output={name}.out
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=sis26_collura

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export REPO={repo}
export OUT=$REPO/data/state_hist
export LOGDIR=$OUT/logs_leonardo
LIST=$REPO/data/bashes/state_hist/{name}.txt
mkdir -p "$LOGDIR"

echo "=== start $(date '+%F %T'), host $(hostname), job ${{SLURM_JOB_ID:-none}}, cpus ${{SLURM_CPUS_PER_TASK:-?}}"
lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|CPU\(s\)|NUMA node\(s\))'
free -g
export TIMEV=""
[ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v"

run_line() {{
    IFS='|' read -r name out threads cmd <<< "$1"
    if [ -e "$OUT/$out" ]; then echo "=== skip $name (output exists)"; return 0; fi
    echo "=== start $(date '+%F %T') $name ($threads threads)"
    SCARS_DATA_DIR=$OUT RAYON_NUM_THREADS=$threads OMP_NUM_THREADS=$threads \
        $TIMEV $REPO/rust/target/release/$cmd > "$LOGDIR/$name.log" 2>&1
    local status=$?
    echo "=== end $(date '+%F %T') $name exit $status"
    return $status
}}
export -f run_line

grep -v '^#' "$LIST" | xargs -d '\n' -n 1 -P {parallel} bash -c 'run_line "$1"' _
RUN_STATUS=$?  # xargs: 123 if a run failed

echo "=== end $(date '+%F %T'); runs without output:"
grep -v '^#' "$LIST" | while IFS='|' read -r name out threads cmd; do [ -e "$OUT/$out" ] || echo "    $name"; done
if [ -n "$SLURM_JOB_ID" ]; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,TotalCPU,MaxRSS,State -P
fi
exit $RUN_STATUS
"""

SUBMIT_SCRIPT = """#!/bin/bash
# {desc}
# Generated by scripts/run_state_hist.py leonardo ({date}).
#
# laptop:   git push   (the state_hist code must be in the pushed commit)
#           scp data/state_hist_to_leonardo.tar.gz <user>@login.leonardo.cineca.it:{repo}/data/
# Leonardo: cd $REPO && git pull && cd rust && cargo build --release --bin state_hist
#           cd $REPO/data && tar xzf state_hist_to_leonardo.tar.gz --skip-old-files   # bashes/state_hist, n̄ files
#           bash $REPO/data/bashes/state_hist/{submit}
# back:     cd $REPO/data && tar czf state_hist_results.tar.gz state_hist/states_* \\
#               state_hist/logs_leonardo bashes/state_hist/*.out bashes/state_hist/*.err
#           laptop: mkdir -p data/state_hist_leonardo && tar xzf state_hist_results.tar.gz -C data/state_hist_leonardo
#                   cp -n data/state_hist_leonardo/state_hist/states_* data/state_hist/   # -n: never overwrites
cd "$(dirname "$0")"
{sbatch}
"""


def partial_rate(path):
    """(Γ_e, σ_Γ, bound) of a run stopped before its end, if its 1/e crossing and the envelope window after it (one
    period) are complete; None otherwise."""
    meta, df, _ = shd.read_run(path)
    df = df.dropna()  # the last line of a killed run can be cut
    t, a = df["t"].to_numpy(), df["a"].to_numpy()
    a_err = df["a_err"].to_numpy() if int(meta["trajectories"]) > 1 else None
    g, s, te, bound = shd.gamma_e(t, a, a_err, a[0])
    if bound or te + shd.PERIOD + 0.1 > t[-1]:
        return None
    return g, s, bound


def run_size(job):
    return int(job.args[job.args.index("--L") + 1])


def leonardo():
    """Writes data/bashes/state_hist/ (run lists, Slurm scripts, submit files) and data/state_hist_to_leonardo.tar.gz
    (those plus the n̄ files) for every run of the grid that is not finished here; no budget drops (`plan["drop"]` is
    ignored). Scan decisions: from the finished runs here, and for L = 16 from the stopped runs in partial/ if their
    1/e window is complete (the complete L = 16 runs are in the jobs; same noise seeds, so the same trajectories)."""
    plan = json.loads(PLAN.read_text())
    b = plan["bench"]
    method = lambda l: "exact" if l <= 10 else "traj"
    ntraj = lambda l, n: ["--trajectories", str(n)] if l > 10 else []
    est = lambda l, t, n: t * cost_per_t(b, method(l), l) * (n if l > 10 else 1)

    rates = scan_rates(GAMMAS)
    init0, tag0 = coherent(0.0)
    jobs, notes = [], []
    for g in GAMMAS:
        r = rates[g]
        if 12 in r and 14 in r and not shd.agree(r[12][0], r[12][1], r[14][0], r[14][1]) and 16 not in r:
            jobs.append(state_job("traj", 16, g, init0, tag0, est(16, scar_t(g, T_MAX_SCAN), TRAJ_SCAN),
                                  ["--t-max", fmt(T_MAX_SCAN), "--dt", "0.1"] + ntraj(16, TRAJ_SCAN), data=LEO_OUT))
            part = PARTIAL / f"{jobs[-1].name}.csv.part"
            pr = partial_rate(part) if part.exists() else None
            if pr is None:
                notes.append(f"γ = {g}: L = 16 scan not usable yet; scar runs of this γ not generated (run the "
                             f"L = 16 scan, then this command again)")
                continue
            r[16] = pr
            notes.append(f"γ = {g}: L = 16 from the stopped run {part.name}")
        s = saturation(r)
        line = ", ".join(f"L={l}: {v[0]:.4f} ± {v[1]:.4f}" for l, v in sorted(r.items()))
        notes.append(f"scan γ = {g}: {line} -> sizes {s if s else 'not converged (no scar runs)'}")
        plan.setdefault("scan_sizes", {})[str(g)] = s
        for l in (s or []):
            g_scan = r[l][0]
            t_max = math.ceil(30 / g_scan) / 10  # 3/Γ_e, as in `run`
            for c in COS_ALL:
                init_c, tag_c = coherent(c)
                jobs.append(state_job(method(l), l, g, init_c, tag_c,
                                      est(l, min(t_max, 1.5 / g_scan + HOLD), TRAJ_SCAR),
                                      ["--t-max", fmt(t_max), "--dt", "0.1"] + ntraj(l, TRAJ_SCAR), data=LEO_OUT))
    for s in range(1, N_BULK + 1):
        for l in [10, 12, 16]:
            init, tag = bulk(s, l, LEO_OUT)
            jobs.append(state_job("pure", l, 0.0, init, tag, T_MAX_BULK * cost_per_t(b, "pure", l, 0.15),
                                  ["--t-max", fmt(T_MAX_BULK)] + BULK_GRID, data=LEO_OUT))
        for g in GAMMAS:
            init, tag = bulk(s, 10, LEO_OUT)
            jobs.append(state_job("exact", 10, g, init, tag, T_MAX_BULK * cost_per_t(b, "exact", 10, 0.15),
                                  ["--t-max", fmt(T_MAX_BULK)] + BULK_GRID, data=LEO_OUT))
            init, tag = bulk(s, 12, LEO_OUT)
            jobs.append(state_job("traj", 12, g, init, tag, T_MAX_BULK * cost_per_t(b, "traj", 12, 0.15) * TRAJ_BULK,
                                  ["--t-max", fmt(T_MAX_BULK), "--trajectories", str(TRAJ_BULK)] + BULK_GRID,
                                  data=LEO_OUT))
    seen, todo = set(), []
    for j in jobs:  # first occurrence wins: the L = 16 scan run is also the cos θ = 0 scar run at L = 16
        if j.name in seen or (OUT / f"{j.name}.csv").exists():
            continue
        seen.add(j.name)
        if j.binary == "state_hist" and "traj" in j.args and run_size(j) >= 14:
            j.threads = MULTI_THREADS
        todo.append(j)

    date = time.strftime("%Y-%m-%d %H:%M")
    LEO_JOBS.mkdir(parents=True, exist_ok=True)
    packs = [
        ("state_hist_single", [j for j in todo if j.threads == 1], NODE_CORES, "128G",
         "single-threaded runs: bulk class (exact ρ L = 10, trajectories L = 12) and scar class at L ≤ 12"),
        ("state_hist_multi", [j for j in todo if j.threads > 1], NODE_CORES // MULTI_THREADS, "64G",
         f"trajectory runs at L ≥ 14 ({MULTI_THREADS} threads each): L = 16 size scan and scar class"),
    ]
    print("\n".join(notes))
    print(f"\n{'job':20s} {'runs':>5s} {'est. wall h':>11s} {'limit':>8s}   (estimate: laptop speed under full load)")
    for name, pack, parallel, mem, desc in packs:
        pack.sort(key=lambda j: -j.est)  # longest first
        wall_h = max([sum(j.est for j in pack) / NODE_CORES] + [j.est / j.threads for j in pack] + [0]) / 3600
        limit_h = min(24, math.ceil(4 * wall_h + 2))  # Leonardo under full load is not benchmarked
        lines = [f"# {desc}"] + [f"{j.name}|{j.name}.csv|{j.threads}|{j.binary} {' '.join(j.args)}" for j in pack]
        (LEO_JOBS / f"{name}.txt").write_text("\n".join(lines) + "\n")
        (LEO_JOBS / f"{name}.sh").write_text(JOB_SCRIPT.format(
            desc=desc, date=date, name=name, parallel=parallel, mem=mem, time=f"{limit_h:02d}:00:00", repo=LEO_REPO))
        counts = {}
        for j in pack:
            key = re.sub(r"_s\d+$|_cos-?[0-9.]+$", "", j.name)
            counts[key] = counts.get(key, 0) + 1
        print(f"{name:20s} {len(pack):5d} {wall_h:11.2f} {limit_h:6d} h")
        for key, n in sorted(counts.items()):
            print(f"    {n:4d}  {key}")
    for submit, names, desc in [("submit_state_hist.sh", ["state_hist_single", "state_hist_multi"],
                                 "Remaining state_hist runs (decay-rate histograms, depolarizing noise)")]:
        (LEO_JOBS / submit).write_text(SUBMIT_SCRIPT.format(
            desc=desc, date=date, repo=LEO_REPO, submit=submit, sbatch="\n".join(f"sbatch {n}.sh" for n in names)))
    plan["leonardo"] = dict(generated=date, runs=len(todo), notes=notes)
    PLAN.write_text(json.dumps(plan, indent=1))
    nbar = sorted(p.relative_to(REPO / "data") for p in OUT.glob("nbar_L*_initbulk-product_s*.csv"))
    subprocess.run(["tar", "czf", str(TARBALL), "-C", str(REPO / "data"),
                    str(LEO_JOBS.relative_to(REPO / "data"))] + [str(p) for p in nbar], check=True)
    for f in sorted(LEO_JOBS.glob("*.sh")):
        subprocess.run(["bash", "-n", str(f)], check=True)
    print(f"\nwritten: {LEO_JOBS}/ and {TARBALL} ({len(nbar)} n̄ files); bash -n passed")


# ---------------------------------------------------------------------------------------------------------------------
# per-state thermodynamic rates: every probe state at L = 6 … 16 (Leonardo)

THERMO_SIZES = [6, 8, 10, 12, 14, 16]
T_MAX_SCAR_THERMO = 60.0
THERMO_TAR = REPO / "data" / "state_hist_thermo_to_leonardo.tar.gz"
BULK_BIG_THREADS = 8  # bulk trajectory runs at L ≥ 14 (32 trajectories): 8 threads each, 14 at a time
# Leonardo cost, core-s per unit t (exact, pure: per run; traj: per trajectory), from the batch of 2026-11-18 (jobs
# 64711588, 64711589, a full node of runs): exact L = 10 87 (bulk, fine grid to t = 1.5); traj L = 12 1.0 (1 thread),
# L = 14 3.7, L = 16 18.2 (16 threads); exact L = 6, 8 scaled from L = 10 by L·4^L; pure = one trajectory.
LEO_COST = {("exact", 6): 0.3, ("exact", 8): 4.4, ("exact", 10): 87.0, ("traj", 12): 1.0, ("traj", 14): 3.7,
            ("traj", 16): 18.2, ("pure", 6): 0.02, ("pure", 8): 0.1, ("pure", 10): 0.3, ("pure", 12): 1.0,
            ("pure", 14): 3.7, ("pure", 16): 18.2}

THERMO_JOB_SCRIPT = r"""#!/bin/bash
# {desc}
# Generated by scripts/run_state_hist.py leonardo-thermo ({date}). Run lists, one run per line,
# name|output|threads|binary arguments, run one list after the other: {phase_doc}
# A run whose output exists in $OUT is skipped (resubmitting never redoes data).
# Per-run log (with /usr/bin/time -v) in $OUT/logs_leonardo_thermo/<name>.log.

#SBATCH --nodes=1
#SBATCH --cpus-per-task=112
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --job-name={name}
#SBATCH --error={name}.err
#SBATCH --output={name}.out
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=sis26_collura

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export REPO={repo}
export OUT=$REPO/data/state_hist
export LOGDIR=$OUT/logs_leonardo_thermo
LISTS=$REPO/data/bashes/state_hist
mkdir -p "$LOGDIR"

echo "=== start $(date '+%F %T'), host $(hostname), job ${{SLURM_JOB_ID:-none}}, cpus ${{SLURM_CPUS_PER_TASK:-?}}"
lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|CPU\(s\)|NUMA node\(s\))'
free -g
export TIMEV=""
[ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v"

run_line() {{
    IFS='|' read -r name out threads cmd <<< "$1"
    if [ -e "$OUT/$out" ]; then echo "=== skip $name (output exists)"; return 0; fi
    echo "=== start $(date '+%F %T') $name ($threads threads)"
    SCARS_DATA_DIR=$OUT RAYON_NUM_THREADS=$threads OMP_NUM_THREADS=$threads \
        $TIMEV $REPO/rust/target/release/$cmd > "$LOGDIR/$name.log" 2>&1
    local status=$?
    echo "=== end $(date '+%F %T') $name exit $status"
    return $status
}}
export -f run_line

RUN_STATUS=0
run_list() {{  # list file, runs at a time
    echo "=== list $1 ($2 at a time) $(date '+%F %T')"
    grep -v '^#' "$LISTS/$1" | xargs -d '\n' -n 1 -P "$2" bash -c 'run_line "$1"' _ || RUN_STATUS=$?  # 123: a run failed
}}
{phases}

echo "=== end $(date '+%F %T'); runs without output:"
for f in {lists}; do
    grep -v '^#' "$LISTS/$f" | while IFS='|' read -r name out threads cmd; do [ -e "$OUT/$out" ] || echo "    $name"; done
done
if [ -n "$SLURM_JOB_ID" ]; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,TotalCPU,MaxRSS,State -P
fi
exit $RUN_STATUS
"""

THERMO_SUBMIT = """#!/bin/bash
# Per-state thermodynamic rates: every probe state at L = 6 … 16 (decay-rate histograms, depolarizing noise)
# Generated by scripts/run_state_hist.py leonardo-thermo ({date}). The three jobs are independent.
#
# laptop:   git push   (state_hist --write-nbar must be in the pushed commit)
#           scp data/state_hist_thermo_to_leonardo.tar.gz <user>@login.leonardo.cineca.it:{repo}/data/
# Leonardo: cd $REPO && git pull && cd rust && cargo build --release --bin state_hist
#           cd $REPO/data && tar xzf state_hist_thermo_to_leonardo.tar.gz --skip-old-files   # job files, n̄ L = 6 … 16
#           bash $REPO/data/bashes/state_hist/submit_thermo.sh
# back:     cd $REPO/data/bashes/state_hist && sacct -u $USER -S <date> \\
#               --format=JobID,JobName%30,Elapsed,TotalCPU,MaxRSS,State -P > sacct_thermo.txt
#           cd $REPO/data && tar czf state_hist_thermo_results.tar.gz state_hist/states_* state_hist/nbar_L14_* \\
#               state_hist/logs_leonardo_thermo bashes/state_hist/thermo_*.out bashes/state_hist/thermo_*.err \\
#               bashes/state_hist/sacct_thermo.txt
#           laptop: mkdir -p data/state_hist_leonardo_thermo
#                   tar xzf data/state_hist_thermo_results.tar.gz -C data/state_hist_leonardo_thermo
#                   cp -n data/state_hist_leonardo_thermo/state_hist/{{states_*,nbar_L14_*}} data/state_hist/
cd "$(dirname "$0")"
{sbatch}
"""


def leonardo_thermo():
    """Writes the job files of the per-state thermodynamic rates (data/bashes/state_hist/thermo_*,
    submit_thermo.sh) and data/state_hist_thermo_to_leonardo.tar.gz (those plus the n̄ files); runs finished here are
    left out (see the module docs for the grid)."""
    plan = json.loads(PLAN.read_text())
    rates = scan_rates(GAMMAS)
    method = lambda l: "exact" if l <= 10 else "traj"

    def scar_t(g, l):
        """Expected run length: adaptive stop at 1.5/Γ + hold, Γ from the cos θ = 0 scan at the nearest size, × 0.7
        for the slower θ."""
        near = min(rates[g], key=lambda m: abs(m - l))
        return min(T_MAX_SCAR_THERMO, 1.5 / (0.7 * rates[g][near][0]) + HOLD)

    def est(meth, l, t, n=1):
        return t * LEO_COST[(meth, l)] * n

    jobs = []
    for g in GAMMAS:
        for l in THERMO_SIZES:
            n = TRAJ_SCAR if l > 10 else 1
            for c in COS_ALL:
                init_c, tag_c = coherent(c)
                extra = ["--t-max", fmt(T_MAX_SCAR_THERMO), "--dt", "0.1"] + (["--trajectories", str(n)] if l > 10 else [])
                jobs.append(state_job(method(l), l, g, init_c, tag_c, est(method(l), l, scar_t(g, l), n), extra,
                                      data=LEO_OUT))
    missing_nbar = []
    for s in range(1, N_BULK + 1):
        for l in THERMO_SIZES:
            init = ["--init", "bulk-product", "--init-seed", str(s)]
            tag = f"initbulk-product_s{s}"
            nbar = LEO_OUT / f"nbar_L{l}_{tag}.csv"
            if l != 14 and not shd.nbar_file(l, s).exists():
                missing_nbar.append(f"L{l}_s{s}")
            write = ["--write-nbar", fmt(shd.LATE)] if l == 14 else []
            jobs.append(state_job("pure", l, 0.0, init, tag, est("pure", l, T_MAX_BULK),
                                  ["--t-max", fmt(T_MAX_BULK)] + BULK_GRID + write, data=LEO_OUT))
            n = TRAJ_BULK if l > 10 else 1
            for g in GAMMAS:
                extra = ["--t-max", fmt(T_MAX_BULK)] + BULK_GRID + (["--trajectories", str(n)] if l > 10 else [])
                jobs.append(state_job(method(l), l, g, init + ["--nbar", str(nbar)], tag,
                                      est(method(l), l, T_MAX_BULK, n), extra, data=LEO_OUT))
    assert not missing_nbar, f"n̄ files missing here: {missing_nbar[:5]} …"
    seen, todo = set(), []
    for j in jobs:
        if j.name in seen or (OUT / f"{j.name}.csv").exists():
            continue
        seen.add(j.name)
        todo.append(j)

    def kind(j):
        big = "--method" in j.args and j.args[j.args.index("--method") + 1] == "traj" and run_size(j) >= 14
        if not big:
            return "pure14" if ("--write-nbar" in j.args) else "single"
        return "scar_big" if "coherent" in j.args else "bulk_big"

    for j in todo:
        k = kind(j)
        j.threads = {"scar_big": MULTI_THREADS, "bulk_big": BULK_BIG_THREADS}.get(k, 1)
    groups = {k: sorted([j for j in todo if kind(j) == k], key=lambda j: -j.est)
              for k in ["single", "pure14", "bulk_big", "scar_big"]}
    date = time.strftime("%Y-%m-%d %H:%M")
    LEO_JOBS.mkdir(parents=True, exist_ok=True)
    specs = [  # job, [(list name, runs, runs at a time)], memory, description
        ("thermo_single", [("thermo_single.txt", groups["single"], NODE_CORES)], "128G",
         "single-threaded runs: exact ρ L = 6, 8, 10, scar trajectories L = 12, γ = 0 runs L = 6, 8"),
        ("thermo_bulk_big", [("thermo_bulk_big_1.txt", groups["pure14"], NODE_CORES),
                             ("thermo_bulk_big_2.txt", groups["bulk_big"], NODE_CORES // BULK_BIG_THREADS)], "64G",
         f"bulk class at L = 14, 16: first the γ = 0 runs at L = 14 (they write n̄), then the trajectory runs "
         f"({BULK_BIG_THREADS} threads each)"),
        ("thermo_scar_big", [("thermo_scar_big.txt", groups["scar_big"], NODE_CORES // MULTI_THREADS)], "64G",
         f"scar class, trajectory runs at L = 14, 16 ({MULTI_THREADS} threads each)"),
    ]
    print(f"{'job':18s} {'list':24s} {'runs':>5s} {'core-h':>8s} {'wall h':>7s} {'limit':>6s}")
    total_core_h = 0.0
    for name, phases, mem, desc in specs:
        wall_h, lines_phases = 0.0, []
        for list_name, pack, parallel in phases:
            core = sum(j.est for j in pack)
            w = max([core / NODE_CORES] + [j.est / j.threads for j in pack] + [0]) / 3600
            wall_h += w
            total_core_h += core / 3600
            lines = [f"# {desc}"] + [f"{j.name}|{j.name}.csv|{j.threads}|{j.binary} {' '.join(j.args)}" for j in pack]
            (LEO_JOBS / list_name).write_text("\n".join(lines) + "\n")
            lines_phases.append(f"run_list {list_name} {parallel}")
            counts = {}
            for j in pack:
                key = re.sub(r"_s\d+$|_cos-?[0-9.]+$", "", j.name)
                counts[key] = counts.get(key, 0) + 1
            print(f"{name:18s} {list_name:24s} {len(pack):5d} {core / 3600:8.1f} {w:7.2f}")
            for key, n in sorted(counts.items()):
                print(f"{'':18s}     {n:4d}  {key}")
        limit_h = min(24, math.ceil(1.5 * wall_h + 1))
        print(f"{name:18s} {'(job)':24s} {'':5s} {'':8s} {wall_h:7.2f} {limit_h:4d} h")
        (LEO_JOBS / f"{name}.sh").write_text(THERMO_JOB_SCRIPT.format(
            desc=desc, date=date, name=name, mem=mem, time=f"{limit_h:02d}:00:00", repo=LEO_REPO,
            phase_doc=", then ".join(f"{l} ({p} at a time)" for l, _, p in phases),
            phases="\n".join(lines_phases), lists=" ".join(l for l, _, _ in phases)))
    (LEO_JOBS / "submit_thermo.sh").write_text(THERMO_SUBMIT.format(
        date=date, repo=LEO_REPO, sbatch="\n".join(f"sbatch {n}.sh" for n, _, _, _ in specs)))
    print(f"total {total_core_h:.0f} core-h = {total_core_h / NODE_CORES:.1f} node-h (Leonardo costs of 2026-11-18)")
    plan["thermo"] = dict(generated=date, runs=len(todo), core_h=round(total_core_h, 1))
    PLAN.write_text(json.dumps(plan, indent=1))
    nbar = sorted(p.relative_to(REPO / "data") for p in OUT.glob("nbar_L*_initbulk-product_s*.csv"))
    files = sorted(p.relative_to(REPO / "data") for p in LEO_JOBS.glob("thermo_*")) + \
        [(LEO_JOBS / "submit_thermo.sh").relative_to(REPO / "data")]
    subprocess.run(["tar", "czf", str(THERMO_TAR), "-C", str(REPO / "data")] + [str(p) for p in files + nbar],
                   check=True)
    for f in sorted(LEO_JOBS.glob("thermo_*.sh")) + [LEO_JOBS / "submit_thermo.sh"]:
        subprocess.run(["bash", "-n", str(f)], check=True)
    print(f"written: {LEO_JOBS}/thermo_*, submit_thermo.sh and {THERMO_TAR} ({len(nbar)} n̄ files); bash -n passed")


def status():
    files = sorted(OUT.glob("states_*.csv"))
    parts = sorted(OUT.glob("*.part"))
    print(f"{len(files)} finished runs, {len(parts)} in progress or interrupted (.part)")
    for p in parts:
        print("  ", p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["bench", "run", "status", "kick", "leonardo", "leonardo-thermo"])
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--deadline-hours", type=float, default=BUDGET_H)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "bench":
        b = bench(args.workers)
        drop, total = budget(b, args.workers)
        PLAN.write_text(json.dumps(dict(bench=b, drop=drop, estimate_h=total, workers=args.workers), indent=1))
        log(f"plan written: {PLAN}")
    elif args.command == "run":
        run(args.workers, args.deadline_hours)
    elif args.command == "kick":
        kick(args.workers)
    elif args.command == "leonardo":
        leonardo()
    elif args.command == "leonardo-thermo":
        leonardo_thermo()
    else:
        status()


if __name__ == "__main__":
    main()
