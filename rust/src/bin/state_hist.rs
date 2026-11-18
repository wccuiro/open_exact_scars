//! Decay of the transverse deviation a(t) from one initial state, for the decay-rate histograms (scar class:
//! coherent states in the tower; bulk class: bulk product states). One run per call:
//!
//!     cargo run --release --bin state_hist -- --method exact|traj|pure --L 10 [--gamma 0.1] [--noise depol|z]
//!         --init coherent (--cos-theta c | --theta θ) [--phi π]  |  --init bulk-product --init-seed N
//!         [--nbar FILE] [--write-nbar T] [--trajectories 64] [--dt 0.1] [--dt-fine 0.02 --t-fine 0] [--t-max 10]
//!         [--stop-level 1.5] [--stop-hold 2] [--seed 1]
//!
//! Methods: exact = ρ as a dense 2^L × 2^L matrix (dmatrix.rs, Taylor steps summed to round-off; the same generator
//! as liouville.rs, ≈ 10× cheaper per application in the full basis; L ≤ 11 in practice); traj = diffusive
//! trajectories (`Trajectory::with_channel`; for --noise depol one isotropic rotation per site and step), averaged;
//! pure = one state vector at γ = 0 (--gamma is not accepted).
//! Noise (noise.rs): --noise depol, jumps √(γ/3) σ^α_j (free-spin rate 4γ/3); default z, √γ σᶻ_j.
//! Initial states (initial.rs): coherent (every spin along (sin θ cos φ, sin θ sin φ, cos θ), default φ = π so that
//! θ = π/2 is |−⟩^⊗L) or bulk-product (Haar-random product state, Q-projected, |⟨H⟩/L| ≤ 0.25; --init-seed is the
//! seed of that state only, --seed the random-c realization as in every binary).
//!
//! Observable: per site n⊥_j = (⟨σˣ_j⟩, ⟨σʸ_j⟩), a(t) = (1/L) Σ_j |n⊥_j(t) − n̄⊥_j| from the (trajectory-averaged)
//! per-site values. n̄⊥_j: 0 for coherent states (tower states: exact); for bulk states the file --nbar (columns
//! j, nx, ny; from `state_hist_ed` or a late-time average), or 0 without it (the column a is then |n⊥| averaged,
//! to be recomputed from the per-site columns). Trajectories: standard error of a by the delta method,
//! (1/L) Σ_j u_j·δn⊥_j with u_j the unit vector of n⊥_j − n̄⊥_j, evaluated per trajectory. Also the tower weight
//! W(t) = Σ_n ⟨Φ_n|ρ|Φ_n⟩.
//! --write-nbar T (pure runs of bulk states only): when the run ends, also write n̄⊥_j = the average of the per-site
//! ⟨σˣ_j⟩, ⟨σʸ_j⟩ over the output times t ≥ T (the late-time n̄ of the γ = 0 run, for sizes without ED) to
//! nbar_L<L>[_seed<N>]_<init>.csv (columns j, nx, ny; the name `state_hist_ed` uses), unless that file exists.
//!
//! Output grid: step --dt-fine up to --t-fine, then --dt (each at most 0.05/γ). Run length: γ > 0 stops once
//! a/a(0) has stayed below e^{−stop-level} for stop-hold + 2π/h, i.e. once the envelope (running maximum of a over the
//! next period 2π/h) has stayed below e^{−stop-level} for stop-hold; at the latest at --t-max. γ = 0 runs to --t-max.
//!
//! Output (data/ or $SCARS_DATA_DIR), written to <name>.part and renamed when the run ends, so an existing <name>
//! is a finished run; the binary does nothing if it exists (a new seed never overwrites another one):
//!   exact  states_L<L>_g<γ><noise><seed>_<init>.csv
//!   traj   states_traj_L<L>_g<γ><noise><seed>_<init>.csv
//!   pure   states_pure_L<L>_g0<seed>_<init>.csv
//! with <noise> = _noisedepol (empty for σᶻ), <seed> = _seed<N> for a realization N ≠ 1, <init> =
//! initcoherent_cos<cos θ>[_phi<φ>] or initbulk-product_s<init seed>. First line: `# key=value …` (method, L, γ,
//! noise, init, E, E/L, a0, W0, nbar_norm = (1/L) Σ_j |n̄⊥_j|, nbar source, …); last line `# end t=… reason=stop|t_max
//! wall_s=… setup_s=…`; columns L, gamma, trajectories, t, a, a_err, W, W_err, mx, my, mz, x0 … x{L−1}, y0 … y{L−1}.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::dmatrix::{DensityLiouvillian, DensityPropagator};
use open_exact_scars::initial::{bulk_product, coherent_bloch, deviation, Init};
use open_exact_scars::model::{seed_suffix, Params};
use open_exact_scars::noise::Channel;
use open_exact_scars::tower;
use open_exact_scars::trajectory::{product_amplitudes, Amp, Hamiltonian, Rng, Trajectory};

/// The output step is at most DT_GAMMA / γ (as in `states`).
const DT_GAMMA: f64 = 0.05;

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Exact,
    Traj,
    Pure,
}

fn main() {
    let args = Options::parse(&[
        "method", "L", "gamma", "noise", "init", "theta", "cos-theta", "phi", "init-seed", "nbar", "trajectories", "dt",
        "dt-fine", "t-fine", "t-max", "stop-level", "stop-hold", "seed", "write-nbar",
    ]);
    let method = match args.str("method") {
        Some("exact") => Method::Exact,
        Some("traj") => Method::Traj,
        Some("pure") => Method::Pure,
        m => panic!("--method exact, traj or pure, got {m:?}"),
    };
    assert!(args.has("L"), "--L is required");
    let l = args.usize("L", 0);
    assert!(method != Method::Pure || !args.has("gamma"), "--method pure is γ = 0: no --gamma");
    let gamma = args.f64("gamma", 0.0);
    assert!(gamma >= 0.0);
    let channel = Channel::parse_noise(args.str("noise").unwrap_or("z"));
    let init = match args.str("init") {
        Some("coherent") => {
            let theta = match (args.has("theta"), args.has("cos-theta")) {
                (true, false) => args.f64("theta", 0.0),
                (false, true) => {
                    let c = args.f64("cos-theta", 0.0);
                    assert!((-1.0..=1.0).contains(&c));
                    c.acos()
                }
                _ => panic!("--init coherent needs --theta or --cos-theta (one of them)"),
            };
            Init::Coherent { theta, phi: args.f64("phi", std::f64::consts::PI) }
        }
        Some("bulk-product") => {
            assert!(args.has("init-seed"), "--init bulk-product needs --init-seed");
            Init::BulkProduct { seed: args.usize("init-seed", 0) as u64 }
        }
        i => panic!("--init coherent or bulk-product, got {i:?}"),
    };
    let seed = args.usize("seed", 1) as u64;
    let run = RunOpts {
        n_traj: if method == Method::Traj && gamma > 0.0 { args.usize("trajectories", 64) } else { 1 },
        dt: args.f64("dt", 0.1),
        dt_fine: args.f64("dt-fine", 0.02),
        t_fine: args.f64("t-fine", 0.0),
        t_max: args.f64("t-max", 10.0),
        stop_level: args.f64("stop-level", 1.5),
        stop_hold: args.f64("stop-hold", 2.0),
    };

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");
    let noise_tag = if method == Method::Pure { String::new() } else { channel.suffix() };
    let name = match method {
        Method::Exact => format!("states_L{l}_g{gamma}{noise_tag}{}_{}.csv", seed_suffix(seed), init.tag()),
        Method::Traj => format!("states_traj_L{l}_g{gamma}{noise_tag}{}_{}.csv", seed_suffix(seed), init.tag()),
        Method::Pure => format!("states_pure_L{l}_g0{}_{}.csv", seed_suffix(seed), init.tag()),
    };
    let file = data.join(&name);
    if file.exists() {
        println!("{name} exists: nothing to do");
        return;
    }

    let start = Instant::now();
    let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
    let ham = Hamiltonian::new(&params);
    let (psi0, bloch, energy, attempts): (Option<Vec<Amp>>, Vec<[f64; 3]>, f64, usize) = match init {
        Init::Coherent { theta, phi } => (None, coherent_bloch(l, theta, phi), 0.5 * params.h * l as f64 * theta.cos(), 0),
        Init::BulkProduct { seed } => {
            let b = bulk_product(&ham, seed);
            (Some(b.psi), b.bloch, b.energy, b.attempts)
        }
    };
    let (nbar, nbar_source): (Vec<[f64; 2]>, String) = match (init, args.str("nbar")) {
        (Init::Coherent { .. }, None) => (vec![[0.0; 2]; l], "tower".into()),
        (Init::Coherent { .. }, Some(_)) => panic!("--nbar: n̄ = 0 exactly for coherent (tower) states"),
        (_, None) => (vec![[0.0; 2]; l], "none".into()),
        (_, Some(f)) => (read_nbar(Path::new(f), l), f.to_string()),
    };
    let nbar_norm = nbar.iter().map(|n| n[0].hypot(n[1])).sum::<f64>() / l as f64;
    let (theta, phi, init_seed) = match init {
        Init::Coherent { theta, phi } => (theta, phi, -1i64),
        Init::BulkProduct { seed } => (f64::NAN, f64::NAN, seed as i64),
    };
    let write_nbar = args.has("write-nbar").then(|| args.f64("write-nbar", 0.0));
    if write_nbar.is_some() {
        assert!(method == Method::Pure && init_seed >= 0, "--write-nbar: pure runs of bulk states only");
        assert!(write_nbar.unwrap() < run.t_max, "--write-nbar T needs T < --t-max");
    }
    let meta = format!(
        "method={} L={l} gamma={gamma} noise={} init={} theta={theta} phi={phi} init_seed={init_seed} seed={seed} \
         E={energy:.12e} E_per_L={:.12e} attempts={attempts} nbar_norm={nbar_norm:.6e} nbar_source={nbar_source} \
         trajectories={} dt={} dt_fine={} t_fine={} t_max={} stop_level={} stop_hold={}",
        args.str("method").unwrap(),
        if method == Method::Pure { "none" } else { args.str("noise").unwrap_or("z") },
        init.tag(),
        energy / l as f64,
        run.n_traj,
        run.dt,
        run.dt_fine,
        run.t_fine,
        run.t_max,
        run.stop_level,
        run.stop_hold
    );

    let part = data.join(format!("{name}.part"));
    let mut out = Output::new(&part, l, &meta, write_nbar);
    let setup = start.elapsed().as_secs_f64();
    let (t_end, reason, setup_more) = match method {
        Method::Exact => {
            let t0 = Instant::now();
            let liou = DensityLiouvillian::new(&params, channel);
            let psi = psi0.clone().unwrap_or_else(|| product_amplitudes(&bloch));
            let rho0 = liou.pure(&psi);
            let s = t0.elapsed().as_secs_f64();
            let (t, r) = evolve_exact(&liou, &params, rho0, gamma, &nbar, &run, &mut out);
            (t, r, s)
        }
        Method::Traj | Method::Pure => {
            let psi = psi0.unwrap_or_else(|| product_amplitudes(&bloch));
            let (t, r) = evolve_traj(&ham, &params, &psi, gamma, &channel, seed, &init, &nbar, &run, &mut out);
            (t, r, 0.0)
        }
    };
    let wall = start.elapsed().as_secs_f64() - setup - setup_more;
    let late = out.late_average();
    out.finish(t_end, reason, wall, setup + setup_more);
    fs::rename(&part, &file).unwrap();
    if let (Some(t_late), Some((nbar_late, rows))) = (write_nbar, late) {
        let nbar_name = format!("nbar_L{l}{}_{}.csv", seed_suffix(seed), init.tag());
        let nbar_file = data.join(&nbar_name);
        if nbar_file.exists() {
            println!("{nbar_name} exists: not written");
        } else {
            let tmp = data.join(format!("{nbar_name}.part.{}", std::process::id()));
            let mut csv = BufWriter::new(File::create(&tmp).unwrap());
            let norm = nbar_late.iter().map(|n| n[0].hypot(n[1])).sum::<f64>() / l as f64;
            writeln!(csv, "# L={l} seed={seed} init_seed={init_seed} E={energy:.12e} source=late t_late={t_late} rows={rows} \
                           nbar_norm={norm:.6e} from={name}").unwrap();
            writeln!(csv, "j,nx,ny").unwrap();
            for (j, n) in nbar_late.iter().enumerate() {
                writeln!(csv, "{j},{:.15e},{:.15e}", n[0], n[1]).unwrap();
            }
            csv.into_inner().unwrap().sync_all().unwrap();
            fs::rename(&tmp, &nbar_file).unwrap();
            println!("{nbar_name}: late-time n̄ over {rows} output times t ≥ {t_late}, (1/L) Σ |n̄⊥_j| = {norm:.4}");
        }
    }
    println!("{name}: E/L = {:.4}, stopped at t = {t_end:.3} ({reason}), {wall:.1} s evolution, {:.1} s setup", energy / l as f64, setup + setup_more);
}

struct RunOpts {
    n_traj: usize,
    dt: f64,
    dt_fine: f64,
    t_fine: f64,
    t_max: f64,
    stop_level: f64,
    stop_hold: f64,
}

/// n̄⊥_j from a CSV with columns j, nx, ny (more columns allowed).
fn read_nbar(path: &Path, l: usize) -> Vec<[f64; 2]> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("--nbar {}: {e}", path.display()));
    let mut lines = text.lines().filter(|s| !s.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("empty n̄ file").split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap_or_else(|| panic!("n̄ file: no column {name}"));
    let (cj, cx, cy) = (col("j"), col("nx"), col("ny"));
    let mut nbar = vec![[f64::NAN; 2]; l];
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let j: usize = f[cj].parse().unwrap();
        nbar[j] = [f[cx].parse().unwrap(), f[cy].parse().unwrap()];
    }
    assert!(nbar.iter().all(|n| n[0].is_finite() && n[1].is_finite()), "n̄ file: not every site 0 … {} given", l - 1);
    nbar
}

/// The CSV being written (`.part`).
struct Output {
    csv: BufWriter<File>,
    /// (t from which per-site x, y are summed, Σx_j, Σy_j, number of rows) for --write-nbar
    late: Option<(f64, Vec<f64>, Vec<f64>, usize)>,
}

impl Output {
    fn new(path: &Path, l: usize, meta: &str, late_from: Option<f64>) -> Self {
        let mut csv = BufWriter::new(File::create(path).unwrap());
        writeln!(csv, "# {meta}").unwrap();
        let cols = |p: &str| (0..l).map(|j| format!("{p}{j}")).collect::<Vec<_>>().join(",");
        writeln!(csv, "L,gamma,trajectories,t,a,a_err,W,W_err,mx,my,mz,{},{}", cols("x"), cols("y")).unwrap();
        Output { csv, late: late_from.map(|t| (t, vec![0.0; l], vec![0.0; l], 0)) }
    }

    /// Per-site average of (x, y) over the rows with t ≥ the --write-nbar time, and the number of those rows.
    fn late_average(&self) -> Option<(Vec<[f64; 2]>, usize)> {
        let (_, sx, sy, n) = self.late.as_ref()?;
        assert!(*n > 0, "--write-nbar: no output time after it");
        Some((sx.iter().zip(sy).map(|(x, y)| [x / *n as f64, y / *n as f64]).collect(), *n))
    }

    #[allow(clippy::too_many_arguments)]
    fn row(&mut self, l: usize, gamma: f64, n: usize, t: f64, a: f64, a_err: f64, w: f64, w_err: f64, m: [f64; 3], x: &[f64], y: &[f64]) {
        let join = |v: &[f64]| v.iter().map(|v| format!("{v:.10e}")).collect::<Vec<_>>().join(",");
        writeln!(
            self.csv,
            "{l},{gamma},{n},{t:.6},{a:.12e},{a_err:.4e},{w:.12e},{w_err:.4e},{:.12e},{:.12e},{:.12e},{},{}",
            m[0],
            m[1],
            m[2],
            join(x),
            join(y)
        )
        .unwrap();
        self.csv.flush().unwrap();
        if let Some((t_late, sx, sy, count)) = self.late.as_mut() {
            if t >= *t_late - 1e-9 {
                sx.iter_mut().zip(x).for_each(|(s, v)| *s += v);
                sy.iter_mut().zip(y).for_each(|(s, v)| *s += v);
                *count += 1;
            }
        }
    }

    fn finish(mut self, t: f64, reason: &str, wall: f64, setup: f64) {
        writeln!(self.csv, "# end t={t:.6} reason={reason} wall_s={wall:.3} setup_s={setup:.3}").unwrap();
        self.csv.into_inner().unwrap().sync_all().unwrap();
    }
}

/// Output times and the adaptive stop shared by both methods.
struct Clock {
    dt: f64,
    dt_fine: f64,
    n_fine: usize,
    step: usize,
    a0: Option<f64>,
    threshold_level: f64,
    hold: f64,
    below_since: Option<f64>,
}

impl Clock {
    fn new(run: &RunOpts, gamma: f64, h: f64) -> Self {
        let cap = if gamma > 0.0 { DT_GAMMA / gamma } else { f64::INFINITY };
        let (dt, dt_fine) = (run.dt.min(cap), run.dt_fine.min(cap));
        let n_fine = if run.t_fine > 0.0 { (run.t_fine / dt_fine).round() as usize } else { 0 };
        Clock {
            dt,
            dt_fine,
            n_fine,
            step: 0,
            a0: None,
            threshold_level: (-run.stop_level).exp(),
            hold: run.stop_hold + 2.0 * std::f64::consts::PI / h,
            below_since: None,
        }
    }

    fn t(&self) -> f64 {
        if self.step <= self.n_fine {
            self.step as f64 * self.dt_fine
        } else {
            self.n_fine as f64 * self.dt_fine + (self.step - self.n_fine) as f64 * self.dt
        }
    }

    /// Step to the next output time; returns its length.
    fn advance(&mut self) -> f64 {
        let dt = if self.step < self.n_fine { self.dt_fine } else { self.dt };
        self.step += 1;
        dt
    }

    /// Records a(t); true once a/a(0) has stayed below e^{−stop-level} for the hold time (γ > 0 only).
    fn done(&mut self, a: f64, gamma: f64) -> bool {
        let t = self.t();
        let a0 = *self.a0.get_or_insert(a);
        if gamma == 0.0 {
            return false;
        }
        if a >= self.threshold_level * a0 {
            self.below_since = None;
            false
        } else {
            t - *self.below_since.get_or_insert(t) >= self.hold - 1e-9
        }
    }
}

fn evolve_exact(
    liou: &DensityLiouvillian,
    params: &Params,
    mut rho: Vec<Amp>,
    gamma: f64,
    nbar: &[[f64; 2]],
    run: &RunOpts,
    out: &mut Output,
) -> (f64, &'static str) {
    let l = params.l;
    let mut clock = Clock::new(run, gamma, params.h);
    let mut fine = DensityPropagator::new(liou, gamma, clock.dt_fine);
    let mut coarse = DensityPropagator::new(liou, gamma, clock.dt);
    loop {
        let t = clock.t();
        let site = liou.per_site(&rho);
        let a = deviation(&site[0], &site[1], nbar);
        let w = liou.tower_weight(&rho);
        let m = [0, 1, 2].map(|k| site[k].iter().sum::<f64>() / l as f64);
        out.row(l, gamma, 1, t, a, 0.0, w, 0.0, m, &site[0], &site[1]);
        if clock.done(a, gamma) {
            return (t, "stop");
        }
        if t >= run.t_max - 1e-9 {
            return (t, "t_max");
        }
        clock.advance();
        if clock.step <= clock.n_fine {
            fine.step(&mut rho);
        } else {
            coarse.step(&mut rho);
        }
    }
}

/// FNV-1a hash of the init tag, mixed into the noise seed so that different initial states have independent noise.
fn tag_hash(tag: &str) -> u64 {
    tag.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

#[allow(clippy::too_many_arguments)]
fn evolve_traj(
    ham: &Hamiltonian,
    params: &Params,
    psi0: &[Amp],
    gamma: f64,
    channel: &Channel,
    seed: u64,
    init: &Init,
    nbar: &[[f64; 2]],
    run: &RunOpts,
    out: &mut Output,
) -> (f64, &'static str) {
    let l = params.l;
    let n = run.n_traj;
    let mut clock = Clock::new(run, gamma, params.h);
    let base = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ ((l as u64) << 48) ^ gamma.to_bits().rotate_left(20) ^ tag_hash(&init.tag());
    let mut trajs: Vec<Trajectory> = (0..n)
        .map(|k| {
            let traj = Trajectory::from_state(ham, psi0.to_vec(), base ^ k as u64);
            if gamma > 0.0 {
                traj.with_channel(channel)
            } else {
                traj
            }
        })
        .collect();
    let nf = n as f64;
    loop {
        let t = clock.t();
        // per trajectory: ⟨σˣ_j⟩, ⟨σʸ_j⟩, m_z and the tower weight
        let per: Vec<([Vec<f64>; 3], f64)> =
            trajs.par_iter().map(|traj| (traj.per_site(), tower::weight(&traj.psi, l))).collect();
        let mut mean = [vec![0.0; l], vec![0.0; l], vec![0.0; l]];
        for (site, _) in &per {
            for k in 0..3 {
                mean[k].iter_mut().zip(&site[k]).for_each(|(m, s)| *m += s / nf);
            }
        }
        let a = deviation(&mean[0], &mean[1], nbar);
        // delta method: a ≈ a(mean) + (1/L) Σ_j u_j · (n⊥_j − mean), per trajectory
        let u: Vec<[f64; 2]> = (0..l)
            .map(|j| {
                let (dx, dy) = (mean[0][j] - nbar[j][0], mean[1][j] - nbar[j][1]);
                let r = dx.hypot(dy);
                if r > 0.0 {
                    [dx / r, dy / r]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let q: Vec<f64> = per
            .iter()
            .map(|(site, _)| (0..l).map(|j| u[j][0] * site[0][j] + u[j][1] * site[1][j]).sum::<f64>() / l as f64)
            .collect();
        let se = |x: &[f64]| -> f64 {
            if n < 2 {
                return 0.0;
            }
            let m = x.iter().sum::<f64>() / nf;
            (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (nf - 1.0) / nf).sqrt()
        };
        let ws: Vec<f64> = per.iter().map(|(_, w)| *w).collect();
        let w = ws.iter().sum::<f64>() / nf;
        let m = [0, 1, 2].map(|k| mean[k].iter().sum::<f64>() / l as f64);
        out.row(l, gamma, n, t, a, se(&q), w, se(&ws), m, &mean[0], &mean[1]);
        if clock.done(a, gamma) {
            return (t, "stop");
        }
        if t >= run.t_max - 1e-9 {
            return (t, "t_max");
        }
        let dt = clock.advance();
        trajs.par_iter_mut().for_each(|traj| if gamma > 0.0 { traj.step(gamma, dt) } else { traj.unitary(dt) });
    }
}
