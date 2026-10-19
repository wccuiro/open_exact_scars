//! Trajectory counterpart of `states`, for sizes beyond the exact ρ evolution (L > 14): the same initial
//! states, the same random-c realization and the same output columns, from pure-state trajectories of
//!
//!     dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ)
//!
//!     minus-x   |−⟩^⊗L          (in the tower: the scar)
//!     neel      |↑↓↑↓…⟩         (even L; not a scar)
//!     neel-x    |+−+−…⟩         (even L; not a scar)
//!
//!     cargo run --release --bin states_traj -- --state minus-x|neel|neel-x [--L 16 | --lmin 4 --lmax 8]
//!                                              [--gammas 0,0.01,0.1] [--trajectories 112] [--dt 0.1]
//!                                              [--t-max 1000] [--t-zero 50] [--stop 1e-3] [--seed 1]
//!                                              [--unravel diffusive|jump] [--traj-seed N]
//!
//! Model coefficients: src/parameters.rs. H has random c (Params::random_c, generator seeded with
//! 1000 · seed + L, the same realization as `states` for the same parameters).
//!
//! Unravelling (`trajectory.rs`): diffusive (default; real noise on every spin, one splitting step per
//! output step) or jump (exact unitary evolution between σᶻ kicks at total rate γL; from the scar state the
//! evolution before the first kick is the analytic reference 1). --traj-seed sets the noise of the
//! trajectories only (default: --seed, as before it existed), so one (L, γ) can be split over several jobs
//! with the same realization; it adds _s<N> to the file name.
//!
//! All trajectories of one γ advance together, one output step at a time, so every row of the CSV is the
//! average over all of them and is written as the run goes (a run cut by a time limit keeps its early
//! part). Memory ≈ 3 · 16 B · 2^L per trajectory.
//!
//! Run length: γ = 0 is deterministic (one trajectory) and runs to --t-zero. γ > 0 runs until the signal
//! (m_⊥ for minus-x, |s_z| for neel, s_⊥ for neel-x) has stayed below max(--stop, 3 × its standard error)
//! for a time STOP_WINDOW, or to --t-max. Step: min(dt, DT_GAMMA / γ), as in `states`; the error of the
//! average is O(dt²).
//!
//! Output (data/ or $SCARS_DATA_DIR), one row per step, flushed every row:
//!   states_traj_<state>_L<L>_g<γ>[_jump][_s<traj-seed>].csv   L, gamma, trajectories, t, mx, my, mz, sx, sy,
//!       sz, then the standard errors mx_err … sz_err (_jump: jump unravelling)
//!       m_α = (1/L) Σ_j ⟨σ^α_j⟩,  s_α = (1/L) Σ_j (−1)^j ⟨σ^α_j⟩, averaged over the trajectories
//!   states_realization_L<L>.csv         j, and the c coefficients of bond (j, j+1)

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::model::{write_realization, Params};
use open_exact_scars::trajectory::{Hamiltonian, JumpTrajectory, Rng, Trajectory};

/// γ > 0: stop once the signal has stayed below threshold for this long (longer than the period 2π/h).
const STOP_WINDOW: f64 = 10.0;
/// The step is at most DT_GAMMA / γ.
const DT_GAMMA: f64 = 0.05;

#[derive(Clone, Copy, PartialEq)]
enum State {
    MinusX,
    Neel,
    NeelX,
}

#[derive(Clone, Copy, PartialEq)]
enum Unravel {
    Diffusive,
    Jump,
}

enum Traj<'a> {
    Diffusive(Trajectory<'a>),
    Jump(JumpTrajectory<'a>),
}

impl Traj<'_> {
    /// Advance from t_new − dt to t_new.
    fn advance(&mut self, gamma: f64, dt: f64, t_new: f64) {
        match self {
            Traj::Diffusive(traj) => traj.step(gamma, dt),
            Traj::Jump(traj) => traj.advance_to(t_new),
        }
    }

    fn per_site(&self) -> [Vec<f64>; 3] {
        match self {
            Traj::Diffusive(traj) => traj.per_site(),
            Traj::Jump(traj) => traj.per_site(),
        }
    }
}

fn main() {
    let args = Options::parse(&[
        "state", "L", "lmin", "lmax", "gammas", "trajectories", "dt", "t-max", "t-zero", "stop", "seed", "unravel",
        "traj-seed",
    ]);
    let unravel = match args.str("unravel") {
        None | Some("diffusive") => Unravel::Diffusive,
        Some("jump") => Unravel::Jump,
        u => panic!("--unravel diffusive or jump, got {u:?}"),
    };
    let state = match args.str("state") {
        Some("minus-x") => State::MinusX,
        Some("neel") => State::Neel,
        Some("neel-x") => State::NeelX,
        s => panic!("--state minus-x, neel or neel-x, got {s:?}"),
    };
    let name = match state {
        State::MinusX => "minus-x",
        State::Neel => "neel",
        State::NeelX => "neel-x",
    };
    let sizes = args.sizes(4, 8);
    let gammas = args.list("gammas").unwrap_or_else(|| vec![0.0, 0.01]);
    assert!(gammas.iter().all(|&g| g >= 0.0), "γ must be ≥ 0");
    let n_traj = args.usize("trajectories", 112);
    let dt = args.f64("dt", 0.1);
    let (t_max, t_zero, stop) = (args.f64("t-max", 1000.0), args.f64("t-zero", 50.0), args.f64("stop", 1e-3));
    let seed = args.usize("seed", 1) as u64;
    let traj_seed = args.has("traj-seed").then(|| args.usize("traj-seed", 0) as u64);
    let suffix = format!(
        "{}{}",
        if unravel == Unravel::Jump { "_jump" } else { "" },
        traj_seed.map_or(String::new(), |s| format!("_s{s}"))
    );

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");

    for l in sizes {
        if state != State::MinusX && l % 2 == 1 {
            println!("L = {l}: skipped, the Néel state needs even L");
            continue;
        }
        let start = Instant::now();
        let mut rng = Rng::new(1000 * seed + l as u64);
        let params = Params::random_c(l, &mut rng);
        write_realization(&data, &params);
        let ham = Hamiltonian::new(&params);
        println!("L = {l}: H built in {:.1?}, {} threads", start.elapsed(), rayon::current_num_threads());

        let bloch: Vec<[f64; 3]> = (0..l)
            .map(|j| match state {
                State::MinusX => [-1.0, 0.0, 0.0],
                State::Neel => [0.0, 0.0, if j % 2 == 0 { 1.0 } else { -1.0 }],
                State::NeelX => [if j % 2 == 0 { 1.0 } else { -1.0 }, 0.0, 0.0],
            })
            .collect();

        // one γ at a time: the parallelism is over the trajectories (and inside H ψ at large L)
        for &g in &gammas {
            let file = data.join(format!("states_traj_{name}_L{l}_g{g}{suffix}.csv"));
            let n = if g == 0.0 { 1 } else { n_traj };
            let t_run = if g == 0.0 { t_zero } else { t_max };
            let noise_seed = traj_seed.unwrap_or(seed);
            let t_end = run(&ham, &params, state, &bloch, unravel, g, n, dt, t_run, stop, noise_seed, &file);
            println!("L = {l}, gamma = {g}: stopped at t = {t_end:.3}, {:.1?} since the start of L = {l}", start.elapsed());
        }
    }
}

/// Evolves n trajectories together and writes one CSV row per step; returns the final time.
#[allow(clippy::too_many_arguments)]
fn run(
    ham: &Hamiltonian,
    params: &Params,
    state: State,
    bloch: &[[f64; 3]],
    unravel: Unravel,
    gamma: f64,
    n: usize,
    dt: f64,
    t_max: f64,
    stop: f64,
    seed: u64,
    file: &Path,
) -> f64 {
    let l = params.l;
    let dt = if gamma > 0.0 { dt.min(DT_GAMMA / gamma) } else { dt };
    let steps = (t_max / dt).round() as usize;
    let mut csv = BufWriter::new(File::create(file).unwrap());
    writeln!(csv, "L,gamma,trajectories,t,mx,my,mz,sx,sy,sz,mx_err,my_err,mz_err,sx_err,sy_err,sz_err").unwrap();

    // independent noise for every (seed, L, γ, trajectory); Rng::new mixes the bits
    let base = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ ((l as u64) << 48) ^ gamma.to_bits().rotate_left(20);
    let mut trajs: Vec<Traj> = (0..n)
        .map(|k| match unravel {
            Unravel::Diffusive => Traj::Diffusive(Trajectory::product(ham, bloch, base ^ k as u64)),
            Unravel::Jump => Traj::Jump(JumpTrajectory::new(
                ham,
                params.h,
                gamma,
                bloch,
                state == State::MinusX,
                base ^ k as u64,
            )),
        })
        .collect();

    let (mut below_since, mut t, mut check) = (None, 0.0, 0.0f64);
    let start = Instant::now();
    for step in 0..=steps {
        t = step as f64 * dt;
        if step > 0 {
            trajs.par_iter_mut().for_each(|traj| traj.advance(gamma, dt, t));
        }

        // per trajectory: [mx, my, mz, sx, sy, sz] and their squares, summed over the trajectories
        let sums = trajs
            .par_iter()
            .map(|traj| {
                let site = traj.per_site();
                let mut r = [0.0f64; 12];
                for a in 0..3 {
                    r[a] = site[a].iter().sum::<f64>() / l as f64;
                    r[a + 3] = site[a].iter().enumerate().map(|(j, &x)| if j % 2 == 0 { x } else { -x }).sum::<f64>() / l as f64;
                }
                for a in 0..6 {
                    r[a + 6] = r[a] * r[a];
                }
                if n == 1 && gamma == 0.0 && state == State::MinusX {
                    // exact reference 1: every spin follows (−cos ht, −sin ht, 0)
                    let (s, c) = (params.h * t).sin_cos();
                    let dev = (0..l).map(|j| (site[0][j] + c).abs().max((site[1][j] + s).abs()).max(site[2][j].abs()));
                    return (r, dev.fold(0.0, f64::max));
                }
                (r, 0.0)
            })
            .reduce(
                || ([0.0; 12], 0.0),
                |(mut a, da), (b, db)| {
                    a.iter_mut().zip(&b).for_each(|(p, q)| *p += q);
                    (a, da.max(db))
                },
            );
        let (s, dev) = sums;
        check = check.max(dev);
        let nf = n as f64;
        let mean: [f64; 6] = std::array::from_fn(|a| s[a] / nf);
        // standard error of the mean (sample standard deviation of one trajectory / √n)
        let err: [f64; 6] = std::array::from_fn(|a| {
            ((s[a + 6] / nf - mean[a] * mean[a]).max(0.0) / (nf - 1.0).max(1.0)).sqrt()
        });
        let [mx, my, mz, sx, sy, sz] = mean;
        let e = err.map(|x| format!("{x:.4e}")).join(",");
        writeln!(csv, "{l},{gamma},{n},{t:.6},{mx:.12e},{my:.12e},{mz:.12e},{sx:.12e},{sy:.12e},{sz:.12e},{e}").unwrap();
        csv.flush().unwrap();

        if gamma > 0.0 {
            let (signal, signal_err) = match state {
                State::MinusX => (mx.hypot(my), err[0].hypot(err[1])),
                State::Neel => (sz.abs(), err[5]),
                State::NeelX => (sx.hypot(sy), err[3].hypot(err[4])),
            };
            if signal >= stop.max(3.0 * signal_err) {
                below_since = None;
            } else if t - *below_since.get_or_insert(t) >= STOP_WINDOW {
                break;
            }
        }
        // progress about once per unit of time
        if step > 0 && step % (1.0 / dt).round().max(1.0) as usize == 0 {
            println!("  L = {l}, gamma = {gamma}: t = {t:.2} of at most {t_max}, {:.1?}", start.elapsed());
        }
    }
    if n == 1 && gamma == 0.0 && state == State::MinusX {
        println!("L = {l}, gamma = 0: largest per-site deviation from (-cos ht, -sin ht, 0) is {check:.1e}");
    }
    t
}
