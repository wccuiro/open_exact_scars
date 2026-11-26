//! Trajectory counterpart of `floquet`: the same Floquet step F (src/floquet.rs), unravelled on state vectors. One
//! step of a trajectory is the L gates of F, each followed by the random single-site rotations of the sites whose
//! field and noise are folded into it (θ_α ~ N(0, r_α γ dt) per jump √(r_α γ) σ^α), so the average over trajectories is F^N ρ(0) for any dt, with
//! statistical error only. Same states, random-c realization, observables, output columns and stop rule as
//! `states_traj`.
//!
//!     cargo run --release --bin floquet_traj -- --state minus-x|neel|neel-x [--L 16 | --lmin 8 --lmax 12]
//!                                               [--gammas 0,0.01] [--trajectories 112] [--dt 0.1]
//!                                               [--channel z|x|y|xyz|dep] [--noise depol] [--t-max 1000] [--t-zero 50]
//!                                               [--stop 1e-3] [--stop-te 0] [--seed 1] [--traj-seed N]
//!
//! Trotter step = output step: dt · min(1, 0.5/γ), as `floquet`. γ = 0 is one trajectory (deterministic), run to
//! --t-zero. γ > 0 runs until the signal (m_⊥ for minus-x, |s_z| for neel, s_⊥ for neel-x) has stayed below
//! max(--stop, 3 × its standard error) for 10 time units, or to --t-max, or (--stop-te F > 0) to F · t₁, t₁ the start
//! of the latest stretch with the signal below 1/e. --traj-seed sets the noise only (default --seed) and adds _s<N>.
//! Trajectories run in parallel, one per thread; memory 16 B · 2^L per trajectory.
//!
//! Output (data/ or $SCARS_DATA_DIR), one row per step, flushed every row:
//!   floquet_traj_<state>_L<L>_g<γ>_dt<dt><ch>[_seed<N>][_s<traj-seed>].csv   L, gamma, trajectories, t, mx, my, mz,
//!       sx, sy, sz, the standard errors mx_err … sz_err, and the covariances of the means mxmy_cov, sxsy_cov.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::floquet::{step_at, FloquetTraj};
use open_exact_scars::model::{seed_suffix, Params};
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::{per_site_expectations, product_amplitudes, Amp, Rng};

const STOP_WINDOW: f64 = 10.0;

/// splitmix64 finalizer.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn main() {
    let args = Options::parse(&[
        "state", "L", "lmin", "lmax", "gammas", "trajectories", "dt", "t-max", "t-zero", "stop", "stop-te", "seed",
        "traj-seed", "channel", "noise",
    ]);
    let name = args.str("state").unwrap_or("minus-x").to_string();
    assert!(["minus-x", "neel", "neel-x"].contains(&name.as_str()), "--state minus-x, neel or neel-x");
    let channel = Channel::from_options(args.str("channel"), args.str("noise"), 1.0);
    let sizes = args.sizes(8, 8);
    let gammas = args.list("gammas").unwrap_or_else(|| vec![0.0, 0.01]);
    assert!(gammas.iter().all(|&g| g >= 0.0), "γ must be ≥ 0");
    let n_traj = args.usize("trajectories", 112);
    let dt = args.f64("dt", 0.1);
    let (t_max, t_zero, stop) = (args.f64("t-max", 1000.0), args.f64("t-zero", 50.0), args.f64("stop", 1e-3));
    let stop_te = args.f64("stop-te", 0.0);
    let seed = args.usize("seed", 1) as u64;
    let traj_seed = args.has("traj-seed").then(|| args.usize("traj-seed", 0) as u64);
    let suffix = format!("{}{}{}", channel.suffix(), seed_suffix(seed), traj_seed.map_or(String::new(), |s| format!("_s{s}")));

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");

    for l in sizes {
        if name != "minus-x" && l % 2 == 1 {
            println!("L = {l}: skipped, the Néel state needs even L");
            continue;
        }
        let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
        let bloch: Vec<[f64; 3]> = (0..l)
            .map(|j| {
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                match name.as_str() {
                    "minus-x" => [-1.0, 0.0, 0.0],
                    "neel" => [0.0, 0.0, sign],
                    _ => [sign, 0.0, 0.0],
                }
            })
            .collect();
        let psi0 = product_amplitudes(&bloch);
        for &gamma in &gammas {
            let step_dt = if gamma > 0.0 { step_at(dt, gamma) } else { dt };
            let floquet = FloquetTraj::new(&params, &channel, gamma, step_dt);
            let n = if gamma == 0.0 { 1 } else { n_traj };
            let base = mix(mix(mix(traj_seed.unwrap_or(seed)) ^ l as u64) ^ gamma.to_bits());
            let mut trajs: Vec<(Vec<Amp>, Rng)> = (0..n).map(|k| (psi0.clone(), Rng::new(base ^ mix(k as u64 + 1)))).collect();
            println!(
                "L = {l}, gamma = {gamma}: {} layers, {n} trajectories, step {step_dt}, {} threads",
                floquet.layers,
                rayon::current_num_threads()
            );
            let file = data.join(format!("floquet_traj_{name}_L{l}_g{gamma}_dt{dt}{suffix}.csv"));
            let mut csv = BufWriter::new(File::create(&file).unwrap());
            writeln!(csv, "L,gamma,trajectories,t,mx,my,mz,sx,sy,sz,mx_err,my_err,mz_err,sx_err,sy_err,sz_err,mxmy_cov,sxsy_cov")
                .unwrap();
            let steps = ((if gamma == 0.0 { t_zero } else { t_max }) / step_dt).round() as usize;
            let (mut below_since, mut t_cross, mut check, mut t) = (None, None::<f64>, 0.0f64, 0.0);
            let (start, mut done, mut step_secs) = (Instant::now(), 0, 0.0);
            for step in 0..=steps {
                t = step as f64 * step_dt;
                if step > 0 {
                    let s0 = Instant::now();
                    trajs.par_iter_mut().for_each(|(psi, rng)| floquet.step(psi, rng));
                    step_secs += s0.elapsed().as_secs_f64();
                    done += 1;
                }
                // per trajectory: [mx, my, mz, sx, sy, sz], their squares, mx·my and sx·sy, summed
                let (s, dev) = trajs
                    .par_iter()
                    .map(|(psi, _)| {
                        let site = per_site_expectations(psi, l);
                        let mut r = [0.0f64; 14];
                        for a in 0..3 {
                            r[a] = site[a].iter().sum::<f64>() / l as f64;
                            r[a + 3] =
                                site[a].iter().enumerate().map(|(j, &x)| if j % 2 == 0 { x } else { -x }).sum::<f64>() / l as f64;
                        }
                        for a in 0..6 {
                            r[a + 6] = r[a] * r[a];
                        }
                        r[12] = r[0] * r[1];
                        r[13] = r[3] * r[4];
                        let mut dev = 0.0f64;
                        if gamma == 0.0 && name == "minus-x" {
                            let (sn, c) = (params.h * t).sin_cos();
                            for j in 0..l {
                                dev = dev.max((site[0][j] + c).abs()).max((site[1][j] + sn).abs()).max(site[2][j].abs());
                            }
                        }
                        (r, dev)
                    })
                    .reduce(
                        || ([0.0; 14], 0.0),
                        |(mut a, da), (b, db)| {
                            a.iter_mut().zip(&b).for_each(|(p, q)| *p += q);
                            (a, da.max(db))
                        },
                    );
                check = check.max(dev);
                let nf = n as f64;
                let mean: [f64; 6] = std::array::from_fn(|a| s[a] / nf);
                let err: [f64; 6] =
                    std::array::from_fn(|a| ((s[a + 6] / nf - mean[a] * mean[a]).max(0.0) / (nf - 1.0).max(1.0)).sqrt());
                let cov = |a: usize, b: usize, k: usize| (s[k] / nf - mean[a] * mean[b]) / (nf - 1.0).max(1.0);
                let [mx, my, mz, sx, sy, sz] = mean;
                let e = err.map(|x| format!("{x:.4e}")).join(",");
                writeln!(
                    csv,
                    "{l},{gamma},{n},{t:.6},{mx:.12e},{my:.12e},{mz:.12e},{sx:.12e},{sy:.12e},{sz:.12e},{e},{:.4e},{:.4e}",
                    cov(0, 1, 12),
                    cov(3, 4, 13)
                )
                .unwrap();
                csv.flush().unwrap();
                if gamma > 0.0 {
                    let (signal, signal_err) = match name.as_str() {
                        "minus-x" => (mx.hypot(my), err[0].hypot(err[1])),
                        "neel" => (sz.abs(), err[5]),
                        _ => (sx.hypot(sy), err[3].hypot(err[4])),
                    };
                    if signal >= (-1.0f64).exp() {
                        t_cross = None;
                    } else if t_cross.is_none() {
                        t_cross = Some(t);
                    }
                    if stop_te > 0.0 && t_cross.is_some_and(|tc| t >= stop_te * tc) {
                        break;
                    }
                    if signal >= stop.max(3.0 * signal_err) {
                        below_since = None;
                    } else if t - *below_since.get_or_insert(t) >= STOP_WINDOW {
                        break;
                    }
                }
            }
            let secs = start.elapsed().as_secs_f64();
            println!(
                "L = {l}, gamma = {gamma}: t = {t:.4} in {secs:.1} s, {done} steps, {:.4} s per step for all {n} trajectories \
                 (observables: {:.0} % of the time) -> {}",
                step_secs / done.max(1) as f64,
                100.0 * (1.0 - step_secs / secs),
                file.display()
            );
            if gamma == 0.0 && name == "minus-x" {
                println!("L = {l}, gamma = 0: largest per-site deviation from (-cos ht, -sin ht, 0) is {check:.1e}");
            }
        }
    }
}
