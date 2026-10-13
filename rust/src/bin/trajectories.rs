//! Pure-state trajectories (diffusive unravelling with real noise, see `trajectory.rs`) for
//!
//!     dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ),     ρ(0) = |ψ₀⟩⟨ψ₀|, |ψ₀⟩ = |−⟩^⊗L.
//!
//!     cargo run --release --bin trajectories -- [--lmin 4] [--lmax 8] [--gamma 0.01]
//!                         [--trajectories 1000] [--t-max 100] [--dt 0.1] [--seed 1]
//!
//! Output: data/trajectories_L<L>.csv with columns
//!   L, gamma, trajectories, t, mx, my, mz, mx_err, my_err, mz_err, mx_std, my_std, mz_std
//! m_α = (1/L) Σ_j ⟨σ^α_j⟩ averaged over the trajectories, *_err its standard error, *_std the
//! spread of a single trajectory (so that err = std / √trajectories).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::model::Params;
use open_exact_scars::trajectory::{Hamiltonian, Trajectory};

fn main() {
    let args = options();
    let get = |key: &str, default: f64| -> f64 {
        args.get(key).map_or(default, |s| s.parse().unwrap_or_else(|_| panic!("cannot parse --{key} {s}")))
    };
    let (lmin, lmax) = (get("lmin", 4.0) as usize, get("lmax", 8.0) as usize);
    let gamma = get("gamma", 0.01);
    let n_traj = get("trajectories", 1000.0) as usize;
    let (t_max, dt) = (get("t-max", 100.0), get("dt", 0.1));
    let seed = get("seed", 1.0) as u64;
    let steps = (t_max / dt).round() as usize;

    // output goes to <repo>/data, or to $SCARS_DATA_DIR if set
    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");

    for l in lmin..=lmax {
        let start = Instant::now();
        let ham = Hamiltonian::new(&Params::baseline(l));

        // per time: Σ m and Σ m² over the trajectories; the trajectories run in parallel
        let zero = || vec![[0.0f64; 6]; steps + 1];
        let sums = (0..n_traj)
            .into_par_iter()
            .map(|k| {
                let mut traj = Trajectory::new(&ham, seed.wrapping_mul(0x9e3779b97f4a7c15) ^ ((l as u64) << 48) ^ k as u64);
                let mut record = zero();
                for (step, r) in record.iter_mut().enumerate() {
                    if step > 0 {
                        traj.step(gamma, dt);
                    }
                    let [mx, my, mz] = traj.magnetization();
                    *r = [mx, my, mz, mx * mx, my * my, mz * mz];
                }
                record
            })
            .reduce(zero, |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| x.iter_mut().zip(y).for_each(|(p, q)| *p += q));
                a
            });

        let mut csv = BufWriter::new(File::create(data.join(format!("trajectories_L{l}.csv"))).unwrap());
        writeln!(csv, "L,gamma,trajectories,t,mx,my,mz,mx_err,my_err,mz_err,mx_std,my_std,mz_std").unwrap();
        let n = n_traj as f64;
        for (step, s) in sums.iter().enumerate() {
            let mean = [s[0] / n, s[1] / n, s[2] / n];
            // sample standard deviation of one trajectory
            let std = [0, 1, 2].map(|a| ((s[a + 3] / n - mean[a] * mean[a]).max(0.0) * n / (n - 1.0).max(1.0)).sqrt());
            let err = std.map(|x| x / n.sqrt());
            writeln!(
                csv,
                "{l},{gamma},{n_traj},{:.6},{:.10e},{:.10e},{:.10e},{:.3e},{:.3e},{:.3e},{:.4e},{:.4e},{:.4e}",
                step as f64 * dt,
                mean[0], mean[1], mean[2], err[0], err[1], err[2], std[0], std[1], std[2]
            )
            .unwrap();
        }
        let wall = start.elapsed().as_secs_f64();
        println!(
            "L = {l}: {n_traj} trajectories, {steps} steps, {wall:.1} s wall on {} threads = {:.3} core-seconds per trajectory",
            rayon::current_num_threads(),
            wall * rayon::current_num_threads().min(n_traj) as f64 / n_traj as f64
        );
    }
}

/// `--key value` pairs from the command line.
fn options() -> HashMap<String, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    argv.chunks(2)
        .map(|pair| match pair {
            [key, value] if key.starts_with("--") => (key[2..].to_string(), value.clone()),
            _ => panic!("options are given as --key value, got {pair:?}"),
        })
        .collect()
}
