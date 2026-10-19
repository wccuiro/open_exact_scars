//! Damage of one σᶻ kick on the precessing scar state, evolved with H only (no noise):
//!
//!     |ψ_{m,k}(τ)⟩ = e^{−iHτ} σᶻ_k |ψ₀(t′)⟩,   t′ = φ_m / h,   φ_m = 2πm/N,  m = 0 … N−1,
//!
//! with |ψ₀(t′)⟩ the noiseless scar state at time t′ (reference 1, every spin along (−cos ht′, −sin ht′, 0)).
//! Per site, c_j = (⟨σˣ_j⟩ + i⟨σʸ_j⟩) / (reference-1 value at t′ + τ), which is 1 without the kick, and the
//! damage D(τ) = Σ_j [1 − Re c_j]: D(0) = 2 (the kicked spin points the other way), and with H_I = 0 D stays 2.
//! For weak dephasing, m_⊥(t) ≈ exp[−γ ∫₀ᵗ D(τ) dτ].
//!
//!     cargo run --release --bin single_flip -- [--L 12 | --lmin 4 --lmax 8] [--phases 16] [--sites all|0,3,6]
//!                                              [--t-max 25] [--dt 0.1] [--seed 1]
//!
//! H: src/parameters.rs with random c (Params::random_c, generator seeded with 1000 · seed + L, the same
//! realization as `states` and `states_traj`). All runs (phases × sites) advance together, one output step at
//! a time (as the trajectories of `states_traj`); memory 3 · 16 B · 2^L per run.
//!
//! Output (data/ or $SCARS_DATA_DIR), written while the run goes on:
//!   single_flip_L<L>.csv (all sites) or single_flip_L<L>_sites<k1-k2-…>.csv (a site list)
//!       L, m, phi, k, tau, D, then per site re_c0 … re_c{L−1}, im_c0 …, z0 … (z_j = ⟨σᶻ_j⟩)
//!   states_realization_L<L>.csv   j, and the c coefficients of bond (j, j+1)

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::model::{write_realization, Params};
use open_exact_scars::trajectory::{coherences, reference_bloch, Hamiltonian, Rng, Trajectory};

fn main() {
    let args = Options::parse(&["L", "lmin", "lmax", "phases", "sites", "t-max", "dt", "seed"]);
    let sizes = args.sizes(4, 8);
    let n_phases = args.usize("phases", 16);
    let (t_max, dt) = (args.f64("t-max", 25.0), args.f64("dt", 0.1));
    let seed = args.usize("seed", 1) as u64;
    let site_list: Option<Vec<usize>> = match args.str("sites") {
        None | Some("all") => None,
        Some(s) => Some(s.split(',').map(|k| k.trim().parse().expect("--sites all or a comma-separated list")).collect()),
    };

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");

    for l in sizes {
        let start = Instant::now();
        let mut rng = Rng::new(1000 * seed + l as u64);
        let params = Params::random_c(l, &mut rng);
        write_realization(&data, &params);
        let ham = Hamiltonian::new(&params);
        let h = params.h;
        let sites = site_list.clone().unwrap_or_else(|| (0..l).collect());
        assert!(sites.iter().all(|&k| k < l), "--sites: every site must be < L = {l}");
        let file = match &site_list {
            None => data.join(format!("single_flip_L{l}.csv")),
            Some(s) => data.join(format!(
                "single_flip_L{l}_sites{}.csv",
                s.iter().map(|k| k.to_string()).collect::<Vec<_>>().join("-")
            )),
        };

        // one run per (phase m, site k): t′ = φ_m / h
        let runs: Vec<(usize, f64, usize)> = (0..n_phases)
            .flat_map(|m| {
                let phi = 2.0 * std::f64::consts::PI * m as f64 / n_phases as f64;
                sites.iter().map(move |&k| (m, phi, k))
            })
            .collect();
        let mut trajs: Vec<Trajectory> = runs
            .par_iter()
            .map(|&(_, phi, k)| Trajectory::product(&ham, &reference_bloch(l, h, phi / h, Some(k)), 0))
            .collect();
        println!(
            "L = {l}: H built, {} runs ({n_phases} phases × {} sites), {:.1?}, {} threads",
            runs.len(),
            sites.len(),
            start.elapsed(),
            rayon::current_num_threads()
        );

        let mut csv = BufWriter::new(File::create(&file).unwrap());
        let cols = |p: &str| (0..l).map(|j| format!("{p}{j}")).collect::<Vec<_>>().join(",");
        writeln!(csv, "L,m,phi,k,tau,D,{},{},{}", cols("re_c"), cols("im_c"), cols("z")).unwrap();
        let steps = (t_max / dt).round() as usize;
        for step in 0..=steps {
            let tau = step as f64 * dt;
            if step > 0 {
                trajs.par_iter_mut().for_each(|traj| traj.unitary(dt));
            }
            let rows: Vec<String> = trajs
                .par_iter()
                .zip(&runs)
                .map(|(traj, &(m, phi, k))| {
                    let site = traj.per_site();
                    let c = coherences(&site, h, phi / h + tau);
                    let d: f64 = c.iter().map(|c| 1.0 - c[0]).sum();
                    let re = c.iter().map(|c| format!("{:.10e}", c[0])).collect::<Vec<_>>().join(",");
                    let im = c.iter().map(|c| format!("{:.10e}", c[1])).collect::<Vec<_>>().join(",");
                    let z = site[2].iter().map(|z| format!("{z:.10e}")).collect::<Vec<_>>().join(",");
                    format!("{l},{m},{phi:.10},{k},{tau:.6},{d:.12e},{re},{im},{z}")
                })
                .collect();
            for row in rows {
                writeln!(csv, "{row}").unwrap();
            }
            csv.flush().unwrap();
            if step > 0 && step % (1.0 / dt).round().max(1.0) as usize == 0 {
                println!("  L = {l}, single flip: tau = {tau:.2} of {t_max}, {:.1?}", start.elapsed());
            }
        }
        println!("L = {l}, single flip: stopped at t = {t_max:.3}, {:.1?} since the start of L = {l}", start.elapsed());
    }
}
