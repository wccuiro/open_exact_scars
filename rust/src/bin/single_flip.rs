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
//!                                              [--t-max 25] [--dt 0.1] [--seed 1] [--ent-dt 0] [--ent-max L/2]
//!                                              [--theta θ | --cos-theta c] [--phi π] [--kick x,y,z]
//!
//! --theta / --cos-theta, --kick (decay-rate histograms, depolarizing noise): the reference state is the coherent
//! state at θ, every spin along (sin θ cos(φ + ht), sin θ sin(φ + ht), cos θ) (θ = π/2, φ = π: the scar state), the
//! kick σ^α_k with α in the --kick list (default z), c_j = (⟨σˣ_j⟩ + i⟨σʸ_j⟩) / (sin θ e^{i(φ + h(t′ + τ))}) and
//! D_αα(τ) = Σ_j [1 − Re c_j] for the kick α: D_zz(0) = 2, D_xx(0) + D_yy(0) = 2 on average over the phase, and with
//! H_I = 0 (D_xx + D_yy + D_zz)/3 = 4/3. For depolarizing noise a(t) ≈ a(0) exp[−(γ/3) ∫₀ᵗ (D_xx + D_yy + D_zz) dτ].
//! Output single_flip_L<L>_cos<cos θ>[_phi<φ>]_kick<axes>[_sites…][_seed<N>].csv with a column kick (x, y or z) after
//! k, one row per (m, k, kick, τ); without these options the binary and its output are as before.
//!
//! --ent-dt Δ > 0: every Δ the entanglement entropy (von Neumann and Rényi-2) of the blocks A = sites 0 … ℓ − 1,
//! ℓ = 1 … --ent-max, of every run (a 2^ℓ Hermitian eigenvalue problem per block). A seed N ≠ 1 (another random-c
//! realization) adds _seed<N> to the file names.
//!
//! H: src/parameters.rs with random c (Params::random_c, generator seeded with 1000 · seed + L, the same
//! realization as `states` and `states_traj`). All runs (phases × sites) advance together, one output step at
//! a time (as the trajectories of `states_traj`); memory 3 · 16 B · 2^L per run.
//!
//! Output (data/ or $SCARS_DATA_DIR), written while the run goes on:
//!   single_flip_L<L>[_sites<k1-k2-…>][_seed<N>].csv (all sites, or a site list)
//!       L, m, phi, k, tau, D, then per site re_c0 … re_c{L−1}, im_c0 …, z0 … (z_j = ⟨σᶻ_j⟩)
//!   single_flip_ent_L<L>[…].csv (with --ent-dt)   L, m, phi, k, tau, ell, S, S2
//!   states_realization_L<L>[_seed<N>].csv   j, and the c coefficients of bond (j, j+1)

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::entanglement::{block_spectrum, block_spectrum_par, entropies};
use open_exact_scars::model::{seed_suffix, write_realization, Params};
use open_exact_scars::trajectory::{
    coherences, coherences_theta, coherent_reference, reference_bloch, Hamiltonian, Rng, Trajectory,
};


fn main() {
    let args = Options::parse(&[
        "L", "lmin", "lmax", "phases", "sites", "t-max", "dt", "seed", "ent-dt", "ent-max", "theta", "cos-theta", "phi",
        "kick",
    ]);
    // --theta / --kick: coherent reference state at θ and kicks σ^α (None: the original scar-state σᶻ runs)
    let general = args.has("theta") || args.has("cos-theta") || args.has("kick");
    let theta = match (args.has("theta"), args.has("cos-theta")) {
        (true, true) => panic!("give --theta or --cos-theta"),
        (false, true) => args.f64("cos-theta", 0.0).acos(),
        _ => args.f64("theta", std::f64::consts::FRAC_PI_2),
    };
    let phi0 = args.f64("phi", std::f64::consts::PI);
    let kicks: Vec<usize> = args
        .str("kick")
        .unwrap_or("z")
        .split(',')
        .map(|a| match a.trim() {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            k => panic!("--kick: comma-separated x, y, z, got {k}"),
        })
        .collect();
    assert!(!general || !args.has("ent-dt"), "--ent-dt is not written for --theta / --kick runs");
    let ent_dt = args.f64("ent-dt", 0.0);
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
        write_realization(&data, &params, seed);
        let ent_max = args.usize("ent-max", l / 2).min(l / 2);
        let ham = Hamiltonian::new(&params);
        let h = params.h;
        let sites = site_list.clone().unwrap_or_else(|| (0..l).collect());
        assert!(sites.iter().all(|&k| k < l), "--sites: every site must be < L = {l}");
        let sites_tag = match &site_list {
            None => String::new(),
            Some(s) => format!("_sites{}", s.iter().map(|k| k.to_string()).collect::<Vec<_>>().join("-")),
        };
        let tag = if general {
            let mut c = (theta.cos() * 1e6).round() / 1e6;
            if c == 0.0 {
                c = 0.0;
            }
            let phi_tag = if (phi0 - std::f64::consts::PI).abs() < 1e-12 { String::new() } else { format!("_phi{phi0}") };
            let axes: String = kicks.iter().map(|&a| ["x", "y", "z"][a]).collect();
            format!("L{l}_cos{c}{phi_tag}_kick{axes}{sites_tag}{}", seed_suffix(seed))
        } else {
            format!("L{l}{sites_tag}{}", seed_suffix(seed))
        };
        let file = data.join(format!("single_flip_{tag}.csv"));
        let mut ent_csv = (ent_dt > 0.0).then(|| {
            let mut w = BufWriter::new(File::create(data.join(format!("single_flip_ent_{tag}.csv"))).unwrap());
            writeln!(w, "L,m,phi,k,tau,ell,S,S2").unwrap();
            w
        });
        let ent_every = if ent_dt > 0.0 { (ent_dt / dt).round().max(1.0) as usize } else { 0 };

        // one run per (phase m, site k, kick axis): t′ = φ_m / h
        let runs: Vec<(usize, f64, usize, usize)> = (0..n_phases)
            .flat_map(|m| {
                let phi = 2.0 * std::f64::consts::PI * m as f64 / n_phases as f64;
                let kicks = &kicks;
                sites.iter().flat_map(move |&k| kicks.iter().map(move |&a| (m, phi, k, a)))
            })
            .collect();
        let mut trajs: Vec<Trajectory> = runs
            .par_iter()
            .map(|&(_, phi, k, a)| {
                let bloch = if general {
                    coherent_reference(l, h, phi / h, theta, phi0, Some((k, a)))
                } else {
                    reference_bloch(l, h, phi / h, Some(k))
                };
                Trajectory::product(&ham, &bloch, 0)
            })
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
        if general {
            writeln!(csv, "L,m,phi,k,kick,tau,D,{},{},{}", cols("re_c"), cols("im_c"), cols("z")).unwrap();
        } else {
            writeln!(csv, "L,m,phi,k,tau,D,{},{},{}", cols("re_c"), cols("im_c"), cols("z")).unwrap();
        }
        let steps = (t_max / dt).round() as usize;
        for step in 0..=steps {
            let tau = step as f64 * dt;
            if step > 0 {
                trajs.par_iter_mut().for_each(|traj| traj.unitary(dt));
            }
            let rows: Vec<String> = trajs
                .par_iter()
                .zip(&runs)
                .map(|(traj, &(m, phi, k, a))| {
                    let site = traj.per_site();
                    let c = if general {
                        coherences_theta(&site, h, phi / h + tau, theta, phi0)
                    } else {
                        coherences(&site, h, phi / h + tau)
                    };
                    let d: f64 = c.iter().map(|c| 1.0 - c[0]).sum();
                    let re = c.iter().map(|c| format!("{:.10e}", c[0])).collect::<Vec<_>>().join(",");
                    let im = c.iter().map(|c| format!("{:.10e}", c[1])).collect::<Vec<_>>().join(",");
                    let z = site[2].iter().map(|z| format!("{z:.10e}")).collect::<Vec<_>>().join(",");
                    if general {
                        format!("{l},{m},{phi:.10},{k},{},{tau:.6},{d:.12e},{re},{im},{z}", ["x", "y", "z"][a])
                    } else {
                        format!("{l},{m},{phi:.10},{k},{tau:.6},{d:.12e},{re},{im},{z}")
                    }
                })
                .collect();
            for row in rows {
                writeln!(csv, "{row}").unwrap();
            }
            csv.flush().unwrap();
            if let Some(w) = ent_csv.as_mut() {
                if step % ent_every == 0 {
                    // many runs: one per thread; few runs (fewer than the threads): parallel inside each
                    let few = 2 * runs.len() <= rayon::current_num_threads();
                    let ent: Vec<Vec<(f64, f64)>> = if few {
                        trajs
                            .iter()
                            .map(|traj| (1..=ent_max).map(|ell| entropies(&block_spectrum_par(&traj.psi, l, ell))).collect())
                            .collect()
                    } else {
                        trajs
                            .par_iter()
                            .map(|traj| (1..=ent_max).map(|ell| entropies(&block_spectrum(&traj.psi, l, ell))).collect())
                            .collect()
                    };
                    for (e, &(m, phi, k, _)) in ent.iter().zip(&runs) {
                        for (i, (s1, s2)) in e.iter().enumerate() {
                            writeln!(w, "{l},{m},{phi:.10},{k},{tau:.6},{},{s1:.10e},{s2:.10e}", i + 1).unwrap();
                        }
                    }
                    w.flush().unwrap();
                }
            }
            if step > 0 && step % (1.0 / dt).round().max(1.0) as usize == 0 {
                println!("  L = {l}, single flip: tau = {tau:.2} of {t_max}, {:.1?}", start.elapsed());
            }
        }
        println!("L = {l}, single flip: stopped at t = {t_max:.3}, {:.1?} since the start of L = {l}", start.elapsed());
    }
}
