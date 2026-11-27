//! One jump (Poisson) trajectory of the Floquet circuit with depolarizing noise D(ρ) = γ Σ_j Σ_α (σ^α_j ρ σ^α_j − ρ)
//! (jumps √γ σ^α, `--channel xyz`), from |−⟩^⊗L. Each cycle: the noiseless Floquet gates (src/floquet.rs, U_d … U_1
//! then the field), then every site is kicked with probability p = (3/4)(1 − e^{−4γ dt}) by σˣ, σʸ or σᶻ (1/3 each).
//! Averaged over trajectories this is exactly the circuit's channel ρ → (1−p)ρ + (p/3) Σ_α σ^α ρ σ^α per site (the
//! noise sits at the end of the cycle). γ is set by the mean number of kicks per cycle on the chain, L p =
//! kicks_per_cycle. Since 2026-11-26; before, the γ column was that of `dep` (√(γ/2) σ^α, p = (3/4)(1 − e^{−2γ dt})),
//! twice this γ for the same kicks: those files have no channel column.
//! Output: singlet weight ⟨R_{j,j+1}⟩ = ‖R_{j,j+1} ψ‖² of every bond after every cycle, and the kicks.
//! From rust/: cargo run --release --example floquet_traj_bond -- [L] [kicks_per_cycle] [dt] [cycles] [seed] [parallel]
//!   > ../data/floquet_traj_bond.csv
//! parallel = 1 (default): gates, kicks and measurement split over threads (one large trajectory); 0: sequential.
//! Both give the same numbers (same operations per amplitude; the bond sums are added in a different order).
//! Leonardo comparison: --quench L gamma dt cycles seed, or
//! --single L 0 dt cycles seed site axis phase_radians. These write a wide row per cycle.
//! The isolated seed acts at age zero on the scar at the specified precession phase.
use std::io::{self, BufWriter, Write};
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::floquet::FloquetTraj;
use open_exact_scars::model::Params;
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::{per_site_expectations, product_amplitudes, Amp, Rng};

/// psi ← σ^axis_site psi, up to a global phase (Y = iXZ).
fn kick(psi: &mut [Amp], site: usize, axis: usize) {
    let bit = 1usize << site;
    if axis >= 1 {
        // σᶻ (and the Z of Y = iXZ)
        psi.par_iter_mut().enumerate().filter(|(s, _)| s & bit != 0).for_each(|(_, a)| *a = [-a[0], -a[1]]);
    }
    if axis <= 1 {
        // σˣ (and the X of Y)
        psi.par_chunks_exact_mut(2 * bit).for_each(|pair| {
            let (up, down) = pair.split_at_mut(bit);
            up.swap_with_slice(down);
        });
    }
}

/// ⟨ψ|R_{j,j+1}|ψ⟩ = Σ_a |ψ(a) − ψ(b)|² / 2, a with site j up and j+1 down, b = a with both flipped.
fn singlet_weight(psi: &[Amp], l: usize, j: usize) -> f64 {
    let (bj, bk) = (1usize << j, 1usize << ((j + 1) % l));
    let term = |a: usize| {
        let b = a ^ bj ^ bk;
        let d = [psi[a][0] - psi[b][0], psi[a][1] - psi[b][1]];
        (d[0] * d[0] + d[1] * d[1]) / 2.0
    };
    if psi.len() >= 1 << 15 {
        (0..psi.len()).into_par_iter().filter(|s| s & bj == 0 && s & bk != 0).map(term).sum()
    } else {
        (0..psi.len()).filter(|s| s & bj == 0 && s & bk != 0).map(term).sum()
    }
}

fn main() -> io::Result<()> {
    let mut argv = std::env::args().skip(1).peekable();
    let mode = argv.next_if(|arg| arg == "--quench" || arg == "--single");
    let sweep = mode.is_some();
    let args: Vec<f64> = argv.map(|a| a.parse().expect("numbers")).collect();
    let single = mode.as_deref() == Some("--single");
    if sweep { assert_eq!(args.len(), if single { 8 } else { 5 }); }
    let get = |i: usize, d: f64| args.get(i).copied().unwrap_or(d);
    let (l, noise_arg, dt) = (get(0, 12.0) as usize, get(1, 0.5), get(2, 0.1));
    let (cycles, seed, parallel) = (get(3, 300.0) as usize, get(4, 1.0) as u64, get(5, 1.0) != 0.0);
    let (p, gamma) = if sweep {
        assert!(noise_arg.is_finite() && noise_arg >= 0.0 && dt.is_finite() && dt > 0.0);
        (-0.75 * (-4.0 * noise_arg * dt).exp_m1(), noise_arg)
    } else {
        let p = noise_arg / l as f64;
        (p, -(1.0 - 4.0 * p / 3.0).ln() / (4.0 * dt))
    };
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    // γ = 0: the gates without noise; the kicks are applied below.
    let floquet = FloquetTraj::new(&params, &Channel::from_options(Some("xyz"), None, 1.0), 0.0, dt);
    let mut rng = Rng::new(seed);
    let phase = if single { args[7] } else { 0.0 };
    let spin = if phase == 0.0 { [-1.0, 0.0, 0.0] } else { [-phase.cos(), -phase.sin(), 0.0] };
    let mut psi = product_amplitudes(&vec![spin; l]);
    if single {
        assert!(gamma == 0.0 && args[5] >= 0.0 && args[5] < l as f64 && args[5].fract() == 0.0);
        assert!([0.0, 1.0, 2.0].contains(&args[6]) && phase.is_finite());
    }
    eprintln!("L = {l}, dt = {dt}, p = {p:.6} per site per cycle, gamma = {gamma:.6}, {cycles} cycles, seed {seed}");
    let mut csv = BufWriter::new(io::stdout().lock());
    if sweep {
        write!(csv, "L,gamma,dt,seed,phase,cycle,t,mx,my")?;
        for j in 0..l { write!(csv, ",r{j},kick{j}")?; }
        writeln!(csv)?;
    } else {
        writeln!(csv, "L,gamma,dt,cycle,t,bond,singlet,kick_axis,channel")?;
    }
    let mut n_kicks = 0;
    let start = Instant::now();
    for n in 0..=cycles {
        // kick_axis[j] = axis of the kick on site j in this cycle (-1: none)
        let mut kicked = vec![-1i32; l];
        if single && n == 0 {
            let (site, axis) = (args[5] as usize, args[6] as usize);
            kick(&mut psi, site, axis);
            kicked[site] = axis as i32;
            n_kicks += 1;
        }
        if n > 0 {
            if parallel && !sweep {
                floquet.step_parallel(&mut psi, &mut rng);
            } else {
                floquet.step(&mut psi, &mut rng);
            }
            for (j, k) in kicked.iter_mut().enumerate() {
                if rng.uniform() <= p {
                    let axis = ((rng.uniform() * 3.0) as usize).min(2);
                    kick(&mut psi, j, axis);
                    *k = axis as i32;
                    n_kicks += 1;
                }
            }
        }
        let t = n as f64 * dt;
        if sweep {
            let [mx, my, _] = per_site_expectations(&psi, l).map(|x| x.iter().sum::<f64>() / l as f64);
            write!(csv, "{l},{gamma},{dt},{seed},{phase:.12},{n},{t:.8},{mx:.12e},{my:.12e}")?;
            for (j, axis) in kicked.iter().enumerate() {
                write!(csv, ",{:.12e},{axis}", singlet_weight(&psi, l, j))?;
            }
            writeln!(csv)?;
        } else {
            for j in 0..l {
                let r = singlet_weight(&psi, l, j);
                writeln!(csv, "{l},{gamma},{dt},{n},{t:.4},{j},{r:.12e},{},xyz", kicked[j])?;
            }
        }
        if n % 50 == 0 {
            csv.flush()?;
            eprintln!("cycle {n}: {n_kicks} kicks, {:.1} s", start.elapsed().as_secs_f64());
        }
    }
    eprintln!("{n_kicks} kicks in {cycles} cycles, {:.1} s", start.elapsed().as_secs_f64());
    csv.flush()
}
