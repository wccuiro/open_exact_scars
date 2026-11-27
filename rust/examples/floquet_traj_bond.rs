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
use std::io::{self, BufWriter, Write};
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::floquet::FloquetTraj;
use open_exact_scars::model::Params;
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::{product_amplitudes, Amp, Rng};

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
    let args: Vec<f64> = std::env::args().skip(1).map(|a| a.parse().expect("numbers")).collect();
    let get = |i: usize, d: f64| args.get(i).copied().unwrap_or(d);
    let (l, per_cycle, dt) = (get(0, 12.0) as usize, get(1, 0.5), get(2, 0.1));
    let (cycles, seed, parallel) = (get(3, 300.0) as usize, get(4, 1.0) as u64, get(5, 1.0) != 0.0);
    let p = per_cycle / l as f64;
    let gamma = -(1.0 - 4.0 * p / 3.0).ln() / (4.0 * dt);
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    // γ = 0: the gates without noise; the kicks are applied below.
    let floquet = FloquetTraj::new(&params, &Channel::from_options(Some("xyz"), None, 1.0), 0.0, dt);
    let mut rng = Rng::new(seed);
    let mut psi = product_amplitudes(&vec![[-1.0, 0.0, 0.0]; l]);
    eprintln!("L = {l}, dt = {dt}, p = {p:.6} per site per cycle, gamma = {gamma:.6}, {cycles} cycles, seed {seed}");
    let mut csv = BufWriter::new(io::stdout().lock());
    writeln!(csv, "L,gamma,dt,cycle,t,bond,singlet,kick_axis,channel")?;
    let mut n_kicks = 0;
    let start = Instant::now();
    for n in 0..=cycles {
        // kick_axis[j] = axis of the kick on site j in this cycle (-1: none)
        let mut kicked = vec![-1i32; l];
        if n > 0 {
            if parallel {
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
        for j in 0..l {
            let r = singlet_weight(&psi, l, j);
            writeln!(csv, "{l},{gamma},{dt},{n},{:.4},{j},{r:.12e},{},xyz", n as f64 * dt, kicked[j])?;
        }
        if n % 50 == 0 {
            csv.flush()?;
            eprintln!("cycle {n}: {n_kicks} kicks, {:.1} s", start.elapsed().as_secs_f64());
        }
    }
    eprintln!("{n_kicks} kicks in {cycles} cycles, {:.1} s", start.elapsed().as_secs_f64());
    csv.flush()
}
