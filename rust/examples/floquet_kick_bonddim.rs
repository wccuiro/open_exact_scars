//! Bond dimension after one depolarizing jump on site k of |−⟩^⊗L at t = 0, under the noiseless Floquet circuit
//! (src/floquet.rs; γ = 0 between jumps, the channel `xyz` only labels the circuit). The jump splits the state into two
//! pure branches (as in floquet_kicks_bond.rs): σˣ_k|−⟩^⊗L = |−⟩^⊗L (weight 1/3, the scar state ψ₀) and
//! σᶻ_k|−⟩^⊗L ∝ σʸ_k|−⟩^⊗L (weight 2/3, the damaged state). Each branch is evolved with the circuit, and its Schmidt
//! spectrum across every cut of the chain (block A = sites 0 … ℓ − 1, ℓ = 1 … L − 1: the MPS bond between sites ℓ − 1
//! and ℓ, the ring's link L−1|0 being the long-range bond of the MPS) gives
//!   chi_e = smallest number of Schmidt values whose discarded weight is ≤ e (e = 1e-10, 1e-6), S, S2.
//! The damaged branch is measured every `every` cycles, the scar branch every `every_scar` cycles (0: never).
//! From rust/: cargo run --release --example floquet_kick_bonddim -- L dt cycles site every every_scar > out.csv
use std::io::{self, BufWriter, Write};
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::entanglement::{block_spectrum_par, entropies};
use open_exact_scars::floquet::FloquetTraj;
use open_exact_scars::model::Params;
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::{product_amplitudes, Rng};

/// Smallest number of the largest weights whose sum is ≥ (1 − eps) of the total.
fn chi(weights: &[f64], eps: f64) -> usize {
    let mut w: Vec<f64> = weights.iter().map(|&x| x.max(0.0)).collect();
    w.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = w.iter().sum();
    let mut kept = 0.0;
    for (k, x) in w.iter().enumerate() {
        kept += x;
        if total - kept <= eps * total {
            return k + 1;
        }
    }
    w.len()
}

fn main() -> io::Result<()> {
    let args: Vec<f64> = std::env::args().skip(1).map(|a| a.parse().expect("numbers")).collect();
    assert_eq!(args.len(), 6, "usage: floquet_kick_bonddim L dt cycles site every every_scar");
    let (l, dt, cycles, site) = (args[0] as usize, args[1], args[2] as usize, args[3] as usize);
    let (every, every_scar) = (args[4] as usize, args[5] as usize);
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    let floquet = FloquetTraj::new(&params, &Channel::from_options(Some("xyz"), None, 1.0), 0.0, dt);
    let mut rng = Rng::new(1); // unused at γ = 0
    let scar = product_amplitudes(&vec![[-1.0, 0.0, 0.0]; l]);
    let mut damaged = scar.clone();
    let bit = 1usize << site;
    damaged.par_iter_mut().enumerate().filter(|(s, _)| s & bit != 0).for_each(|(_, a)| *a = [-a[0], -a[1]]);
    let mut branches = [("x", 1.0 / 3.0, scar, every_scar), ("y|z", 2.0 / 3.0, damaged, every)];
    eprintln!("L = {l}, dt = {dt}, {cycles} cycles, jump on site {site} at t = 0, every {every} / {every_scar} cycles");
    let mut csv = BufWriter::new(io::stdout().lock());
    writeln!(csv, "L,dt,cycle,t,site,branch,weight,ell,chi_max,chi_1e-10,chi_1e-6,S,S2")?;
    let start = Instant::now();
    for n in 0..=cycles {
        if n > 0 {
            for b in branches.iter_mut() {
                floquet.step_parallel(&mut b.2, &mut rng);
            }
        }
        for (name, weight, psi, ev) in branches.iter() {
            if *ev == 0 || n % ev != 0 {
                continue;
            }
            let t0 = Instant::now();
            for ell in 1..l {
                let w = block_spectrum_par(psi, l, ell);
                let (s1, s2) = entropies(&w);
                let chi_max = 1usize << ell.min(l - ell);
                writeln!(
                    csv,
                    "{l},{dt},{n},{:.4},{site},{name},{weight:.6},{ell},{chi_max},{},{},{s1:.10},{s2:.10}",
                    n as f64 * dt,
                    chi(&w, 1e-10),
                    chi(&w, 1e-6)
                )?;
            }
            csv.flush()?;
            eprintln!(
                "cycle {n} (t = {:.3}), branch {name}: all cuts in {:.1} s, total {:.1} s",
                n as f64 * dt,
                t0.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!("done, {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}
