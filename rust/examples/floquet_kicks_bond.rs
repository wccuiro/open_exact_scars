//! Scheduled jumps on |−⟩^⊗L under the noiseless Floquet circuit (src/floquet.rs: interaction layers U_d … U_1, then
//! the field; no noise layer). A jump "n:k:α" acts on site k right after cycle n (n = 0: before the first cycle):
//! α = x, y, z applies σ^α_k; α = d is the full depolarizing jump ρ → (1/3) Σ_α σ^α_k ρ σ^α_k, done by branching every
//! state into its three images with weight 1/3 each (branches equal up to a phase are merged). ⟨R⟩ is linear in ρ,
//! so the output is the weighted average over the branches.
//! Output (stdout): singlet weight ⟨R_{j,j+1}⟩ = Tr(ρ R_{j,j+1}) of every bond after every cycle (columns of
//! floquet_traj_bond, gamma = 0; kick_axis = 0, 1, 2 for x, y, z, 3 for d on the jumped site in that cycle, −1
//! elsewhere). Optional 5th argument: a CSV with every branch (cycle, branch, label, weight, bond, singlet; "-" for
//! none). Optional 6th: a CSV with the fidelity F(n) = Σ_b w_b |⟨ψ₀(n)|ψ_b(n)⟩|² against the noiseless scar state
//! ψ₀(n) = ⊗_j (|↑⟩ − e^{iφ}|↓⟩)/√2, φ = h n dt (columns L, dt, cycle, t, fidelity, branches = label:weight:F_b;…).
//! From rust/: cargo run --release --example floquet_kicks_bond -- L dt cycles n:k:α[,n:k:α…] [branches.csv|-] [fidelity.csv]
//!   e.g. 24 0.1 1100 0:6:d,300:18:d ../data/floquet_kicks_bond_L24_branches.csv > ../data/floquet_kicks_bond_L24.csv
use std::fs::File;
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
        psi.par_iter_mut().enumerate().filter(|(s, _)| s & bit != 0).for_each(|(_, a)| *a = [-a[0], -a[1]]);
    }
    if axis <= 1 {
        psi.par_chunks_exact_mut(2 * bit).for_each(|pair| {
            let (up, down) = pair.split_at_mut(bit);
            up.swap_with_slice(down);
        });
    }
}

/// ⟨ψ|R_{j,j+1}|ψ⟩ = Σ_a |ψ(a) − ψ(b)|² / 2, a with site j up and j+1 down, b = a with both flipped.
fn singlet_weight(psi: &[Amp], l: usize, j: usize) -> f64 {
    let (bj, bk) = (1usize << j, 1usize << ((j + 1) % l));
    (0..psi.len())
        .into_par_iter()
        .filter(|s| s & bj == 0 && s & bk != 0)
        .map(|a| {
            let b = a ^ bj ^ bk;
            let d = [psi[a][0] - psi[b][0], psi[a][1] - psi[b][1]];
            (d[0] * d[0] + d[1] * d[1]) / 2.0
        })
        .sum()
}

/// |⟨ψ₀|ψ⟩|² for ψ₀ = ⊗_j (|↑⟩ − e^{iφ}|↓⟩)/√2 (the γ = 0 scar state at φ = h t; the circuit acts on the tower only
/// through the field). The amplitude of ψ₀ on a basis state with k down spins is 2^{−L/2} (−e^{iφ})^k, so only the
/// sums S_k of ψ over the states with k down spins are needed.
fn scar_fidelity(psi: &[Amp], l: usize, phi: f64) -> f64 {
    let sums = psi
        .par_iter()
        .enumerate()
        .fold(
            || vec![[0.0f64; 2]; l + 1],
            |mut acc, (s, a)| {
                let k = s.count_ones() as usize;
                acc[k][0] += a[0];
                acc[k][1] += a[1];
                acc
            },
        )
        .reduce(
            || vec![[0.0f64; 2]; l + 1],
            |mut x, y| {
                for (p, q) in x.iter_mut().zip(y) {
                    p[0] += q[0];
                    p[1] += q[1];
                }
                x
            },
        );
    // ⟨ψ₀|ψ⟩ = 2^{−L/2} Σ_k (−e^{−iφ})^k S_k
    let (mut re, mut im) = (0.0, 0.0);
    for (k, sk) in sums.iter().enumerate() {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let (c, sn) = (sign * (k as f64 * phi).cos(), -sign * (k as f64 * phi).sin());
        re += c * sk[0] - sn * sk[1];
        im += c * sk[1] + sn * sk[0];
    }
    (re * re + im * im) / 2f64.powi(l as i32)
}

/// |⟨a|b⟩|².
fn overlap2(a: &[Amp], b: &[Amp]) -> f64 {
    let (re, im) = a
        .par_iter()
        .zip(b)
        .map(|(x, y)| (x[0] * y[0] + x[1] * y[1], x[0] * y[1] - x[1] * y[0]))
        .reduce(|| (0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1));
    re * re + im * im
}

struct Branch {
    label: String,
    weight: f64,
    psi: Vec<Amp>,
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        (4..=6).contains(&args.len()),
        "usage: floquet_kicks_bond L dt cycles n:k:axis[,n:k:axis...] [branches.csv|-] [fidelity.csv]"
    );
    let l: usize = args[0].parse().unwrap();
    let dt: f64 = args[1].parse().unwrap();
    let cycles: usize = args[2].parse().unwrap();
    let schedule: Vec<(usize, usize, usize)> = args[3]
        .split(',')
        .map(|k| {
            let f: Vec<&str> = k.split(':').collect();
            let axis = ["x", "y", "z", "d"].iter().position(|&a| a == f[2]).expect("axis x, y, z or d");
            let site: usize = f[1].parse().unwrap();
            assert!(site < l, "site must be < L");
            (f[0].parse().unwrap(), site, axis)
        })
        .collect();
    let mut branch_csv =
        args.get(4).filter(|p| *p != "-").map(|p| BufWriter::new(File::create(p).expect("cannot create branch file")));
    let mut fid_csv = args.get(5).map(|p| BufWriter::new(File::create(p).expect("cannot create fidelity file")));
    if let Some(f) = fid_csv.as_mut() {
        writeln!(f, "L,dt,cycle,t,fidelity,branches")?;
    }
    if let Some(f) = branch_csv.as_mut() {
        writeln!(f, "cycle,branch,label,weight,bond,singlet")?;
    }
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    // γ = 0 (no noise between the scheduled jumps), so the channel only labels the circuit
    let floquet = FloquetTraj::new(&params, &Channel::from_options(Some("xyz"), None, 1.0), 0.0, dt);
    let mut rng = Rng::new(1); // unused at γ = 0 (no random rotations)
    let mut branches =
        vec![Branch { label: String::new(), weight: 1.0, psi: product_amplitudes(&vec![[-1.0, 0.0, 0.0]; l]) }];
    eprintln!("L = {l}, dt = {dt}, {} layers, {cycles} cycles, jumps {schedule:?} (cycle, site, axis 0-3 = x y z d)", floquet.layers);
    let mut csv = BufWriter::new(io::stdout().lock());
    writeln!(csv, "L,gamma,dt,cycle,t,bond,singlet,kick_axis")?;
    let start = Instant::now();
    for n in 0..=cycles {
        if n > 0 {
            for b in branches.iter_mut() {
                floquet.step_parallel(&mut b.psi, &mut rng);
            }
        }
        let mut kicked = vec![-1i32; l];
        for &(_, site, axis) in schedule.iter().filter(|k| k.0 == n) {
            kicked[site] = axis as i32;
            let axes: Vec<usize> = if axis == 3 { vec![0, 1, 2] } else { vec![axis] };
            let mut children: Vec<Branch> = Vec::new();
            for parent in branches.drain(..) {
                for &a in &axes {
                    let mut psi = parent.psi.clone();
                    kick(&mut psi, site, a);
                    let w = parent.weight / axes.len() as f64;
                    let tag = format!("{}{}{site}", parent.label, ["x", "y", "z"][a]);
                    if let Some(same) = children.iter_mut().find(|c| overlap2(&c.psi, &psi) > 1.0 - 1e-10) {
                        same.weight += w;
                        same.label = format!("{}|{tag}", same.label);
                    } else {
                        children.push(Branch { label: tag, weight: w, psi });
                    }
                }
            }
            branches = children;
            eprintln!(
                "cycle {n}: jump on site {site} -> {} branches: {}",
                branches.len(),
                branches.iter().map(|b| format!("{} ({:.4})", b.label, b.weight)).collect::<Vec<_>>().join(", ")
            );
        }
        let per_branch: Vec<Vec<f64>> =
            branches.iter().map(|b| (0..l).map(|j| singlet_weight(&b.psi, l, j)).collect()).collect();
        for j in 0..l {
            let r: f64 = branches.iter().zip(&per_branch).map(|(b, s)| b.weight * s[j]).sum();
            writeln!(csv, "{l},0,{dt},{n},{:.4},{j},{r:.12e},{}", n as f64 * dt, kicked[j])?;
        }
        if let Some(f) = fid_csv.as_mut() {
            let phi = params.h * n as f64 * dt;
            let fb: Vec<f64> = branches.iter().map(|b| scar_fidelity(&b.psi, l, phi)).collect();
            let total: f64 = branches.iter().zip(&fb).map(|(b, x)| b.weight * x).sum();
            let parts: Vec<String> =
                branches.iter().zip(&fb).map(|(b, x)| format!("{}:{:.6}:{x:.12e}", b.label, b.weight)).collect();
            writeln!(f, "{l},{dt},{n},{:.4},{total:.12e},{}", n as f64 * dt, parts.join(";"))?;
        }
        if let Some(f) = branch_csv.as_mut() {
            for (i, (b, s)) in branches.iter().zip(&per_branch).enumerate() {
                for (j, r) in s.iter().enumerate() {
                    writeln!(f, "{n},{i},{},{:.6},{j},{r:.12e}", b.label, b.weight)?;
                }
            }
        }
        if n % 50 == 0 {
            csv.flush()?;
            if let Some(f) = branch_csv.as_mut() {
                f.flush()?;
            }
            if let Some(f) = fid_csv.as_mut() {
                f.flush()?;
            }
            eprintln!("cycle {n}: {} branches, {:.1} s", branches.len(), start.elapsed().as_secs_f64());
        }
    }
    csv.flush()?;
    if let Some(f) = branch_csv.as_mut() {
        f.flush()?;
    }
    if let Some(f) = fid_csv.as_mut() {
        f.flush()?;
    }
    eprintln!("done, {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}
