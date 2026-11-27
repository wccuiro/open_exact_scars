//! One depolarizing kick on |−⟩^⊗L at a random site k: σ^α_k with α = x, y, z (probability 1/3 each), then
//! evolution under H only (random c, seed 1). Output: singlet weight ⟨R_{j,j+1}⟩ = ‖R_{j,j+1} ψ‖² on every
//! bond of the ring against time, for each axis (the depolarizing result is the mean over the three).
//! From rust/: cargo run --release --example kjma_bond > ../data/kjma_bond.csv
use std::io::{self, BufWriter, Write};

use open_exact_scars::model::Params;
use open_exact_scars::trajectory::{Amp, Hamiltonian, Rng, Trajectory};

/// psi ← σ^axis_site psi, up to a global phase (Y = iXZ).
fn kick(state: &mut Trajectory<'_>, site: usize, axis: char) {
    if axis == 'z' || axis == 'y' {
        state.flip(site);
    }
    if axis == 'x' || axis == 'y' {
        let bit = 1 << site;
        for pair in state.psi.chunks_exact_mut(2 * bit) {
            let (up, down) = pair.split_at_mut(bit);
            up.swap_with_slice(down);
        }
    }
}

/// ⟨ψ|R_{j,j+1}|ψ⟩ = Σ_a |ψ(a) − ψ(b)|² / 2, a with site j up and j+1 down, b = a with both flipped.
fn singlet_weight(psi: &[Amp], l: usize, j: usize) -> f64 {
    let (bj, bk) = (1usize << j, 1usize << ((j + 1) % l));
    (0..psi.len())
        .filter(|s| s & bj == 0 && s & bk != 0)
        .map(|a| {
            let b = a ^ bj ^ bk;
            let d = [psi[a][0] - psi[b][0], psi[a][1] - psi[b][1]];
            (d[0] * d[0] + d[1] * d[1]) / 2.0
        })
        .sum()
}

fn main() -> io::Result<()> {
    let l = 16;
    let (dt, t_max): (f64, f64) = (0.05, 12.0);
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    let ham = Hamiltonian::new(&params);
    let site = (Rng::new(7).uniform() * l as f64) as usize % l;
    eprintln!("L = {l}, kicked site {site}");
    let mut csv = BufWriter::new(io::stdout().lock());
    writeln!(csv, "L,kick_site,axis,t,bond,singlet")?;
    let steps = (t_max / dt).round() as usize;
    for axis in ['x', 'y', 'z'] {
        let mut state = Trajectory::new(&ham, 0); // |−⟩^⊗L
        kick(&mut state, site, axis);
        for step in 0..=steps {
            if step > 0 {
                state.unitary(dt);
            }
            let t = step as f64 * dt;
            for j in 0..l {
                writeln!(csv, "{l},{site},{axis},{t:.2},{j},{:.12}", singlet_weight(&state.psi, l, j))?;
            }
        }
    }
    csv.flush()
}
