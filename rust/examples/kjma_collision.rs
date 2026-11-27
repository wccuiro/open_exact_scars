//! A controlled collision and its two single-kick controls in the same disorder realization.
//! From rust/: cargo run --release --example kjma_collision > ../data/kjma_collision.csv
//! Add `-- depol` and use a separate CSV for the three equally likely Pauli axes per kick.
use std::io::{self, BufWriter, Write};

use open_exact_scars::model::Params;
use open_exact_scars::trajectory::{coherences, reference_bloch, Hamiltonian, Rng, Trajectory};

fn kick(state: &mut Trajectory<'_>, site: usize, axis: char) {
    if axis == 'z' || axis == 'y' {
        state.flip(site);
    }
    if axis == 'x' || axis == 'y' {
        // Swap up/down for X. Applying X after Z gives Y up to a global phase.
        let bit = 1 << site;
        for pair in state.psi.chunks_exact_mut(2 * bit) {
            let (up, down) = pair.split_at_mut(bit);
            up.swap_with_slice(down);
        }
    }
}

fn main() -> io::Result<()> {
    let l = 14;
    let dt = 0.05;
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    let ham = Hamiltonian::new(&params);
    let initial = reference_bloch(l, params.h, 0.0, None);
    let mut csv = BufWriter::new(io::stdout().lock());
    let depol = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => false,
        [mode] if mode == "depol" => true,
        _ => panic!("usage: kjma_collision [depol]"),
    };
    let axes: &[char] = if depol { &['x', 'y', 'z'] } else { &['z'] };
    let extra_header = if depol { ",axis_a,axis_b" } else { "" };
    writeln!(csv, "run,t,site,kick,damage{extra_header}")?;
    for (name, use_a, use_b) in [("A", true, false), ("B", false, true), ("AB", true, true)] {
        let axes_a = if use_a { axes } else { &['-'] };
        let axes_b = if use_b { axes } else { &['-'] };
        for &axis_a in axes_a {
            for &axis_b in axes_b {
                let labels = if depol {
                    format!(",{axis_a},{axis_b}")
                } else {
                    String::new()
                };
                let mut state = Trajectory::product(&ham, &initial, 0);
                for step in 0..=60 {
                    let t = step as f64 * dt;
                    if step > 0 {
                        state.unitary(dt);
                    }
                    let event = match step {
                        4 if use_a => Some((3, axis_a)),
                        16 if use_b => Some((10, axis_b)),
                        _ => None,
                    };
                    if let Some((site, axis)) = event {
                        kick(&mut state, site, axis);
                    }
                    for (j, c) in coherences(&state.per_site(), params.h, t)
                        .iter()
                        .enumerate()
                    {
                        let kicked = event.is_some_and(|(site, _)| site == j);
                        writeln!(
                            csv,
                            "{name},{t:.2},{j},{},{:.10}{labels}",
                            u8::from(kicked),
                            1.0 - c[0]
                        )?;
                    }
                }
            }
        }
    }
    csv.flush()
}
