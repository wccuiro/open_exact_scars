//! Exact time evolution of ρ(0) = |ψ₀⟩⟨ψ₀| under
//!
//!     dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ),
//!
//! for each L from lmin to lmax (or one --L), at γ = 0 and at one fixed small γ.
//! Model coefficients: src/parameters.rs.
//!
//!     cargo run --release -- [--gamma 0.01] [--L 6 | --lmin 4 --lmax 8] [--t-max 100] [--dt 0.1]
//!                            [--init minus-x | random] [--seed 1]
//!
//! --init minus-x (default): |ψ₀⟩ = |−⟩^⊗L, the scar state; at γ = 0 the oscillations are perfect (check).
//! --init random: a product state with an independent, uniformly random Bloch vector on each site
//!                (generator seeded with 1000 · seed + L).
//!
//! Output: data/evolution_L<L>.csv (minus-x) or data/evolution_random_L<L>.csv (random), with
//! columns  L, gamma, t, mx, my, mz  where m_α = (1/L) Σ_j ⟨σ^α_j⟩.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator};
use open_exact_scars::model::Params;
use open_exact_scars::trajectory::{random_direction, Rng};

fn main() {
    let args = Options::parse(&["gamma", "L", "lmin", "lmax", "t-max", "dt", "init", "seed"]);
    let gamma = args.f64("gamma", 0.01);
    let sizes = args.sizes(4, 8);
    let (t_max, dt) = (args.f64("t-max", 100.0), args.f64("dt", 0.1));
    let steps = (t_max / dt).round() as usize;
    let seed = args.usize("seed", 1) as u64;
    let random = match args.str("init").unwrap_or("minus-x") {
        "minus-x" => false,
        "random" => true,
        other => panic!("--init is minus-x or random, got {other}"),
    };

    // output goes to <repo>/data, or to $SCARS_DATA_DIR if set
    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");

    // the sizes are independent, so they run in parallel
    sizes.into_par_iter().for_each(|l| {
        let start = Instant::now();
        let params = Params::baseline(l);
        let liou = Liouvillian::new(&params);
        let magnetization = Magnetization::new(&liou);
        let bloch: Vec<[f64; 3]> = if random {
            let mut rng = Rng::new(1000 * seed + l as u64);
            (0..l).map(|_| random_direction(&mut rng)).collect()
        } else {
            vec![[-1.0, 0.0, 0.0]; l]
        };

        let name = if random { format!("evolution_random_L{l}.csv") } else { format!("evolution_L{l}.csv") };
        let mut csv = BufWriter::new(File::create(data.join(name)).unwrap());
        writeln!(csv, "L,gamma,t,mx,my,mz").unwrap();
        let mut check = 0.0f64;
        let mut m0 = [0.0; 3];
        for g in [0.0, gamma] {
            let mut prop = Propagator::new(&liou, g, dt);
            let mut v = if random { liou.product_state(&bloch) } else { liou.initial_state() };
            for step in 0..=steps {
                if step > 0 {
                    prop.step(&mut v);
                }
                let t = step as f64 * dt;
                let [mx, my, mz] = magnetization.measure(&v);
                writeln!(csv, "{l},{g},{t:.6},{mx:.12e},{my:.12e},{mz:.12e}").unwrap();
                if step == 0 {
                    m0 = [mx, my, mz];
                }
                if g == 0.0 && !random {
                    // exact: m_x = −cos(ht), m_y = −sin(ht), m_z = 0
                    let (sin, cos) = (params.h * t).sin_cos();
                    check = check.max((mx + cos).abs()).max((my + sin).abs()).max(mz.abs());
                }
            }
        }
        if random {
            println!(
                "L = {l}: dimension {}, {:.1?}; initial (m_x, m_y, m_z) = ({:.4}, {:.4}, {:.4})",
                liou.dim(),
                start.elapsed(),
                m0[0],
                m0[1],
                m0[2]
            );
        } else {
            println!(
                "L = {l}: dimension {}, {:.1?}; at gamma = 0 the largest deviation from (-cos ht, -sin ht, 0) is {check:.1e}",
                liou.dim(),
                start.elapsed()
            );
        }
    });
}
