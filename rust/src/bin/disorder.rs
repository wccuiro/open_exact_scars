//! Exact evolution without translation symmetry: random c coefficients and product initial states
//! on a cone around −x with random azimuths,
//!
//!     dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ),
//!
//! for each L from lmin to lmax, at γ = 0 and at one fixed small γ, in the full Pauli basis (4^L).
//!
//! Model coefficients: src/parameters.rs.
//!
//!     cargo run --release --bin disorder -- [--gamma 0.01] [--L 6 | --lmin 4 --lmax 8] [--t-max 100] [--dt 0.1]
//!                                           [--thetas 0,0.125,0.25,0.375,0.5] [--seed 1]
//!
//! H: baseline, but on every bond j each nonzero c[α][β] is multiplied by an independent factor
//!    uniform in [0.5, 1.5] (Params::random_c). Every term still annihilates the tower.
//! ρ(0): spin j along n_j = (−cos θ, sin θ cos φ_j, sin θ sin φ_j), φ_j uniform in [0, 2π).
//!    θ = 0 is the scar state |−⟩^⊗L; θ > 0 is not in the tower. --thetas is in units of π.
//! One realization per L (generator seeded with 1000 · seed + L): first the c factors, then the φ_j.
//! The same c and φ_j are used for every θ and γ.
//!
//! Output: data/disorder_L<L>.csv with columns
//!     L, gamma, theta, t, mx, my, mz, x0 … x{L−1}, y0 …, z0 …
//! where m_α = (1/L) Σ_j ⟨σ^α_j⟩ and xj, yj, zj = ⟨σ^α_j⟩ on site j (bit j of the basis states).
//! data/disorder_realization_L<L>.csv: j, phi, and the c coefficients of bond (j, j+1).

use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator};
use open_exact_scars::model::Params;
use open_exact_scars::trajectory::Rng;

fn main() {
    let args = Options::parse(&["gamma", "L", "lmin", "lmax", "t-max", "dt", "thetas", "seed"]);
    let gamma = args.f64("gamma", 0.01);
    let sizes = args.sizes(4, 8);
    let (t_max, dt) = (args.f64("t-max", 100.0), args.f64("dt", 0.1));
    let steps = (t_max / dt).round() as usize;
    let seed = args.usize("seed", 1) as u64;
    let thetas: Vec<f64> =
        args.list("thetas").unwrap_or_else(|| vec![0.0, 0.125, 0.25, 0.375, 0.5]).iter().map(|x| x * PI).collect();

    // output goes to <repo>/data, or to $SCARS_DATA_DIR if set
    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");

    sizes.into_par_iter().for_each(|l| {
        let start = Instant::now();
        let mut rng = Rng::new(1000 * seed + l as u64);
        let params = Params::random_c(l, &mut rng);
        let phi: Vec<f64> = (0..l).map(|_| 2.0 * PI * rng.uniform()).collect();
        write_realization(&data, &params, &phi);

        let liou = Liouvillian::new(&params);
        let magnetization = Magnetization::new(&liou);

        // every (θ, γ) run is independent; rows are collected and written in order
        let runs: Vec<(f64, f64)> = thetas.iter().flat_map(|&th| [(th, 0.0), (th, gamma)]).collect();
        let results: Vec<(String, f64)> = runs
            .par_iter()
            .map(|&(theta, g)| {
                let (sin, cos) = theta.sin_cos();
                let bloch: Vec<[f64; 3]> =
                    phi.iter().map(|p| [-cos, sin * p.cos(), sin * p.sin()]).collect();
                let mut v = liou.product_state(&bloch);
                let mut prop = Propagator::new(&liou, g, dt);
                let (mut rows, mut check) = (String::new(), 0.0f64);
                for step in 0..=steps {
                    if step > 0 {
                        prop.step(&mut v);
                    }
                    let t = step as f64 * dt;
                    let site = magnetization.per_site(&v);
                    let [mx, my, mz] = site.clone().map(|m| m.iter().sum::<f64>() / l as f64);
                    rows += &format!("{l},{g},{theta:.6},{t:.6},{mx:.12e},{my:.12e},{mz:.12e}");
                    for m in site.iter().flatten() {
                        rows += &format!(",{m:.12e}");
                    }
                    rows.push('\n');
                    if theta == 0.0 && g == 0.0 {
                        // scar state: every spin follows (−cos ht, −sin ht, 0) exactly
                        let (s, c) = (params.h * t).sin_cos();
                        for j in 0..l {
                            check = check.max((site[0][j] + c).abs()).max((site[1][j] + s).abs()).max(site[2][j].abs());
                        }
                    }
                }
                (rows, check)
            })
            .collect();

        let mut csv = BufWriter::new(File::create(data.join(format!("disorder_L{l}.csv"))).unwrap());
        let header: Vec<String> = ["x", "y", "z"].iter().flat_map(|a| (0..l).map(move |j| format!("{a}{j}"))).collect();
        writeln!(csv, "L,gamma,theta,t,mx,my,mz,{}", header.join(",")).unwrap();
        let mut check = 0.0f64;
        for (rows, c) in &results {
            csv.write_all(rows.as_bytes()).unwrap();
            check = check.max(*c);
        }
        println!(
            "L = {l}: dimension {}, {:.1?}; theta = 0, gamma = 0: largest per-site deviation from (-cos ht, -sin ht, 0) is {check:.1e}",
            liou.dim(),
            start.elapsed()
        );
    });
}

/// φ_j and the c coefficients of bond (j, j+1), one row per site.
fn write_realization(data: &PathBuf, params: &Params, phi: &[f64]) {
    let c_bond = params.c_bond.as_ref().unwrap();
    let axes = ["x", "y", "z"];
    // only the entries that are nonzero in the baseline are random
    let entries: Vec<(usize, usize)> =
        (0..3).flat_map(|a| (0..3).map(move |b| (a, b))).filter(|&(a, b)| params.c[a][b] != 0.0).collect();
    let mut csv = BufWriter::new(File::create(data.join(format!("disorder_realization_L{}.csv", params.l))).unwrap());
    let names: Vec<String> = entries.iter().map(|&(a, b)| format!("c_{}{}", axes[a], axes[b])).collect();
    writeln!(csv, "j,phi,{}", names.join(",")).unwrap();
    for (j, p) in phi.iter().enumerate() {
        let c: Vec<String> = entries.iter().map(|&(a, b)| format!("{:.12}", c_bond[j][a][b])).collect();
        writeln!(csv, "{j},{p:.12},{}", c.join(",")).unwrap();
    }
}
