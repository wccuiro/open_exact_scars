//! Exact time evolution of ρ(0) = |ψ₀⟩⟨ψ₀|, |ψ₀⟩ = |−⟩^⊗L, under
//!
//!     dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ),
//!
//! for each L from lmin to lmax, at γ = 0 (check: perfect oscillations) and at one fixed small γ.
//!
//!     cargo run --release -- [--gamma 0.01] [--lmin 4] [--lmax 8] [--t-max 100] [--dt 0.1]
//!
//! Output: data/evolution_L<L>.csv with columns  L, gamma, t, mx, my, mz
//! where m_α = (1/L) Σ_j ⟨σ^α_j⟩.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator};
use open_exact_scars::model::Params;

fn main() {
    let args = options();
    let get = |key: &str, default: f64| -> f64 {
        args.get(key).map_or(default, |s| s.parse().unwrap_or_else(|_| panic!("cannot parse --{key} {s}")))
    };
    let gamma = get("gamma", 0.01);
    let (lmin, lmax) = (get("lmin", 4.0) as usize, get("lmax", 8.0) as usize);
    let (t_max, dt) = (get("t-max", 100.0), get("dt", 0.1));
    let steps = (t_max / dt).round() as usize;

    // output goes to <repo>/data, or to $SCARS_DATA_DIR if set
    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");

    // the sizes are independent, so they run in parallel
    (lmin..=lmax).into_par_iter().for_each(|l| {
        let start = Instant::now();
        let params = Params::baseline(l);
        let liou = Liouvillian::new(&params);
        let magnetization = Magnetization::new(&liou);

        let mut csv = BufWriter::new(File::create(data.join(format!("evolution_L{l}.csv"))).unwrap());
        writeln!(csv, "L,gamma,t,mx,my,mz").unwrap();
        let mut check = 0.0f64;
        for g in [0.0, gamma] {
            let mut prop = Propagator::new(&liou, g, dt);
            let mut v = liou.initial_state();
            for step in 0..=steps {
                if step > 0 {
                    prop.step(&mut v);
                }
                let t = step as f64 * dt;
                let [mx, my, mz] = magnetization.measure(&v);
                writeln!(csv, "{l},{g},{t:.6},{mx:.12e},{my:.12e},{mz:.12e}").unwrap();
                if g == 0.0 {
                    // exact: m_x = −cos(ht), m_y = −sin(ht), m_z = 0
                    let (sin, cos) = (params.h * t).sin_cos();
                    check = check.max((mx + cos).abs()).max((my + sin).abs()).max(mz.abs());
                }
            }
        }
        println!(
            "L = {l}: dimension {}, {:.1?}; at gamma = 0 the largest deviation from (-cos ht, -sin ht, 0) is {check:.1e}",
            liou.dim(),
            start.elapsed()
        );
    });
}

/// `--key value` pairs from the command line.
fn options() -> HashMap<String, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    argv.chunks(2)
        .map(|pair| match pair {
            [key, value] if key.starts_with("--") => (key[2..].to_string(), value.clone()),
            _ => panic!("options are given as --key value, got {pair:?}"),
        })
        .collect()
}
