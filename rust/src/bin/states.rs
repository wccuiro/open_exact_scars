//! Scar state against a thermal state under dephasing: exact evolution of ρ from
//!
//!     minus-x   ρ₀ = |−⟩⟨−|^⊗L        (in the tower: the scar)
//!     neel      ρ₀ = |Néel⟩⟨Néel|,  |Néel⟩ = |↑↓↑↓…⟩, spin j up for even j (even L; not a scar)
//!     neel-x    ρ₀ = |+−+−…⟩⟨+−+−…|, spin j along +x for even j, −x for odd j (even L; not a scar)
//!
//! with dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ), one run per γ, in the full Pauli basis (4^L).
//!
//!     cargo run --release --bin states -- --state minus-x|neel|neel-x [--L 8 | --lmin 4 --lmax 8]
//!                                         [--gammas 0,0.01,0.1] [--dt 0.1] [--t-max 1000] [--t-zero 50]
//!                                         [--stop 1e-3] [--seed 1]
//!
//! Model coefficients: src/parameters.rs. H has random c (Params::random_c, generator seeded with
//! 1000 · seed + L, the same realization as `disorder`, `probes` and `ed` for the same parameters).
//!
//! Run length: γ = 0 runs to --t-zero. γ > 0 runs until the decay signal (m_⊥ = √(m_x² + m_y²) for
//! minus-x, |s_z| for neel, s_⊥ = √(s_x² + s_y²) for neel-x) has stayed below --stop for a time
//! STOP_WINDOW, or to --t-max.
//!
//! z-Néel is diagonal in the σᶻ basis, so the dissipator leaves it (and s_z) alone until H_I rotates weight
//! into σˣ, σʸ. x-Néel is transverse like the scar: with H_I = 0 its s_⊥ decays as e^(−2γt), as m_⊥ does.
//! Output step: min(dt, DT_GAMMA / γ), so that the fast decay at large γ is resolved.
//!
//! Output (data/ or $SCARS_DATA_DIR), written while the run goes on:
//!   states_<state>_L<L>_g<γ>.csv    L, gamma, t, mx, my, mz, sx, sy, sz
//!       m_α = (1/L) Σ_j ⟨σ^α_j⟩,  s_α = (1/L) Σ_j (−1)^j ⟨σ^α_j⟩
//!   states_realization_L<L>.csv     j, and the c coefficients of bond (j, j+1)

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator};
use open_exact_scars::model::Params;
use open_exact_scars::trajectory::Rng;

/// γ > 0: stop once the signal has stayed below --stop for this long (longer than the period 2π/h).
const STOP_WINDOW: f64 = 10.0;
/// The output step is at most DT_GAMMA / γ.
const DT_GAMMA: f64 = 0.05;

#[derive(Clone, Copy, PartialEq)]
enum State {
    MinusX,
    Neel,
    NeelX,
}

fn main() {
    let args =
        Options::parse(&["state", "L", "lmin", "lmax", "gammas", "dt", "t-max", "t-zero", "stop", "seed"]);
    let state = match args.str("state") {
        Some("minus-x") => State::MinusX,
        Some("neel") => State::Neel,
        Some("neel-x") => State::NeelX,
        s => panic!("--state minus-x, neel or neel-x, got {s:?}"),
    };
    let name = match state {
        State::MinusX => "minus-x",
        State::Neel => "neel",
        State::NeelX => "neel-x",
    };
    let sizes = args.sizes(4, 8);
    let gammas = args.list("gammas").unwrap_or_else(|| vec![0.0, 0.01]);
    assert!(gammas.iter().all(|&g| g >= 0.0), "γ must be ≥ 0");
    let dt = args.f64("dt", 0.1);
    let (t_max, t_zero, stop) = (args.f64("t-max", 1000.0), args.f64("t-zero", 50.0), args.f64("stop", 1e-3));
    let seed = args.usize("seed", 1) as u64;

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");

    for l in sizes {
        if state != State::MinusX && l % 2 == 1 {
            println!("L = {l}: skipped, the Néel state needs even L");
            continue;
        }
        let start = Instant::now();
        let mut rng = Rng::new(1000 * seed + l as u64);
        let params = Params::random_c(l, &mut rng);
        write_realization(&data, &params);
        let liou = Liouvillian::new(&params);
        println!(
            "L = {l}: dimension {}, Liouvillian built in {:.1?}{}",
            liou.dim(),
            start.elapsed(),
            if liou.is_matrix_free() { " (matrix-free)" } else { "" }
        );

        let bloch: Vec<[f64; 3]> = (0..l)
            .map(|j| match state {
                State::MinusX => [-1.0, 0.0, 0.0],
                State::Neel => [0.0, 0.0, if j % 2 == 0 { 1.0 } else { -1.0 }],
                State::NeelX => [if j % 2 == 0 { 1.0 } else { -1.0 }, 0.0, 0.0],
            })
            .collect();
        let rho0 = liou.product_state(&bloch);

        // the γ runs are independent; each one also uses the threads inside 𝓛 v
        gammas.par_iter().for_each(|&g| {
            let file = data.join(format!("states_{name}_L{l}_g{g}.csv"));
            let t_end = run(&liou, &params, state, &rho0, g, dt, if g == 0.0 { t_zero } else { t_max }, stop, &file);
            println!("L = {l}, gamma = {g}: stopped at t = {t_end:.3}, {:.1?} since the start of L = {l}", start.elapsed());
        });
    }
}

/// Evolves ρ₀ and writes one CSV row per output step; returns the final time.
#[allow(clippy::too_many_arguments)]
fn run(
    liou: &Liouvillian,
    params: &Params,
    state: State,
    rho0: &[f64],
    gamma: f64,
    dt: f64,
    t_max: f64,
    stop: f64,
    file: &Path,
) -> f64 {
    let l = params.l;
    let dt = if gamma > 0.0 { dt.min(DT_GAMMA / gamma) } else { dt };
    let steps = (t_max / dt).round() as usize;
    let mut csv = BufWriter::new(File::create(file).unwrap());
    writeln!(csv, "L,gamma,t,mx,my,mz,sx,sy,sz").unwrap();
    let magnetization = Magnetization::new(liou);
    let mut prop = Propagator::new(liou, gamma, dt);
    let mut v = rho0.to_vec();
    let (mut below_since, mut t, mut check) = (None, 0.0, 0.0f64);
    let start = Instant::now();
    for step in 0..=steps {
        if step > 0 {
            prop.step(&mut v);
        }
        t = step as f64 * dt;
        let site = magnetization.per_site(&v);
        let [mx, my, mz] = site.clone().map(|m| m.iter().sum::<f64>() / l as f64);
        let [sx, sy, sz] =
            site.clone().map(|m| m.iter().enumerate().map(|(j, &x)| if j % 2 == 0 { x } else { -x }).sum::<f64>() / l as f64);
        writeln!(csv, "{l},{gamma},{t:.6},{mx:.12e},{my:.12e},{mz:.12e},{sx:.12e},{sy:.12e},{sz:.12e}").unwrap();
        if step % 1000 == 0 {
            csv.flush().unwrap();
        }
        if gamma == 0.0 && state == State::MinusX {
            // exact reference 1: every spin follows (−cos ht, −sin ht, 0)
            let (s, c) = (params.h * t).sin_cos();
            for j in 0..l {
                check = check.max((site[0][j] + c).abs()).max((site[1][j] + s).abs()).max(site[2][j].abs());
            }
        }
        if gamma > 0.0 {
            let signal = match state {
                State::MinusX => mx.hypot(my),
                State::Neel => sz.abs(),
                State::NeelX => sx.hypot(sy),
            };
            if signal >= stop {
                below_since = None;
            } else if t - *below_since.get_or_insert(t) >= STOP_WINDOW {
                break;
            }
        }
        if step > 0 && step % (steps / 20).max(1) == 0 {
            println!("  L = {l}, gamma = {gamma}: t = {t:.2} of at most {t_max}, {:.1?}", start.elapsed());
        }
    }
    csv.flush().unwrap();
    if gamma == 0.0 && state == State::MinusX {
        println!("L = {l}, gamma = 0: largest per-site deviation from (-cos ht, -sin ht, 0) is {check:.1e}");
    }
    t
}

/// The c coefficients of bond (j, j+1), one row per bond (only the entries nonzero in parameters.rs).
fn write_realization(data: &Path, params: &Params) {
    let c_bond = params.c_bond.as_ref().unwrap();
    let axes = ["x", "y", "z"];
    let entries: Vec<(usize, usize)> =
        (0..3).flat_map(|a| (0..3).map(move |b| (a, b))).filter(|&(a, b)| params.c[a][b] != 0.0).collect();
    // several jobs can write the same file at once: write a private copy and rename it (atomic)
    let file = data.join(format!("states_realization_L{}.csv", params.l));
    let tmp = data.join(format!("states_realization_L{}.csv.{}", params.l, std::process::id()));
    let mut csv = BufWriter::new(File::create(&tmp).unwrap());
    let names: Vec<String> = entries.iter().map(|&(a, b)| format!("c_{}{}", axes[a], axes[b])).collect();
    writeln!(csv, "j,{}", names.join(",")).unwrap();
    for (j, c) in c_bond.iter().enumerate() {
        let c: Vec<String> = entries.iter().map(|&(a, b)| format!("{:.12}", c[a][b])).collect();
        writeln!(csv, "{j},{}", c.join(",")).unwrap();
    }
    csv.into_inner().unwrap().sync_all().unwrap();
    fs::rename(&tmp, &file).unwrap();
}
