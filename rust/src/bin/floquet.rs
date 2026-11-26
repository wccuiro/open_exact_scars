//! Trotterized (Floquet) evolution of ρ: the runs of `states` (same states, random-c realization, observables and
//! stop rule) with e^{𝓛 dt} replaced by one Floquet step
//!
//!     F = 𝓔 ∘ 𝓑 ∘ 𝓤_1 ∘ 𝓤_2 ∘ ⋯ ∘ 𝓤_d,     r(N dt) = F^N r(0)
//!
//! (src/floquet.rs, floquet-circuit.md: d layers of 4-site interaction gates, d = smallest divisor of L that is ≥ 4,
//! each layer the previous one shifted by one site, U_d first; then the uniform field 𝓑 = e^{−i h dt [Sᶻ_tot, ·]} and
//! the noise 𝓔 on every site). The γ = 0 scar is exact for any dt.
//!
//!     cargo run --release --bin floquet -- --state minus-x|neel|neel-x [--L 8 | --lmin 8 --lmax 12]
//!                                          [--gammas 0,0.01] [--dt 0.1] [--channel z|x|y|xyz|dep] [--noise depol]
//!                                          [--t-max 1000] [--t-zero 50] [--stop 1e-3] [--stop-te 0] [--seed 1]
//!
//! --channel: a Hermitian Pauli channel (noise.rs): z (default, √γ σᶻ), x, y, xyz (√γ σ^α, α = x, y, z: every Pauli
//! factor decays at 4γ), dep (√(γ/2) σ^α); --noise depol (√(γ/3) σ^α).
//! Trotter step = output step: dt · min(1, 0.5/γ) (= dt up to γ = 0.5, then ∝ 1/γ so that the decay stays resolved;
//! a run with half the --dt halves it at every γ). γ = 0 runs to --t-zero; γ > 0 until the signal (m_⊥ for minus-x,
//! |s_z| for neel, s_⊥ for neel-x) has stayed below --stop for 10 time units, or to --t-max, or (--stop-te F > 0) to
//! F · t₁, t₁ the start of the latest stretch with the signal below 1/e (as `states`).
//!
//! Output (data/ or $SCARS_DATA_DIR): floquet_<state>_L<L>_g<γ>_dt<dt><ch>[_seed<N>].csv (dt = the --dt given, <ch>
//! the channel suffix of `states`, empty for σᶻ) with the columns of `states`: L, gamma, t, mx, my, mz, sx, sy, sz.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use open_exact_scars::cli::Options;
use open_exact_scars::floquet::{per_site, product_state, step_at, Floquet};
use open_exact_scars::model::{seed_suffix, Params};
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::Rng;

/// γ > 0: stop once the signal has stayed below --stop for this long (as `states`).
const STOP_WINDOW: f64 = 10.0;

fn main() {
    let args = Options::parse(&[
        "state", "L", "lmin", "lmax", "gammas", "dt", "t-max", "t-zero", "stop", "stop-te", "seed", "channel", "noise",
    ]);
    let name = args.str("state").unwrap_or("minus-x").to_string();
    assert!(["minus-x", "neel", "neel-x"].contains(&name.as_str()), "--state minus-x, neel or neel-x");
    let channel = Channel::from_options(args.str("channel"), args.str("noise"), 1.0);
    let sizes = args.sizes(8, 8);
    let gammas = args.list("gammas").unwrap_or_else(|| vec![0.0, 0.01]);
    assert!(gammas.iter().all(|&g| g >= 0.0), "γ must be ≥ 0");
    let dt = args.f64("dt", 0.1);
    let (t_max, t_zero, stop) = (args.f64("t-max", 1000.0), args.f64("t-zero", 50.0), args.f64("stop", 1e-3));
    let stop_te = args.f64("stop-te", 0.0);
    let seed = args.usize("seed", 1) as u64;

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create the data directory");

    for l in sizes {
        if name != "minus-x" && l % 2 == 1 {
            println!("L = {l}: skipped, the Néel state needs even L");
            continue;
        }
        let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
        let bloch: Vec<[f64; 3]> = (0..l)
            .map(|j| {
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                match name.as_str() {
                    "minus-x" => [-1.0, 0.0, 0.0],
                    "neel" => [0.0, 0.0, sign],
                    _ => [sign, 0.0, 0.0],
                }
            })
            .collect();
        for &gamma in &gammas {
            let start = Instant::now();
            let step_dt = if gamma > 0.0 { step_at(dt, gamma) } else { dt };
            let floquet = Floquet::new(&params, &channel, gamma, step_dt);
            let mut r = product_state(&bloch);
            println!(
                "L = {l}, gamma = {gamma}: {} layers, {l} gates built and ρ(0) set in {:.1?}, step {step_dt}",
                floquet.layers,
                start.elapsed()
            );
            let file = data.join(format!("floquet_{name}_L{l}_g{gamma}_dt{dt}{}{}.csv", channel.suffix(), seed_suffix(seed)));
            let mut csv = BufWriter::new(File::create(&file).unwrap());
            writeln!(csv, "L,gamma,t,mx,my,mz,sx,sy,sz").unwrap();
            let steps = ((if gamma == 0.0 { t_zero } else { t_max }) / step_dt).round() as usize;
            let (mut below_since, mut t_cross, mut check, mut t) = (None, None::<f64>, 0.0f64, 0.0);
            let run_start = Instant::now();
            let mut done = 0;
            for step in 0..=steps {
                if step > 0 {
                    floquet.step(&mut r);
                    done += 1;
                }
                t = step as f64 * step_dt;
                let site = per_site(&r, l);
                let [mx, my, mz] = site.clone().map(|m| m.iter().sum::<f64>() / l as f64);
                let [sx, sy, sz] = site
                    .clone()
                    .map(|m| m.iter().enumerate().map(|(j, &x)| if j % 2 == 0 { x } else { -x }).sum::<f64>() / l as f64);
                writeln!(csv, "{l},{gamma},{t:.6},{mx:.12e},{my:.12e},{mz:.12e},{sx:.12e},{sy:.12e},{sz:.12e}").unwrap();
                csv.flush().unwrap();
                if gamma == 0.0 && name == "minus-x" {
                    // exact reference 1, for any dt
                    let (s, c) = (params.h * t).sin_cos();
                    for j in 0..l {
                        check = check.max((site[0][j] + c).abs()).max((site[1][j] + s).abs()).max(site[2][j].abs());
                    }
                }
                if gamma > 0.0 {
                    let signal = match name.as_str() {
                        "minus-x" => mx.hypot(my),
                        "neel" => sz.abs(),
                        _ => sx.hypot(sy),
                    };
                    if signal >= (-1.0f64).exp() {
                        t_cross = None;
                    } else if t_cross.is_none() {
                        t_cross = Some(t);
                    }
                    if stop_te > 0.0 && t_cross.is_some_and(|tc| t >= stop_te * tc) {
                        break;
                    }
                    if signal >= stop {
                        below_since = None;
                    } else if t - *below_since.get_or_insert(t) >= STOP_WINDOW {
                        break;
                    }
                }
            }
            let secs = run_start.elapsed().as_secs_f64();
            println!(
                "L = {l}, gamma = {gamma}: t = {t:.4} in {secs:.1} s ({done} steps, {:.3} s per step) -> {}",
                secs / done.max(1) as f64,
                file.display()
            );
            if gamma == 0.0 && name == "minus-x" {
                println!("L = {l}, gamma = 0: largest per-site deviation from (-cos ht, -sin ht, 0) is {check:.1e}");
            }
        }
    }
}
