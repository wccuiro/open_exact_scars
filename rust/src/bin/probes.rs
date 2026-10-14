//! Scar vs thermal decay under dephasing (scar-vs-thermal-decay-plan.md): exact evolution of three
//! probes with the same Lindbladian, at γ = 0 and one γ > 0, for each L from lmin to lmax.
//!
//!     cargo run --release --bin probes -- [--gamma 0.01] [--L 6 | --lmin 4 --lmax 8] [--dt 0.1] [--seed 1]
//!                                         [--t-max T] [--gammas 0.002,0.005,... --tag name]
//!
//! Model coefficients: src/parameters.rs.
//! --gammas runs several γ > 0 (instead of --gamma); γ = 0 is always run. Without --t-max each probe
//! runs to its own T_* below, stretched by 0.01/γ for γ < 0.01; --t-max T runs every probe at every
//! γ to T. --tag appends _name to the output file names.
//!
//! H: baseline with random c, the realization of the `disorder` binary (Params::random_c with the
//! generator seeded with 1000 · seed + L). Full Pauli basis.
//!
//! Every operator X is stored by its coordinates x_P = Tr(X P) / 2^L and evolved forward with 𝓛.
//! Then Tr(A B) = 2^L Σ_P a_P b_P, and for Hermitian A, Tr[A e^{𝓛†t}(B)] = Tr[e^{𝓛t}(A) B].
//!
//! free  H_I = 0, ρ₀ = |−⟩⟨−|^⊗L: m_⊥ must be e^{−2γt} (sanity check).   t ≤ T_FREE
//! P1    ρ₀ = |−⟩⟨−|^⊗L: m_x, m_y, m_z, m_⊥ = √(m_x² + m_y²) and the tower weight
//!       w = Tr(P_tower ρ).                                                  t ≤ T_P1
//! P2    H̃ = H − h Sᶻ_tot P_tower: C(t) = Tr[H̃ e^{𝓛†t}(H̃)] / Tr H̃².        t ≤ T_P2
//! P3    O^x = Σ_j (−1)^j σˣ_j and O^y = Σ_j (−1)^j σʸ_j (even L), each evolved separately.
//!       With O⁺ = (O^x + i O^y)/2, C⁺(t) = Tr[(O⁺)† e^{𝓛†t}(O⁺)] / Tr[(O⁺)† O⁺]
//!       = [xx + yy + i (xy − yx)] / (2L), where ab = Tr[e^{𝓛t}(O^a) O^b] / 2^L; |C⁺| removes the
//!       precession in the field h. C_xx = xx / L is the plain autocorrelation.   t ≤ T_P3
//!
//! Output (data/ or $SCARS_DATA_DIR):
//!   probe_free[_tag].csv  L, gamma, t, m_perp, exact
//!   probe_P1[_tag].csv    L, gamma, t, mx, my, mz, m_perp, w
//!   probe_P2[_tag].csv    L, gamma, t, C
//!   probe_P3[_tag].csv    L, gamma, k, t, C_xx, C_re, C_im, C_abs
//! The checks of the plan (identities 4 and 5, conservation at γ = 0, initial rate of P2,
//! Tr(O_π H) = 0) are printed.

use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::liouville::{Liouvillian, Magnetization, Propagator};
use open_exact_scars::model::{hamiltonian_terms, Params};
use open_exact_scars::pauli::Pauli;
use open_exact_scars::tower;
use open_exact_scars::trajectory::Rng;

const T_FREE: f64 = 100.0;
const T_P1: f64 = 200.0;
const T_P2: f64 = 300.0;
const T_P3: f64 = 100.0;
/// Below this γ the run lengths grow as 1/γ.
const GAMMA_REF: f64 = 0.01;

/// One model size: the random-c Lindbladian and the tower traces.
struct Size {
    l: usize,
    params: Params,
    phi: Vec<f64>,
    liou: Liouvillian,
    /// Tr(P_tower P) / 2^L
    tower: Vec<f64>,
    /// Tr(Sᶻ_tot P_tower P) / 2^L
    tower_sz: Vec<f64>,
}

impl Size {
    fn new(l: usize, seed: u64) -> Self {
        let mut rng = Rng::new(1000 * seed + l as u64);
        let params = Params::random_c(l, &mut rng);
        let phi = (0..l).map(|_| 2.0 * PI * rng.uniform()).collect();
        let liou = Liouvillian::new(&params);
        let scale = 0.5f64.powi(l as i32);
        let tower = tower::traces(l, &vec![1.0; l + 1]).iter().map(|x| x * scale).collect();
        let sz: Vec<f64> = (0..=l).map(|n| l as f64 / 2.0 - n as f64).collect();
        let tower_sz = tower::traces(l, &sz).iter().map(|x| x * scale).collect();
        Size { l, params, phi, liou, tower, tower_sz }
    }

    /// Coordinates of Σ_j coef_j σ^axis_j.
    fn single_site(&self, axis: usize, coef: impl Fn(usize) -> f64) -> Vec<f64> {
        let mut v = vec![0.0; self.liou.dim()];
        for j in 0..self.l {
            v[self.liou.sector.orbit(Pauli::single(axis, j))] = coef(j);
        }
        v
    }

    /// Coordinates of H.
    fn hamiltonian(&self) -> Vec<f64> {
        let mut v = vec![0.0; self.liou.dim()];
        for (p, h) in hamiltonian_terms(&self.params) {
            v[self.liou.sector.orbit(p)] = h;
        }
        v
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Probe {
    Free,
    P1,
    P2,
    P3x,
    P3y,
}

fn main() {
    let args = Options::parse(&["gamma", "gammas", "L", "lmin", "lmax", "dt", "t-max", "seed", "tag"]);
    assert!(!(args.has("gamma") && args.has("gammas")), "give either --gamma or --gammas");
    let gammas: Vec<f64> = args.list("gammas").unwrap_or_else(|| vec![args.f64("gamma", 0.01)]);
    assert!(gammas.iter().all(|&g| g > 0.0), "γ = 0 is always run; list only γ > 0");
    let tag = args.str("tag").map_or(String::new(), |t| format!("_{t}"));
    let sizes = args.sizes(4, 8);
    let dt = args.f64("dt", 0.1);
    let t_max = args.has("t-max").then(|| args.f64("t-max", 0.0));
    let seed = args.usize("seed", 1) as u64;

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");

    let sizes: Vec<Size> = sizes.into_par_iter().map(|l| Size::new(l, seed)).collect();
    for s in &sizes {
        static_checks(s, gammas[0]);
    }
    let all_gammas: Vec<f64> = [0.0].into_iter().chain(gammas.iter().copied()).collect();

    let mut tasks = Vec::new();
    for s in &sizes {
        for &g in &all_gammas {
            tasks.extend([(s, g, Probe::P1), (s, g, Probe::P2)]);
            if s.l % 2 == 0 {
                tasks.extend([(s, g, Probe::P3x), (s, g, Probe::P3y)]);
            }
        }
        for &g in &gammas {
            tasks.push((s, g, Probe::Free));
        }
    }
    // largest systems and longest runs first, so that the small ones fill the gaps
    tasks.sort_by(|a, b| (b.0.l, a.1).partial_cmp(&(a.0.l, b.1)).unwrap());
    let results: Vec<((usize, f64, Probe), Vec<Vec<f64>>)> =
        tasks.par_iter().map(|&(s, g, probe)| ((s.l, g, probe), run(s, g, probe, dt, t_max))).collect();
    let find = |l: usize, g: f64, p: Probe| &results.iter().find(|r| r.0 == (l, g, p)).unwrap().1;

    let mut free = csv(&data, &format!("probe_free{tag}.csv"), "L,gamma,t,m_perp,exact");
    let mut p1 = csv(&data, &format!("probe_P1{tag}.csv"), "L,gamma,t,mx,my,mz,m_perp,w");
    let mut p2 = csv(&data, &format!("probe_P2{tag}.csv"), "L,gamma,t,C");
    let mut p3 = csv(&data, &format!("probe_P3{tag}.csv"), "L,gamma,k,t,C_xx,C_re,C_im,C_abs");
    for s in &sizes {
        let l = s.l;
        let mut worst = 0.0f64;
        for &g in &gammas {
            for row in find(l, g, Probe::Free) {
                let exact = (-2.0 * g * row[0]).exp();
                worst = worst.max((row[1] - exact).abs());
                writeln!(free, "{l},{g},{:.6},{:.12e},{exact:.12e}", row[0], row[1]).unwrap();
            }
        }
        println!("L = {l}: free spins, largest |m_perp − e^(−2γt)| = {worst:.1e}");
        for &g in &all_gammas {
            for row in find(l, g, Probe::P1) {
                let [t, mx, my, mz, w] = row[..] else { unreachable!() };
                writeln!(p1, "{l},{g},{t:.6},{mx:.12e},{my:.12e},{mz:.12e},{:.12e},{w:.12e}", mx.hypot(my)).unwrap();
            }
            for row in find(l, g, Probe::P2) {
                writeln!(p2, "{l},{g},{:.6},{:.12e}", row[0], row[1]).unwrap();
            }
            if l % 2 == 0 {
                let (x, y) = (find(l, g, Probe::P3x), find(l, g, Probe::P3y));
                for (rx, ry) in x.iter().zip(y) {
                    // rows: t, overlap with O^x, overlap with O^y
                    let lf = l as f64;
                    let (re, im) = ((rx[1] + ry[2]) / (2.0 * lf), (rx[2] - ry[1]) / (2.0 * lf));
                    writeln!(
                        p3,
                        "{l},{g},{PI:.6},{:.6},{:.12e},{re:.12e},{im:.12e},{:.12e}",
                        rx[0],
                        rx[1] / lf,
                        re.hypot(im)
                    )
                    .unwrap();
                }
            }
            if g == 0.0 {
                let p1_dev = find(l, g, Probe::P1)
                    .iter()
                    .map(|r| (r[1] + (s.params.h * r[0]).cos()).abs().max((r[2] + (s.params.h * r[0]).sin()).abs()))
                    .fold(0.0, f64::max);
                let p2_dev = find(l, g, Probe::P2).iter().map(|r| (r[1] - 1.0).abs()).fold(0.0, f64::max);
                println!("L = {l}, gamma = 0: P1 largest deviation from (−cos ht, −sin ht) {p1_dev:.1e}; P2 largest |C − 1| {p2_dev:.1e}");
            }
        }
    }
}

/// Checks that need no time evolution.
fn static_checks(s: &Size, gamma: f64) {
    let l = s.l;
    let liou = &s.liou;
    let mut lv = vec![0.0; liou.dim()];

    // identity 5: dw/dt at t = 0 is −γ(L − 1) for ρ₀ = |−⟩⟨−|^⊗L
    let rho0 = liou.initial_state();
    liou.apply(gamma, &rho0, &mut lv);
    let w0 = dot(&s.tower, &rho0);
    let dw = dot(&s.tower, &lv);
    println!(
        "L = {l}: identity 5: w(0) = {w0:.12}, dw/dt(0) / gamma = {:.10} (expected {})",
        dw / gamma,
        -(l as f64 - 1.0)
    );

    // identity 4: tower part of m_z for the cone state at θ = π/2 of the disorder runs
    let bloch: Vec<[f64; 3]> = s.phi.iter().map(|p| [0.0, p.cos(), p.sin()]).collect();
    let cone = liou.product_state(&bloch);
    let mz0 = 2.0 / l as f64 * dot(&s.single_site(2, |_| 0.5), &cone);
    let mz_tower = 2.0 / l as f64 * dot(&s.tower_sz, &cone);
    println!("L = {l}: identity 4: cone θ = π/2: m_z(0) = {mz0:.4}, tower part (2/L) Tr(P_tower Sᶻ ρ₀) = {mz_tower:.4}");

    // P2: H̃ has no tower part; predicted initial rate 2γ Σ n_xy x_P² / Σ x_P²
    let ht = h_tilde(s);
    let norm2 = dot(&ht, &ht);
    let rate: f64 = (0..liou.dim()).map(|i| liou.n_xy[i] * ht[i] * ht[i]).sum::<f64>() * 2.0 / norm2;
    let h = s.hamiltonian();
    let rate_h: f64 = (0..liou.dim()).map(|i| liou.n_xy[i] * h[i] * h[i]).sum::<f64>() * 2.0 / dot(&h, &h);
    println!(
        "L = {l}: P2: Tr(H̃ P_tower) = {:.1e}, Tr(H̃ Sᶻ P_tower) = {:.1e}; tower weight of H, Tr(hSᶻP_tower)² / Tr H² = {:.4}; \
         predicted Γ(0)/γ = {rate:.4} for H̃, {rate_h:.4} for H (h = {})",
        dot(&s.tower, &ht),
        dot(&s.tower_sz, &ht),
        1.0 - norm2 / dot(&h, &h),
        s.params.h
    );

    // P3: Tr(O_π H) = 0 exactly for k ≠ 0
    if l % 2 == 0 {
        let ox = s.single_site(0, |j| if j % 2 == 0 { 1.0 } else { -1.0 });
        println!("L = {l}: P3: Tr(O_π H) / (‖O_π‖ ‖H‖) = {:.1e}", dot(&ox, &h) / (dot(&ox, &ox) * dot(&h, &h)).sqrt());
    }
}

/// Coordinates of H̃ = H − h Sᶻ_tot P_tower.
fn h_tilde(s: &Size) -> Vec<f64> {
    s.hamiltonian().iter().zip(&s.tower_sz).map(|(h, t)| h - s.params.h * t).collect()
}

/// One evolution to `t_max_cli` if given, otherwise to the probe's own length; returns one row per step.
fn run(s: &Size, gamma: f64, probe: Probe, dt: f64, t_max_cli: Option<f64>) -> Vec<Vec<f64>> {
    let free_params;
    let free_liou;
    let (liou, t_max) = match probe {
        Probe::Free => {
            free_params = Params::free_spins(s.l);
            free_liou = Liouvillian::new(&free_params);
            (&free_liou, T_FREE)
        }
        Probe::P1 => (&s.liou, T_P1),
        Probe::P2 => (&s.liou, T_P2),
        Probe::P3x | Probe::P3y => (&s.liou, T_P3),
    };
    let t_max = match t_max_cli {
        Some(t) => t,
        None if gamma > 0.0 => t_max * (GAMMA_REF / gamma).max(1.0),
        None => t_max,
    };
    let stagger = |j: usize| if j % 2 == 0 { 1.0 } else { -1.0 };
    let (o_x, o_y) = (s.single_site(0, stagger), s.single_site(1, stagger));
    let mut v = match probe {
        Probe::Free | Probe::P1 => liou.initial_state(),
        Probe::P2 => h_tilde(s),
        Probe::P3x => o_x.clone(),
        Probe::P3y => o_y.clone(),
    };
    let v0_norm2 = dot(&v, &v);
    let magnetization = Magnetization::new(liou);
    let mut prop = Propagator::new(liou, gamma, dt);
    let steps = (t_max / dt).round() as usize;
    let mut rows = Vec::with_capacity(steps + 1);
    let v0 = v.clone();
    for step in 0..=steps {
        if step > 0 {
            prop.step(&mut v);
        }
        let t = step as f64 * dt;
        rows.push(match probe {
            Probe::Free => {
                let [mx, my, _] = magnetization.measure(&v);
                vec![t, mx.hypot(my)]
            }
            Probe::P1 => {
                let [mx, my, mz] = magnetization.measure(&v);
                vec![t, mx, my, mz, dot(&s.tower, &v)]
            }
            Probe::P2 => vec![t, dot(&v0, &v) / v0_norm2],
            Probe::P3x | Probe::P3y => vec![t, dot(&v, &o_x), dot(&v, &o_y)],
        });
    }
    rows
}

fn csv(data: &PathBuf, name: &str, header: &str) -> BufWriter<File> {
    let mut f = BufWriter::new(File::create(data.join(name)).unwrap());
    writeln!(f, "{header}").unwrap();
    f
}
