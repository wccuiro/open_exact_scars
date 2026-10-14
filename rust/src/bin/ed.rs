//! Full exact diagonalization of the Lindbladian (LAPACK from Intel MKL, linked by build.rs).
//!
//!     cargo run --release --bin ed -- [--gamma 0.01] [--L 6 | --lmin 4 --lmax 6] [--seed 1]
//!
//! Model coefficients: src/parameters.rs. No time evolution here, so no --t-max or --dt.
//!
//! H: baseline with random c, the realization of the `disorder` and `probes` binaries.
//!
//! γ > 0: 𝓛 is a dense real 4^L × 4^L matrix in the Pauli basis (coordinates r_P of ρ = 2^{−L} Σ r_P P);
//!   dgeev gives all eigenvalues λ and the right and left eigenvectors.
//! γ = 0: 𝓛 = −i[H, ·] is massively degenerate (every |E⟩⟨E| has λ = 0), so eigenvectors from a
//!   general solver are arbitrary mixtures. Instead H (2^L × 2^L) is diagonalized with zheevd, and the
//!   modes are |m⟩⟨n| with λ = −i(E_m − E_n).
//!
//! For every right eigenvector R (as a 2^L × 2^L matrix) with P = P_tower, Q = 1 − P:
//!   tower = ‖P R P‖ / ‖R‖,  mixed = √(‖P R Q‖² + ‖Q R P‖²) / ‖R‖,  bulk = ‖Q R Q‖ / ‖R‖
//! (Frobenius norms, tower² + mixed² + bulk² = 1), and for the left eigenvector U
//!   energy_overlap = |Tr(H U)| / (‖H‖ ‖U‖).
//!
//! Output: data/spectrum_L<L>.csv with columns  L, gamma, re, im, tower, mixed, bulk, energy_overlap
//! (both γ = 0 and the chosen γ). Checks are printed.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::liouville::Liouvillian;
use open_exact_scars::model::{hamiltonian_terms, Params};
use open_exact_scars::pauli::Pauli;
use open_exact_scars::tower;
use open_exact_scars::trajectory::Rng;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct C64 {
    re: f64,
    im: f64,
}

const COL_MAJOR: i32 = 102;

extern "C" {
    fn LAPACKE_dgeev(
        layout: i32,
        jobvl: u8,
        jobvr: u8,
        n: i32,
        a: *mut f64,
        lda: i32,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: i32,
        vr: *mut f64,
        ldvr: i32,
    ) -> i32;
    fn LAPACKE_zheevd(layout: i32, jobz: u8, uplo: u8, n: i32, a: *mut C64, lda: i32, w: *mut f64) -> i32;
}

/// i^k for k in 0..4, as (re, im).
const I_POW: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// One mode: eigenvalue and the four weights.
struct Mode {
    re: f64,
    im: f64,
    tower: f64,
    mixed: f64,
    bulk: f64,
    energy: f64,
}

fn main() {
    let args = Options::parse(&["gamma", "L", "lmin", "lmax", "seed"]);
    let gamma = args.f64("gamma", 0.01);
    let sizes = args.sizes(4, 6);
    let seed = args.usize("seed", 1) as u64;
    assert!(*sizes.end() <= 7, "dense ED needs 8 · 16^L bytes per matrix");

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");

    for l in sizes {
        let start = Instant::now();
        let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
        let phi = tower::states(l);

        let closed = closed_modes(&params, &phi);
        let liou = Liouvillian::new(&params);
        let open = open_modes(&params, &liou, &phi, gamma);

        // γ = 0 cross-check at small L: the dense 𝓛 at γ = 0 has the spectrum −i(E_m − E_n)
        if l <= 5 {
            let (wr, wi, _, _) = dgeev(liou.dense(0.0), liou.dim(), false);
            let mut a: Vec<f64> = wi.clone();
            let mut b: Vec<f64> = closed.iter().map(|m| m.im).collect();
            a.sort_by(f64::total_cmp);
            b.sort_by(f64::total_cmp);
            let dev = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            let re = wr.iter().map(|x| x.abs()).fold(0.0, f64::max);
            println!("L = {l}: gamma = 0 cross-check, dense dgeev vs −i(E_m − E_n): largest |Δ Im λ| = {dev:.1e}, largest |Re λ| = {re:.1e}");
        }

        // checks at γ > 0: one steady state, −2γL ≤ Re λ ≤ 0, Σ λ = Tr 𝓛 = −2γ Σ_P n_xy(P) = −γ L 4^L
        let max_re = open.iter().map(|m| m.re).fold(f64::MIN, f64::max);
        let n_zero = open.iter().filter(|m| m.re.hypot(m.im) < 1e-10).count();
        let min_re = open.iter().map(|m| m.re).fold(f64::MAX, f64::min);
        let trace: f64 = open.iter().map(|m| m.re).sum();
        let expected = -gamma * l as f64 * 4f64.powi(l as i32);
        println!(
            "L = {l}: gamma = {gamma}: max Re λ = {max_re:.1e}, {n_zero} eigenvalue(s) with |λ| < 1e-10, \
             min Re λ / (2γL) = {:.4}, Σ Re λ / (−γ L 4^L) = {:.12}",
            min_re / (2.0 * gamma * l as f64),
            trace / expected
        );
        let n_tower = open.iter().filter(|m| m.tower > 0.9).count();
        println!("L = {l}: {n_tower} modes with tower weight > 0.9 (the tower has (L + 1)² = {} coherences)", (l + 1) * (l + 1));

        let mut csv = BufWriter::new(File::create(data.join(format!("spectrum_L{l}.csv"))).unwrap());
        writeln!(csv, "L,gamma,re,im,tower,mixed,bulk,energy_overlap").unwrap();
        for (g, modes) in [(0.0, &closed), (gamma, &open)] {
            for m in modes.iter() {
                writeln!(
                    csv,
                    "{l},{g},{:.12e},{:.12e},{:.6e},{:.6e},{:.6e},{:.6e}",
                    m.re, m.im, m.tower, m.mixed, m.bulk, m.energy
                )
                .unwrap();
            }
        }
        println!("L = {l}: dimension {}, {:.1?}", liou.dim(), start.elapsed());
    }
}

/// γ = 0: modes |m⟩⟨n| from the eigenstates of H.
fn closed_modes(params: &Params, phi: &[Vec<f64>]) -> Vec<Mode> {
    let l = params.l;
    let n = 1usize << l;
    let terms = hamiltonian_terms(params);
    // dense H, column-major: H[t, s] += h i^k for P|s⟩ = i^k |t⟩
    let mut h = vec![C64::default(); n * n];
    for &(p, coef) in &terms {
        for s in 0..n as u32 {
            let (t, k) = p.apply(s);
            let (re, im) = I_POW[k as usize];
            let e = &mut h[s as usize * n + t as usize];
            e.re += coef * re;
            e.im += coef * im;
        }
    }
    let h_norm = h.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>().sqrt();
    let mut energies = vec![0.0; n];
    let info = unsafe { LAPACKE_zheevd(COL_MAJOR, b'V', b'U', n as i32, h.as_mut_ptr(), n as i32, energies.as_mut_ptr()) };
    assert_eq!(info, 0, "zheevd failed");

    // tower weight of each eigenstate, Σ_n |⟨Φ_n|E⟩|²
    let w: Vec<f64> = (0..n)
        .map(|m| {
            let v = &h[m * n..(m + 1) * n];
            phi.iter()
                .map(|f| {
                    let (re, im) = f.iter().zip(v).fold((0.0, 0.0), |(a, b), (x, z)| (a + x * z.re, b + x * z.im));
                    re * re + im * im
                })
                .sum()
        })
        .collect();
    let scars: Vec<usize> = (0..n).filter(|&m| w[m] > 0.5).collect();
    let dev = w.iter().map(|x| x.min(1.0 - x)).fold(0.0, f64::max);
    let mut e_scar: Vec<f64> = scars.iter().map(|&m| energies[m]).collect();
    e_scar.sort_by(f64::total_cmp);
    let e_dev = e_scar
        .iter()
        .enumerate()
        .map(|(i, e)| (e - params.h * (i as f64 - l as f64 / 2.0)).abs())
        .fold(0.0, f64::max);
    println!(
        "L = {l}: H: {} eigenstates in the tower (expected {}), tower weights within {dev:.1e} of 0 or 1, \
         scar energies within {e_dev:.1e} of h(L/2 − n)",
        scars.len(),
        l + 1
    );

    let mut modes = Vec::with_capacity(n * n);
    for a in 0..n {
        for b in 0..n {
            let (wa, wb) = (w[a], w[b]);
            modes.push(Mode {
                re: 0.0,
                im: -(energies[a] - energies[b]),
                tower: (wa * wb).sqrt(),
                mixed: (wa * (1.0 - wb) + (1.0 - wa) * wb).max(0.0).sqrt(),
                bulk: ((1.0 - wa) * (1.0 - wb)).max(0.0).sqrt(),
                energy: if a == b { energies[a].abs() / h_norm } else { 0.0 },
            });
        }
    }
    modes
}

/// γ > 0: all eigenpairs of the dense real 𝓛.
fn open_modes(params: &Params, liou: &Liouvillian, phi: &[Vec<f64>], gamma: f64) -> Vec<Mode> {
    let dim = liou.dim();
    let reps = &liou.sector.reps;
    let (wr, wi, vl, vr) = dgeev(liou.dense(gamma), dim, true);
    let mut h = vec![0.0; dim];
    for (p, coef) in hamiltonian_terms(params) {
        h[liou.sector.orbit(p)] = coef;
    }
    let h_norm = h.iter().map(|x| x * x).sum::<f64>().sqrt();

    (0..dim)
        .into_par_iter()
        .map(|j| {
            // complex eigenvector j: for a pair λ_j, λ_{j+1} = conj(λ_j) (wi[j] > 0) LAPACK stores
            // re in column j and im in column j + 1
            let column = |v: &[f64]| -> Vec<(f64, f64)> {
                if wi[j] == 0.0 {
                    v[j * dim..(j + 1) * dim].iter().map(|&x| (x, 0.0)).collect()
                } else {
                    let (c, sign) = if wi[j] > 0.0 { (j, 1.0) } else { (j - 1, -1.0) };
                    (0..dim).map(|i| (v[c * dim + i], sign * v[(c + 1) * dim + i])).collect()
                }
            };
            let (right, left) = (column(&vr), column(&vl));
            let (tower, mixed, bulk) = weights(&right, reps, phi);
            let (re, im) = left.iter().zip(&h).fold((0.0, 0.0), |(a, b), (u, x)| (a + x * u.0, b + x * u.1));
            let u_norm = left.iter().map(|u| u.0 * u.0 + u.1 * u.1).sum::<f64>().sqrt();
            Mode { re: wr[j], im: wi[j], tower, mixed, bulk, energy: re.hypot(im) / (h_norm * u_norm) }
        })
        .collect()
}

/// (tower, mixed, bulk) weights of R = Σ_P r_P P.
fn weights(r: &[(f64, f64)], reps: &[Pauli], phi: &[Vec<f64>]) -> (f64, f64, f64) {
    let n = phi[0].len();
    let mut m = vec![(0.0, 0.0); n * n]; // row-major, m[t * n + s] = ⟨t|R|s⟩
    for (&(a, b), p) in r.iter().zip(reps) {
        if a == 0.0 && b == 0.0 {
            continue;
        }
        for s in 0..n as u32 {
            let (t, k) = p.apply(s);
            let (c, d) = I_POW[k as usize];
            let e = &mut m[t as usize * n + s as usize];
            e.0 += a * c - b * d;
            e.1 += a * d + b * c;
        }
    }
    let abs2 = |z: (f64, f64)| z.0 * z.0 + z.1 * z.1;
    let total: f64 = m.iter().map(|&z| abs2(z)).sum();
    // B = Φ R (rows Φ_a), C = R Φ^T, A = Φ R Φ^T
    let mut b_norm = 0.0;
    let mut a_mat = vec![(0.0, 0.0); phi.len() * phi.len()];
    for (ia, fa) in phi.iter().enumerate() {
        let row: Vec<(f64, f64)> = (0..n)
            .map(|s| (0..n).fold((0.0, 0.0), |acc, t| (acc.0 + fa[t] * m[t * n + s].0, acc.1 + fa[t] * m[t * n + s].1)))
            .collect();
        b_norm += row.iter().map(|&z| abs2(z)).sum::<f64>();
        for (ib, fb) in phi.iter().enumerate() {
            a_mat[ia * phi.len() + ib] = row.iter().zip(fb).fold((0.0, 0.0), |acc, (z, x)| (acc.0 + x * z.0, acc.1 + x * z.1));
        }
    }
    let mut c_norm = 0.0;
    for fb in phi {
        for t in 0..n {
            let z = (0..n).fold((0.0, 0.0), |acc, s| (acc.0 + m[t * n + s].0 * fb[s], acc.1 + m[t * n + s].1 * fb[s]));
            c_norm += abs2(z);
        }
    }
    let a_norm: f64 = a_mat.iter().map(|&z| abs2(z)).sum();
    let mixed = (b_norm - a_norm) + (c_norm - a_norm);
    let bulk = total - b_norm - c_norm + a_norm;
    ((a_norm / total).sqrt(), (mixed.max(0.0) / total).sqrt(), (bulk.max(0.0) / total).sqrt())
}

/// All eigenvalues of a real column-major matrix, and the left and right eigenvectors if `vectors`.
fn dgeev(mut a: Vec<f64>, n: usize, vectors: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
    let size = if vectors { n * n } else { 1 };
    let (mut vl, mut vr) = (vec![0.0; size], vec![0.0; size]);
    let job = if vectors { b'V' } else { b'N' };
    let ld = if vectors { n } else { 1 } as i32;
    let info = unsafe {
        LAPACKE_dgeev(
            COL_MAJOR,
            job,
            job,
            n as i32,
            a.as_mut_ptr(),
            n as i32,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            ld,
            vr.as_mut_ptr(),
            ld,
        )
    };
    assert_eq!(info, 0, "dgeev failed");
    (wr, wi, vl, vr)
}
