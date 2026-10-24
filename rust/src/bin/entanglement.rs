//! Entanglement entropy of every eigenstate of H (γ = 0), against its energy.
//!
//!     cargo run --release --bin entanglement -- [--L 12 | --lmin 8 --lmax 12] [--seed 1]
//!
//! Model coefficients: src/parameters.rs. H has random c (Params::random_c, generator seeded with
//! 1000 · seed + L, the same realization as `states`, `ed` and the other binaries). Random c removes
//! translation, so H has no exact degeneracies apart from accidental ones and every eigenvector is
//! well defined (with uniform c the k, −k pairs would be degenerate and their eigenvectors arbitrary).
//!
//! H (2^L × 2^L, complex) is diagonalized densely with zheevd (LAPACK from Intel MKL, linked by build.rs).
//! For each eigenstate |E⟩ the subsystem A is the block of sites 0 … L/2 − 1 (bits 0 … L/2 − 1, two
//! boundaries on the ring), ρ_A = Tr_B |E⟩⟨E|, and S = −Σ p ln p over the eigenvalues p of ρ_A.
//! The scars are the eigenstates with tower weight Σ_n |⟨Φ_n|E⟩|² > 1/2.
//!
//! Checks printed: number of scars (L + 1), their energies h(L/2 − n), and their entropies against the
//! exact Dicke-state value, p_k = C(L_A, k) C(L − L_A, n − k) / C(L, n).
//!
//! Output: data/entanglement_L<L>.csv (or $SCARS_DATA_DIR) with columns  L, L_A, E, S, tower
//! (one row per eigenstate, ascending E). Page value for comparison (L_A ≤ L − L_A):
//! L_A ln 2 − 2^{L_A} / 2^{L − L_A + 1}, i.e. L_A ln 2 − 1/2 for even L.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::model::{hamiltonian_terms, Params};
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
    fn LAPACKE_zheevd(layout: i32, jobz: u8, uplo: u8, n: i32, a: *mut C64, lda: i32, w: *mut f64) -> i32;
    fn MKL_Get_Max_Threads() -> i32;
    fn MKL_Set_Num_Threads(n: i32);
}

/// i^k for k in 0..4, as (re, im).
const I_POW: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

fn main() {
    let args = Options::parse(&["L", "lmin", "lmax", "seed"]);
    let sizes = args.sizes(8, 12);
    let seed = args.usize("seed", 1) as u64;
    assert!(*sizes.end() <= 14, "dense H needs 16 · 4^L bytes (4.3 GB at L = 14)");

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");
    let mkl_threads = unsafe { MKL_Get_Max_Threads() };

    for l in sizes {
        let start = Instant::now();
        let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
        let n = 1usize << l;
        let l_a = l / 2;

        // dense H, column-major: H[t, s] += coef i^k for P|s⟩ = i^k |t⟩
        let mut h = vec![C64::default(); n * n];
        for (p, coef) in hamiltonian_terms(&params) {
            for s in 0..n as u32 {
                let (t, k) = p.apply(s);
                let (re, im) = I_POW[k as usize];
                let e = &mut h[s as usize * n + t as usize];
                e.re += coef * re;
                e.im += coef * im;
            }
        }
        let mut energies = vec![0.0; n];
        unsafe { MKL_Set_Num_Threads(mkl_threads) };
        let info =
            unsafe { LAPACKE_zheevd(COL_MAJOR, b'V', b'U', n as i32, h.as_mut_ptr(), n as i32, energies.as_mut_ptr()) };
        assert_eq!(info, 0, "zheevd failed");
        let t_diag = start.elapsed();

        // one small zheevd per eigenstate, from the rayon threads: MKL itself sequential
        unsafe { MKL_Set_Num_Threads(1) };
        let phi = tower::states(l);
        let (entropy, weight): (Vec<f64>, Vec<f64>) = (0..n)
            .into_par_iter()
            .map(|m| {
                let v = &h[m * n..(m + 1) * n];
                let w: f64 = phi
                    .iter()
                    .map(|f| {
                        let (re, im) = f.iter().zip(v).fold((0.0, 0.0), |(a, b), (x, z)| (a + x * z.re, b + x * z.im));
                        re * re + im * im
                    })
                    .sum();
                (entropy_half(v, l, l_a), w)
            })
            .unzip();

        // checks: the scars
        let scars: Vec<usize> = (0..n).filter(|&m| weight[m] > 0.5).collect();
        let w_dev = weight.iter().map(|x| x.min(1.0 - x)).fold(0.0, f64::max);
        let mut e_dev: f64 = 0.0;
        let mut s_dev: f64 = 0.0;
        for &m in &scars {
            let down = (l as f64 / 2.0 - energies[m] / params.h).round() as usize; // E = h(L/2 − n)
            e_dev = e_dev.max((energies[m] - params.h * (l as f64 / 2.0 - down as f64)).abs());
            s_dev = s_dev.max((entropy[m] - dicke_entropy(l, l_a, down)).abs());
        }
        println!(
            "L = {l}: {} eigenstates in the tower (expected {}), tower weights within {w_dev:.1e} of 0 or 1, \
             scar energies within {e_dev:.1e} of h(L/2 − n), scar entropies within {s_dev:.1e} of the Dicke value",
            scars.len(),
            l + 1
        );
        let page = l_a as f64 * std::f64::consts::LN_2 - 0.5 * (1u64 << l_a) as f64 / (1u64 << (l - l_a)) as f64;
        let s_max = entropy.iter().cloned().fold(0.0, f64::max);
        println!("L = {l}: L_A = {l_a}, largest S = {s_max:.4}, Page value = {page:.4}");

        let mut csv = BufWriter::new(File::create(data.join(format!("entanglement_L{l}.csv"))).unwrap());
        writeln!(csv, "L,L_A,E,S,tower").unwrap();
        for m in 0..n {
            writeln!(csv, "{l},{l_a},{:.12e},{:.12e},{:.6e}", energies[m], entropy[m], weight[m]).unwrap();
        }
        println!("L = {l}: dimension {n}, diagonalization {t_diag:.1?}, total {:.1?}", start.elapsed());
    }
}

/// −Tr ρ_A ln ρ_A for the sites 0 … l_a − 1 (the low bits of the basis index).
fn entropy_half(v: &[C64], l: usize, l_a: usize) -> f64 {
    let (da, db) = (1usize << l_a, 1usize << (l - l_a));
    // ρ_A[a, a'] = Σ_b ψ(a, b) ψ(a', b)*, ψ(a, b) = v[a + b · d_A]; column-major, upper triangle
    let mut rho = vec![C64::default(); da * da];
    for ap in 0..da {
        for a in 0..=ap {
            let (mut re, mut im) = (0.0, 0.0);
            for b in 0..db {
                let (x, y) = (v[a + b * da], v[ap + b * da]);
                re += x.re * y.re + x.im * y.im;
                im += x.im * y.re - x.re * y.im;
            }
            rho[a + ap * da] = C64 { re, im };
        }
    }
    let mut p = vec![0.0; da];
    let info = unsafe { LAPACKE_zheevd(COL_MAJOR, b'N', b'U', da as i32, rho.as_mut_ptr(), da as i32, p.as_mut_ptr()) };
    assert_eq!(info, 0, "zheevd failed");
    p.iter().filter(|&&x| x > 1e-300).map(|&x| -x * x.ln()).sum()
}

/// Exact entropy of the Dicke state with n down spins, cut into l_a and l − l_a sites.
fn dicke_entropy(l: usize, l_a: usize, n: usize) -> f64 {
    let binomial = |l: usize, k: usize| -> f64 {
        if k > l {
            0.0
        } else {
            (0..k).fold(1.0, |acc, i| acc * (l - i) as f64 / (i + 1) as f64)
        }
    };
    (0..=n.min(l_a))
        .map(|k| binomial(l_a, k) * binomial(l - l_a, n - k) / binomial(l, n))
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}
