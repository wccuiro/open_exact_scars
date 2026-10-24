//! Operator entanglement of the right eigenmatrices of the Lindbladian (LAPACK from Intel MKL, linked by build.rs).
//!
//!     cargo run --release --bin lindblad_entanglement -- [--gamma 0.01 | --gamma 0] [--L 6 | --lmin 4 --lmax 6] [--seed 1]
//!                                                       [--channel z|x|y|dep|decay|thermal] [--nbar 1]
//!
//! --channel: the single-site noise (noise.rs; default σᶻ dephasing; file name with the channel suffix). The edge-mode
//! check below holds for σᶻ dephasing only (|Φ_0⟩, |Φ_L⟩ are its dark states) and is skipped for the other channels.
//!
//! Model coefficients: src/parameters.rs. H has random c (Params::random_c, generator seeded with
//! 1000 · seed + L, the same realization as `states`, `ed` and `entanglement`).
//!
//! 𝓛 = −i[H, ·] + γ Σ_j (σᶻ_j · σᶻ_j − ·) is the dense real 4^L × 4^L matrix of `ed` (full Pauli basis,
//! coordinates r_P), diagonalized with dgeev (right eigenvectors only). For each eigenvalue λ the right
//! eigenmatrix is X = Σ_P r_P P (2^L × 2^L, complex), and
//!
//!   ρ_L = X X† / Tr(X X†),   ρ_R = X† X / Tr(X† X),   E_L = Tr(ρ_L H),   E_R = Tr(ρ_R H).
//!
//! Operator entanglement S_op(ρ) across the cut A = sites 0 … L/2 − 1 (bits 0 … L/2 − 1) | B: write
//! ρ = Σ_k s_k A_k ⊗ B_k with Hilbert–Schmidt orthonormal A_k, B_k, w_k = s_k² / Σ s_k², S_op = −Σ w_k ln w_k.
//! The s_k are the singular values of M[(a, a′), (b, b′)] = ⟨a b|ρ|a′ b′⟩, obtained from the eigenvalues of
//! M M† (4^{L_A} × 4^{L_A}, zheevd). For a pure ρ = |ψ⟩⟨ψ|, S_op = 2 S(ψ).
//! tower = ‖P X P‖ / ‖X‖ (Frobenius, P the tower projector), as in `ed`.
//!
//! Degenerate eigenvalues (|Δλ| < DEGENERATE): the eigenspace basis is not unique, so ρ_L, ρ_R are not either.
//! There the basis is the one that diagonalizes the left energy form (A c = μ B c, A_ij = Tr(X_i† H X_j),
//! B_ij = Tr(X_i† X_j)), so E_L is stationary within the eigenspace. The degenerate eigenspaces are the kernel
//! (𝟙, |Φ_0⟩⟨Φ_0|, |Φ_L⟩⟨Φ_L|) and the edge pairs |Φ_0⟩⟨Φ_n|, |Φ_{L−n}⟩⟨Φ_L| (λ = −2nγ − ihn, 0 < n < L) and
//! their adjoints; this basis separates each pair exactly (the cross term of A vanishes).
//!
//! γ = 0 (𝓛 = −i[H, ·], eigenmatrices |n⟩⟨m|, λ = −i(E_n − E_m)): the kernel holds all 2^L |n⟩⟨n|, and the
//! equally spaced tower makes |Φ_a⟩⟨Φ_{a+k}| degenerate for each k. In the |n⟩⟨m| basis of such an eigenspace
//! A_ij = E_{n_i} δ_ij, so the same rule returns the single |n⟩⟨m| as long as H has no degenerate levels:
//! ρ_L = |n⟩⟨n|, ρ_R = |m⟩⟨m|, S_L = 2 S_n, S_R = 2 S_m (S_n of the `entanglement` binary).
//!
//! Check printed (γ > 0): the exact modes X = |Φ_a⟩⟨Φ_b| with (a, b) = (0, n), (L − n, L) (λ = −2nγ − i h n) and their
//! adjoints have ρ_L = |Φ_a⟩⟨Φ_a|, ρ_R = |Φ_b⟩⟨Φ_b|, so E_L = h(L/2 − a), S_L = 2 S_Dicke(a), and the same for b.
//! Check printed (γ = 0): ρ_L and ρ_R pure (Tr ρ² = 1), Re λ = 0, Im λ = −(E_L − E_R), E_L and E_R in the spectrum
//! of H (zheevd of H), and every level of H appearing 2^L times as E_L.
//!
//! Output: data/lindblad_entanglement_L<L>_g<γ>.csv (or $SCARS_DATA_DIR) with columns
//!   L, L_A, gamma, re, im, E_L, S_L, E_R, S_R, tower      (one row per eigenvalue)

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use open_exact_scars::cli::Options;
use open_exact_scars::liouville::Liouvillian;
use open_exact_scars::noise::Channel;
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
    fn LAPACKE_zhegv(
        layout: i32,
        itype: i32,
        jobz: u8,
        uplo: u8,
        n: i32,
        a: *mut C64,
        lda: i32,
        b: *mut C64,
        ldb: i32,
        w: *mut f64,
    ) -> i32;
    fn LAPACKE_zheevd(layout: i32, jobz: u8, uplo: u8, n: i32, a: *mut C64, lda: i32, w: *mut f64) -> i32;
    fn MKL_Get_Max_Threads() -> i32;
    fn MKL_Set_Num_Threads(n: i32);
}

/// i^k for k in 0..4, as (re, im).
const I_POW: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// One mode: eigenvalue, energies and operator entanglement of ρ_L and ρ_R, tower weight.
struct Mode {
    re: f64,
    im: f64,
    e_l: f64,
    s_l: f64,
    e_r: f64,
    s_r: f64,
    tower: f64,
    /// Tr ρ_L², Tr ρ_R² (for the γ = 0 check, not written)
    purity: (f64, f64),
}

fn main() {
    let args = Options::parse(&["gamma", "L", "lmin", "lmax", "seed", "channel", "nbar"]);
    let channel = Channel::parse(args.str("channel").unwrap_or("z"), args.f64("nbar", 1.0));
    let gamma = args.f64("gamma", 0.01);
    let sizes = args.sizes(4, 6);
    let seed = args.usize("seed", 1) as u64;
    assert!(gamma >= 0.0, "γ must be non-negative");
    assert!(*sizes.end() <= 7, "dense ED needs 8 · 16^L bytes per matrix");

    let data = std::env::var_os("SCARS_DATA_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data"), PathBuf::from);
    fs::create_dir_all(&data).expect("cannot create data/");
    let mkl_threads = unsafe { MKL_Get_Max_Threads() };

    for l in sizes {
        let start = Instant::now();
        let params = Params::random_c(l, &mut Rng::new(1000 * seed + l as u64));
        let mut liou = Liouvillian::new(&params);
        if channel != Channel::Z {
            liou.set_channel(channel);
        }
        let dim = liou.dim();
        assert_eq!(dim, 1 << (2 * l), "random c: full Pauli basis expected");

        unsafe { MKL_Set_Num_Threads(mkl_threads) };
        let (wr, wi, vr) = dgeev(liou.dense(gamma), dim);
        let t_diag = start.elapsed();

        let h = dense_h(&params);
        let phi = tower::states(l);
        let reps = &liou.sector.reps;
        // complex eigenvector j: for a pair λ_j, λ_{j+1} = conj(λ_j) (wi[j] > 0) LAPACK stores
        // re in column j and im in column j + 1
        let column = |j: usize| -> Vec<C64> {
            if wi[j] == 0.0 {
                vr[j * dim..(j + 1) * dim].iter().map(|&x| C64 { re: x, im: 0.0 }).collect()
            } else {
                let (c, sign) = if wi[j] > 0.0 { (j, 1.0) } else { (j - 1, -1.0) };
                (0..dim).map(|i| C64 { re: vr[c * dim + i], im: sign * vr[(c + 1) * dim + i] }).collect()
            }
        };

        // exactly degenerate eigenvalues: dgeev returns an arbitrary basis of the eigenspace, replaced by the
        // basis that diagonalizes the left energy form (E_L stationary within the eigenspace)
        let clusters = degenerate_clusters(&wr, &wi);
        let mut fixed: HashMap<usize, Vec<C64>> = HashMap::new();
        for cluster in &clusters {
            let r: Vec<Vec<C64>> = cluster.iter().map(|&j| column(j)).collect();
            fixed.extend(cluster.iter().copied().zip(energy_basis(&r, reps, l, &h)));
        }
        let mut sizes: Vec<usize> = clusters.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        println!("L = {l}: degenerate eigenvalues (|Δλ| < {DEGENERATE:.0e}): {} clusters of sizes {sizes:?}", clusters.len());

        // one small zheevd per mode, from the rayon threads: MKL itself sequential
        unsafe { MKL_Set_Num_Threads(1) };
        let modes: Vec<Mode> = (0..dim)
            .into_par_iter()
            .map(|j| {
                let r = fixed.get(&j).cloned().unwrap_or_else(|| column(j));
                let x = matrix(&r, reps, l);
                let (rho_l, rho_r) = gram(&x, l);
                Mode {
                    re: wr[j],
                    im: wi[j],
                    e_l: energy(&rho_l, &h),
                    s_l: operator_entropy(&rho_l, l, l / 2),
                    e_r: energy(&rho_r, &h),
                    s_r: operator_entropy(&rho_r, l, l / 2),
                    tower: tower_weight(&x, &phi),
                    purity: (purity(&rho_l), purity(&rho_r)),
                }
            })
            .collect();

        if gamma > 0.0 && channel != Channel::Z {
            println!("L = {l}: edge-mode check skipped (channel {channel:?}: |Φ_0⟩, |Φ_L⟩ are dark states of σᶻ dephasing only)");
        } else if gamma > 0.0 {
            // check: λ = −2nγ − ihn is the eigenspace of |Φ_0⟩⟨Φ_n| and |Φ_{L−n}⟩⟨Φ_L| (both dark-state edges,
            // E_0 − E_n = E_{L−n} − E_L = hn; one mode for n = L); λ = −2nγ + ihn holds the adjoints
            let (mut dev_e, mut dev_s, mut dev_lambda): (f64, f64, f64) = (0.0, 0.0, 0.0);
            let mut count_ok = true;
            let energy_n = |n: usize| params.h * (l as f64 / 2.0 - n as f64);
            let s_op = |n: usize| 2.0 * dicke_entropy(l, l / 2, n);
            for n in 1..=l {
                for sign in [-1.0, 1.0] {
                    let (lr, li) = (-2.0 * n as f64 * gamma, sign * params.h * n as f64);
                    let found: Vec<&Mode> = modes.iter().filter(|m| (m.re - lr).hypot(m.im - li) < 1e-8).collect();
                    count_ok &= found.len() == if n == l { 1 } else { 2 };
                    // (E_L, S_L, E_R, S_R) of X = |Φ_a⟩⟨Φ_b|: ρ_L = |Φ_a⟩⟨Φ_a|, ρ_R = |Φ_b⟩⟨Φ_b|
                    let (pairs, other) = ([(0, n), (l - n, l)], [(n, 0), (l, l - n)]);
                    for &(a, b) in if sign < 0.0 { &pairs } else { &other } {
                        let want = [energy_n(a), s_op(a), energy_n(b), s_op(b)];
                        let Some(m) = found.iter().min_by(|x, y| {
                            let d = |m: &Mode| (m.e_l - want[0]).abs() + (m.e_r - want[2]).abs();
                            d(x).total_cmp(&d(y))
                        }) else {
                            dev_lambda = f64::INFINITY;
                            continue;
                        };
                        dev_lambda = dev_lambda.max((m.re - lr).hypot(m.im - li));
                        dev_e = dev_e.max((m.e_l - want[0]).abs()).max((m.e_r - want[2]).abs());
                        dev_s = dev_s.max((m.s_l - want[1]).abs()).max((m.s_r - want[3]).abs());
                    }
                }
            }
            println!(
                "L = {l}: edge modes |Φ_0⟩⟨Φ_n|, |Φ_{{L−n}}⟩⟨Φ_L| and adjoints: multiplicities {}, λ within {dev_lambda:.1e} of \
                 −2nγ ∓ ihn, E_L, E_R within {dev_e:.1e}, S_L, S_R within {dev_s:.1e} of the exact values",
                if count_ok { "as expected" } else { "WRONG" }
            );
        } else {
            check_closed(&modes, &h, l);
        }
        let n_tower = modes.iter().filter(|m| m.tower > 0.9).count();
        println!("L = {l}: gamma = {gamma}: {n_tower} modes with tower weight > 0.9 (the tower has (L + 1)² = {} coherences)", (l + 1) * (l + 1));

        let name = format!("lindblad_entanglement_L{l}_g{gamma}{}.csv", channel.suffix());
        let mut csv = BufWriter::new(File::create(data.join(name)).unwrap());
        writeln!(csv, "L,L_A,gamma,re,im,E_L,S_L,E_R,S_R,tower").unwrap();
        for m in &modes {
            writeln!(
                csv,
                "{l},{},{gamma},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.6e}",
                l / 2, m.re, m.im, m.e_l, m.s_l, m.e_r, m.s_r, m.tower
            )
            .unwrap();
        }
        println!("L = {l}: dimension {dim}, diagonalization {t_diag:.1?}, total {:.1?}", start.elapsed());
    }
}

/// Eigenvalues closer than this are treated as one degenerate eigenspace.
const DEGENERATE: f64 = 1e-8;

/// Groups of indices (size ≥ 2) whose eigenvalues coincide within DEGENERATE.
fn degenerate_clusters(wr: &[f64], wi: &[f64]) -> Vec<Vec<usize>> {
    let mut assigned = vec![false; wr.len()];
    let mut clusters = Vec::new();
    for i in 0..wr.len() {
        if assigned[i] {
            continue;
        }
        let cluster: Vec<usize> =
            (i..wr.len()).filter(|&j| !assigned[j] && (wr[j] - wr[i]).hypot(wi[j] - wi[i]) < DEGENERATE).collect();
        for &j in &cluster {
            assigned[j] = true;
        }
        if cluster.len() > 1 {
            clusters.push(cluster);
        }
    }
    clusters
}

/// Basis of the eigenspace spanned by the eigenvectors r_i (Pauli coordinates) that diagonalizes the left
/// energy form: A c = μ B c with A_ij = Tr(X_i† H X_j), B_ij = Tr(X_i† X_j) (zhegv), X' = Σ_j c_j X_j.
fn energy_basis(r: &[Vec<C64>], reps: &[Pauli], l: usize, h: &[C64]) -> Vec<Vec<C64>> {
    let n = 1usize << l;
    let k = r.len();
    let xs: Vec<Vec<C64>> = r.iter().map(|v| matrix(v, reps, l)).collect();
    let hx: Vec<Vec<C64>> = xs.iter().map(|x| product(h, x, n)).collect();
    // Tr(Y† Z) = Σ conj(Y) Z
    let inner = |y: &[C64], z: &[C64]| {
        y.iter().zip(z).fold(C64::default(), |acc, (a, b)| C64 {
            re: acc.re + a.re * b.re + a.im * b.im,
            im: acc.im + a.re * b.im - a.im * b.re,
        })
    };
    let (mut a, mut b) = (vec![C64::default(); k * k], vec![C64::default(); k * k]);
    for i in 0..k {
        for j in 0..k {
            a[i + j * k] = inner(&xs[i], &hx[j]);
            b[i + j * k] = inner(&xs[i], &xs[j]);
        }
    }
    let mut w = vec![0.0; k];
    let info = unsafe {
        LAPACKE_zhegv(COL_MAJOR, 1, b'V', b'U', k as i32, a.as_mut_ptr(), k as i32, b.as_mut_ptr(), k as i32, w.as_mut_ptr())
    };
    assert_eq!(info, 0, "zhegv failed");
    (0..k)
        .map(|mu| {
            (0..r[0].len())
                .map(|p| {
                    (0..k).fold(C64::default(), |acc, j| {
                        let (c, v) = (a[j + mu * k], r[j][p]);
                        C64 { re: acc.re + c.re * v.re - c.im * v.im, im: acc.im + c.re * v.im + c.im * v.re }
                    })
                })
                .collect()
        })
        .collect()
}

/// Product of two row-major n × n matrices.
fn product(a: &[C64], b: &[C64], n: usize) -> Vec<C64> {
    let mut c = vec![C64::default(); n * n];
    for t in 0..n {
        for k in 0..n {
            let x = a[t * n + k];
            if x.re == 0.0 && x.im == 0.0 {
                continue;
            }
            for s in 0..n {
                let y = b[k * n + s];
                let e = &mut c[t * n + s];
                e.re += x.re * y.re - x.im * y.im;
                e.im += x.re * y.im + x.im * y.re;
            }
        }
    }
    c
}

/// γ = 0: every X should be |n⟩⟨m| (pure ρ_L, ρ_R), λ = −i(E_n − E_m), E_n and E_m levels of H, and every level
/// should appear 2^L times as E_L.
fn check_closed(modes: &[Mode], h: &[C64], l: usize) {
    let n = 1usize << l;
    let mut a = h.to_vec(); // row-major Hermitian = column-major of its conjugate: same eigenvalues
    let mut levels = vec![0.0; n];
    let info = unsafe { LAPACKE_zheevd(COL_MAJOR, b'N', b'U', n as i32, a.as_mut_ptr(), n as i32, levels.as_mut_ptr()) };
    assert_eq!(info, 0, "zheevd failed");
    let gap = levels.windows(2).map(|w| w[1] - w[0]).fold(f64::MAX, f64::min);
    // nearest level and its distance
    let nearest = |e: f64| -> (usize, f64) {
        let i = levels.partition_point(|&x| x < e);
        [i.saturating_sub(1), i.min(n - 1)].into_iter().map(|k| (k, (levels[k] - e).abs())).min_by(|p, q| p.1.total_cmp(&q.1)).unwrap()
    };
    let (mut dev_pure, mut dev_lambda, mut dev_level): (f64, f64, f64) = (0.0, 0.0, 0.0);
    let mut count = vec![0usize; n];
    for m in modes {
        dev_pure = dev_pure.max((m.purity.0 - 1.0).abs()).max((m.purity.1 - 1.0).abs());
        dev_lambda = dev_lambda.max(m.re.abs()).max((m.im + m.e_l - m.e_r).abs());
        let (k, d) = nearest(m.e_l);
        count[k] += 1;
        dev_level = dev_level.max(d).max(nearest(m.e_r).1);
    }
    println!(
        "L = {l}: gamma = 0: ρ_L, ρ_R pure within {dev_pure:.1e}, λ within {dev_lambda:.1e} of −i(E_L − E_R), \
         E_L, E_R within {dev_level:.1e} of levels of H (smallest level spacing {gap:.1e}), every level {} times as E_L",
        if count.iter().all(|&c| c == n) { format!("{n}") } else { "NOT".to_string() }
    );
}

/// Tr ρ² for a Hermitian ρ (row-major).
fn purity(rho: &[C64]) -> f64 {
    rho.iter().map(|z| z.re * z.re + z.im * z.im).sum()
}

/// Dense H, row-major: H[t * n + s] = ⟨t|H|s⟩.
fn dense_h(params: &Params) -> Vec<C64> {
    let n = 1usize << params.l;
    let mut h = vec![C64::default(); n * n];
    for (p, coef) in hamiltonian_terms(params) {
        for s in 0..n as u32 {
            let (t, k) = p.apply(s);
            let (re, im) = I_POW[k as usize];
            let e = &mut h[t as usize * n + s as usize];
            e.re += coef * re;
            e.im += coef * im;
        }
    }
    h
}

/// X = Σ_P r_P P, row-major: X[t * n + s] = ⟨t|X|s⟩.
fn matrix(r: &[C64], reps: &[Pauli], l: usize) -> Vec<C64> {
    let n = 1usize << l;
    let mut x = vec![C64::default(); n * n];
    for (c, p) in r.iter().zip(reps) {
        if c.re == 0.0 && c.im == 0.0 {
            continue;
        }
        for s in 0..n as u32 {
            let (t, k) = p.apply(s);
            let (a, b) = I_POW[k as usize];
            let e = &mut x[t as usize * n + s as usize];
            e.re += c.re * a - c.im * b;
            e.im += c.re * b + c.im * a;
        }
    }
    x
}

/// (X X† / Tr, X† X / Tr), row-major.
fn gram(x: &[C64], l: usize) -> (Vec<C64>, Vec<C64>) {
    let n = 1usize << l;
    let mut left = vec![C64::default(); n * n];
    let mut right = vec![C64::default(); n * n];
    for t in 0..n {
        for u in 0..n {
            let (mut lr, mut li, mut rr, mut ri) = (0.0, 0.0, 0.0, 0.0);
            for s in 0..n {
                // (X X†)[t, u] = Σ_s X[t, s] X[u, s]*
                let (a, b) = (x[t * n + s], x[u * n + s]);
                lr += a.re * b.re + a.im * b.im;
                li += a.im * b.re - a.re * b.im;
                // (X† X)[t, u] = Σ_s X[s, t]* X[s, u]
                let (a, b) = (x[s * n + t], x[s * n + u]);
                rr += a.re * b.re + a.im * b.im;
                ri += a.re * b.im - a.im * b.re;
            }
            left[t * n + u] = C64 { re: lr, im: li };
            right[t * n + u] = C64 { re: rr, im: ri };
        }
    }
    let trace: f64 = (0..n).map(|t| left[t * n + t].re).sum(); // Tr X X† = Tr X† X = ‖X‖²
    for z in left.iter_mut().chain(right.iter_mut()) {
        z.re /= trace;
        z.im /= trace;
    }
    (left, right)
}

/// Tr(ρ H) (real for Hermitian ρ and H).
fn energy(rho: &[C64], h: &[C64]) -> f64 {
    let n = (rho.len() as f64).sqrt() as usize;
    (0..n).map(|t| (0..n).map(|s| rho[t * n + s].re * h[s * n + t].re - rho[t * n + s].im * h[s * n + t].im).sum::<f64>()).sum()
}

/// Operator entanglement of ρ (row-major, 2^l × 2^l) for the cut sites 0 … l_a − 1 | the rest.
fn operator_entropy(rho: &[C64], l: usize, l_a: usize) -> f64 {
    let (da, db) = (1usize << l_a, 1usize << (l - l_a));
    let n = da * db;
    // M[(a, a′), (b, b′)] = ρ[a + b d_A, a′ + b′ d_A]; G = M M†, column-major (Hermitian)
    let d = da * da;
    let row = |i: usize, k: usize| -> C64 {
        let (a, ap, b, bp) = (i % da, i / da, k % db, k / db);
        rho[(a + b * da) * n + ap + bp * da]
    };
    let mut g = vec![C64::default(); d * d];
    for i2 in 0..d {
        for i1 in 0..=i2 {
            let (mut re, mut im) = (0.0, 0.0);
            for k in 0..db * db {
                let (x, y) = (row(i1, k), row(i2, k));
                re += x.re * y.re + x.im * y.im;
                im += x.im * y.re - x.re * y.im;
            }
            g[i1 + i2 * d] = C64 { re, im };
        }
    }
    let mut w = vec![0.0; d];
    let info = unsafe { LAPACKE_zheevd(COL_MAJOR, b'N', b'U', d as i32, g.as_mut_ptr(), d as i32, w.as_mut_ptr()) };
    assert_eq!(info, 0, "zheevd failed");
    let total: f64 = w.iter().map(|x| x.max(0.0)).sum();
    w.iter().map(|x| x / total).filter(|&x| x > 1e-300).map(|x| -x * x.ln()).sum()
}

/// ‖P X P‖ / ‖X‖ with P = Σ_n |Φ_n⟩⟨Φ_n| (Φ_n real).
fn tower_weight(x: &[C64], phi: &[Vec<f64>]) -> f64 {
    let n = phi[0].len();
    let total: f64 = x.iter().map(|z| z.re * z.re + z.im * z.im).sum();
    let mut inner = 0.0;
    for fa in phi {
        // row vector Φ_a X
        let row: Vec<C64> = (0..n)
            .map(|s| (0..n).fold(C64::default(), |acc, t| C64 { re: acc.re + fa[t] * x[t * n + s].re, im: acc.im + fa[t] * x[t * n + s].im }))
            .collect();
        for fb in phi {
            let (re, im) = row.iter().zip(fb).fold((0.0, 0.0), |(a, b), (z, f)| (a + f * z.re, b + f * z.im));
            inner += re * re + im * im;
        }
    }
    (inner / total).sqrt()
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

/// All eigenvalues and the right eigenvectors of a real column-major matrix.
fn dgeev(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
    let mut vr = vec![0.0; n * n];
    let mut vl = vec![0.0; 1];
    let info = unsafe {
        LAPACKE_dgeev(
            COL_MAJOR,
            b'N',
            b'V',
            n as i32,
            a.as_mut_ptr(),
            n as i32,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            1,
            vr.as_mut_ptr(),
            n as i32,
        )
    };
    assert_eq!(info, 0, "dgeev failed");
    (wr, wi, vr)
}
