//! Entanglement and operator statistics, without LAPACK (the cluster and workstation builds do not link MKL).
//!
//! - [`sym_eigenvalues`]: eigenvalues of a real symmetric matrix (Householder tridiagonalization, then implicit
//!   QL with Wilkinson shifts), O(n³); the trailing updates run on rayon threads for n ≥ PAR_MIN.
//! - [`block_spectrum`]: eigenvalues of ρ_A of a pure state for A = sites 0 … ℓ − 1 (the lowest ℓ bits), from the
//!   smaller of ΨΨ† and ΨᵀΨ*, Ψ_{ab} = ψ(a + 2^ℓ b). A complex Hermitian n × n matrix M = X + iY is diagonalized
//!   through the real symmetric 2n × 2n matrix [[X, −Y], [Y, X]], which has every eigenvalue of M twice.
//! - [`operator_entanglement`]: operator Schmidt weights of ρ = 2^{−L} Σ_P r_P P across the same cut. The Pauli
//!   strings factorize, P = P_A ⊗ P_B, and are orthogonal with equal norms, so the weights are the squared
//!   singular values of the matrix R[P_A, P_B] = r_P, normalized (eigenvalues of R Rᵀ on the smaller side).
//! - [`entropies`]: von Neumann and Rényi-2 entropies of normalized weights.
//! - [`operator_stats`]: purity and the size, X/Y-count and support distributions of ρ, weights ∝ r_P².

use rayon::prelude::*;

use crate::trajectory::Amp;

/// Matrices from this size on use rayon for the Householder updates.
const PAR_MIN: usize = 512;

/// Eigenvalues (ascending) of the real symmetric n × n matrix `a` (row-major, both triangles filled); the updates
/// run on rayon threads for n ≥ PAR_MIN. Use [`sym_eigenvalues_seq`] inside a parallel loop.
pub fn sym_eigenvalues(a: Vec<f64>, n: usize) -> Vec<f64> {
    eigenvalues_impl(a, n, n >= PAR_MIN)
}

/// As [`sym_eigenvalues`], always on the calling thread (for calls from inside a parallel loop over trajectories or
/// runs, where nested parallel updates only add scheduling overhead).
pub fn sym_eigenvalues_seq(a: Vec<f64>, n: usize) -> Vec<f64> {
    eigenvalues_impl(a, n, false)
}

fn eigenvalues_impl(mut a: Vec<f64>, n: usize, parallel: bool) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    if n == 0 {
        return vec![];
    }
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n]; // e[k] couples k and k + 1
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    for k in 0..n.saturating_sub(1) {
        d[k] = a[k * n + k];
        let m = n - k - 1; // size of the trailing block
        let x = &a[k * n + k + 1..k * n + n]; // column k below the diagonal (= row k, symmetric)
        // scale by the largest entry first: for tiny columns ‖x‖² would underflow and β = 2/(v·v) overflow
        let xmax = x.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if m == 1 || xmax == 0.0 {
            e[k] = x[0];
            continue;
        }
        // Householder vector v = x/xmax − α e₁ (H = 1 − β v vᵀ does not depend on the scale of v)
        for (vi, xi) in v[..m].iter_mut().zip(x) {
            *vi = xi / xmax;
        }
        let norm = v[..m].iter().map(|v| v * v).sum::<f64>().sqrt(); // ≥ 1
        let alpha = if v[0] > 0.0 { -norm } else { norm };
        e[k] = alpha * xmax;
        v[0] -= alpha;
        let vv: f64 = v[..m].iter().map(|v| v * v).sum(); // ≥ 1
        let beta = 2.0 / vv;
        // p = β B v, B the trailing block; w = p − (β vᵀp / 2) v; B ← B − v wᵀ − w vᵀ
        let off = k + 1;
        let row = |i: usize, a: &[f64]| -> f64 {
            let r = &a[(off + i) * n + off..(off + i) * n + n];
            beta * r.iter().zip(&v[..m]).map(|(x, y)| x * y).sum::<f64>()
        };
        if parallel && m >= PAR_MIN {
            let a_ref = &a;
            p[..m].par_iter_mut().enumerate().for_each(|(i, pi)| *pi = row(i, a_ref));
        } else {
            for i in 0..m {
                p[i] = row(i, &a);
            }
        }
        let kk = beta * v[..m].iter().zip(&p[..m]).map(|(x, y)| x * y).sum::<f64>() / 2.0;
        for i in 0..m {
            p[i] -= kk * v[i];
        }
        let (vs, ws) = (&v[..m], &p[..m]);
        let update = |(i, r): (usize, &mut [f64])| {
            let r = &mut r[off..];
            let (vi, wi) = (vs[i], ws[i]);
            for j in 0..m {
                r[j] -= vi * ws[j] + wi * vs[j];
            }
        };
        let rows = &mut a[off * n..];
        if parallel && m >= PAR_MIN {
            rows.par_chunks_mut(n).enumerate().for_each(update);
        } else {
            rows.chunks_mut(n).enumerate().for_each(update);
        }
    }
    d[n - 1] = a[(n - 1) * n + n - 1];
    if let Some(k) = (0..n).find(|&k| !d[k].is_finite() || !e[k].is_finite()) {
        panic!("Householder stage produced a non-finite entry at k = {k} of n = {n}");
    }
    tridiagonal_eigenvalues(&mut d, &mut e);
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    d
}

/// Implicit QL with Wilkinson shifts on the symmetric tridiagonal matrix (diagonal d, off-diagonal e[k] between k
/// and k + 1); d is overwritten with the eigenvalues.
fn tridiagonal_eigenvalues(d: &mut [f64], e: &mut [f64]) {
    let n = d.len();
    if n > 0 {
        e[n - 1] = 0.0;
    }
    // absolute floor of the deflation test: eigenvalues near 0 (low-rank ρ_A) never pass the relative test alone
    let scale = (0..n).map(|i| d[i].abs() + e[i].abs()).fold(0.0, f64::max);
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd || e[m].abs() <= f64::EPSILON * scale {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(iter < 100, "tridiagonal QL did not converge: l = {l}, m = {m}, d[l..=m] = {:?}, e[l..m] = {:?}, scale = {scale}", &d[l..=m.min(l + 4)], &e[l..m.min(l + 4)]);
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut deflated = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    deflated = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if deflated {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

/// Eigenvalues of the Hermitian n × n matrix with real part `re` and imaginary part `im` (row-major), on the calling
/// thread.
pub fn hermitian_eigenvalues(re: &[f64], im: &[f64], n: usize) -> Vec<f64> {
    hermitian_impl(re, im, n, false)
}

fn hermitian_impl(re: &[f64], im: &[f64], n: usize, parallel: bool) -> Vec<f64> {
    let n2 = 2 * n;
    let mut a = vec![0.0; n2 * n2];
    for i in 0..n {
        for j in 0..n {
            let (x, y) = (re[i * n + j], im[i * n + j]);
            a[i * n2 + j] = x;
            a[(i + n) * n2 + j + n] = x;
            a[i * n2 + j + n] = -y;
            a[(i + n) * n2 + j] = y;
        }
    }
    // every eigenvalue appears twice: keep one of each pair
    eigenvalues_impl(a, n2, parallel && n2 >= PAR_MIN).into_iter().step_by(2).collect()
}

/// Eigenvalues of the reduced density matrix of the block A = sites 0 … ell − 1 of the pure state ψ (2^l amplitudes,
/// normalized), computed on the smaller of A and its complement, on the calling thread.
pub fn block_spectrum(psi: &[Amp], l: usize, ell: usize) -> Vec<f64> {
    block_spectrum_impl(psi, l, ell, false)
}

/// As [`block_spectrum`], with the formation of ρ_A and the eigenvalue updates on rayon threads (for few states
/// at a time, e.g. 16 single-flip runs on a 112-core node).
pub fn block_spectrum_par(psi: &[Amp], l: usize, ell: usize) -> Vec<f64> {
    block_spectrum_impl(psi, l, ell, true)
}

fn block_spectrum_impl(psi: &[Amp], l: usize, ell: usize, parallel: bool) -> Vec<f64> {
    assert!(ell >= 1 && ell < l);
    let (da, db) = (1usize << ell, 1usize << (l - ell));
    // rows of the matrix whose Gram matrix is ρ: rows over a (Ψ_{a·}) if d_A ≤ d_B, else over b (Ψ_{·b})
    let (n, len) = if da <= db { (da, db) } else { (db, da) };
    let rows: Vec<Amp> = if da <= db {
        (0..da).flat_map(|a| (0..db).map(move |b| psi[a + b * da])).collect()
    } else {
        psi.to_vec() // row b = ψ(b·d_A .. b·d_A + d_A), already contiguous
    };
    // ρ_ij = Σ ψ_i ψ_j* for j ≤ i, row by row
    let lower_row = |i: usize| -> Vec<(f64, f64)> {
        let ri = &rows[i * len..(i + 1) * len];
        (0..=i)
            .map(|j| {
                let rj = &rows[j * len..(j + 1) * len];
                let (mut x, mut y) = (0.0, 0.0);
                for (u, w) in ri.iter().zip(rj) {
                    x += u[0] * w[0] + u[1] * w[1];
                    y += u[1] * w[0] - u[0] * w[1];
                }
                (x, y)
            })
            .collect()
    };
    let lower: Vec<Vec<(f64, f64)>> =
        if parallel { (0..n).into_par_iter().map(lower_row).collect() } else { (0..n).map(lower_row).collect() };
    let (mut re, mut im) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (i, row) in lower.iter().enumerate() {
        for (j, &(x, y)) in row.iter().enumerate() {
            re[i * n + j] = x;
            im[i * n + j] = y;
            re[j * n + i] = x;
            im[j * n + i] = -y;
        }
    }
    hermitian_impl(&re, &im, n, parallel)
}

/// (von Neumann, Rényi-2) entropies of nonnegative weights (normalized here; tiny negative round-off dropped).
pub fn entropies(weights: &[f64]) -> (f64, f64) {
    let total: f64 = weights.iter().filter(|&&w| w > 0.0).sum();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &w in weights {
        if w > 1e-300 {
            let p = w / total;
            s1 -= p * p.ln();
            s2 += p * p;
        }
    }
    (s1, -s2.ln())
}

/// Operator Schmidt weights of ρ = 2^{−L} Σ_P r_P P (full Pauli basis, index x | z << l) across sites 0 … ell − 1 | rest.
pub fn operator_weights(r: &[f64], l: usize, ell: usize) -> Vec<f64> {
    assert_eq!(r.len(), 1usize << (2 * l), "operator entanglement needs the full Pauli basis");
    let (na, nb) = (1usize << (2 * ell), 1usize << (2 * (l - ell)));
    let (ma, mb) = ((1usize << ell) - 1, (1usize << (l - ell)) - 1);
    // R[a, b] with a = x_A | z_A << ell, b = x_B | z_B << (l − ell)
    let mut m = vec![0.0; na * nb];
    for (i, &v) in r.iter().enumerate() {
        let (x, z) = (i & ((1 << l) - 1), i >> l);
        let a = (x & ma) | (z & ma) << ell;
        let b = ((x >> ell) & mb) | ((z >> ell) & mb) << (l - ell);
        m[a * nb + b] = v;
    }
    // Gram matrix on the smaller side
    let (n, len, rows) = if na <= nb {
        (na, nb, m)
    } else {
        let mut t = vec![0.0; na * nb];
        for a in 0..na {
            for b in 0..nb {
                t[b * na + a] = m[a * nb + b];
            }
        }
        (nb, na, t)
    };
    let gram: Vec<f64> = (0..n * n)
        .into_par_iter()
        .with_min_len(256)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            rows[i * len..(i + 1) * len].iter().zip(&rows[j * len..(j + 1) * len]).map(|(x, y)| x * y).sum()
        })
        .collect();
    sym_eigenvalues(gram, n)
}

/// Purity Tr ρ² and the distributions of ρ over Pauli strings with weights w_P = r_P² / Σ_Q r_Q² (identity included):
/// size n = number of non-identity sites, n_xy = number of σˣ/σʸ sites, support = length of the shortest arc of
/// the ring that contains all non-identity sites (0 for the identity). Full Pauli basis.
pub struct OperatorStats {
    pub purity: f64,
    pub mean_size: f64,
    pub mean_nxy: f64,
    /// P(n), n = 0 … L
    pub size: Vec<f64>,
    /// P(ℓ), ℓ = 0 … L
    pub support: Vec<f64>,
}

pub fn operator_stats(r: &[f64], l: usize) -> OperatorStats {
    assert_eq!(r.len(), 1usize << (2 * l), "operator statistics need the full Pauli basis");
    let mask = (1u32 << l) - 1;
    let (sum, nxy, size, support) = r
        .par_iter()
        .enumerate()
        .with_min_len(4096)
        .fold(
            || (0.0, 0.0, vec![0.0; l + 1], vec![0.0; l + 1]),
            |(mut s, mut nxy, mut size, mut support), (i, &v)| {
                let w = v * v;
                let (x, z) = (i as u32 & mask, (i >> l) as u32);
                let occ = x | z;
                s += w;
                nxy += w * x.count_ones() as f64;
                size[occ.count_ones() as usize] += w;
                support[arc_length(occ, l)] += w;
                (s, nxy, size, support)
            },
        )
        .reduce(
            || (0.0, 0.0, vec![0.0; l + 1], vec![0.0; l + 1]),
            |(s1, n1, mut a1, mut b1), (s2, n2, a2, b2)| {
                a1.iter_mut().zip(&a2).for_each(|(x, y)| *x += y);
                b1.iter_mut().zip(&b2).for_each(|(x, y)| *x += y);
                (s1 + s2, n1 + n2, a1, b1)
            },
        );
    let size: Vec<f64> = size.iter().map(|w| w / sum).collect();
    let support = support.iter().map(|w| w / sum).collect();
    OperatorStats {
        purity: sum / (1u64 << l) as f64,
        mean_size: size.iter().enumerate().map(|(n, p)| n as f64 * p).sum(),
        mean_nxy: nxy / sum,
        size,
        support,
    }
}

/// Shortest arc of the ring of l sites containing every set bit of `occ` (0 if none): l minus the longest circular
/// run of empty sites.
fn arc_length(occ: u32, l: usize) -> usize {
    if occ == 0 {
        return 0;
    }
    let (mut longest, mut run) = (0, 0);
    for k in 0..2 * l {
        if occ >> (k % l) & 1 == 0 {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    l - longest.min(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_lengths() {
        assert_eq!(arc_length(0b0000_0001, 8), 1);
        assert_eq!(arc_length(0b1000_0001, 8), 2); // sites 7 and 0 are neighbours on the ring
        assert_eq!(arc_length(0b0001_0001, 8), 5);
        assert_eq!(arc_length(0b1111_1111, 8), 8);
    }
}
