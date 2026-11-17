//! Dense diagonalization of H (Intel MKL `zheevd`) and the γ = 0 diagonal ensemble of the per-site Bloch vectors,
//! n̄^α_j = Σ_E |⟨E|ψ⟩|² ⟨E|σ^α_j|E⟩ (H with random c has no degeneracies, so this is the infinite-time average).
//!
//! Not part of the library (which never links MKL): included with `#[path]` by `state_hist_ed` (linked by build.rs)
//! and by the tests when build.rs finds MKL (`cfg(mkl)`).

use rayon::prelude::*;

use open_exact_scars::model::{hamiltonian_terms, Params};
use open_exact_scars::trajectory::{per_site_expectations, Amp};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

const COL_MAJOR: i32 = 102;

extern "C" {
    fn LAPACKE_zheevd(layout: i32, jobz: u8, uplo: u8, n: i32, a: *mut C64, lda: i32, w: *mut f64) -> i32;
}

/// i^k for k in 0..4, as (re, im).
const I_POW: [(f64, f64); 4] = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];

/// Eigenvalues, eigenvectors and the per-site Bloch vectors ⟨E|σ^α_j|E⟩ of every eigenstate.
pub struct Eigen {
    pub l: usize,
    pub energies: Vec<f64>,
    /// Column-major: entry (s, k) at k · 2^L + s is ⟨s|E_k⟩.
    pub vectors: Vec<C64>,
    /// bloch[k][α][j] = ⟨E_k|σ^α_j|E_k⟩.
    pub bloch: Vec<[Vec<f64>; 3]>,
}

impl Eigen {
    pub fn new(params: &Params) -> Self {
        let l = params.l;
        let n = 1usize << l;
        let mut h = vec![C64::default(); n * n];
        for (p, coef) in hamiltonian_terms(params) {
            for s in 0..n as u32 {
                let (t, k) = p.apply(s);
                let (re, im) = I_POW[k as usize];
                let e = &mut h[s as usize * n + t as usize];
                e.re += coef * re;
                e.im += coef * im;
            }
        }
        let mut energies = vec![0.0; n];
        let info = unsafe { LAPACKE_zheevd(COL_MAJOR, b'V', b'U', n as i32, h.as_mut_ptr(), n as i32, energies.as_mut_ptr()) };
        assert_eq!(info, 0, "zheevd failed");
        let bloch = h
            .par_chunks(n)
            .map(|col| {
                let psi: Vec<Amp> = col.iter().map(|c| [c.re, c.im]).collect();
                per_site_expectations(&psi, l)
            })
            .collect();
        Eigen { l, energies, vectors: h, bloch }
    }

    /// (n̄_j = [⟨σˣ_j⟩, ⟨σʸ_j⟩, ⟨σᶻ_j⟩] of the diagonal ensemble of ψ, Σ_E |⟨E|ψ⟩|², Σ_E |⟨E|ψ⟩|² E).
    pub fn diagonal_ensemble(&self, psi: &[Amp]) -> (Vec<[f64; 3]>, f64, f64) {
        let n = 1usize << self.l;
        assert_eq!(psi.len(), n);
        let weights: Vec<f64> = self
            .vectors
            .par_chunks(n)
            .map(|col| {
                let (mut re, mut im) = (0.0, 0.0);
                for (v, p) in col.iter().zip(psi) {
                    // conj(v) · p
                    re += v.re * p[0] + v.im * p[1];
                    im += v.re * p[1] - v.im * p[0];
                }
                re * re + im * im
            })
            .collect();
        let mut nbar = vec![[0.0; 3]; self.l];
        for (w, b) in weights.iter().zip(&self.bloch) {
            for (j, nj) in nbar.iter_mut().enumerate() {
                for a in 0..3 {
                    nj[a] += w * b[a][j];
                }
            }
        }
        let total = weights.iter().sum();
        let energy = weights.iter().zip(&self.energies).map(|(w, e)| w * e).sum();
        (nbar, total, energy)
    }
}
