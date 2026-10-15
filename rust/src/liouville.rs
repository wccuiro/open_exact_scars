//! The Lindbladian  dρ/dt = −i[H, ρ] + γ Σ_j (σᶻ_j ρ σᶻ_j − ρ)  in the translation-invariant
//! Pauli basis.
//!
//! Write ρ = 2^{−L} Σ_P r_P P with r_P = Tr[ρ P] real. Then
//!
//!   d r_b / dt = Σ_a h_a κ_ab r_{ab} − 2γ n_xy(b) r_b,      i[P_a, P_b] = κ_ab P_{ab},  κ_ab ∈ {0, ±2},
//!
//! so the generator is a REAL matrix: an antisymmetric Hamiltonian part plus a diagonal dephasing
//! part (σᶻ dephasing damps a string at rate 2γ per site carrying X or Y).
//!
//! For the translation-invariant baseline, H, the dissipator and ρ(0) are translation invariant,
//! so r_P is constant on translation orbits. The basis used then is one orthonormal vector per
//! orbit O of size p_O, with coordinate v_O = √p_O · r_O. The dimension is ≈ 4^L / L (8230 for L = 8).
//! Without translation (site-dependent coefficients, open chain) every string is its own orbit
//! ([`Sector::full`]): p_O = 1, v_P = r_P, dimension 4^L.
//!
//! The Hamiltonian part is stored as a sparse matrix, except in the full basis for L ≥ MATRIX_FREE_MIN_L.
//! There the row of string b is generated on the fly from the Hamiltonian terms P_a that anticommute
//! with P_b (column P_a P_b). The sparse matrix has ≈ 16 L · 4^L nonzeros of 12 bytes (40 GB at L = 12,
//! 750 GB at L = 14).

use rayon::prelude::*;

use crate::model::{hamiltonian_terms, Params};
use crate::pauli::Pauli;

/// Translation orbits of Pauli strings on a ring of `l` sites.
pub struct Sector {
    pub l: usize,
    /// Smallest member of each orbit (ordered by the code x | z << L). Orbit 0 is the identity.
    pub reps: Vec<Pauli>,
    /// Orbit sizes.
    pub period: Vec<u32>,
    lookup: Vec<u32>,
}

impl Sector {
    pub fn new(l: usize) -> Self {
        assert!((1..=12).contains(&l), "the orbit table has 4^L entries (translation orbits: L ≤ 12)");
        let mask = (1usize << l) - 1;
        let mut lookup = vec![u32::MAX; 1usize << (2 * l)];
        let (mut reps, mut period) = (Vec::new(), Vec::new());
        for code in 0..lookup.len() {
            if lookup[code] != u32::MAX {
                continue;
            }
            let rep = Pauli { x: (code & mask) as u32, z: (code >> l) as u32 };
            let (mut p, mut size) = (rep, 0);
            loop {
                lookup[p.x as usize | (p.z as usize) << l] = reps.len() as u32;
                size += 1;
                p = p.translate(l);
                if p == rep {
                    break;
                }
            }
            reps.push(rep);
            period.push(size);
        }
        Sector { l, reps, period, lookup }
    }

    /// No symmetry: every Pauli string is its own orbit.
    pub fn full(l: usize) -> Self {
        assert!((1..=MAX_L_FULL).contains(&l), "the full basis has 4^L entries (L ≤ {MAX_L_FULL})");
        let mask = (1usize << l) - 1;
        let n = 1usize << (2 * l);
        let reps = (0..n).map(|code| Pauli { x: (code & mask) as u32, z: (code >> l) as u32 }).collect();
        Sector { l, reps, period: vec![1; n], lookup: (0..n as u32).collect() }
    }

    pub fn dim(&self) -> usize {
        self.reps.len()
    }

    /// Index of the orbit containing `p`.
    pub fn orbit(&self, p: Pauli) -> usize {
        self.lookup[p.x as usize | (p.z as usize) << self.l] as usize
    }
}

/// Largest L of the full basis (4^L coordinates; 2 GB per vector at L = 14).
pub const MAX_L_FULL: usize = 14;
/// From this L on, the full basis applies the Hamiltonian part matrix-free. Laptop (16 threads), one 𝓛v:
/// L = 10 sparse 67 ms, matrix-free 107 ms; L = 11 sparse 1.7 s, matrix-free 0.56 s (and the sparse
/// matrix takes 27 s to build, against 0.4 s).
pub const MATRIX_FREE_MIN_L: usize = 11;

pub struct Liouvillian {
    pub sector: Sector,
    hamiltonian: HamiltonianPart,
    /// Number of X/Y sites of each orbit; the dephasing part is −2γ times this, on the diagonal.
    pub n_xy: Vec<f64>,
}

/// The Hamiltonian part −i[H, ·] (real antisymmetric in the orthonormal orbit basis).
enum HamiltonianPart {
    Csr { row_ptr: Vec<usize>, col: Vec<u32>, val: Vec<f64> },
    /// Full basis only (index of a string = x | z << L): the terms (x, z, number of Y, coefficient) of H.
    MatrixFree { terms: Vec<(u32, u32, u32, f64)> },
}

impl Liouvillian {
    /// Translation orbits if `params` is translation invariant, the full Pauli basis otherwise.
    /// Matrix-free from MATRIX_FREE_MIN_L on in the full basis.
    pub fn new(params: &Params) -> Self {
        if params.translation_invariant() {
            Self::with_sector(params, Sector::new(params.l))
        } else if params.l >= MATRIX_FREE_MIN_L {
            Self::matrix_free(params)
        } else {
            Self::with_sector(params, Sector::full(params.l))
        }
    }

    /// Full basis, Hamiltonian part applied matrix-free.
    pub fn matrix_free(params: &Params) -> Self {
        let sector = Sector::full(params.l);
        let terms = hamiltonian_terms(params).into_iter().map(|(p, h)| (p.x, p.z, p.n_y(), h)).collect();
        let n_xy = sector.reps.par_iter().map(|p| p.n_xy() as f64).collect();
        Liouvillian { sector, hamiltonian: HamiltonianPart::MatrixFree { terms }, n_xy }
    }

    /// Sparse Hamiltonian part in the given sector.
    pub fn with_sector(params: &Params, sector: Sector) -> Self {
        let terms = hamiltonian_terms(params);
        let n = sector.dim();

        let mut row_ptr = Vec::with_capacity(n + 1);
        let (mut col, mut val) = (Vec::new(), Vec::new());
        let mut row: Vec<(u32, f64)> = Vec::new();
        row_ptr.push(0);
        for b in 0..n {
            row.clear();
            let pb = sector.reps[b];
            for &(pa, ha) in &terms {
                if pa.commutes(pb) {
                    continue;
                }
                // P_a P_b = i^k P_c with k odd, so i[P_a, P_b] = 2 i^{k+1} P_c = ∓2 P_c for k = 1, 3
                let (pc, k) = pa.mul(pb);
                let kappa = if k == 1 { -2.0 } else { 2.0 };
                let c = sector.orbit(pc);
                let scale = (sector.period[b] as f64 / sector.period[c] as f64).sqrt();
                row.push((c as u32, ha * kappa * scale));
            }
            row.sort_unstable_by_key(|e| e.0);
            let mut i = 0;
            while i < row.len() {
                let (c, mut sum) = row[i];
                i += 1;
                while i < row.len() && row[i].0 == c {
                    sum += row[i].1;
                    i += 1;
                }
                if sum.abs() > 1e-14 {
                    col.push(c);
                    val.push(sum);
                }
            }
            row_ptr.push(col.len());
        }
        let n_xy = sector.reps.iter().map(|p| p.n_xy() as f64).collect();
        Liouvillian { sector, hamiltonian: HamiltonianPart::Csr { row_ptr, col, val }, n_xy }
    }

    pub fn is_matrix_free(&self) -> bool {
        matches!(self.hamiltonian, HamiltonianPart::MatrixFree { .. })
    }

    /// Calls f(column, value) for every nonzero of row i of the Hamiltonian part.
    #[inline]
    fn for_each_in_row(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        match &self.hamiltonian {
            HamiltonianPart::Csr { row_ptr, col, val } => {
                for k in row_ptr[i]..row_ptr[i + 1] {
                    f(col[k] as usize, val[k]);
                }
            }
            HamiltonianPart::MatrixFree { terms } => {
                let l = self.sector.l;
                let (bx, bz) = ((i & ((1 << l) - 1)) as u32, (i >> l) as u32);
                let b_ny = (bx & bz).count_ones();
                for &(ax, az, a_ny, h) in terms {
                    // anticommute iff |a.x & b.z| + |a.z & b.x| is odd
                    if ((ax & bz) ^ (az & bx)).count_ones() & 1 == 0 {
                        continue;
                    }
                    // P_a P_b = i^k P_c, k = 1 or 3, and i[P_a, P_b] = ∓2 P_c (as in Pauli::mul)
                    let (cx, cz) = (ax ^ bx, az ^ bz);
                    let k = (a_ny + b_ny + 3 * (cx & cz).count_ones() + 2 * (az & bx).count_ones()) % 4;
                    f(cx as usize | (cz as usize) << l, if k == 1 { -2.0 * h } else { 2.0 * h });
                }
            }
        }
    }

    pub fn dim(&self) -> usize {
        self.sector.dim()
    }

    /// out = 𝓛 v.
    pub fn apply(&self, gamma: f64, v: &[f64], out: &mut [f64]) {
        // rows in parallel; small sectors stay on one thread
        out.par_iter_mut().enumerate().with_min_len(1024).for_each(|(i, o)| {
            let mut acc = 0.0;
            self.for_each_in_row(i, |c, x| acc += x * v[c]);
            *o = acc - 2.0 * gamma * self.n_xy[i] * v[i];
        });
    }

    /// 𝓛 as a dense real matrix in column-major order (entry (i, c) at c · dim + i), for ED.
    pub fn dense(&self, gamma: f64) -> Vec<f64> {
        let n = self.dim();
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            self.for_each_in_row(i, |c, x| m[c * n + i] += x);
            m[i * n + i] -= 2.0 * gamma * self.n_xy[i];
        }
        m
    }

    /// Largest absolute row sum (equal to the largest column sum, by antisymmetry).
    pub fn norm_bound(&self, gamma: f64) -> f64 {
        (0..self.dim())
            .into_par_iter()
            .with_min_len(1024)
            .map(|i| {
                let mut h = 0.0;
                self.for_each_in_row(i, |_, x| h += x.abs());
                h + 2.0 * gamma * self.n_xy[i]
            })
            .reduce(|| 0.0, f64::max)
    }

    /// ρ(0) = |ψ₀⟩⟨ψ₀| with |ψ₀⟩ = |−⟩^⊗L: r_P = (−1)^{#X} for strings made of 1 and X only.
    pub fn initial_state(&self) -> Vec<f64> {
        let s = &self.sector;
        (0..s.dim())
            .map(|o| {
                let p = s.reps[o];
                if p.z != 0 {
                    0.0
                } else {
                    (s.period[o] as f64).sqrt() * if p.x.count_ones() % 2 == 0 { 1.0 } else { -1.0 }
                }
            })
            .collect()
    }

    /// ρ(0) = ⊗_j (1 + n_j·σ_j)/2, a product of pure states with Bloch vectors n_j, averaged over
    /// translations (no averaging in the full basis). In the translation sector, H, the dissipator
    /// and the site-averaged magnetization are translation invariant, so the averaged state gives
    /// the same m_α(t) as the state itself.
    /// r_P = Π_j n_j^{a_j}, and the orbit coordinate is √p_O times the mean of r_P over the orbit.
    pub fn product_state(&self, n: &[[f64; 3]]) -> Vec<f64> {
        let s = &self.sector;
        assert_eq!(n.len(), s.l, "one Bloch vector per site");
        (0..s.dim())
            .into_par_iter()
            .map(|o| {
                let (mut p, mut sum) = (s.reps[o], 0.0);
                for _ in 0..s.period[o] {
                    sum += (0..s.l)
                        .map(|j| match ((p.x >> j) & 1, (p.z >> j) & 1) {
                            (0, 0) => 1.0,
                            (1, 0) => n[j][0],
                            (1, 1) => n[j][1],
                            _ => n[j][2],
                        })
                        .product::<f64>();
                    p = p.translate(s.l);
                }
                sum / (s.period[o] as f64).sqrt()
            })
            .collect()
    }
}

/// Largest row-sum norm ‖𝓛 dt_sub‖_∞ of one Taylor substep. The largest term of the series is then
/// at most 4⁴/4! ≈ 11, so at most one digit is lost to cancellation; the row-sum bound is about 3×
/// the actual norm of 𝓛 at small γ. Compared with a bound of 1 this is ≈ 2× faster, with the same
/// results to 1e-14 (L = 8, γ = 0.01 and 10).
const SUBSTEP_NORM: f64 = 4.0;

/// Exact propagation v → exp(𝓛 dt) v: Taylor series summed to round-off, with dt split into
/// substeps of norm ≤ SUBSTEP_NORM.
pub struct Propagator<'a> {
    liou: &'a Liouvillian,
    gamma: f64,
    dt_sub: f64,
    n_sub: usize,
    term: Vec<f64>,
    tmp: Vec<f64>,
}

impl<'a> Propagator<'a> {
    pub fn new(liou: &'a Liouvillian, gamma: f64, dt: f64) -> Self {
        let n_sub = (liou.norm_bound(gamma) * dt / SUBSTEP_NORM).ceil().max(1.0) as usize;
        let n = liou.dim();
        Propagator { liou, gamma, dt_sub: dt / n_sub as f64, n_sub, term: vec![0.0; n], tmp: vec![0.0; n] }
    }

    pub fn step(&mut self, v: &mut [f64]) {
        for _ in 0..self.n_sub {
            self.term.copy_from_slice(v);
            for order in 1..=80 {
                self.liou.apply(self.gamma, &self.term, &mut self.tmp);
                let factor = self.dt_sub / order as f64;
                let (mut term_sq, mut v_sq) = (0.0, 0.0);
                for ((t, &x), y) in self.term.iter_mut().zip(&self.tmp).zip(v.iter_mut()) {
                    *t = x * factor;
                    *y += *t;
                    term_sq += *t * *t;
                    v_sq += *y * *y;
                }
                if term_sq <= 1e-36 * v_sq {
                    break;
                }
                assert!(order < 80, "Taylor series did not converge");
            }
        }
    }
}

/// Magnetization ⟨σ^α_j⟩ = Tr[σ^α_j ρ], α = x, y, z, read off the orbit vector v: the coordinate of
/// the orbit O containing σ^α_j is √p_O · ⟨σ^α_j⟩ (p_O = L in the translation sector, 1 in the full basis).
pub struct Magnetization {
    /// orbit[α][j] of σ^α_j
    orbit: [Vec<usize>; 3],
    /// 1 / √p_O
    norm: [Vec<f64>; 3],
}

impl Magnetization {
    pub fn new(liou: &Liouvillian) -> Self {
        let s = &liou.sector;
        let orbit = [0, 1, 2].map(|axis| (0..s.l).map(|j| s.orbit(Pauli::single(axis, j))).collect::<Vec<_>>());
        let norm = orbit.clone().map(|o| o.iter().map(|&o| 1.0 / (s.period[o] as f64).sqrt()).collect());
        Magnetization { orbit, norm }
    }

    /// Site averages [m_x, m_y, m_z], m_α = (1/L) Σ_j ⟨σ^α_j⟩.
    pub fn measure(&self, v: &[f64]) -> [f64; 3] {
        self.per_site(v).map(|m| m.iter().sum::<f64>() / m.len() as f64)
    }

    /// [⟨σˣ_j⟩, ⟨σʸ_j⟩, ⟨σᶻ_j⟩] for j = 0 … L − 1.
    pub fn per_site(&self, v: &[f64]) -> [Vec<f64>; 3] {
        [0, 1, 2].map(|a| self.orbit[a].iter().zip(&self.norm[a]).map(|(&o, &n)| v[o] * n).collect())
    }
}
