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
        assert!((1..=12).contains(&l), "the orbit table has 4^L entries");
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
        assert!((1..=12).contains(&l), "the orbit table has 4^L entries");
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

pub struct Liouvillian {
    pub sector: Sector,
    // Hamiltonian part −i[H, ·] in CSR form (real antisymmetric in the orthonormal orbit basis)
    row_ptr: Vec<usize>,
    col: Vec<u32>,
    val: Vec<f64>,
    /// Number of X/Y sites of each orbit; the dephasing part is −2γ times this, on the diagonal.
    pub n_xy: Vec<f64>,
}

impl Liouvillian {
    /// Translation orbits if `params` is translation invariant, the full Pauli basis otherwise.
    pub fn new(params: &Params) -> Self {
        let sector = if params.translation_invariant() { Sector::new(params.l) } else { Sector::full(params.l) };
        Self::with_sector(params, sector)
    }

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
        Liouvillian { sector, row_ptr, col, val, n_xy }
    }

    pub fn dim(&self) -> usize {
        self.sector.dim()
    }

    /// out = 𝓛 v.
    pub fn apply(&self, gamma: f64, v: &[f64], out: &mut [f64]) {
        // rows in parallel; small sectors stay on one thread
        out.par_iter_mut().enumerate().with_min_len(1024).for_each(|(i, o)| {
            let range = self.row_ptr[i]..self.row_ptr[i + 1];
            let acc: f64 = self.col[range.clone()].iter().zip(&self.val[range]).map(|(&c, &x)| x * v[c as usize]).sum();
            *o = acc - 2.0 * gamma * self.n_xy[i] * v[i];
        });
    }

    /// 𝓛 as a dense real matrix in column-major order (entry (i, c) at c · dim + i), for ED.
    pub fn dense(&self, gamma: f64) -> Vec<f64> {
        let n = self.dim();
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                m[self.col[k] as usize * n + i] += self.val[k];
            }
            m[i * n + i] -= 2.0 * gamma * self.n_xy[i];
        }
        m
    }

    /// Largest absolute row sum (equal to the largest column sum, by antisymmetry).
    pub fn norm_bound(&self, gamma: f64) -> f64 {
        (0..self.dim())
            .map(|i| {
                let h: f64 = self.val[self.row_ptr[i]..self.row_ptr[i + 1]].iter().map(|x| x.abs()).sum();
                h + 2.0 * gamma * self.n_xy[i]
            })
            .fold(0.0, f64::max)
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

/// Exact propagation v → exp(𝓛 dt) v: Taylor series summed to round-off, with dt split into
/// substeps of norm ≤ 1 so that no cancellation occurs.
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
        let n_sub = (liou.norm_bound(gamma) * dt).ceil().max(1.0) as usize;
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
