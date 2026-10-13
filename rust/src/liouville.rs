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
//! H, the dissipator and ρ(0) are translation invariant, so r_P is constant on translation orbits.
//! The basis used here is one orthonormal vector per orbit O of size p_O, with coordinate
//! v_O = √p_O · r_O. The dimension is ≈ 4^L / L (8230 for L = 8).

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
    pub fn new(params: &Params) -> Self {
        assert!(params.periodic, "the translation sector needs periodic boundaries");
        let terms = hamiltonian_terms(params);
        let sector = Sector::new(params.l);
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

/// Site-averaged magnetization m_α = (1/L) Σ_j Tr[σ^α_j ρ], α = x, y, z, read off the orbit vector v:
/// the orbit of a single-site σ^α has L members, so m_α = v[orbit] / √L.
pub struct Magnetization {
    orbit: [usize; 3],
    norm: f64,
}

impl Magnetization {
    pub fn new(liou: &Liouvillian) -> Self {
        let s = &liou.sector;
        Magnetization { orbit: [0, 1, 2].map(|axis| s.orbit(Pauli::single(axis, 0))), norm: 1.0 / (s.l as f64).sqrt() }
    }

    /// [m_x, m_y, m_z].
    pub fn measure(&self, v: &[f64]) -> [f64; 3] {
        self.orbit.map(|o| v[o] * self.norm)
    }
}
