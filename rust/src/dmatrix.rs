//! Exact Lindblad evolution of ρ as a dense 2^L × 2^L complex matrix, ρ[t · 2^L + u] = ⟨t|ρ|u⟩, for Hermitian
//! single-site Pauli channels (σᶻ, σˣ, σʸ dephasing, both depolarizing normalizations):
//!
//!     dρ/dt = −i(Hρ − (Hρ)†) + γ Σ_j Σ_α r_α (σ^α_j ρ σ^α_j − ρ),
//!
//! with r_α the rates per unit of γ of `Channel::hermitian_jumps` (depol: 1/3 each). This is the same generator as
//! `liouville.rs` in another basis (checked in tests/state_hist.rs), with a cheaper application in the full basis:
//! H is applied by its flip masks x, (Hρ)(t, u) = Σ_x c_x(t) ρ(t ⊕ x, u) with c_x(t) = Σ_{strings of x} coef
//! (−1)^{|z & (t ⊕ x)|} (≈ 5L masks: L = 10 costs ≈ 50 row axpys of 2^L per row, against ≈ 300 string tests per
//! Pauli coordinate in the matrix-free `liouville.rs`), and the channel acts element by element:
//! σˣ_j ρ σˣ_j (t, u) = ρ(t ⊕ b, u ⊕ b), σʸ_j ρ σʸ_j (t, u) = s ρ(t ⊕ b, u ⊕ b), σᶻ_j ρ σᶻ_j (t, u) = s ρ(t, u),
//! b = 2^j, s = (−1)^{t_j + u_j}. Memory 16 B · 4^L per matrix (16 MB at L = 10), five matrices.
//!
//! Time stepping: Taylor series of exp(𝓛 dt) summed to round-off, as `liouville::Propagator`, with substeps of
//! ‖𝓛 dt_sub‖ ≤ 4 for the estimate ‖𝓛‖ ≤ 2‖H‖ + γ max_P |D(P)| (‖H‖ by power iteration, × 1.05) instead of the
//! row-sum bound (≈ 2.7× larger at L = 10).

use rayon::prelude::*;

use crate::model::{hamiltonian_terms, Params};
use crate::noise::Channel;
use crate::trajectory::{Amp, Rng};

pub struct DensityLiouvillian {
    pub l: usize,
    d: usize,
    /// (x, c_x(t) for t = 0 … 2^L − 1): H(t, t ⊕ x) = c_x(t).
    groups: Vec<(usize, Vec<Amp>)>,
    /// Channel rates per unit of γ on σˣ, σʸ, σᶻ jumps.
    rates: [f64; 3],
    /// Estimate of ‖H‖ (spectral radius), from above.
    h_norm: f64,
    /// Largest damping rate per unit of γ of a Pauli string: L × the largest single-site rate.
    d_max: f64,
}

impl DensityLiouvillian {
    pub fn new(params: &Params, channel: Channel) -> Self {
        let l = params.l;
        let d = 1usize << l;
        let mut rates = [0.0; 3];
        for (axis, rate) in channel.hermitian_jumps().expect("dmatrix: Hermitian Pauli channels only") {
            rates[axis] += rate;
        }
        let mut by_x: std::collections::BTreeMap<usize, Vec<(usize, Amp)>> = std::collections::BTreeMap::new();
        for (p, h) in hamiltonian_terms(params) {
            let coef = [[h, 0.0], [0.0, h], [-h, 0.0], [0.0, -h]][(p.n_y() % 4) as usize];
            by_x.entry(p.x as usize).or_default().push((p.z as usize, coef));
        }
        let groups: Vec<(usize, Vec<Amp>)> = by_x
            .into_iter()
            .map(|(x, terms)| {
                let c = (0..d)
                    .map(|t| {
                        let s = t ^ x;
                        terms.iter().fold([0.0, 0.0], |acc, &(z, c)| {
                            if (z & s).count_ones() & 1 == 0 {
                                [acc[0] + c[0], acc[1] + c[1]]
                            } else {
                                [acc[0] - c[0], acc[1] - c[1]]
                            }
                        })
                    })
                    .collect();
                (x, c)
            })
            .collect();
        let site_max = channel.site_diag().iter().fold(0.0f64, |m, r| m.max(r.abs()));
        let mut liou = DensityLiouvillian { l, d, groups, rates, h_norm: 0.0, d_max: site_max * l as f64 };
        liou.h_norm = 1.05 * liou.power_norm();
        liou
    }

    /// ψ → Hψ on a state vector.
    fn apply_h(&self, psi: &[Amp], out: &mut [Amp]) {
        out.iter_mut().enumerate().for_each(|(t, o)| {
            *o = self.groups.iter().fold([0.0, 0.0], |acc, (x, c)| {
                let (a, b) = (c[t], psi[t ^ x]);
                [acc[0] + a[0] * b[0] - a[1] * b[1], acc[1] + a[0] * b[1] + a[1] * b[0]]
            })
        });
    }

    /// ‖H‖ by power iteration (from below; 200 steps from a fixed pseudo-random vector).
    fn power_norm(&self) -> f64 {
        let mut rng = Rng::new(0x5eed);
        let mut v: Vec<Amp> = (0..self.d).map(|_| [rng.gaussian(), rng.gaussian()]).collect();
        let mut w = vec![[0.0; 2]; self.d];
        let mut norm = 0.0;
        for _ in 0..200 {
            let s = v.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum::<f64>().sqrt();
            v.iter_mut().for_each(|a| *a = [a[0] / s, a[1] / s]);
            self.apply_h(&v, &mut w);
            norm = w.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum::<f64>().sqrt();
            std::mem::swap(&mut v, &mut w);
        }
        norm
    }

    /// Upper estimate of ‖𝓛‖: the spread of the spectrum of H plus the largest damping rate.
    pub fn norm(&self, gamma: f64) -> f64 {
        2.0 * self.h_norm + gamma * self.d_max
    }

    /// out = 𝓛 ρ; `m` is workspace (Hρ).
    pub fn apply(&self, gamma: f64, rho: &[Amp], out: &mut [Amp], m: &mut [Amp]) {
        let d = self.d;
        m.par_chunks_mut(d).enumerate().for_each(|(t, row)| {
            row.iter_mut().for_each(|a| *a = [0.0, 0.0]);
            for (x, c) in &self.groups {
                let a = c[t];
                let r0 = (t ^ x) * d;
                let src = &rho[r0..r0 + d];
                for (o, b) in row.iter_mut().zip(src) {
                    o[0] += a[0] * b[0] - a[1] * b[1];
                    o[1] += a[0] * b[1] + a[1] * b[0];
                }
            }
        });
        let [rx, ry, rz] = self.rates;
        let total = rx + ry + rz;
        let l = self.l;
        let m: &[Amp] = m;
        out.par_chunks_mut(d).enumerate().for_each(|(t, row)| {
            for (u, o) in row.iter_mut().enumerate() {
                // −i (M − M†)(t, u) with M = Hρ
                let (a, b) = (m[t * d + u], m[u * d + t]);
                let mut acc = [a[1] + b[1], b[0] - a[0]];
                if gamma > 0.0 {
                    let r = rho[t * d + u];
                    let mut dis = [-total * l as f64 * r[0], -total * l as f64 * r[1]];
                    let diff = t ^ u;
                    for j in 0..l {
                        let bit = 1usize << j;
                        let s = if diff & bit == 0 { 1.0 } else { -1.0 };
                        dis[0] += rz * s * r[0];
                        dis[1] += rz * s * r[1];
                        if rx + ry > 0.0 {
                            let f = rx + ry * s;
                            let q = rho[(t ^ bit) * d + (u ^ bit)];
                            dis[0] += f * q[0];
                            dis[1] += f * q[1];
                        }
                    }
                    acc[0] += gamma * dis[0];
                    acc[1] += gamma * dis[1];
                }
                *o = acc;
            }
        });
    }

    /// ρ = |ψ⟩⟨ψ|.
    pub fn pure(&self, psi: &[Amp]) -> Vec<Amp> {
        let d = self.d;
        assert_eq!(psi.len(), d);
        (0..d * d)
            .into_par_iter()
            .map(|k| {
                let (a, b) = (psi[k / d], psi[k % d]);
                [a[0] * b[0] + a[1] * b[1], a[1] * b[0] - a[0] * b[1]]
            })
            .collect()
    }

    /// [⟨σˣ_j⟩, ⟨σʸ_j⟩, ⟨σᶻ_j⟩] for j = 0 … L − 1: Tr(ρσˣ_j) = Σ_t ρ(t, t ⊕ b), Tr(ρσʸ_j) = −Σ_t (−1)^{t_j}
    /// Im ρ(t, t ⊕ b), Tr(ρσᶻ_j) = Σ_t (−1)^{t_j} ρ(t, t).
    pub fn per_site(&self, rho: &[Amp]) -> [Vec<f64>; 3] {
        let d = self.d;
        let mut m = [vec![0.0; self.l], vec![0.0; self.l], vec![0.0; self.l]];
        for j in 0..self.l {
            let bit = 1usize << j;
            for t in 0..d {
                let s = if t & bit == 0 { 1.0 } else { -1.0 };
                let r = rho[t * d + (t ^ bit)];
                m[0][j] += r[0];
                m[1][j] -= s * r[1];
                m[2][j] += s * rho[t * d + t][0];
            }
        }
        m
    }

    /// Tr ρ.
    pub fn trace(&self, rho: &[Amp]) -> f64 {
        (0..self.d).map(|t| rho[t * self.d + t][0]).sum()
    }

    /// Tower weight Σ_n ⟨Φ_n|ρ|Φ_n⟩ = Σ_{|t| = |u|} ρ(t, u) / C(L, |t|).
    pub fn tower_weight(&self, rho: &[Amp]) -> f64 {
        let d = self.d;
        let l = self.l;
        let binom: Vec<f64> = (0..=l).map(|n| (0..n).fold(1.0, |acc, i| acc * (l - i) as f64 / (i + 1) as f64)).collect();
        (0..d)
            .into_par_iter()
            .map(|t| {
                let n = t.count_ones();
                (0..d).filter(|u| u.count_ones() == n).map(|u| rho[t * d + u][0]).sum::<f64>() / binom[n as usize]
            })
            .sum()
    }
}

/// Exact propagation ρ → exp(𝓛 dt) ρ (Taylor series summed to round-off, substeps of estimated norm ≤ 4).
pub struct DensityPropagator<'a> {
    liou: &'a DensityLiouvillian,
    gamma: f64,
    dt_sub: f64,
    pub n_sub: usize,
    term: Vec<Amp>,
    tmp: Vec<Amp>,
    work: Vec<Amp>,
}

impl<'a> DensityPropagator<'a> {
    pub fn new(liou: &'a DensityLiouvillian, gamma: f64, dt: f64) -> Self {
        let n_sub = (liou.norm(gamma) * dt / 4.0).ceil().max(1.0) as usize;
        let n = liou.d * liou.d;
        DensityPropagator {
            liou,
            gamma,
            dt_sub: dt / n_sub as f64,
            n_sub,
            term: vec![[0.0; 2]; n],
            tmp: vec![[0.0; 2]; n],
            work: vec![[0.0; 2]; n],
        }
    }

    pub fn step(&mut self, rho: &mut [Amp]) {
        for _ in 0..self.n_sub {
            self.term.copy_from_slice(rho);
            for order in 1..=80 {
                self.liou.apply(self.gamma, &self.term, &mut self.tmp, &mut self.work);
                let f = self.dt_sub / order as f64;
                let (term_sq, rho_sq) = self
                    .term
                    .par_iter_mut()
                    .zip(self.tmp.par_iter())
                    .zip(rho.par_iter_mut())
                    .map(|((t, x), r)| {
                        *t = [x[0] * f, x[1] * f];
                        r[0] += t[0];
                        r[1] += t[1];
                        (t[0] * t[0] + t[1] * t[1], r[0] * r[0] + r[1] * r[1])
                    })
                    .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                if term_sq <= 1e-36 * rho_sq {
                    break;
                }
                assert!(order < 80, "Taylor series did not converge");
            }
        }
    }
}
