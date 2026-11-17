//! Initial states of the decay-rate histograms (`state_hist`, `state_hist_ed`) and the observable a(t).
//!
//! - Coherent state (scar class): every spin along n = (sin θ cos φ, sin θ sin φ, cos θ). It lies entirely in the
//!   tower, E/L = (h/2) cos θ, and θ = π/2, φ = π is the scar state |−⟩^⊗L.
//! - Bulk product state (bulk class): a Haar-random product state (independent uniform Bloch vectors), projected
//!   with Q = 1 − P_tower and renormalized; kept only if |⟨H⟩/L| ≤ `BULK_E_MAX` (rejection sampling, the next
//!   draw of the same generator otherwise). The generator is seeded with the init seed and L only, so the same
//!   (L, init seed) gives the same state in every binary, independently of the realization seed.
//! - a(t) = (1/L) Σ_j |n⊥_j(t) − n̄⊥_j|, n⊥_j = (⟨σˣ_j⟩, ⟨σʸ_j⟩) and n̄⊥_j its γ = 0 diagonal-ensemble value.

use rayon::prelude::*;

use crate::tower;
use crate::trajectory::{product_amplitudes, random_direction, Amp, Hamiltonian, Rng};

/// Bulk product states are kept only if |⟨H⟩/L| is at most this.
pub const BULK_E_MAX: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    Coherent { theta: f64, phi: f64 },
    BulkProduct { seed: u64 },
}

impl Init {
    /// File-name part: `initcoherent_cos<cos θ>` (plus `_phi<φ>` if φ ≠ π) or `initbulk-product_s<seed>`.
    pub fn tag(&self) -> String {
        match *self {
            Init::Coherent { theta, phi } => {
                let mut c = (theta.cos() * 1e6).round() / 1e6;
                if c == 0.0 {
                    c = 0.0; // no "-0"
                }
                let phi_tag = if (phi - std::f64::consts::PI).abs() < 1e-12 { String::new() } else { format!("_phi{phi}") };
                format!("initcoherent_cos{c}{phi_tag}")
            }
            Init::BulkProduct { seed } => format!("initbulk-product_s{seed}"),
        }
    }
}

/// Bloch vectors of the coherent state: every spin along (sin θ cos φ, sin θ sin φ, cos θ).
pub fn coherent_bloch(l: usize, theta: f64, phi: f64) -> Vec<[f64; 3]> {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    vec![[st * cp, st * sp, ct]; l]
}

/// ⟨ψ|H|ψ⟩ (real part) of a normalized state.
pub fn energy(ham: &Hamiltonian, psi: &[Amp]) -> f64 {
    let mut h_psi = vec![[0.0; 2]; psi.len()];
    ham.apply(psi, &mut h_psi);
    psi.iter().zip(&h_psi).map(|(a, b)| a[0] * b[0] + a[1] * b[1]).sum()
}

/// A bulk product state and how it was found.
pub struct BulkState {
    /// Q|p⟩ / ‖Q|p⟩‖, |p⟩ the accepted product state.
    pub psi: Vec<Amp>,
    /// Bloch vectors of |p⟩.
    pub bloch: Vec<[f64; 3]>,
    /// ⟨ψ|H|ψ⟩.
    pub energy: f64,
    /// Number of product states drawn (the last one accepted).
    pub attempts: usize,
}

/// The bulk product state of `init_seed` for the chain of `ham` (see the module docs).
pub fn bulk_product(ham: &Hamiltonian, init_seed: u64) -> BulkState {
    let l = ham.l;
    let mut rng = Rng::new(init_seed.wrapping_mul(0xd1b54a32d192ed03) ^ ((l as u64) << 56) ^ 0x6a09e667f3bcc908);
    for attempts in 1..=10_000 {
        let bloch: Vec<[f64; 3]> = (0..l).map(|_| random_direction(&mut rng)).collect();
        let mut psi = product_amplitudes(&bloch);
        tower::project_out(&mut psi, l);
        let norm = psi.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum::<f64>().sqrt();
        psi.iter_mut().for_each(|a| *a = [a[0] / norm, a[1] / norm]);
        // a second pass removes the round-off left in the tower by the first
        tower::project_out(&mut psi, l);
        let e = energy(ham, &psi);
        if (e / l as f64).abs() <= BULK_E_MAX {
            return BulkState { psi, bloch, energy: e, attempts };
        }
    }
    panic!("no bulk product state with |E/L| ≤ {BULK_E_MAX} in 10000 draws");
}

/// r_P = ⟨ψ|P|ψ⟩ for every Pauli string P, in the order of `Sector::full` (index x | z << L), so that
/// ρ = |ψ⟩⟨ψ| = 2^{−L} Σ_P r_P P. For each flip mask x, P|s⟩ = i^{|x & z|} (−1)^{|z & s|} |s ⊕ x⟩ gives
/// r_P = Re[i^{|x & z|} Σ_s ψ(s ⊕ x)* ψ(s) (−1)^{|z & s|}], a Walsh–Hadamard transform over s: L · 4^L operations.
pub fn pauli_coordinates(psi: &[Amp], l: usize) -> Vec<f64> {
    let d = 1usize << l;
    assert_eq!(psi.len(), d);
    let columns: Vec<Vec<f64>> = (0..d)
        .into_par_iter()
        .map(|x| {
            let mut f: Vec<Amp> = (0..d)
                .map(|s| {
                    let (a, b) = (psi[s ^ x], psi[s]);
                    [a[0] * b[0] + a[1] * b[1], a[0] * b[1] - a[1] * b[0]]
                })
                .collect();
            let mut h = 1;
            while h < d {
                for i in (0..d).step_by(2 * h) {
                    for j in i..i + h {
                        let (u, v) = (f[j], f[j + h]);
                        f[j] = [u[0] + v[0], u[1] + v[1]];
                        f[j + h] = [u[0] - v[0], u[1] - v[1]];
                    }
                }
                h *= 2;
            }
            (0..d)
                .map(|z| {
                    let g = f[z];
                    match (x & z).count_ones() % 4 {
                        0 => g[0],
                        1 => -g[1],
                        2 => -g[0],
                        _ => g[1],
                    }
                })
                .collect()
        })
        .collect();
    let mut r = vec![0.0; d * d];
    for (x, col) in columns.iter().enumerate() {
        for (z, &v) in col.iter().enumerate() {
            r[x | z << l] = v;
        }
    }
    r
}

/// a = (1/L) Σ_j |n⊥_j − n̄⊥_j| from the per-site ⟨σˣ_j⟩, ⟨σʸ_j⟩ and n̄⊥_j.
pub fn deviation(x: &[f64], y: &[f64], nbar: &[[f64; 2]]) -> f64 {
    let l = x.len();
    (0..l).map(|j| (x[j] - nbar[j][0]).hypot(y[j] - nbar[j][1])).sum::<f64>() / l as f64
}
