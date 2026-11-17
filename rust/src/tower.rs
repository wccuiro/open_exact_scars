//! The scar tower |Φ_n⟩ ∝ (S⁻_tot)ⁿ |↑…↑⟩, n = 0 … L: the Dicke states with n down spins,
//! |Φ_n⟩ = C(L, n)^{−1/2} Σ_{|s| = n} |s⟩ (bit j = 1 for |↓⟩ on site j).

use rayon::prelude::*;

use crate::pauli::Pauli;
use crate::trajectory::Amp;

fn binomial(l: usize, n: usize) -> f64 {
    (0..n).fold(1.0, |acc, i| acc * (l - i) as f64 / (i + 1) as f64)
}

/// The tower as an L + 1 by 2^L real matrix, row n = |Φ_n⟩.
pub fn states(l: usize) -> Vec<Vec<f64>> {
    (0..=l)
        .map(|n| {
            let amp = binomial(l, n).powf(-0.5);
            (0..1u32 << l).map(|s| if s.count_ones() as usize == n { amp } else { 0.0 }).collect()
        })
        .collect()
}

/// For every Pauli string P (index x | z << L, the order of `Sector::full`):
/// Σ_n weight[n] ⟨Φ_n|P|Φ_n⟩ = Tr(A P) with A = Σ_n weight[n] |Φ_n⟩⟨Φ_n|.
/// weight = 1 gives Tr(P_tower P); weight[n] = L/2 − n gives Tr(Sᶻ_tot P_tower P).
pub fn traces(l: usize, weight: &[f64]) -> Vec<f64> {
    assert_eq!(weight.len(), l + 1);
    let w: Vec<f64> = (0..=l).map(|n| weight[n] / binomial(l, n)).collect();
    let mask = (1usize << l) - 1;
    (0..1usize << (2 * l))
        .into_par_iter()
        .map(|code| {
            let p = Pauli { x: (code & mask) as u32, z: (code >> l) as u32 };
            // ⟨Φ_n|P|Φ_n⟩ = Σ_{|s| = |s ⊕ x| = n} i^k(s) / C(L, n); P is Hermitian, so the sum is real
            let mut sum = 0.0;
            for s in 0..1u32 << l {
                let (t, k) = p.apply(s);
                let n = s.count_ones();
                if t.count_ones() == n {
                    sum += w[n as usize] * [1.0, 0.0, -1.0, 0.0][k as usize];
                }
            }
            sum
        })
        .collect()
}

/// ⟨Φ_n|ψ⟩ = C(L, n)^{−1/2} Σ_{|s| = n} ψ(s) for n = 0 … L: one pass over the amplitudes, grouped by Hamming weight.
pub fn overlaps(psi: &[Amp], l: usize) -> Vec<Amp> {
    assert_eq!(psi.len(), 1usize << l);
    let mut sums = vec![[0.0f64; 2]; l + 1];
    for (s, a) in psi.iter().enumerate() {
        let n = s.count_ones() as usize;
        sums[n][0] += a[0];
        sums[n][1] += a[1];
    }
    (0..=l).map(|n| {
        let f = binomial(l, n).powf(-0.5);
        [sums[n][0] * f, sums[n][1] * f]
    })
    .collect()
}

/// Tower weight Σ_n |⟨Φ_n|ψ⟩|² of a state vector.
pub fn weight(psi: &[Amp], l: usize) -> f64 {
    overlaps(psi, l).iter().map(|o| o[0] * o[0] + o[1] * o[1]).sum()
}

/// ψ ← Q ψ with Q = 1 − Σ_n |Φ_n⟩⟨Φ_n| (not renormalized): subtract ⟨Φ_n|ψ⟩ C(L, n)^{−1/2} from every amplitude
/// with Hamming weight n.
pub fn project_out(psi: &mut [Amp], l: usize) {
    let shift: Vec<Amp> = overlaps(psi, l)
        .iter()
        .enumerate()
        .map(|(n, o)| {
            let f = binomial(l, n).powf(-0.5);
            [o[0] * f, o[1] * f]
        })
        .collect();
    for (s, a) in psi.iter_mut().enumerate() {
        let d = shift[s.count_ones() as usize];
        a[0] -= d[0];
        a[1] -= d[1];
    }
}
