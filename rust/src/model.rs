//! Model parameters and the Hamiltonian H = H_I + h Sᶻ_tot as a real combination of Pauli strings.
//!
//! H_I = Σ_j [ Σ_α ( a_α R_{j,j+1} σ^α_{j+2} + b_α σ^α_{j−1} R_{j,j+1} )
//!           + Σ_{αβ} c_{αβ} σ^β_{j−1} R_{j,j+1} σ^α_{j+2} ],
//! with the singlet projector R_{j,j+1} = (1 − σ_j·σ_{j+1}) / 4.

use std::collections::BTreeMap;

use crate::parameters;
use crate::pauli::Pauli;
use crate::trajectory::Rng;

#[derive(Clone, Debug)]
pub struct Params {
    pub l: usize,
    /// Lifting field: H|Φ_n⟩ = h (L/2 − n)|Φ_n⟩.
    pub h: f64,
    /// a[α] multiplies R_{j,j+1} σ^α_{j+2}.
    pub a: [f64; 3],
    /// b[α] multiplies σ^α_{j−1} R_{j,j+1}.
    pub b: [f64; 3],
    /// c[α][β] multiplies σ^β_{j−1} R_{j,j+1} σ^α_{j+2}.
    pub c: [[f64; 3]; 3],
    /// Site-dependent c: if set, c_bond[j] replaces c in the term on bond (j, j+1). Breaks translation.
    pub c_bond: Option<Vec<[[f64; 3]; 3]>>,
    /// Periodic ring (needs L ≥ 4). Open chain: a term is kept only if all its sites exist.
    pub periodic: bool,
}

impl Params {
    /// Uniform couplings from `parameters.rs` (the current coefficients of CLAUDE.md).
    pub fn baseline(l: usize) -> Self {
        Params {
            l,
            h: parameters::H,
            a: parameters::A,
            b: parameters::B,
            c: parameters::C,
            c_bond: None,
            periodic: parameters::PERIODIC,
        }
    }

    /// Baseline with random c: on every bond, each c[α][β] is multiplied by an independent factor
    /// uniform in `parameters::C_RANDOM_RANGE` (default [0.5, 1.5]; signs kept, zero entries stay zero).
    /// One uniform number is drawn per entry and bond, zero or not, so a realization depends only on
    /// the seed and L.
    pub fn random_c(l: usize, rng: &mut Rng) -> Self {
        let base = Self::baseline(l);
        let (low, high) = parameters::C_RANDOM_RANGE;
        let c_bond = (0..l).map(|_| base.c.map(|row| row.map(|c| c * (low + (high - low) * rng.uniform())))).collect();
        Params { c_bond: Some(c_bond), ..base }
    }

    pub fn translation_invariant(&self) -> bool {
        self.periodic && self.c_bond.is_none()
    }

    /// H_I = 0: independent spins in the field h (exact reference 2).
    pub fn free_spins(l: usize) -> Self {
        Params { a: [0.0; 3], b: [0.0; 3], c: [[0.0; 3]; 3], c_bond: None, ..Self::baseline(l) }
    }
}

/// H = Σ_a coef_a P_a with real coefficients; identical strings are merged, the identity is dropped.
pub fn hamiltonian_terms(p: &Params) -> Vec<(Pauli, f64)> {
    let l = p.l as isize;
    assert!(p.l <= 32);
    assert!(!p.periodic || p.l >= 4, "periodic boundaries need L >= 4");
    let site = |j: isize| -> Option<usize> {
        if p.periodic {
            Some(j.rem_euclid(l) as usize)
        } else {
            (0..l).contains(&j).then_some(j as usize)
        }
    };

    let mut terms: BTreeMap<Pauli, f64> = BTreeMap::new();
    for j in 0..l {
        let (Some(s1), Some(s2)) = (site(j), site(j + 1)) else { continue };
        // coef · R_{s1,s2} · dressing = (coef / 4) (dressing − Σ_δ σ^δ_{s1} σ^δ_{s2} dressing)
        let mut add_dressed = |coef: f64, dressing: Pauli| {
            if coef == 0.0 {
                return;
            }
            *terms.entry(dressing).or_insert(0.0) += coef / 4.0;
            for delta in 0..3 {
                let bond = Pauli::single(delta, s1).times_disjoint(Pauli::single(delta, s2));
                *terms.entry(bond.times_disjoint(dressing)).or_insert(0.0) -= coef / 4.0;
            }
        };
        let (left, right) = (site(j - 1), site(j + 2));
        let c = p.c_bond.as_ref().map_or(&p.c, |c_bond| &c_bond[j as usize]);
        for alpha in 0..3 {
            if let Some(r) = right {
                add_dressed(p.a[alpha], Pauli::single(alpha, r));
            }
            if let Some(lf) = left {
                add_dressed(p.b[alpha], Pauli::single(alpha, lf));
            }
            if let (Some(lf), Some(r)) = (left, right) {
                for beta in 0..3 {
                    add_dressed(c[alpha][beta], Pauli::single(beta, lf).times_disjoint(Pauli::single(alpha, r)));
                }
            }
        }
    }
    for j in 0..p.l {
        *terms.entry(Pauli::single(2, j)).or_insert(0.0) += p.h / 2.0;
    }
    terms.remove(&Pauli::IDENTITY);
    terms.into_iter().filter(|&(_, coef)| coef != 0.0).collect()
}
