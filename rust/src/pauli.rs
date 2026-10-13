//! Pauli strings on up to 32 sites.
//!
//! A string is stored as two bit masks: bit j of `x` is set if site j carries X or Y, bit j of `z`
//! if it carries Z or Y. The operator is the Hermitian tensor product of single-site Pauli matrices,
//! P = i^{|x & z|} X^x Z^z  (per site Y = i X Z).
//!
//! Basis states are bit strings with bit j = 0 for |↑⟩ and 1 for |↓⟩, so σᶻ|s⟩ = (−1)^{s_j}|s⟩.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pauli {
    pub x: u32,
    pub z: u32,
}

impl Pauli {
    pub const IDENTITY: Pauli = Pauli { x: 0, z: 0 };

    /// σ^axis on one site, axis 0, 1, 2 = x, y, z.
    pub fn single(axis: usize, site: usize) -> Pauli {
        let bit = 1u32 << site;
        match axis {
            0 => Pauli { x: bit, z: 0 },
            1 => Pauli { x: bit, z: bit },
            2 => Pauli { x: 0, z: bit },
            _ => panic!("axis must be 0, 1 or 2"),
        }
    }

    /// Tensor product of two strings with disjoint support.
    pub fn times_disjoint(self, other: Pauli) -> Pauli {
        debug_assert_eq!((self.x | self.z) & (other.x | other.z), 0);
        Pauli { x: self.x | other.x, z: self.z | other.z }
    }

    /// Number of Y factors.
    pub fn n_y(self) -> u32 {
        (self.x & self.z).count_ones()
    }

    /// Number of sites carrying X or Y. Dephasing damps the string at rate 2γ per such site.
    pub fn n_xy(self) -> u32 {
        self.x.count_ones()
    }

    pub fn commutes(self, other: Pauli) -> bool {
        ((self.x & other.z).count_ones() + (self.z & other.x).count_ones()) % 2 == 0
    }

    /// Product `self · other = i^k · P`; returns (P, k) with k in 0..4.
    /// k is odd exactly when the two strings anticommute.
    pub fn mul(self, other: Pauli) -> (Pauli, u32) {
        let p = Pauli { x: self.x ^ other.x, z: self.z ^ other.z };
        let k = (self.n_y() + other.n_y() + 3 * p.n_y() + 2 * (self.z & other.x).count_ones()) % 4;
        (p, k)
    }

    /// Action on a basis state: `P|s⟩ = i^k |s'⟩`; returns (s', k).
    pub fn apply(self, s: u32) -> (u32, u32) {
        (s ^ self.x, (self.n_y() + 2 * (self.z & s).count_ones()) % 4)
    }

    /// Translation by one site on a ring of `l` sites (site j → j + 1).
    pub fn translate(self, l: usize) -> Pauli {
        Pauli { x: rotate(self.x, l), z: rotate(self.z, l) }
    }
}

/// Cyclic shift of the lowest `l` bits by one position (bit j → bit j + 1).
pub fn rotate(v: u32, l: usize) -> u32 {
    ((v << 1) | (v >> (l - 1))) & ((1u32 << l) - 1)
}
