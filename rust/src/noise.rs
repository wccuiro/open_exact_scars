//! Uniform single-site noise channels: dρ/dt = −i[H, ρ] + Σ_j Σ_k (L_jk ρ L_jk† − ½{L_jk† L_jk, ρ}).
//!
//! ```text
//!     z (default)   L_j = √γ σᶻ_j                         dephasing
//!     x, y          L_j = √γ σˣ_j, √γ σʸ_j                 dephasing along x or y
//!     dep           L_jα = √(γ/2) σ^α_j, α = x, y, z      depolarizing (every Pauli factor decays at 2γ)
//!     decay         L_j = √γ σ⁻_j, σ⁻ = |↓⟩⟨↑|            amplitude damping towards |↓⟩
//!     thermal       L_j⁻ = √(γ(n̄+1)) σ⁻_j, L_j⁺ = √(γ n̄) σ⁺_j
//! ```
//!
//! In the Pauli coordinates r_P = Tr(ρ P), dr_P/dt = Tr(𝓓†(P) ρ), and the adjoint dissipator acts on each
//! site factor separately. Per unit of γ, on one site (columns: the factor 𝟙, σˣ, σʸ, σᶻ):
//!
//! ```text
//!     z        0    −2       −2       0
//!     x        0     0       −2      −2
//!     y        0    −2        0      −2
//!     dep      0    −2       −2      −2
//!     decay    0    −1/2     −1/2    −1       plus  σᶻ → −𝟙   (𝓓†(σᶻ) = −σᶻ − 𝟙)
//!     thermal  0    −(2n̄+1)/2 (×2)   −(2n̄+1)  plus  σᶻ → −𝟙
//! ```
//!
//! So a string P is damped at the sum over its sites of the diagonal entries, and for decay and thermal the
//! coordinate of P also receives −γ r_{P'} for every site j where P carries σᶻ, with P' = P with that σᶻ replaced
//! by 𝟙 (the "feed"; it drives ⟨σᶻ_j⟩ to −1 or −1/(2n̄+1)).

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Channel {
    Z,
    X,
    Y,
    Depolarizing,
    Decay,
    Thermal { nbar: f64 },
}

impl Channel {
    /// `name`: z, x, y, dep, decay or thermal; `nbar` is used by thermal only.
    pub fn parse(name: &str, nbar: f64) -> Self {
        match name {
            "z" => Channel::Z,
            "x" => Channel::X,
            "y" => Channel::Y,
            "dep" => Channel::Depolarizing,
            "decay" => Channel::Decay,
            "thermal" => {
                assert!(nbar >= 0.0, "--nbar must be ≥ 0");
                Channel::Thermal { nbar }
            }
            _ => panic!("--channel z, x, y, dep, decay or thermal, got {name}"),
        }
    }

    /// Added to output file names: empty for the default z dephasing.
    pub fn suffix(&self) -> String {
        match self {
            Channel::Z => String::new(),
            Channel::X => "_chx".into(),
            Channel::Y => "_chy".into(),
            Channel::Depolarizing => "_chdep".into(),
            Channel::Decay => "_chdecay".into(),
            Channel::Thermal { nbar } => format!("_chthermal_n{nbar}"),
        }
    }

    /// Diagonal rate of the site factors σˣ, σʸ, σᶻ per unit of γ (the identity is not damped).
    pub fn site_diag(&self) -> [f64; 3] {
        match *self {
            Channel::Z => [-2.0, -2.0, 0.0],
            Channel::X => [0.0, -2.0, -2.0],
            Channel::Y => [-2.0, 0.0, -2.0],
            Channel::Depolarizing => [-2.0, -2.0, -2.0],
            Channel::Decay => [-0.5, -0.5, -1.0],
            Channel::Thermal { nbar } => [-(2.0 * nbar + 1.0) / 2.0, -(2.0 * nbar + 1.0) / 2.0, -(2.0 * nbar + 1.0)],
        }
    }

    /// Coefficient (per unit of γ) of r_{P'} in dr_P/dt for each σᶻ of P replaced by 𝟙; 0 for Hermitian channels.
    pub fn feed(&self) -> f64 {
        match self {
            Channel::Decay | Channel::Thermal { .. } => -1.0,
            _ => 0.0,
        }
    }

    /// Diagonal of the dissipator (per unit of γ) for a string with masks (x, z) (bit j of x: σˣ or σʸ on site j,
    /// bit j of z: σᶻ or σʸ).
    pub fn diag(&self, x: u32, z: u32) -> f64 {
        let [dx, dy, dz] = self.site_diag();
        dx * (x & !z).count_ones() as f64 + dy * (x & z).count_ones() as f64 + dz * (z & !x).count_ones() as f64
    }

    /// The Hermitian jump operators as (axis 0, 1, 2 = x, y, z; rate per unit of γ), for the diffusive
    /// unravelling; None for decay and thermal (not Hermitian).
    pub fn hermitian_jumps(&self) -> Option<Vec<(usize, f64)>> {
        match self {
            Channel::Z => Some(vec![(2, 1.0)]),
            Channel::X => Some(vec![(0, 1.0)]),
            Channel::Y => Some(vec![(1, 1.0)]),
            Channel::Depolarizing => Some(vec![(0, 0.5), (1, 0.5), (2, 0.5)]),
            Channel::Decay | Channel::Thermal { .. } => None,
        }
    }
}
