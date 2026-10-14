//! Model parameters: edit the values here and rerun (`cargo run` recompiles automatically).
//! Every binary builds H from these through `Params::baseline` (uniform couplings) or
//! `Params::random_c` (c randomized per bond).
//!
//!   H = H_I + h Sᶻ_tot
//!   H_I = Σ_j [ Σ_α ( a_α R_{j,j+1} σ^α_{j+2} + b_α σ^α_{j−1} R_{j,j+1} )
//!             + Σ_{α,β} c_{αβ} σ^β_{j−1} R_{j,j+1} σ^α_{j+2} ],      R_{j,j+1} = 1/4 − S_j·S_{j+1}
//!
//! Every term of H_I annihilates the ferromagnetic tower for any values, so the scars survive any
//! choice here; h sets the tower spacing, H|Φ_n⟩ = h (L/2 − n)|Φ_n⟩.
//! L, the evolution time, dt, γ and the seed are command-line options of the binaries.

/// Lifting field h (tower spacing and precession frequency).
pub const H: f64 = 1.0;

/// a = (a_x, a_y, a_z): a_α multiplies R_{j,j+1} σ^α_{j+2}.
pub const A: [f64; 3] = [1.0, 1.0, 1.0];

/// b = (b_x, b_y, b_z): b_α multiplies σ^α_{j−1} R_{j,j+1}.
pub const B: [f64; 3] = [1.0, 1.0, 1.0];

/// C[α][β] = c_{αβ} multiplies σ^β_{j−1} R_{j,j+1} σ^α_{j+2}.
/// Rows α (the right dressing σ^α_{j+2}), columns β (the left dressing σ^β_{j−1}), order x, y, z.
pub const C: [[f64; 3]; 3] = [
    //  β = x   β = y   β = z
    [1.00, 0.00, 0.00], // α = x   (c_xx)
    [0.00, 1.00, 0.00], // α = y   (c_yy)
    [0.00, 0.00, 1.00], // α = z   (c_zz)
];

/// Random c (`Params::random_c`): on every bond, each c_{αβ} is multiplied by an independent factor
/// uniform in [low, high]. Zero entries stay zero.
pub const C_RANDOM_RANGE: (f64, f64) = (0.5, 1.5);

/// Periodic ring (needs L ≥ 4). false: open chain, a term is kept only if all its sites exist.
/// The translation-orbit basis needs periodic boundaries; open chains use the full Pauli basis.
pub const PERIODIC: bool = true;
