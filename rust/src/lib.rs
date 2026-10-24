//! Exact Lindblad evolution of the ferromagnetic scar tower under bulk σᶻ dephasing.
//! The model and conventions are in `CLAUDE.md` at the repository root.
//!
//! - [`parameters`]: the model coefficients h, a, b, c (edit there).
//! - [`cli`]: command-line options shared by the binaries.
//! - [`entanglement`]: eigenvalues without LAPACK, block and operator entanglement, operator size and support.
//! - [`pauli`]: Pauli strings as bit masks, products with exact phases.
//! - [`model`]: parameters and the Hamiltonian as a real combination of Pauli strings.
//! - [`liouville`]: the Lindbladian in the Pauli basis (translation orbits or full), propagation, observables.
//! - [`noise`]: the single-site noise channels (σᶻ, σˣ, σʸ dephasing, depolarizing, decay, thermal).
//! - [`tower`]: the scar tower |Φ_n⟩ and traces Tr(P_tower P) of Pauli strings.
//! - [`trajectory`]: pure-state trajectories (diffusive unravelling of the dephasing).

pub mod cli;
pub mod entanglement;
pub mod liouville;
pub mod model;
pub mod noise;
pub mod parameters;
pub mod pauli;
pub mod tower;
pub mod trajectory;
