//! Exact Lindblad evolution of the ferromagnetic scar tower under bulk σᶻ dephasing.
//! The model and conventions are in `CLAUDE.md` at the repository root.
//!
//! - [`pauli`]: Pauli strings as bit masks, products with exact phases.
//! - [`model`]: parameters and the Hamiltonian as a real combination of Pauli strings.
//! - [`liouville`]: the Lindbladian in the translation-invariant Pauli basis, propagation, observables.
//! - [`trajectory`]: pure-state trajectories (diffusive unravelling of the dephasing).

pub mod liouville;
pub mod model;
pub mod pauli;
pub mod trajectory;
