# open_exact_scars

Lindblad evolution of the ferromagnetic scar tower under bulk σᶻ dephasing (model and conventions
in `CLAUDE.md`), starting from |−⟩^⊗L and measuring ⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩. Three methods:

```
rust/           exact evolution of ρ (default binary) and pure-state trajectories (--bin trajectories)
julia/          ITensor: ρ as an MPS in the Pauli basis, evolved with TDVP
plot_scripts/   Python: figures, fits and tables from the CSV files
data/           generated output, not tracked by git
```

```bash
cd rust
cargo run --release                          # exact: L = 4 … 8, γ = 0 (check) and γ = 0.01, t up to 100
cargo run --release -- --lmin 9 --lmax 10    # larger sizes: about 8 and 11 minutes
cargo run --release --bin trajectories       # trajectories: L = 4 … 8, 1000 per size
cargo test --release                         # the two exact references of CLAUDE.md

cd ../julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=. lindblad_tdvp.jl 8 16,32,64         # TDVP at L = 8 for three bond dimensions

cd ../plot_scripts                           # needs numpy, pandas, matplotlib
python plot_evolution.py                     # ⟨σˣ⟩, ⟨σʸ⟩, ⟨σᶻ⟩ against time, one curve per L
python fit_decay.py                          # exponential fit per L, then the rate against L
python compare_methods.py                    # TDVP and trajectories against the exact evolution
python resource_tables.py                    # time and RAM estimates for the three methods
```

Options are listed at the top of each program. The Rust binaries write to `data/`, or to
`$SCARS_DATA_DIR` if it is set.

Methods:
- **Exact**: ρ is expanded in Pauli strings, where the Lindbladian is a real matrix and the dephasing
  is diagonal; translation invariance reduces the dimension to about 4^L / L. The propagator is
  applied as a Taylor series summed to round-off.
- **Trajectories**: the jump operators σᶻ_j are Hermitian, so the dephasing is unravelled as a
  fluctuating z field on every spin: each trajectory is a normalized state evolving under H with
  random z rotations, and the average over trajectories obeys the Lindblad equation.
- **TDVP**: the same Pauli representation as an MPS with 4 states per site and a real Lindbladian
  MPO (periodic boundaries as long-range terms), two-site TDVP with a maximum bond dimension.
